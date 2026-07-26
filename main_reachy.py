"""
Reachy 2019 — yeux + assistance horizontale de la tête.
Version de retour stable : Kalman conservé, affichage du scan virtuel retiré.

Pipeline conserve :
- couleur seule (ex. "vert") : OpenCV HSV uniquement ;
- objet seul (ex. "chapeau") : YOLO26n-seg + MobileCLIP ;
- objet + couleur (ex. "chapeau vert") : OpenCV HSV + YOLO/MobileCLIP.

Commande robot :
- les yeux réels sont activés automatiquement, comme dans l'ancien code fonctionnel ;
- la tête complète horizontalement le mouvement des yeux ;
- le mouvement vertical de la tête reste désactivé ;
- l'option --dry-run permet de tester sans moteur ;
- aucune commande de bras ou de main n'est envoyée.

Le fichier doit être placé dans le même dossier que head_control.py.
La caméra des yeux de Reachy est ouverte comme une webcam OpenCV.
Fermer Cheese et tout programme utilisant la caméra ou /dev/ttyUSB0 avant le lancement.

Touches :
  t : saisir une cible libre
  p : cell phone
  k : keyboard
  b : bottle
  c : cup
  h : recentrer les yeux et la tête, puis arrêter la cible
  r : réinitialiser la cible sans déplacer le robot

Affichage :
- le panneau de droite montre la demande utilisateur exacte ;
- la cible comprise par le programme est affichée séparément.
  q : quitter
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import torch
    from PIL import Image
    import mobileclip
except Exception as exc:  # MobileCLIP reste optionnel pour tester YOLO seul.
    torch = None
    Image = None
    mobileclip = None
    MOBILECLIP_IMPORT_ERROR = exc
else:
    MOBILECLIP_IMPORT_ERROR = None

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None


# -----------------------------------------------------------------------------
# Configuration générale
# -----------------------------------------------------------------------------
CAMERA_ID = 0
YOLO_MODEL = "yolo26n-seg.pt"
YOLO_CONF = 0.20
YOLO_IMGSZ = 416
DETECT_EVERY = 3

WINDOW_NAME = "Reachy - yeux + tete horizontale + OpenCV/YOLO/MobileCLIP"

# Le flux caméra reste affiché en 960 x 720 environ.
# Une colonne séparée est ajoutée à droite pour les indices sémantiques.
SEMANTIC_PANEL_W = 300
WINDOW_W = 1410
WINDOW_H = 720

# -----------------------------------------------------------------------------
# Contrôle Reachy 2019 — yeux + assistance horizontale de la tête
# -----------------------------------------------------------------------------
EYE_KP_X = 0.040
EYE_KP_Y = 0.040
EYE_DEADZONE_X = 45
EYE_DEADZONE_Y = 45
EYE_MIN_STEP = 1
EYE_MAX_STEP = 8
EYE_CONTROL_COOLDOWN = 0.040

# Scan lent et prudent lorsque la cible n'est pas encore trouvée.
EYE_SCAN_ENABLED = True
EYE_SCAN_STEP = 7
EYE_SCAN_COOLDOWN = 0.040
EYE_SCAN_HORIZONTAL_COUNT = 45
EYE_SCAN_VERTICAL_COUNT = 30

# À modifier seulement si les yeux partent dans le sens opposé sur le robot.
EYE_INVERT_X = False
EYE_INVERT_Y = False

# Assistance de la tête reprise du code qui fonctionnait au laboratoire.
# La tête complète horizontalement les yeux ; son axe vertical reste désactivé.
HEAD_ASSIST_ENABLED = True
HEAD_VERTICAL_ENABLED = False
HEAD_INVERT_X = False
HEAD_INVERT_Y = False

HEAD_SPEED = 60
HEAD_KP_X = 0.030
HEAD_MIN_STEP = 2
HEAD_MAX_STEP = 12
HEAD_DEADZONE_X = 50
HEAD_TRACK_COOLDOWN = 0.060

HEAD_SCAN_STEP = 6
HEAD_SCAN_COOLDOWN = 1.10
HEAD_H_LIMIT = 220
HEAD_V_LIMIT = 80

# MobileCLIP S2 : compromis précision/vitesse.
MOBILECLIP_ENABLED = True
MOBILECLIP_MODEL = "mobileclip_s2"
MOBILECLIP_REPO = "apple/MobileCLIP-S2"
MOBILECLIP_FILENAME = "mobileclip_s2.pt"
MOBILECLIP_LOCAL_PATH = "ml-mobileclip/checkpoints/mobileclip_s2.pt"
MOBILECLIP_AUTO_DOWNLOAD = True

# MobileCLIP n'est pas lancé à chaque frame.
MOBILECLIP_EVERY = 12
MOBILECLIP_MAX_OBJECTS = 6
MOBILECLIP_MIN_TARGET_SCORE = 0.28
MOBILECLIP_MIN_MARGIN = 0.035
MOBILECLIP_CACHE_TIME = 4.0

# Stabilisation de la décision MobileCLIP sans tracker.
MOBILECLIP_MULTIVIEW = True
SCORE_WEIGHT_SEMANTIC = 0.60
SCORE_WEIGHT_MARGIN = 0.15
SCORE_WEIGHT_POSITION = 0.07
SCORE_WEIGHT_SIZE = 0.04
SCORE_WEIGHT_YOLO = 0.04
SCORE_WEIGHT_REGION = 0.10

# Une requête open-vocabulary vise ici un objet localisable.
# Une grande box "person" peut contenir l'objet sans le localiser : on la refuse.
MOBILECLIP_EXCLUDED_CONTEXT_CLASSES = {"person"}
MOBILECLIP_IDEAL_AREA_RATIO = 0.08
MOBILECLIP_LARGE_BOX_RATIO = 0.30

# False = pas de logs MobileCLIP à chaque analyse dans le terminal.
# Mets True seulement pour déboguer les scores.
MOBILECLIP_DEBUG = False

TARGET_LOCK_TIME = 2.5
EDGE_MARGIN = 25

# Mouvement apparent de la cible dans l'image.
# Comme la caméra est placée sur Reachy, cette vitesse combine le déplacement
# de l'objet et les mouvements éventuels des yeux/de la tête.
MOTION_EMA_ALPHA = 0.35
MOTION_STOP_THRESHOLD = 12.0
MOTION_SLOW_THRESHOLD = 70.0
MOTION_MEDIUM_THRESHOLD = 180.0
MOTION_MAX_DT = 1.0

# Détection de couleur OpenCV : cette route ne lance aucune inférence YOLO
# ni MobileCLIP lorsque la cible saisie est une couleur.
COLOR_DETECT_EVERY = 1
COLOR_MIN_AREA_RATIO = 0.0025
COLOR_MAX_AREA_RATIO = 0.72
COLOR_MIN_BOX_SIDE = 12
COLOR_MORPH_KERNEL = 5

# Requête hybride objet + couleur.
# Un candidat doit contenir une proportion minimale de la couleur demandée.
HYBRID_COLOR_MIN_RATIO = 0.14
HYBRID_NEUTRAL_COLOR_MIN_RATIO = 0.22
HYBRID_MAX_COLOR_REGIONS = 8

# Plages HSV OpenCV (H dans [0, 179]). Certaines couleurs ont plusieurs plages.
COLOR_HSV_RANGES = {
    "rouge": [((0, 85, 55), (9, 255, 255)), ((170, 85, 55), (179, 255, 255))],
    "orange": [((9, 90, 55), (22, 255, 255))],
    "jaune": [((22, 75, 70), (36, 255, 255))],
    "vert": [((36, 55, 40), (86, 255, 255))],
    "bleu": [((86, 65, 40), (130, 255, 255))],
    "violet": [((130, 50, 40), (156, 255, 255))],
    "rose": [((156, 45, 65), (170, 255, 255))],
    "marron": [((5, 60, 25), (25, 255, 170))],
    "noir": [((0, 0, 0), (179, 255, 58))],
    "gris": [((0, 0, 58), (179, 48, 205))],
    "blanc": [((0, 0, 190), (179, 48, 255))],
}

QUERY_TRANSLATIONS = {
    "pingouin": "penguin",
    "manette": "game controller",
    "manette ps4": "PlayStation 4 controller",
    "ps4": "PlayStation 4 controller",
    "playstation": "PlayStation console",
    "playstation 4": "PlayStation 4 console",
    "chapeau": "hat",
    "casquette": "cap",
    "peluche": "stuffed toy",
    "stylo": "pen",
    "lunettes": "glasses",
    "montre": "wristwatch",
    "tondeuse": "electric hair clipper",
}

ALIASES = {
    "phone": "cell phone",
    "cellphone": "cell phone",
    "cell phone": "cell phone",
    "telephone": "cell phone",
    "téléphone": "cell phone",
    "portable": "cell phone",
    "smartphone": "cell phone",
    "clavier": "keyboard",
    "keyboard": "keyboard",
    "souris": "mouse",
    "mouse": "mouse",
    "laptop": "laptop",
    "ordinateur": "laptop",
    "ordi": "laptop",
    "pc": "laptop",
    "bottle": "bottle",
    "water bottle": "bottle",
    "bouteille": "bottle",
    "gourde": "bottle",
    "cup": "cup",
    "tasse": "cup",
    "book": "book",
    "livre": "book",
    "chair": "chair",
    "chaise": "chair",
    "person": "person",
    "personne": "person",
    "remote": "remote",
    "telecommande": "remote",
    "télécommande": "remote",
    "teddy": "teddy bear",
    "teddy bear": "teddy bear",
    "ours": "teddy bear",
    "ours en peluche": "teddy bear",
    "nounours": "teddy bear",
}


# -----------------------------------------------------------------------------
# Utilitaires de cible/couleur
# -----------------------------------------------------------------------------
COLOR_WORDS = {
    "rouge": "rouge", "rouges": "rouge", "red": "rouge",
    "bleu": "bleu", "bleue": "bleu", "bleus": "bleu", "bleues": "bleu", "blue": "bleu",
    "vert": "vert", "verte": "vert", "verts": "vert", "vertes": "vert", "green": "vert",
    "jaune": "jaune", "jaunes": "jaune", "yellow": "jaune",
    "orange": "orange", "oranges": "orange",
    "noir": "noir", "noire": "noir", "noirs": "noir", "noires": "noir", "black": "noir",
    "blanc": "blanc", "blanche": "blanc", "blancs": "blanc", "blanches": "blanc", "white": "blanc",
    "gris": "gris", "grise": "gris", "grises": "gris", "gray": "gris", "grey": "gris",
    "rose": "rose", "roses": "rose", "pink": "rose",
    "violet": "violet", "violette": "violet", "violets": "violet", "violettes": "violet", "purple": "violet",
    "marron": "marron", "marrons": "marron", "brown": "marron",
}

LEADING_WORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "l", "the", "a", "an",
    "objet", "object", "moi", "me",
}


def _clean_free_text(text: str) -> str:
    raw = text.strip().lower().replace("’", "'")
    raw = raw.replace("-", " ")
    raw = re.sub(r"[^a-zà-ÿ0-9' ]+", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()

    prefixes = (
        "détecte moi ", "detecte moi ", "détecte ", "detecte ", "detect ",
        "trouve moi ", "trouve ", "cherche moi ", "cherche ",
    )
    for prefix in prefixes:
        if raw.startswith(prefix):
            raw = raw[len(prefix):].strip()
            break
    return raw


def _strip_leading_words(words: list[str]) -> list[str]:
    while words and words[0] in LEADING_WORDS:
        words.pop(0)
    return words


def normalize_target(text: str) -> str:
    """Encode trois modes : objet, couleur seule, ou objet + couleur."""
    raw = _clean_free_text(text)
    words = raw.split()

    found_color: str | None = None
    object_words: list[str] = []
    for word in words:
        if word in COLOR_WORDS and found_color is None:
            found_color = COLOR_WORDS[word]
        elif word not in COLOR_WORDS:
            object_words.append(word)

    object_words = _strip_leading_words(object_words)
    object_phrase = " ".join(object_words).strip()

    if found_color and not object_phrase:
        return "color:" + found_color

    normalized_object = ALIASES.get(object_phrase, object_phrase)
    if found_color and normalized_object:
        # Le troisième champ conserve le texte objet lisible pour l'affichage.
        return f"hybrid|{normalized_object}|{found_color}|{object_phrase}"

    return normalized_object


def is_color_target(target: str | None) -> bool:
    return isinstance(target, str) and target.startswith("color:")


def is_hybrid_target(target: str | None) -> bool:
    return isinstance(target, str) and target.startswith("hybrid|")


def hybrid_target_parts(target: str) -> tuple[str, str, str]:
    parts = target.split("|", 3)
    if len(parts) != 4:
        return "", "", ""
    return parts[1], parts[2], parts[3]


def hybrid_object_name(target: str | None) -> str:
    return hybrid_target_parts(target)[0] if is_hybrid_target(target) else ""


def hybrid_color_name(target: str | None) -> str:
    return hybrid_target_parts(target)[1] if is_hybrid_target(target) else ""


def target_display_name(target: str | None) -> str:
    if target is None:
        return "-"
    if is_color_target(target):
        return target.split(":", 1)[1]
    if is_hybrid_target(target):
        object_name, color_name, display_object = hybrid_target_parts(target)
        return f"{display_object or object_name} {color_name}".strip()
    return target


def build_opencv_color_mask(frame: np.ndarray, color_name: str) -> np.ndarray:
    """Construit un masque HSV nettoyé pour une couleur donnée."""
    ranges = COLOR_HSV_RANGES.get(color_name)
    height, width = frame.shape[:2]
    if not ranges:
        return np.zeros((height, width), dtype=np.uint8)

    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    raw_mask = np.zeros((height, width), dtype=np.uint8)

    for lower, upper in ranges:
        raw_mask = cv2.bitwise_or(
            raw_mask,
            cv2.inRange(
                hsv,
                np.array(lower, dtype=np.uint8),
                np.array(upper, dtype=np.uint8),
            ),
        )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (COLOR_MORPH_KERNEL, COLOR_MORPH_KERNEL),
    )
    clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return clean_mask


def build_opencv_color_objects(
    frame: np.ndarray,
    color_name: str,
) -> list[dict[str, Any]]:
    """Segmente les zones de la couleur demandée uniquement avec OpenCV/HSV.

    Cette fonction ne dépend ni de YOLO ni de MobileCLIP. Elle renvoie des
    dictionnaires compatibles avec le reste du pipeline pour conserver le
    panneau d'indices sémantiques et le viseur virtuel.
    """
    if color_name not in COLOR_HSV_RANGES:
        return []

    height, width = frame.shape[:2]
    frame_area = max(1, height * width)
    clean_mask = build_opencv_color_mask(frame, color_name)

    contours, _ = cv2.findContours(
        clean_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    min_area = max(400.0, COLOR_MIN_AREA_RATIO * frame_area)
    max_area = COLOR_MAX_AREA_RATIO * frame_area
    objects: list[dict[str, Any]] = []

    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < min_area or contour_area > max_area:
            continue

        x, y, box_w, box_h = cv2.boundingRect(contour)
        if box_w < COLOR_MIN_BOX_SIDE or box_h < COLOR_MIN_BOX_SIDE:
            continue

        component_mask_u8 = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(component_mask_u8, [contour], -1, 255, thickness=cv2.FILLED)
        # Intersection avec le masque HSV pour éviter d'inclure les trous/remplissages
        # qui n'appartiennent pas réellement à la couleur recherchée.
        component_mask_u8 = cv2.bitwise_and(component_mask_u8, clean_mask)
        component_mask = component_mask_u8 > 0
        area_px = int(component_mask.sum())
        if area_px < min_area:
            continue

        x1, y1 = int(x), int(y)
        x2, y2 = min(width - 1, x + box_w), min(height - 1, y + box_h)
        features = extract_mask_features(frame, component_mask, [x1, y1, x2, y2])

        box_area = max(1, box_w * box_h)
        fill_ratio = float(area_px / box_area)
        area_ratio = float(area_px / frame_area)
        # Score OpenCV indicatif : densité de couleur + taille raisonnable.
        size_term = min(1.0, area_ratio / 0.12)
        large_penalty = max(0.0, (area_ratio - 0.45) / 0.27)
        color_score = float(np.clip(0.72 * fill_ratio + 0.28 * size_term - 0.35 * large_penalty, 0.0, 1.0))

        features["color_label"] = color_name
        features["color_ratio"] = fill_ratio
        features["opencv_color_score"] = color_score

        objects.append(
            {
                "id": len(objects) + 1,
                "source": "OpenCV",
                "classe": f"zone {color_name}",
                "conf": color_score,
                "mask": component_mask,
                "box": [x1, y1, x2, y2],
                "center": features["center"],
                "position": position_label(*features["center"], width, height),
                "features": features,
                "area": area_px,
                "color": color_name,
                "opencv_color_score": color_score,
            }
        )

    return objects


def select_opencv_color_object(
    objects: list[dict[str, Any]],
    target_lock: "TargetLock",
    frame_shape: tuple[int, ...],
) -> dict[str, Any] | None:
    """Choisit une zone colorée en privilégiant surface et continuité spatiale."""
    if not objects:
        return None

    frame_area = max(1.0, float(frame_shape[0] * frame_shape[1]))

    def score(obj: dict[str, Any]) -> float:
        continuity = target_lock.continuity(obj, frame_shape)
        area_ratio = float(obj.get("area", 0.0)) / frame_area
        area_score = min(1.0, area_ratio / 0.12)
        return (
            0.52 * float(obj.get("opencv_color_score", 0.0))
            + 0.23 * area_score
            + 0.18 * continuity["position"]
            + 0.07 * continuity["size"]
        )

    return max(objects, key=score)


def hybrid_color_threshold(color_name: str) -> float:
    if color_name in {"noir", "blanc", "gris", "marron"}:
        return HYBRID_NEUTRAL_COLOR_MIN_RATIO
    return HYBRID_COLOR_MIN_RATIO


def annotate_objects_with_requested_color(
    objects: list[dict[str, Any]],
    color_mask_u8: np.ndarray,
    color_name: str,
) -> list[dict[str, Any]]:
    """Ajoute à chaque proposition la proportion HSV de la couleur demandée."""
    annotated: list[dict[str, Any]] = []
    color_mask = color_mask_u8 > 0

    for obj in objects:
        mask = obj.get("mask")
        if mask is None or mask.shape != color_mask.shape:
            x1, y1, x2, y2 = obj["box"]
            mask = np.zeros_like(color_mask, dtype=bool)
            mask[y1:y2 + 1, x1:x2 + 1] = True

        object_area = int(mask.sum())
        if object_area <= 0:
            continue

        overlap_px = int(np.logical_and(mask, color_mask).sum())
        ratio = float(overlap_px / object_area)

        candidate = obj.copy()
        features = dict(obj.get("features", {}))
        features["requested_color"] = color_name
        features["requested_color_ratio"] = ratio
        candidate["features"] = features
        candidate["hybrid_color"] = color_name
        candidate["hybrid_color_ratio"] = ratio
        candidate["hybrid_mode"] = True
        annotated.append(candidate)

    return annotated


def filter_objects_by_requested_color(
    objects: list[dict[str, Any]],
    color_name: str,
) -> list[dict[str, Any]]:
    threshold = hybrid_color_threshold(color_name)
    return [
        obj for obj in objects
        if float(obj.get("hybrid_color_ratio", 0.0)) >= threshold
    ]


def prepare_color_regions_for_hybrid(
    frame: np.ndarray,
    color_name: str,
) -> list[dict[str, Any]]:
    """Transforme les composantes OpenCV colorées en régions candidates MobileCLIP."""
    regions = build_opencv_color_objects(frame, color_name)
    regions = sorted(
        regions,
        key=lambda obj: float(obj.get("opencv_color_score", 0.0)),
        reverse=True,
    )[:HYBRID_MAX_COLOR_REGIONS]

    prepared: list[dict[str, Any]] = []
    for region in regions:
        candidate = region.copy()
        features = dict(region.get("features", {}))
        ratio = float(features.get("color_ratio", 1.0))
        features["requested_color"] = color_name
        features["requested_color_ratio"] = ratio
        candidate["features"] = features
        candidate["hybrid_color"] = color_name
        candidate["hybrid_color_ratio"] = ratio
        candidate["hybrid_mode"] = True
        candidate["hybrid_fallback_region"] = True
        candidate["source"] = "OpenCV+MobileCLIP"
        prepared.append(candidate)
    return prepared


def find_hybrid_yolo_target(
    objects: list[dict[str, Any]],
    object_name: str,
    target_lock: "TargetLock",
    frame_shape: tuple[int, ...],
) -> dict[str, Any] | None:
    """Sélectionne une classe YOLO exacte parmi les objets ayant la bonne couleur."""
    candidates = [obj for obj in objects if obj.get("classe") == object_name]
    if not candidates:
        return None

    def score(obj: dict[str, Any]) -> float:
        continuity = target_lock.continuity(obj, frame_shape)
        return (
            0.56 * float(obj.get("conf", 0.0))
            + 0.28 * float(obj.get("hybrid_color_ratio", 0.0))
            + 0.10 * continuity["position"]
            + 0.06 * continuity["size"]
        )

    selected = max(candidates, key=score).copy()
    selected["hybrid_source"] = "YOLO+OpenCV"
    return selected


def color_label_from_rgb(rgb: list[int]) -> str:
    r, g, b = rgb
    maxc = max(r, g, b)
    minc = min(r, g, b)

    if maxc < 55:
        return "noir"
    if minc > 205:
        return "blanc"
    if maxc - minc < 30:
        return "gris"
    if r > 150 and g < 120 and b < 120:
        return "rouge"
    if g > 120 and r < 140 and b < 140:
        return "vert"
    if b > 130 and r < 140 and g < 150:
        return "bleu"
    if r > 170 and g > 130 and b < 120:
        return "jaune"
    if r > 170 and 60 < g < 160 and b < 120:
        return "orange"
    if r > 150 and b > 120 and g < 130:
        return "rose" if r > b + 30 else "violet"
    if r > 90 and g > 45 and b < 80:
        return "marron"
    return "neutre"


def dominant_color_from_pixels(pixels_bgr: np.ndarray) -> tuple[str, float]:
    """Estime la couleur dominante avec HSV, plus robuste qu'une simple moyenne RGB."""
    if pixels_bgr is None or len(pixels_bgr) == 0:
        return "inconnue", 0.0

    # Limite le cout CPU sur les grands masques.
    max_samples = 12000
    if len(pixels_bgr) > max_samples:
        indices = np.linspace(0, len(pixels_bgr) - 1, max_samples, dtype=np.int32)
        pixels_bgr = pixels_bgr[indices]

    hsv = cv2.cvtColor(
        pixels_bgr.reshape(-1, 1, 3).astype(np.uint8),
        cv2.COLOR_BGR2HSV,
    ).reshape(-1, 3)

    h = hsv[:, 0]
    sat = hsv[:, 1]
    val = hsv[:, 2]
    labels = np.full(len(h), "neutre", dtype=object)

    # Couleurs achromatiques en priorite.
    labels[val < 48] = "noir"
    labels[(val >= 48) & (sat < 38) & (val < 205)] = "gris"
    labels[(sat < 38) & (val >= 205)] = "blanc"

    chromatic = (sat >= 38) & (val >= 48)
    labels[chromatic & (((h < 10) | (h >= 170)))] = "rouge"
    labels[chromatic & (h >= 10) & (h < 22)] = "orange"
    labels[chromatic & (h >= 22) & (h < 35)] = "jaune"
    labels[chromatic & (h >= 35) & (h < 85)] = "vert"
    labels[chromatic & (h >= 85) & (h < 130)] = "bleu"
    labels[chromatic & (h >= 130) & (h < 155)] = "violet"
    labels[chromatic & (h >= 155) & (h < 170)] = "rose"

    # Les tons sombres orange/jaune sont souvent marron.
    brown = chromatic & (h >= 5) & (h < 28) & (val < 165)
    labels[brown] = "marron"

    names, counts = np.unique(labels, return_counts=True)
    valid = [(name, int(count)) for name, count in zip(names, counts) if name != "neutre"]
    if not valid:
        mean_bgr = pixels_bgr.mean(axis=0)
        mean_rgb = [int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0])]
        return color_label_from_rgb(mean_rgb), 0.0

    dominant, count = max(valid, key=lambda item: item[1])
    return str(dominant), float(count / max(1, len(labels)))


def apparent_size_label(area_ratio: float) -> str:
    if area_ratio < 0.025:
        return "petit"
    if area_ratio < 0.12:
        return "moyen"
    return "grand"


def shape_label(aspect_ratio: float, fill_ratio: float) -> str:
    if aspect_ratio >= 1.55:
        base = "allonge horizontal"
    elif aspect_ratio <= 0.65:
        base = "allonge vertical"
    else:
        base = "compact"

    if fill_ratio < 0.38:
        return base + ", contour fin/irregulier"
    return base


def enrich_selected_object_semantics(frame: np.ndarray, obj: dict[str, Any] | None) -> dict[str, Any] | None:
    """Calcule les indices plus coûteux uniquement pour la cible choisie."""
    if obj is None:
        return None

    enriched = obj.copy()
    features = dict(obj.get("features", {}))
    mask = obj.get("mask")

    if mask is not None and mask.shape[:2] == frame.shape[:2]:
        pixels_bgr = frame[mask]
        dominant_color, color_ratio = dominant_color_from_pixels(pixels_bgr)
        features["color_label"] = dominant_color
        features["color_ratio"] = color_ratio
        enriched["color"] = dominant_color

    enriched["features"] = features
    return enriched


def position_label(cx: int, cy: int, width: int, height: int) -> str:
    horizontal = "gauche" if cx < width / 3 else ("droite" if cx > 2 * width / 3 else "centre")
    vertical = "haut" if cy < height / 3 else ("bas" if cy > 2 * height / 3 else "milieu")
    return f"{vertical}-{horizontal}"


# -----------------------------------------------------------------------------
# Construction des objets YOLO
# -----------------------------------------------------------------------------
def extract_mask_features(frame: np.ndarray, mask: np.ndarray, box: list[int]) -> dict[str, Any]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box

    area_px = int(mask.sum())
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    box_area = box_w * box_h

    ys, xs = np.where(mask)
    cx = int(xs.mean()) if len(xs) else int((x1 + x2) / 2)
    cy = int(ys.mean()) if len(ys) else int((y1 + y2) / 2)

    pixels_bgr = frame[mask]
    if len(pixels_bgr):
        mean_bgr = pixels_bgr.mean(axis=0)
        mean_rgb = [int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0])]
    else:
        mean_rgb = [0, 0, 0]

    # Important pour les performances : on ne calcule pas ici la couleur
    # dominante complexe pour toutes les boxes YOLO. Elle sera calculée
    # uniquement pour la cible finalement sélectionnée.
    quick_color = color_label_from_rgb(mean_rgb)

    return {
        "center": [cx, cy],
        "area_px": area_px,
        "area_ratio": area_px / max(1, height * width),
        "box_w": box_w,
        "box_h": box_h,
        "aspect_ratio": box_w / max(1, box_h),
        "fill_ratio": area_px / max(1, box_area),
        "mean_rgb": mean_rgb,
        "color_label": quick_color,
        "color_ratio": 0.0,
        "size_label": apparent_size_label(area_px / max(1, height * width)),
        "shape_label": shape_label(box_w / max(1, box_h), area_px / max(1, box_area)),
    }


def build_yolo_objects(frame: np.ndarray, result: Any) -> list[dict[str, Any]]:
    height, width = frame.shape[:2]
    objects: list[dict[str, Any]] = []

    if result is None or result.boxes is None or len(result.boxes) == 0:
        return objects

    masks_np = None
    if result.masks is not None:
        masks_np = result.masks.data.detach().cpu().numpy()

    for index, box in enumerate(result.boxes):
        cls_id = int(box.cls[0].detach().cpu().item())
        confidence = float(box.conf[0].detach().cpu().item())
        class_name = result.names.get(cls_id, str(cls_id))

        x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        if masks_np is not None and index < len(masks_np):
            mask = cv2.resize(
                masks_np[index].astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ) > 0.5
        else:
            mask = np.zeros((height, width), dtype=bool)
            mask[y1:y2, x1:x2] = True

        features = extract_mask_features(frame, mask, [x1, y1, x2, y2])
        objects.append(
            {
                "id": len(objects) + 1,
                "classe": class_name,
                "conf": confidence,
                "mask": mask,
                "box": [x1, y1, x2, y2],
                "center": features["center"],
                "position": position_label(*features["center"], width, height),
                "features": features,
                "area": features["area_px"],
                "color": features["color_label"],
            }
        )

    return objects


class TargetLock:
    """Mémoire spatiale légère, sans ByteTrack ni identifiant persistant."""

    def __init__(self) -> None:
        self.center: list[int] | None = None
        self.area: float | None = None
        self.class_name: str | None = None
        self.last_seen = 0.0

    def reset(self) -> None:
        self.center = None
        self.area = None
        self.class_name = None
        self.last_seen = 0.0

    def update(self, obj: dict[str, Any]) -> None:
        self.center = list(obj.get("center", [0, 0]))
        self.area = float(obj.get("area", 0.0))
        self.class_name = str(obj.get("classe", ""))
        self.last_seen = time.time()

    def continuity(self, obj: dict[str, Any], frame_shape: tuple[int, ...]) -> dict[str, float]:
        position_similarity = 0.0
        if self.center is not None:
            dx = float(obj["center"][0] - self.center[0])
            dy = float(obj["center"][1] - self.center[1])
            diagonal = max(1.0, float(np.hypot(frame_shape[1], frame_shape[0])))
            distance = float(np.hypot(dx, dy))
            position_similarity = float(np.exp(-distance / (0.22 * diagonal)))

        size_similarity = 0.0
        if self.area is not None and self.area > 0 and float(obj.get("area", 0.0)) > 0:
            ratio = float(obj.get("area", 0.0)) / self.area
            size_similarity = float(np.exp(-abs(np.log(max(ratio, 1e-6)))))

        return {"position": position_similarity, "size": size_similarity}


def find_target_yolo(
    objects: list[dict[str, Any]],
    target: str,
    target_lock: TargetLock,
    frame_shape: tuple[int, ...],
) -> dict[str, Any] | None:
    if is_color_target(target):
        wanted = target.split(":", 1)[1]
        candidates = [obj for obj in objects if obj.get("color") == wanted]
    else:
        candidates = [obj for obj in objects if obj.get("classe") == target]

    if not candidates:
        return None

    def stable_yolo_score(obj: dict[str, Any]) -> float:
        continuity = target_lock.continuity(obj, frame_shape)
        return (
            0.72 * float(obj.get("conf", 0.0))
            + 0.18 * continuity["position"]
            + 0.10 * continuity["size"]
        )

    return max(candidates, key=stable_yolo_score)


def create_masked_crop(frame: np.ndarray, obj: dict[str, Any], padding_ratio: float = 0.08) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = obj["box"]

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad_x = int(box_w * padding_ratio)
    pad_y = int(box_h * padding_ratio)

    x1p = max(0, x1 - pad_x)
    y1p = max(0, y1 - pad_y)
    x2p = min(width - 1, x2 + pad_x)
    y2p = min(height - 1, y2 + pad_y)

    if x2p <= x1p or y2p <= y1p:
        return None

    crop = frame[y1p : y2p + 1, x1p : x2p + 1].copy()
    if crop.size == 0:
        return None

    mask = obj.get("mask")
    if mask is None:
        return crop

    crop_mask = mask[y1p : y2p + 1, x1p : x2p + 1]
    if crop_mask.shape[:2] != crop.shape[:2]:
        return crop

    masked_crop = np.full_like(crop, 127)
    masked_crop[crop_mask] = crop[crop_mask]
    return masked_crop


def create_tight_crop(frame: np.ndarray, obj: dict[str, Any], padding_ratio: float = 0.03) -> np.ndarray | None:
    """Crop serre autour du masque, utile quand la box YOLO contient trop de fond."""
    height, width = frame.shape[:2]
    mask = obj.get("mask")

    if mask is not None and mask.shape[:2] == frame.shape[:2] and np.any(mask):
        ys, xs = np.where(mask)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
    else:
        x1, y1, x2, y2 = obj["box"]

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad_x = int(box_w * padding_ratio)
    pad_y = int(box_h * padding_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width - 1, x2 + pad_x)
    y2 = min(height - 1, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1 : y2 + 1, x1 : x2 + 1].copy()
    return crop if crop.size else None


def create_mobileclip_views(frame: np.ndarray, obj: dict[str, Any]) -> list[np.ndarray]:
    """Deux vues complementaires, encodees ensemble pour limiter la latence."""
    views: list[np.ndarray] = []
    masked = create_masked_crop(frame, obj)
    if masked is not None:
        views.append(masked)

    if MOBILECLIP_MULTIVIEW:
        tight = create_tight_crop(frame, obj)
        if tight is not None:
            views.append(tight)

    return views


# -----------------------------------------------------------------------------
# MobileCLIP
# -----------------------------------------------------------------------------
def translate_mobileclip_query(query: str) -> str:
    normalized = query.strip().lower()
    return QUERY_TRANSLATIONS.get(normalized, normalized)


def build_mobileclip_prompts(query_en: str) -> tuple[list[str], list[str]]:
    positives = [
        f"a photo of a {query_en}",
        f"a close-up photo of a {query_en}",
        f"an object that is a {query_en}",
    ]
    negatives = [
        "a photo of an unrelated object",
        "a photo of the background",
        "an empty scene",
        "a blurry unrecognizable object",
        "a photo of a person",
        "a human face or human body",
        "a person holding an object",
    ]
    return positives + negatives, positives


class MobileClipHelper:
    def __init__(self, checkpoint_argument: str = "", enabled: bool = True) -> None:
        self.ready = False
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.device = None
        self.text_cache: dict[tuple[str, ...], Any] = {}

        self.last_query: str | None = None
        self.last_result: dict[str, Any] | None = None
        self.last_valid_time = 0.0

        if not enabled or not MOBILECLIP_ENABLED:
            print("[MOBILECLIP] Désactivé.")
            return

        if torch is None or Image is None or mobileclip is None:
            print(f"[MOBILECLIP] Import impossible : {MOBILECLIP_IMPORT_ERROR}")
            print("[MOBILECLIP] Installe le dépôt Apple officiel avec : pip install -e .")
            return

        checkpoint = self._resolve_checkpoint(checkpoint_argument)
        if checkpoint is None:
            return

        try:
            # Évite que le CPU ancien utilise trop de threads en parallèle.
            torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[MOBILECLIP] Chargement {MOBILECLIP_MODEL} sur {self.device}...")

            self.model, _, self.preprocess = mobileclip.create_model_and_transforms(
                MOBILECLIP_MODEL,
                pretrained=str(checkpoint),
            )
            self.tokenizer = mobileclip.get_tokenizer(MOBILECLIP_MODEL)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.ready = True
            print(f"[MOBILECLIP] Prêt. Checkpoint : {checkpoint}")
        except Exception as exc:
            print(f"[MOBILECLIP] Erreur de chargement : {exc}")

    def _resolve_checkpoint(self, checkpoint_argument: str) -> Path | None:
        candidates: list[Path] = []
        if checkpoint_argument:
            candidates.append(Path(checkpoint_argument).expanduser())

        script_dir = Path(__file__).resolve().parent
        candidates.extend(
            [
                script_dir / MOBILECLIP_LOCAL_PATH,
                Path.cwd() / MOBILECLIP_LOCAL_PATH,
            ]
        )

        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()

        if MOBILECLIP_AUTO_DOWNLOAD and hf_hub_download is not None:
            try:
                print("[MOBILECLIP] Checkpoint absent : téléchargement officiel Hugging Face...")
                downloaded = hf_hub_download(
                    repo_id=MOBILECLIP_REPO,
                    filename=MOBILECLIP_FILENAME,
                )
                return Path(downloaded)
            except Exception as exc:
                print(f"[MOBILECLIP] Téléchargement impossible : {exc}")

        print("[MOBILECLIP] Checkpoint introuvable.")
        print(f"[MOBILECLIP] Place {MOBILECLIP_FILENAME} dans : {script_dir / 'checkpoints'}")
        print("[MOBILECLIP] Ou lance le script avec --checkpoint CHEMIN_DU_FICHIER")
        return None

    def reset(self) -> None:
        self.last_query = None
        self.last_result = None
        self.last_valid_time = 0.0

    def _text_features(self, prompts: list[str]) -> Any:
        key = tuple(prompts)
        if key in self.text_cache:
            return self.text_cache[key]

        tokens = self.tokenizer(prompts).to(self.device)
        with torch.inference_mode():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        self.text_cache[key] = features
        return features

    def classify_view_groups_batch(
        self,
        view_groups: list[list[np.ndarray]],
        prompts: list[str],
    ) -> list[dict[str, float]]:
        """
        Encode toutes les vues en un seul batch, puis moyenne les embeddings
        des vues appartenant au meme objet avant la comparaison texte-image.
        """
        if not self.ready or not view_groups:
            return []

        try:
            tensors: list[Any] = []
            group_sizes: list[int] = []
            for views in view_groups:
                valid_views = [view for view in views if view is not None and view.size > 0]
                group_sizes.append(len(valid_views))
                for crop_bgr in valid_views:
                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    image_pil = Image.fromarray(crop_rgb).convert("RGB")
                    tensors.append(self.preprocess(image_pil))

            if not tensors or any(size == 0 for size in group_sizes):
                return []

            image_tensor = torch.stack(tensors, dim=0).to(self.device)
            text_features = self._text_features(prompts)

            with torch.inference_mode():
                image_features = self.model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                ).clamp_min(1e-8)

                grouped_features = []
                offset = 0
                for size in group_sizes:
                    group_feature = image_features[offset : offset + size].mean(dim=0, keepdim=True)
                    group_feature = group_feature / group_feature.norm(
                        dim=-1, keepdim=True
                    ).clamp_min(1e-8)
                    grouped_features.append(group_feature)
                    offset += size

                averaged_features = torch.cat(grouped_features, dim=0)
                probabilities = (100.0 * averaged_features @ text_features.T).softmax(dim=-1)

            values = probabilities.detach().cpu().numpy()
            return [
                {prompt: float(score) for prompt, score in zip(prompts, row)}
                for row in values
            ]
        except Exception as exc:
            print(f"[MOBILECLIP] Erreur d'inférence multivue : {exc}")
            return []

    def _ordered_candidates(
        self,
        objects: list[dict[str, Any]],
        frame_shape: tuple[int, ...],
    ) -> list[dict[str, Any]]:
        """Priorise les boxes d'objet et refuse les grandes boxes de contexte."""
        frame_area = max(1.0, float(frame_shape[0] * frame_shape[1]))

        candidates = [
            obj
            for obj in objects
            if str(obj.get("classe", "")).lower() not in MOBILECLIP_EXCLUDED_CONTEXT_CLASSES
        ]

        if not candidates:
            return []

        def candidate_priority(obj: dict[str, Any]) -> float:
            area_ratio = float(obj.get("area", 0.0)) / frame_area
            area_ratio = max(area_ratio, 1e-6)
            # Maximum autour de 8 % de l'image, mais reste tolérant aux petits objets.
            region_score = float(
                np.exp(-0.70 * abs(np.log(area_ratio / MOBILECLIP_IDEAL_AREA_RATIO)))
            )
            large_penalty = max(
                0.0,
                (area_ratio - MOBILECLIP_LARGE_BOX_RATIO)
                / max(1e-6, 1.0 - MOBILECLIP_LARGE_BOX_RATIO),
            )
            color_bonus = float(obj.get("hybrid_color_ratio", 0.0))
            return (
                0.62 * region_score
                + 0.23 * float(obj.get("conf", 0.0))
                + 0.15 * color_bonus
                - 0.70 * large_penalty
            )

        return sorted(candidates, key=candidate_priority, reverse=True)[:MOBILECLIP_MAX_OBJECTS]

    def match_cached_object(
        self,
        objects: list[dict[str, Any]],
        query: str,
        target_lock: TargetLock,
        frame_shape: tuple[int, ...],
    ) -> dict[str, Any] | None:
        if (
            self.last_query != query
            or self.last_result is None
            or time.time() - self.last_valid_time > MOBILECLIP_CACHE_TIME
            or not objects
            or target_lock.center is None
        ):
            return None

        candidates = [
            obj
            for obj in objects
            if str(obj.get("classe", "")).lower() not in MOBILECLIP_EXCLUDED_CONTEXT_CLASSES
        ]
        if not candidates:
            return None

        scored: list[tuple[float, dict[str, Any]]] = []
        for obj in candidates:
            continuity = target_lock.continuity(obj, frame_shape)
            score = 0.78 * continuity["position"] + 0.22 * continuity["size"]
            scored.append((score, obj))

        score, candidate = max(scored, key=lambda item: item[0])
        if score < 0.34:
            return None

        result = candidate.copy()
        for key in (
            "mobileclip_score",
            "mobileclip_margin",
            "mobileclip_prompt",
            "mobileclip_combined_score",
            "mobileclip_position_similarity",
            "mobileclip_size_similarity",
            "mobileclip_region_score",
        ):
            result[key] = self.last_result.get(key)
        result["mobileclip_cached"] = True
        return result

    def find_target(
        self,
        frame: np.ndarray,
        objects: list[dict[str, Any]],
        query: str,
        target_lock: TargetLock,
    ) -> dict[str, Any] | None:
        if not self.ready or not objects:
            return None

        query_en = translate_mobileclip_query(query)
        prompts, positive_prompts = build_mobileclip_prompts(query_en)
        candidates = self._ordered_candidates(objects, frame.shape)

        valid_candidates: list[dict[str, Any]] = []
        view_groups: list[list[np.ndarray]] = []
        for obj in candidates:
            views = create_mobileclip_views(frame, obj)
            if views:
                valid_candidates.append(obj)
                view_groups.append(views)

        all_scores = self.classify_view_groups_batch(view_groups, prompts)
        evaluations: list[dict[str, Any]] = []

        for obj, scores in zip(valid_candidates, all_scores):
            target_score = max(scores[prompt] for prompt in positive_prompts)
            negative_score = max(
                scores[prompt] for prompt in prompts if prompt not in positive_prompts
            )
            margin = target_score - negative_score
            margin_quality = float(np.clip(margin / 0.30, 0.0, 1.0))
            continuity = target_lock.continuity(obj, frame.shape)
            yolo_conf = float(obj.get("conf", 0.0))

            frame_area = max(1.0, float(frame.shape[0] * frame.shape[1]))
            area_ratio = max(float(obj.get("area", 0.0)) / frame_area, 1e-6)
            region_score = float(
                np.exp(-0.70 * abs(np.log(area_ratio / MOBILECLIP_IDEAL_AREA_RATIO)))
            )
            large_penalty = max(
                0.0,
                (area_ratio - MOBILECLIP_LARGE_BOX_RATIO)
                / max(1e-6, 1.0 - MOBILECLIP_LARGE_BOX_RATIO),
            )

            color_ratio = float(obj.get("hybrid_color_ratio", 0.0))
            hybrid_bonus = 0.18 * color_ratio if obj.get("hybrid_mode", False) else 0.0
            combined_score = (
                SCORE_WEIGHT_SEMANTIC * target_score
                + SCORE_WEIGHT_MARGIN * margin_quality
                + SCORE_WEIGHT_POSITION * continuity["position"]
                + SCORE_WEIGHT_SIZE * continuity["size"]
                + SCORE_WEIGHT_YOLO * yolo_conf
                + SCORE_WEIGHT_REGION * region_score
                + hybrid_bonus
                - 0.35 * large_penalty
            )

            evaluation = {
                "obj": obj,
                "target_score": target_score,
                "negative_score": negative_score,
                "margin": margin,
                "combined_score": combined_score,
                "position_similarity": continuity["position"],
                "size_similarity": continuity["size"],
                "region_score": region_score,
                "winning_prompt": max(scores, key=scores.get),
            }
            evaluations.append(evaluation)

            if MOBILECLIP_DEBUG:
                print(
                    f"[MOBILECLIP] query='{query}' "
                    f"yolo='{obj.get('classe')}' semantic={target_score:.3f} "
                    f"margin={margin:.3f} combined={combined_score:.3f}"
                )

        valid = [
            item
            for item in evaluations
            if item["target_score"] >= MOBILECLIP_MIN_TARGET_SCORE
            and item["margin"] >= MOBILECLIP_MIN_MARGIN
        ]
        if not valid:
            return None

        best = max(valid, key=lambda item: item["combined_score"])


        result = best["obj"].copy()
        if result.get("hybrid_mode", False):
            result["hybrid_source"] = (
                "OpenCV+MobileCLIP"
                if result.get("hybrid_fallback_region", False)
                else "YOLO+MobileCLIP+OpenCV"
            )
        result.update(
            {
                "mobileclip_score": best["target_score"],
                "mobileclip_margin": best["margin"],
                "mobileclip_combined_score": best["combined_score"],
                "mobileclip_position_similarity": best["position_similarity"],
                "mobileclip_size_similarity": best["size_similarity"],
                "mobileclip_region_score": best["region_score"],
                "mobileclip_prompt": best["winning_prompt"],
                "mobileclip_query": query,
                "mobileclip_cached": False,
            }
        )

        self.last_query = query
        self.last_result = result.copy()
        self.last_valid_time = time.time()

        if MOBILECLIP_DEBUG:
            print(
                f"[MOBILECLIP MATCH] '{query}' -> "
                f"yolo='{result.get('classe')}' semantic={result['mobileclip_score']:.3f} "
                f"combined={result['mobileclip_combined_score']:.3f}"
            )
        return result


# -----------------------------------------------------------------------------
# Kalman corrigé : predict() avant correct() lorsqu'une mesure arrive
# -----------------------------------------------------------------------------
class KalmanTracker2D:
    def __init__(self) -> None:
        self.filter = cv2.KalmanFilter(4, 2)
        self.filter.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        self.filter.processNoiseCov = np.diag([0.03, 0.03, 0.25, 0.25]).astype(np.float32)
        self.filter.measurementNoiseCov = np.eye(2, dtype=np.float32) * 3.0
        self.filter.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False
        self.last_time: float | None = None

    def reset(self) -> None:
        self.initialized = False
        self.last_time = None

    def _set_transition(self, dt: float) -> None:
        self.filter.transitionMatrix = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )

    def update(self, x: int, y: int, now: float | None = None) -> tuple[int, int]:
        now = time.perf_counter() if now is None else now
        measurement = np.array([[np.float32(x)], [np.float32(y)]], dtype=np.float32)

        if not self.initialized:
            state = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.filter.statePost = state.copy()
            self.filter.statePre = state.copy()
            self.filter.errorCovPost = np.eye(4, dtype=np.float32)
            self.initialized = True
            self.last_time = now
            return x, y

        dt = max(1e-3, min(now - (self.last_time or now), 0.25))
        self._set_transition(dt)
        self.filter.predict()
        corrected = self.filter.correct(measurement)
        self.last_time = now
        return int(corrected[0, 0]), int(corrected[1, 0])

    def predict(self, now: float | None = None) -> tuple[int, int] | None:
        if not self.initialized:
            return None

        now = time.perf_counter() if now is None else now
        dt = max(1e-3, min(now - (self.last_time or now), 0.25))
        self._set_transition(dt)
        prediction = self.filter.predict()
        self.last_time = now
        return int(prediction[0, 0]), int(prediction[1, 0])


# -----------------------------------------------------------------------------
# Analyse du mouvement apparent de la cible dans l'image
# -----------------------------------------------------------------------------
class MotionAnalyzer:
    """Estime une direction et une vitesse en pixels/seconde.

    L'estimation utilise les centres filtrés par Kalman et un lissage
    exponentiel de la vitesse. Il s'agit d'un mouvement apparent dans l'image :
    si Reachy bouge les yeux ou la tête, cela influence aussi la mesure.
    """

    def __init__(self) -> None:
        self.previous_center: np.ndarray | None = None
        self.previous_time: float | None = None
        self.velocity = np.zeros(2, dtype=np.float32)
        self.info = self._empty_info()

    @staticmethod
    def _empty_info() -> dict[str, Any]:
        return {
            "direction": "initialisation",
            "speed_px_s": 0.0,
            "speed_label": "immobile",
            "vx_px_s": 0.0,
            "vy_px_s": 0.0,
        }

    def reset(self) -> None:
        self.previous_center = None
        self.previous_time = None
        self.velocity[:] = 0.0
        self.info = self._empty_info()

    def current_info(self) -> dict[str, Any]:
        return dict(self.info)

    @staticmethod
    def _direction_label(vx: float, vy: float, speed: float) -> str:
        if speed < MOTION_STOP_THRESHOLD:
            return "immobile"

        abs_x = abs(vx)
        abs_y = abs(vy)

        if abs_x >= 1.6 * abs_y:
            return "vers la droite" if vx > 0 else "vers la gauche"
        if abs_y >= 1.6 * abs_x:
            return "vers le bas" if vy > 0 else "vers le haut"

        horizontal = "droite" if vx > 0 else "gauche"
        vertical = "bas" if vy > 0 else "haut"
        return f"vers le {vertical}-{horizontal}"

    @staticmethod
    def _speed_label(speed: float) -> str:
        if speed < MOTION_STOP_THRESHOLD:
            return "immobile"
        if speed < MOTION_SLOW_THRESHOLD:
            return "lente"
        if speed < MOTION_MEDIUM_THRESHOLD:
            return "moyenne"
        return "rapide"

    def update(
        self,
        center: list[int] | tuple[int, int],
        now: float | None = None,
    ) -> dict[str, Any]:
        now = time.perf_counter() if now is None else float(now)
        current = np.asarray(center, dtype=np.float32)

        if self.previous_center is None or self.previous_time is None:
            self.previous_center = current
            self.previous_time = now
            self.info = self._empty_info()
            return self.current_info()

        dt = now - self.previous_time
        if dt <= 1e-3 or dt > MOTION_MAX_DT:
            self.previous_center = current
            self.previous_time = now
            self.velocity[:] = 0.0
            self.info = self._empty_info()
            return self.current_info()

        instantaneous_velocity = (current - self.previous_center) / dt
        alpha = float(np.clip(MOTION_EMA_ALPHA, 0.0, 1.0))
        self.velocity = (
            alpha * instantaneous_velocity
            + (1.0 - alpha) * self.velocity
        ).astype(np.float32)

        vx = float(self.velocity[0])
        vy = float(self.velocity[1])
        speed = float(np.hypot(vx, vy))

        self.info = {
            "direction": self._direction_label(vx, vy, speed),
            "speed_px_s": speed,
            "speed_label": self._speed_label(speed),
            "vx_px_s": vx,
            "vy_px_s": vy,
        }

        self.previous_center = current
        self.previous_time = now
        return self.current_info()


# -----------------------------------------------------------------------------
# Contrôleur Reachy 2019 : yeux + assistance horizontale de la tête
# -----------------------------------------------------------------------------
class ReachyEyeHeadController:
    """Contrôle les yeux et complète leur mouvement avec la tête horizontale.

    Les commandes des yeux reprennent exactement la logique fonctionnelle :
    ``go_left``, ``go_right``, ``go_up`` et ``go_down``.

    Pour la tête, le même ``Robot_Interface`` commande ``BLOW_HORIZONTAL``
    avec ``set_position``. L'axe vertical de la tête reste désactivé par défaut.
    """

    def __init__(self, robot_enabled: bool = True, head_enabled: bool = True) -> None:
        self.robot_enabled = bool(robot_enabled)
        self.head_enabled = bool(head_enabled and HEAD_ASSIST_ENABLED and robot_enabled)
        self.robot = None
        self.hd = None

        # La touche h ramène les yeux à la position enregistrée au démarrage.
        self.eye_h_id: int | None = None
        self.eye_v_id: int | None = None
        self.eye_h_center: int | None = None
        self.eye_v_center: int | None = None
        self.eye_center_ready = False

        self.visual_gaze: np.ndarray | None = None
        self.last_move_time = 0.0
        self.last_scan_time = 0.0
        self.scan_index = 0

        self.head_ready = False
        self.head_h: int | None = None
        self.head_v: int | None = None
        self.head_h_min = 0
        self.head_h_max = 0
        self.head_v_min = 0
        self.head_v_max = 0
        self.last_head_track_time = 0.0
        self.last_head_scan_time = 0.0
        self.head_scan_dir = 1

        if self.robot_enabled:
            current_dir = Path(__file__).resolve().parent
            if str(current_dir) not in sys.path:
                sys.path.insert(0, str(current_dir))

            try:
                import head_control as hd
            except Exception as exc:
                raise RuntimeError(
                    "Impossible d'importer head_control.py. Place ce script dans le même "
                    "dossier que head_control.py."
                ) from exc

            self.hd = hd

            try:
                self.robot = hd.Robot_Interface()
            except Exception as exc:
                raise RuntimeError(
                    "Connexion aux moteurs impossible. Ferme les autres scripts utilisant "
                    "/dev/ttyUSB0 et vérifie le câble/baudrate."
                ) from exc

            self._setup_head_assist()
            self._setup_eye_center()
            head_status = "active" if self.head_ready else "indisponible"
            print(
                "[REACHY] Yeux connectés. "
                f"Assistance tête horizontale : {head_status}. "
                "Bras et main non utilisés."
            )
        else:
            print("[REACHY] DRY-RUN : aucun moteur ne bougera.")

    def _setup_head_assist(self) -> None:
        """Initialise l'assistance horizontale avec les constantes de head_control."""
        if not self.head_enabled or self.robot is None or self.hd is None:
            self.head_ready = False
            return

        required = [
            "BLOW_HORIZONTAL",
            "BLOW_VERTICAL",
            "INIT_POS",
        ]
        missing = [name for name in required if not hasattr(self.hd, name)]
        if missing:
            print(
                "[REACHY] Assistance tête désactivée : constantes absentes dans "
                f"head_control.py : {', '.join(missing)}"
            )
            self.head_ready = False
            return

        try:
            self.robot.set_speed(self.hd.BLOW_HORIZONTAL, HEAD_SPEED)
            # Même initialisation que dans l'ancien code. L'axe vertical ne bougera
            # pas tant que HEAD_VERTICAL_ENABLED reste False.
            self.robot.set_speed(self.hd.BLOW_VERTICAL, HEAD_SPEED)

            head_h_init = int(self.hd.INIT_POS[0])
            head_v_init = int(self.hd.INIT_POS[1])

            try:
                self.head_h = int(self.robot.get_position(self.hd.BLOW_HORIZONTAL))
                self.head_v = int(self.robot.get_position(self.hd.BLOW_VERTICAL))
            except Exception as exc:
                print(
                    "[REACHY] Lecture position tête impossible, utilisation de INIT_POS : "
                    f"{exc}"
                )
                self.head_h = head_h_init
                self.head_v = head_v_init

            self.head_h_min = head_h_init - HEAD_H_LIMIT
            self.head_h_max = head_h_init + HEAD_H_LIMIT
            self.head_v_min = head_v_init - HEAD_V_LIMIT
            self.head_v_max = head_v_init + HEAD_V_LIMIT
            self.head_ready = True
        except Exception as exc:
            print(f"[REACHY] Initialisation de la tête impossible : {exc}")
            self.head_ready = False

    def _setup_eye_center(self) -> None:
        """Mémorise la position des yeux au lancement comme position centrale."""
        if self.robot is None or self.hd is None:
            self.eye_center_ready = False
            return

        # Les identifiants confirmés dans head_control.py sont normalement 6 et 5.
        self.eye_h_id = int(getattr(self.hd, "EYE_HORIZONTAL", 6))
        self.eye_v_id = int(getattr(self.hd, "EYE_VERTICAL", 5))

        try:
            self.eye_h_center = int(self.robot.get_position(self.eye_h_id))
            self.eye_v_center = int(self.robot.get_position(self.eye_v_id))
            self.eye_center_ready = True
            print(
                "[REACHY] Position de recentrage des yeux enregistrée : "
                f"H={self.eye_h_center}, V={self.eye_v_center}"
            )
        except Exception as exc:
            self.eye_center_ready = False
            print(f"[REACHY] Position initiale des yeux illisible : {exc}")

    def recenter(self, frame_shape: tuple[int, ...] | None = None) -> None:
        """Recentre les yeux et l'axe horizontal de la tête.

        Les yeux reviennent à leur position enregistrée au démarrage. La tête
        revient à INIT_POS sur son axe horizontal. L'axe vertical de la tête
        reste volontairement désactivé, comme dans le reste du pipeline.
        """
        self.reset()

        if frame_shape is not None:
            height, width = frame_shape[:2]
            self.visual_gaze = np.array(
                [width / 2.0, height / 2.0],
                dtype=np.float32,
            )

        if self.robot is None:
            print("[REACHY] DRY-RUN : recentrage simulé.")
            return

        eye_ok = False
        head_ok = False

        try:
            if (
                self.eye_center_ready
                and self.eye_h_id is not None
                and self.eye_v_id is not None
                and self.eye_h_center is not None
                and self.eye_v_center is not None
            ):
                self.robot.set_position(self.eye_h_id, int(self.eye_h_center))
                self.robot.set_position(self.eye_v_id, int(self.eye_v_center))
                eye_ok = True

            if (
                self.head_ready
                and self.hd is not None
                and hasattr(self.hd, "INIT_POS")
            ):
                self._set_head_h(int(self.hd.INIT_POS[0]))
                head_ok = True

            print(
                "[REACHY] Recentrage terminé | "
                f"yeux={'OK' if eye_ok else 'indisponibles'} | "
                f"tête horizontale={'OK' if head_ok else 'indisponible'}"
            )
        except Exception as exc:
            print(f"[REACHY] Erreur pendant le recentrage : {exc}")

    def reset(self) -> None:
        self.visual_gaze = None
        self.last_move_time = 0.0
        self.last_scan_time = 0.0
        self.scan_index = 0
        self.last_head_track_time = 0.0
        self.last_head_scan_time = 0.0
        self.head_scan_dir = 1

    def test_eyes(self) -> None:
        """Petit test prudent des yeux uniquement."""
        if self.robot is None:
            print("[REACHY] Test des yeux ignoré en dry-run.")
            return

        print("[REACHY] Test rapide des yeux...")
        sequence = [
            (self._left, 4),
            (self._right, 8),
            (self._left, 4),
            (self._up, 4),
            (self._down, 8),
            (self._up, 4),
        ]
        for command, step in sequence:
            command(step)
            time.sleep(0.25)
        print("[REACHY] Test des yeux terminé.")

    # Compatibilité avec l'ancien nom utilisé dans main().
    def test_motors(self) -> None:
        self.test_eyes()

    def test_head(self) -> None:
        """Petit test horizontal de la tête avec retour à la position de départ."""
        if not self.head_ready or self.head_h is None:
            print("[REACHY] Test tête ignoré : assistance tête indisponible ou désactivée.")
            return

        print("[REACHY] Test rapide de la tête horizontale...")
        start = int(self.head_h)
        try:
            self._set_head_h(start + 8)
            time.sleep(0.45)
            self._set_head_h(start - 8)
            time.sleep(0.45)
            self._set_head_h(start)
            time.sleep(0.45)
            print("[REACHY] Test de la tête terminé.")
        except Exception as exc:
            print(f"[REACHY] Erreur test tête : {exc}")

    def _update_visual_target(
        self,
        center: list[int] | tuple[int, int],
        frame_shape: tuple[int, ...],
    ) -> None:
        height, width = frame_shape[:2]
        if self.visual_gaze is None:
            self.visual_gaze = np.array([width / 2.0, height / 2.0], dtype=np.float32)

        target = np.array(center, dtype=np.float32)
        error = target - self.visual_gaze
        if float(np.linalg.norm(error)) > 1.0:
            self.visual_gaze += np.clip(error * 0.15, -18.0, 18.0)

        self.visual_gaze[0] = np.clip(self.visual_gaze[0], 0, width - 1)
        self.visual_gaze[1] = np.clip(self.visual_gaze[1], 0, height - 1)

    # ------------------------------------------------------------------
    # Yeux : mêmes appels que dans le code qui fonctionnait.
    # ------------------------------------------------------------------
    def _left(self, step: int) -> None:
        if self.robot is None:
            return
        method = self.robot.go_right if EYE_INVERT_X else self.robot.go_left
        method(step=int(step))

    def _right(self, step: int) -> None:
        if self.robot is None:
            return
        method = self.robot.go_left if EYE_INVERT_X else self.robot.go_right
        method(step=int(step))

    def _up(self, step: int) -> None:
        if self.robot is None:
            return
        method = self.robot.go_down if EYE_INVERT_Y else self.robot.go_up
        method(step=int(step))

    def _down(self, step: int) -> None:
        if self.robot is None:
            return
        method = self.robot.go_up if EYE_INVERT_Y else self.robot.go_down
        method(step=int(step))

    @staticmethod
    def _proportional_eye_step(error: float, gain: float) -> int:
        raw = int(abs(error) * gain)
        return max(EYE_MIN_STEP, min(raw, EYE_MAX_STEP))

    # ------------------------------------------------------------------
    # Tête : logique horizontale reprise de l'ancien EyeController.
    # ------------------------------------------------------------------
    def _set_head_h(self, position: int) -> None:
        if (
            not self.head_ready
            or self.robot is None
            or self.hd is None
            or self.head_h is None
        ):
            return

        position = int(max(self.head_h_min, min(position, self.head_h_max)))
        if position != self.head_h:
            self.robot.set_position(self.hd.BLOW_HORIZONTAL, position)
            self.head_h = position

    def _set_head_v(self, position: int) -> None:
        if (
            not self.head_ready
            or not HEAD_VERTICAL_ENABLED
            or self.robot is None
            or self.hd is None
            or self.head_v is None
        ):
            return

        position = int(max(self.head_v_min, min(position, self.head_v_max)))
        if position != self.head_v:
            self.robot.set_position(self.hd.BLOW_VERTICAL, position)
            self.head_v = position

    def head_track_assist(self, err_x: int, err_y: int) -> None:
        """Complète les yeux avec un déplacement horizontal lent de la tête."""
        if not self.head_ready or self.head_h is None:
            return

        now = time.time()
        if now - self.last_head_track_time < HEAD_TRACK_COOLDOWN:
            return

        moved = False
        if abs(err_x) > HEAD_DEADZONE_X:
            raw_step = abs(err_x) * HEAD_KP_X
            head_step = max(HEAD_MIN_STEP, min(int(raw_step), HEAD_MAX_STEP))

            # Sens repris à l'identique de l'ancien code fonctionnel.
            delta = -head_step if err_x > 0 else head_step
            if HEAD_INVERT_X:
                delta = -delta
            self._set_head_h(self.head_h + delta)
            moved = True

        # Disponible mais volontairement désactivé par défaut.
        if HEAD_VERTICAL_ENABLED and self.head_v is not None and abs(err_y) > HEAD_DEADZONE_X:
            raw_step_y = abs(err_y) * HEAD_KP_X
            head_step_y = max(HEAD_MIN_STEP, min(int(raw_step_y), HEAD_MAX_STEP))
            delta_y = -head_step_y if err_y > 0 else head_step_y
            if HEAD_INVERT_Y:
                delta_y = -delta_y
            self._set_head_v(self.head_v + delta_y)
            moved = True

        if moved:
            self.last_head_track_time = now

    def head_scan_assist(self) -> None:
        """Balayage horizontal lent de la tête pendant la recherche."""
        if not self.head_ready or self.head_h is None:
            return

        now = time.time()
        if now - self.last_head_scan_time < HEAD_SCAN_COOLDOWN:
            return

        if self.head_h >= self.head_h_max - 5:
            self.head_scan_dir = -1
        elif self.head_h <= self.head_h_min + 5:
            self.head_scan_dir = 1

        delta_h = HEAD_SCAN_STEP * self.head_scan_dir
        if HEAD_INVERT_X:
            delta_h = -delta_h
        self._set_head_h(self.head_h + delta_h)
        self.last_head_scan_time = now

    def update_target(
        self,
        center: list[int] | tuple[int, int],
        frame_shape: tuple[int, ...],
    ) -> None:
        """Centre la cible avec les yeux, puis complète avec la tête horizontale."""
        self._update_visual_target(center, frame_shape)

        height, width = frame_shape[:2]
        cx, cy = int(center[0]), int(center[1])
        err_x = cx - width // 2
        err_y = cy - height // 2

        action_h = "CENTER"
        action_v = "CENTER"
        step_x = 0
        step_y = 0

        if abs(err_x) > EYE_DEADZONE_X:
            action_h = "RIGHT" if err_x > 0 else "LEFT"
            step_x = self._proportional_eye_step(err_x, EYE_KP_X)
        if abs(err_y) > EYE_DEADZONE_Y:
            action_v = "DOWN" if err_y > 0 else "UP"
            step_y = self._proportional_eye_step(err_y, EYE_KP_Y)

        now = time.time()
        if now - self.last_move_time < EYE_CONTROL_COOLDOWN:
            return
        self.last_move_time = now

        try:
            if action_h == "RIGHT":
                self._right(step_x)
            elif action_h == "LEFT":
                self._left(step_x)

            if action_v == "DOWN":
                self._down(step_y)
            elif action_v == "UP":
                self._up(step_y)

            # Exactement comme dans l'ancien code : la tête est commandée après
            # les yeux à chaque cycle de contrôle autorisé par le cooldown.
            self.head_track_assist(err_x, err_y)
        except Exception as exc:
            print(f"[REACHY] Erreur commande yeux/tête : {exc}")

        if self.robot is not None:
            print(
                f"[EYES+HEAD REAL] error=({err_x},{err_y}) "
                f"eyes={action_h}({step_x})/{action_v}({step_y}) "
                f"head_h={self.head_h if self.head_ready else 'OFF'}"
            )

    def _scan_action(self) -> str:
        sequence = (
            ["LEFT"] * EYE_SCAN_HORIZONTAL_COUNT
            + ["RIGHT"] * (2 * EYE_SCAN_HORIZONTAL_COUNT)
            + ["LEFT"] * EYE_SCAN_HORIZONTAL_COUNT
            + ["UP"] * EYE_SCAN_VERTICAL_COUNT
            + ["DOWN"] * (2 * EYE_SCAN_VERTICAL_COUNT)
            + ["UP"] * EYE_SCAN_VERTICAL_COUNT
        )
        action = sequence[self.scan_index % len(sequence)]
        self.scan_index += 1
        return action

    def scan(self, frame_shape: tuple[int, ...]) -> None:
        """Les yeux scannent rapidement et la tête complète lentement à l'horizontale."""
        height, width = frame_shape[:2]
        if self.visual_gaze is None:
            self.visual_gaze = np.array([width / 2.0, height / 2.0], dtype=np.float32)

        self.visual_gaze[0] += 4.0 * (1 if (self.scan_index // 60) % 2 == 0 else -1)
        self.visual_gaze[0] = np.clip(self.visual_gaze[0], width * 0.12, width * 0.88)
        self.visual_gaze[1] = height / 2.0 + 0.12 * height * np.sin(time.time() * 1.2)

        if not EYE_SCAN_ENABLED:
            self.head_scan_assist()
            return

        now = time.time()
        if now - self.last_scan_time < EYE_SCAN_COOLDOWN:
            # La tête possède son propre cooldown et doit continuer à être évaluée.
            self.head_scan_assist()
            return
        self.last_scan_time = now

        action = self._scan_action()
        try:
            if action == "LEFT":
                self._left(EYE_SCAN_STEP)
            elif action == "RIGHT":
                self._right(EYE_SCAN_STEP)
            elif action == "UP":
                self._up(EYE_SCAN_STEP)
            elif action == "DOWN":
                self._down(EYE_SCAN_STEP)

            self.head_scan_assist()
        except Exception as exc:
            print(f"[REACHY] Erreur scan yeux/tête : {exc}")

    def direction_text(
        self,
        target_center: list[int] | None,
        frame_shape: tuple[int, ...],
    ) -> str:
        if self.robot_enabled:
            head_mode = "HEAD ON" if self.head_ready else "HEAD OFF"
            mode = f"{head_mode}"
        else:
            mode = "DRY-RUN EYES + HEAD"

        if target_center is None:
            return f"{mode}: SCAN"

        height, width = frame_shape[:2]
        err_x = target_center[0] - width // 2
        err_y = target_center[1] - height // 2
        horizontal = (
            "RIGHT"
            if err_x > EYE_DEADZONE_X
            else ("LEFT" if err_x < -EYE_DEADZONE_X else "CENTER")
        )
        vertical = (
            "DOWN"
            if err_y > EYE_DEADZONE_Y
            else ("UP" if err_y < -EYE_DEADZONE_Y else "CENTER")
        )
        return f"{mode}: {horizontal} / {vertical}"

    def close(self) -> None:
        if self.robot is not None:
            try:
                self.robot.shutdown()
            except Exception as exc:
                print(f"[REACHY] Erreur fermeture : {exc}")
            finally:
                self.robot = None


# -----------------------------------------------------------------------------
# Affichage
# -----------------------------------------------------------------------------
def wrap_panel_text(text: str, max_chars: int = 34) -> list[str]:
    """Découpe un texte en plusieurs lignes adaptées au panneau latéral."""
    cleaned = " ".join(str(text).strip().split())
    if not cleaned:
        return ["Aucune demande"]

    words = cleaned.split(" ")
    lines: list[str] = []
    current = ""

    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            # Coupe également un mot exceptionnellement long.
            while len(word) > max_chars:
                lines.append(word[:max_chars])
                word = word[max_chars:]
            current = word

    if current:
        lines.append(current)

    return lines or ["Aucune demande"]


def draw_scene(
    frame: np.ndarray,
    objects: list[dict[str, Any]],
    target_text: str | None,
    user_request_text: str | None,
    tracking_center: list[int] | None,
    state: str,
    fps: float,
    gaze: ReachyEyeHeadController,
    is_predicted: bool = False,
    semantic_obj: dict[str, Any] | None = None,
    target_obj: dict[str, Any] | None = None,
    motion_info: dict[str, Any] | None = None,
) -> np.ndarray:
    vis = frame.copy()

    # Panneau séparé à droite : il ne masque plus le flux caméra.
    semantic_panel = np.full(
        (vis.shape[0], SEMANTIC_PANEL_W, 3),
        (24, 24, 24),
        dtype=np.uint8,
    )

    cv2.line(
        semantic_panel,
        (0, 0),
        (0, semantic_panel.shape[0] - 1),
        (210, 210, 210),
        2,
    )

    cv2.putText(
        semantic_panel,
        "INDICES SEMANTIQUES",
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )

    # Affiche exactement ce que l'utilisateur a saisi, avant normalisation,
    # traduction ou séparation objet/couleur.
    cv2.putText(
        semantic_panel,
        "DEMANDE UTILISATEUR",
        (14, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.37,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )

    request_lines = wrap_panel_text(
        user_request_text if user_request_text else "Aucune demande",
        max_chars=34,
    )

    request_y = 80
    request_line_height = 19
    for index, request_line in enumerate(request_lines):
        cv2.putText(
            semantic_panel,
            request_line,
            (14, request_y + index * request_line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (250, 250, 250),
            1,
            cv2.LINE_AA,
        )

    separator_y = request_y + len(request_lines) * request_line_height + 7
    cv2.line(
        semantic_panel,
        (12, separator_y),
        (SEMANTIC_PANEL_W - 12, separator_y),
        (90, 90, 90),
        1,
    )

    panel_content_y = separator_y + 24

    if target_obj is None:
        cv2.putText(
            semantic_panel,
            "Aucune cible verrouillee",
            (14, panel_content_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (185, 185, 185),
            1,
            cv2.LINE_AA,
        )

        if target_text is not None:
            cv2.putText(
                semantic_panel,
                f"Cible comprise : {target_display_name(target_text)}",
                (14, panel_content_y + 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

    for obj in objects:
        x1, y1, x2, y2 = obj["box"]
        mask = obj.get("mask")
        color = (0, 190, 0)
        if mask is not None:
            vis[mask] = (0.72 * vis[mask] + 0.28 * np.array(color)).astype(np.uint8)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            vis,
            f"{obj['classe']} {obj['conf']:.2f}",
            (x1, max(18, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )

    if target_obj is not None:
        x1, y1, x2, y2 = target_obj["box"]
        is_opencv_color = target_obj.get("source") == "OpenCV" and not target_obj.get("hybrid_mode", False)
        is_hybrid = bool(target_obj.get("hybrid_mode", False))
        if is_opencv_color:
            target_color = (0, 255, 255)
            source_label = "OPENCV COULEUR"
        elif is_hybrid:
            target_color = (255, 0, 255)
            source_label = "OBJET+COULEUR"
        elif semantic_obj is not None:
            target_color = (255, 0, 0)
            source_label = "MOBILECLIP"
        else:
            target_color = (0, 215, 255)
            source_label = "YOLO"
        features = target_obj.get("features", {})
        detected_color = (
            features.get("requested_color", "?")
            if is_hybrid
            else features.get("color_label", "?")
        )
        cv2.rectangle(vis, (x1, y1), (x2, y2), target_color, 3)
        cv2.putText(
            vis,
            f"{source_label}: {target_display_name(target_text)} | couleur: {detected_color}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            target_color,
            2,
        )

    if tracking_center is not None:
        point_color = (0, 165, 255) if is_predicted else (0, 255, 255)
        cv2.circle(vis, tuple(map(int, tracking_center)), 8, point_color, -1)
        cv2.putText(
            vis,
            "KALMAN" if is_predicted else "LOCKED",
            (tracking_center[0] + 10, tracking_center[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            point_color,
            2,
        )

    cv2.putText(vis, f"TARGET: {target_display_name(target_text)}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)
    cv2.putText(vis, state, (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
    cv2.putText(vis, gaze.direction_text(tracking_center, frame.shape), (15, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    cv2.putText(vis, f"fps: {fps:.1f}", (15, vis.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)

    if target_obj is not None:
        features = target_obj.get("features", {})
        mean_rgb = features.get("mean_rgb", [0, 0, 0])
        color_name = features.get("color_label", "inconnue")
        color_ratio = float(features.get("color_ratio", 0.0))
        area_ratio = float(features.get("area_ratio", 0.0))
        is_hybrid = bool(target_obj.get("hybrid_mode", False))
        is_opencv_color = target_obj.get("source") == "OpenCV" and not is_hybrid

        if is_hybrid:
            hybrid_source = target_obj.get("hybrid_source", "YOLO+MobileCLIP+OpenCV")
            source_name = {
                "YOLO+OpenCV": "YOLO + OpenCV (HSV)",
                "YOLO+MobileCLIP+OpenCV": "MobileCLIP + OpenCV (HSV)",
                "OpenCV+MobileCLIP": "Zones OpenCV + MobileCLIP",
            }.get(str(hybrid_source), str(hybrid_source))
        else:
            source_name = "OpenCV (HSV)" if is_opencv_color else ("MobileCLIP" if semantic_obj is not None else "YOLO")

        lines = [
            "INDICES SEMANTIQUES",
            f"Cible comprise : {target_display_name(target_text)}",
            f"Source : {source_name}",
        ]
        if is_opencv_color:
            lines.extend([
                f"Zone detectee : {target_obj.get('classe', '?')}",
                f"Couleur recherchee : {color_name}",
                f"Densite couleur : {color_ratio * 100:.0f}%",
                f"RGB moyen : {mean_rgb}",
                f"Position : {target_obj.get('position', '?')}",
                f"Taille apparente : {features.get('size_label', '?')}",
                f"Forme : {features.get('shape_label', '?')}",
                f"Score OpenCV : {target_obj.get('opencv_color_score', 0.0):.2f}",
                "YOLO/MobileCLIP : non utilises",
            ])
        elif is_hybrid:
            requested_color = features.get("requested_color", target_obj.get("hybrid_color", "?"))
            requested_ratio = float(features.get("requested_color_ratio", target_obj.get("hybrid_color_ratio", 0.0)))
            object_name = hybrid_object_name(target_text)
            lines.extend([
                f"Objet recherche : {object_name}",
                f"Couleur validee : {requested_color} ({requested_ratio * 100:.0f}%)",
                f"Classe/region : {target_obj.get('classe', '?')}",
                f"Position : {target_obj.get('position', '?')}",
                f"Taille apparente : {features.get('size_label', '?')} ",
                f"Forme : {features.get('shape_label', '?')}",
            ])
        else:
            lines.extend([
                f"Classe YOLO : {target_obj.get('classe', '?')}",
                f"Couleur dominante : {color_name} ({color_ratio * 100:.0f}%)",
                f"RGB moyen : {mean_rgb}",
                f"Position : {target_obj.get('position', '?')}",
                f"Taille apparente : {features.get('size_label', '?')} ",
                f"Forme : {features.get('shape_label', '?')}",
            ])
        if motion_info is not None:
            speed_label = str(motion_info.get("speed_label", "?"))
            speed_px_s = float(motion_info.get("speed_px_s", 0.0))
            lines.append(
                f"Vitesse image : {speed_label} ({speed_px_s:.0f} px/s)"
            )

        if semantic_obj is not None:
            cache_label = "oui" if semantic_obj.get("mobileclip_cached", False) else "non"
            lines.append(f"Cache MobileCLIP : {cache_label}")

        # Affichage dans la colonne séparée, sans masquer l'image caméra.
        line_height = 24
        start_y = panel_content_y

        # lines[0] contient déjà le titre, affiché en haut du panneau.
        for index, line in enumerate(lines[1:]):
            y = start_y + index * line_height
            cv2.putText(
                semantic_panel,
                line,
                (14, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.39,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )

        swatch_bgr = (
            int(mean_rgb[2]),
            int(mean_rgb[1]),
            int(mean_rgb[0]),
        )

        swatch_y = min(
            semantic_panel.shape[0] - 28,
            start_y + len(lines[1:]) * line_height + 18,
        )

        cv2.putText(
            semantic_panel,
            "Couleur moyenne",
            (14, swatch_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        cv2.rectangle(
            semantic_panel,
            (SEMANTIC_PANEL_W - 58, swatch_y - 18),
            (SEMANTIC_PANEL_W - 20, swatch_y + 8),
            swatch_bgr,
            -1,
        )
        cv2.rectangle(
            semantic_panel,
            (SEMANTIC_PANEL_W - 58, swatch_y - 18),
            (SEMANTIC_PANEL_W - 20, swatch_y + 8),
            (255, 255, 255),
            1,
        )

    cv2.putText(
        vis,
        "t=target | h=recentrer | r=reset | q=quit",
        (15, vis.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (230, 230, 230),
        1,
    )
    # Le panneau est placé à droite du flux caméra.
    final_view = np.hstack((vis, semantic_panel))
    return final_view


def open_camera(camera_id: int) -> cv2.VideoCapture:
    if os.name == "nt":
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()

    return cv2.VideoCapture(camera_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=CAMERA_ID, help="Index caméra des yeux Reachy, souvent 0 ou 1")
    parser.add_argument("--checkpoint", default="", help="Chemin vers mobileclip_s2.pt")
    parser.add_argument("--yolo", default=YOLO_MODEL, help="Modèle Ultralytics")
    parser.add_argument("--no-mobileclip", action="store_true", help="Tester seulement YOLO")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Teste la vision sans envoyer de commande aux moteurs des yeux ni de la tête",
    )
    parser.add_argument(
        "--target",
        default="",
        help='Cible définie au lancement, ex. --target "chapeau vert"',
    )
    parser.add_argument(
        "--test-eyes",
        action="store_true",
        help="Effectue un petit test gauche/droite/haut/bas des yeux au démarrage",
    )
    parser.add_argument(
        "--test-head",
        action="store_true",
        help="Effectue un petit test horizontal de la tête au démarrage",
    )
    parser.add_argument(
        "--no-head",
        action="store_true",
        help="Désactive uniquement l'assistance horizontale de la tête",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[YOLO] Chargement : {args.yolo}")
    model = YOLO(args.yolo)
    yolo_names = set(model.names.values())

    clip_helper = MobileClipHelper(
        checkpoint_argument=args.checkpoint,
        enabled=not args.no_mobileclip,
    )
    kalman = KalmanTracker2D()
    motion = MotionAnalyzer()
    target_lock = TargetLock()
    gaze = ReachyEyeHeadController(
        robot_enabled=not args.dry_run,
        head_enabled=not args.no_head,
    )
    if args.test_eyes:
        gaze.test_eyes()
    if args.test_head:
        gaze.test_head()

    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Impossible d'ouvrir la caméra {args.camera}. Essaie --camera 1 ou ferme les applications qui utilisent la webcam."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)

    target_text: str | None = None
    user_request_text: str | None = None
    state = "Press t"
    typing_mode = False
    typing_buffer = ""

    frame_index = 0
    fps = 0.0
    fps_count = 0
    fps_start = time.perf_counter()

    objects: list[dict[str, Any]] = []
    semantic_obj: dict[str, Any] | None = None
    target_obj: dict[str, Any] | None = None
    active_center: list[int] | None = None
    motion_info: dict[str, Any] = motion.current_info()
    is_predicted = False
    last_locked_time = 0.0
    last_mobileclip_frame = -10_000

    def set_target(word: str) -> None:
        nonlocal target_text, user_request_text, state
        nonlocal semantic_obj, target_obj, active_center
        nonlocal motion_info, is_predicted, last_locked_time, last_mobileclip_frame

        # Garde la phrase exacte saisie pour l'affichage dans le panneau.
        user_request_text = word.strip()
        target_text = normalize_target(word)
        if is_color_target(target_text):
            state = f"OPENCV SEARCHING COLOR: {target_display_name(target_text)}"
        elif is_hybrid_target(target_text):
            state = f"SEARCHING OBJECT + COLOR: {target_display_name(target_text)}"
        else:
            state = f"SEARCHING: {target_display_name(target_text)}"
        semantic_obj = None
        target_obj = None
        active_center = None
        is_predicted = False
        last_locked_time = 0.0
        last_mobileclip_frame = -10_000
        kalman.reset()
        motion.reset()
        motion_info = motion.current_info()
        target_lock.reset()
        gaze.reset()
        clip_helper.reset()
        if is_color_target(target_text):
            mode = "OpenCV couleur"
        elif is_hybrid_target(target_text):
            mode = "objet + couleur (YOLO/MobileCLIP + OpenCV)"
        else:
            mode = "YOLO/MobileCLIP"
        print(f"[TARGET] {target_display_name(target_text)} | mode={mode}")

    if args.target.strip():
        set_target(args.target.strip())

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[CAMERA] Lecture interrompue.")
                break

            frame_index += 1
            fps_count += 1
            now_wall = time.time()
            now_perf = time.perf_counter()

            color_mode = is_color_target(target_text)
            hybrid_mode = is_hybrid_target(target_text)
            ran_detection = False

            # ROUTE 1 : cible = couleur -> OpenCV uniquement.
            # Aucune inférence YOLO ni MobileCLIP n'est appelée dans ce bloc.
            if color_mode and frame_index % COLOR_DETECT_EVERY == 0:
                ran_detection = True
                wanted_color = target_display_name(target_text)
                objects = build_opencv_color_objects(frame, wanted_color)
                selected_obj = select_opencv_color_object(objects, target_lock, frame.shape)
                semantic_obj = None

                if selected_obj is not None:
                    target_obj = selected_obj
                    cx, cy = selected_obj["center"]
                    if EDGE_MARGIN < cx < frame.shape[1] - EDGE_MARGIN and EDGE_MARGIN < cy < frame.shape[0] - EDGE_MARGIN:
                        filtered_x, filtered_y = kalman.update(cx, cy, now_perf)
                        active_center = [filtered_x, filtered_y]
                        motion_info = motion.update(active_center, now_perf)
                        last_locked_time = now_wall
                        target_lock.update(selected_obj)
                        is_predicted = False
                        state = f"OPENCV COLOR LOCKED: {wanted_color}"
                    else:
                        prediction = kalman.predict(now_perf)
                        active_center = list(prediction) if prediction is not None else None
                        is_predicted = prediction is not None
                        state = f"OPENCV EDGE: {wanted_color}"
                elif now_wall - last_locked_time < TARGET_LOCK_TIME:
                    prediction = kalman.predict(now_perf)
                    active_center = list(prediction) if prediction is not None else None
                    is_predicted = prediction is not None
                    state = f"OPENCV MEMORY: {wanted_color}"
                else:
                    active_center = None
                    target_obj = None
                    kalman.reset()
                    motion.reset()
                    motion_info = motion.current_info()
                    target_lock.reset()
                    state = f"OPENCV SEARCHING COLOR: {wanted_color}"

            # ROUTE 2 : cible = objet -> fonctionnement normal YOLO/MobileCLIP.
            elif not color_mode and frame_index % DETECT_EVERY == 0:
                ran_detection = True
                results = model(
                    frame,
                    imgsz=YOLO_IMGSZ,
                    conf=YOLO_CONF,
                    retina_masks=False,
                    verbose=False,
                )
                objects = build_yolo_objects(frame, results[0] if results else None)

                if target_text is not None:
                    selected_obj: dict[str, Any] | None = None
                    semantic_obj = None

                    if hybrid_mode:
                        object_query = hybrid_object_name(target_text)
                        wanted_color = hybrid_color_name(target_text)
                        color_mask_u8 = build_opencv_color_mask(frame, wanted_color)
                        annotated_objects = annotate_objects_with_requested_color(
                            objects,
                            color_mask_u8,
                            wanted_color,
                        )
                        color_candidates = filter_objects_by_requested_color(
                            annotated_objects,
                            wanted_color,
                        )

                        # Si la classe existe directement dans YOLO, on commence par
                        # une sélection exacte parmi les objets de la bonne couleur.
                        if object_query in yolo_names:
                            selected_obj = find_hybrid_yolo_target(
                                color_candidates,
                                object_query,
                                target_lock,
                                frame.shape,
                            )

                        # Sinon (ou si YOLO n'a pas fourni la bonne classe),
                        # MobileCLIP choisit uniquement parmi les régions validées
                        # par OpenCV. En dernier recours, les composantes de couleur
                        # OpenCV deviennent elles-mêmes des crops MobileCLIP.
                        if selected_obj is None:
                            cached = clip_helper.match_cached_object(
                                color_candidates,
                                object_query,
                                target_lock,
                                frame.shape,
                            )
                            if frame_index - last_mobileclip_frame >= MOBILECLIP_EVERY:
                                if color_candidates:
                                    selected_obj = clip_helper.find_target(
                                        frame,
                                        color_candidates,
                                        object_query,
                                        target_lock,
                                    )

                                if selected_obj is None:
                                    color_regions = prepare_color_regions_for_hybrid(
                                        frame,
                                        wanted_color,
                                    )
                                    selected_obj = clip_helper.find_target(
                                        frame,
                                        color_regions,
                                        object_query,
                                        target_lock,
                                    )

                                last_mobileclip_frame = frame_index
                                if selected_obj is None:
                                    selected_obj = cached
                            else:
                                selected_obj = cached
                            semantic_obj = selected_obj
                    elif target_text in yolo_names:
                        selected_obj = find_target_yolo(
                            objects,
                            target_text,
                            target_lock,
                            frame.shape,
                        )
                    else:
                        cached = clip_helper.match_cached_object(
                            objects,
                            target_text,
                            target_lock,
                            frame.shape,
                        )
                        if frame_index - last_mobileclip_frame >= MOBILECLIP_EVERY:
                            selected_obj = clip_helper.find_target(
                                frame,
                                objects,
                                target_text,
                                target_lock,
                            )
                            last_mobileclip_frame = frame_index
                            if selected_obj is None:
                                selected_obj = cached
                        else:
                            selected_obj = cached
                        semantic_obj = selected_obj

                    if selected_obj is not None:
                        selected_obj = enrich_selected_object_semantics(frame, selected_obj)
                        target_obj = selected_obj
                        if semantic_obj is not None:
                            semantic_obj = selected_obj
                        cx, cy = selected_obj["center"]
                        if EDGE_MARGIN < cx < frame.shape[1] - EDGE_MARGIN and EDGE_MARGIN < cy < frame.shape[0] - EDGE_MARGIN:
                            filtered_x, filtered_y = kalman.update(cx, cy, now_perf)
                            active_center = [filtered_x, filtered_y]
                            motion_info = motion.update(active_center, now_perf)
                            last_locked_time = now_wall
                            target_lock.update(selected_obj)
                            is_predicted = False
                            if hybrid_mode:
                                state = f"OBJECT + COLOR LOCKED: {target_display_name(target_text)}"
                            elif semantic_obj is not None:
                                state = (
                                    f"MOBILECLIP LOCKED: {target_display_name(target_text)} "
                                    f"({semantic_obj.get('mobileclip_combined_score', 0.0):.2f})"
                                )
                            else:
                                state = f"YOLO LOCKED: {target_display_name(target_text)}"
                        else:
                            prediction = kalman.predict(now_perf)
                            active_center = list(prediction) if prediction is not None else None
                            is_predicted = prediction is not None
                            state = f"EDGE PREDICTION: {target_display_name(target_text)}"
                    elif now_wall - last_locked_time < TARGET_LOCK_TIME:
                        prediction = kalman.predict(now_perf)
                        active_center = list(prediction) if prediction is not None else None
                        is_predicted = prediction is not None
                        state = f"KALMAN MEMORY: {target_display_name(target_text)}"
                    else:
                        active_center = None
                        semantic_obj = None
                        target_obj = None
                        kalman.reset()
                        motion.reset()
                        motion_info = motion.current_info()
                        target_lock.reset()
                        object_for_readiness = hybrid_object_name(target_text) if hybrid_mode else target_text
                        if object_for_readiness not in yolo_names and not clip_helper.ready:
                            state = "MOBILECLIP NOT READY"
                        elif hybrid_mode:
                            state = f"SEARCHING OBJECT + COLOR: {target_display_name(target_text)}"
                        else:
                            state = f"SCANNING: {target_display_name(target_text)}"

            elif target_text is not None and kalman.initialized and now_wall - last_locked_time < TARGET_LOCK_TIME:
                prediction = kalman.predict(now_perf)
                if prediction is not None:
                    active_center = list(prediction)
                    is_predicted = True

            if active_center is not None:
                gaze.update_target(active_center, frame.shape)
            elif target_text is not None:
                gaze.scan(frame.shape)

            elapsed = now_perf - fps_start
            if elapsed >= 1.0:
                fps = fps_count / elapsed
                fps_count = 0
                fps_start = now_perf

            vis = draw_scene(
                frame,
                objects,
                target_text,
                user_request_text,
                active_center,
                state,
                fps,
                gaze,
                is_predicted=is_predicted,
                semantic_obj=semantic_obj,
                target_obj=target_obj,
                motion_info=motion_info,
            )

            if typing_mode:
                cv2.putText(
                    vis,
                    f"TYPE: {typing_buffer}_",
                    (15, 116),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow(WINDOW_NAME, cv2.resize(vis, (WINDOW_W, WINDOW_H)))
            key = cv2.waitKey(1) & 0xFF

            if typing_mode:
                if key in (13, 10):
                    if typing_buffer.strip():
                        set_target(typing_buffer.strip())
                    typing_mode = False
                    typing_buffer = ""
                elif key == 27:
                    typing_mode = False
                    typing_buffer = ""
                elif key in (8, 127):
                    typing_buffer = typing_buffer[:-1]
                elif 32 <= key <= 126:
                    typing_buffer += chr(key)
            else:
                if key == ord("q"):
                    break
                if key == ord("t"):
                    typing_mode = True
                    typing_buffer = ""
                elif key == ord("h"):
                    target_text = None
                    user_request_text = None
                    active_center = None
                    semantic_obj = None
                    target_obj = None
                    state = "RECENTERED - Press t"
                    is_predicted = False
                    last_locked_time = 0.0
                    last_mobileclip_frame = -10_000
                    kalman.reset()
                    motion.reset()
                    motion_info = motion.current_info()
                    target_lock.reset()
                    clip_helper.reset()
                    gaze.recenter(frame.shape)
                elif key == ord("r"):
                    target_text = None
                    user_request_text = None
                    active_center = None
                    semantic_obj = None
                    target_obj = None
                    state = "Press t or use --target"
                    is_predicted = False
                    kalman.reset()
                    motion.reset()
                    motion_info = motion.current_info()
                    target_lock.reset()
                    gaze.reset()
                    clip_helper.reset()
                elif key == ord("p"):
                    set_target("cell phone")
                elif key == ord("k"):
                    set_target("keyboard")
                elif key == ord("b"):
                    set_target("bottle")
                elif key == ord("c"):
                    set_target("cup")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        gaze.close()


if __name__ == "__main__":
    main()