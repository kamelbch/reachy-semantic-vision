# Résultats expérimentaux

Les images de ce dossier proviennent des essais réalisés avec le flux vidéo du robot Reachy.

## Modes validés visuellement

### Couleur seule — OpenCV HSV

- `color_red_opencv.png`
- détection d'une région rouge sans appel à YOLO ni MobileCLIP.

### Objet connu — YOLO

- `closed_set_bottle_yolo.png`
- sélection directe d'une classe appartenant au vocabulaire YOLO.

### Objet libre — MobileCLIP

- `open_vocab_penguin.png`
- `open_vocab_ps4_controller.png`
- `open_vocab_usb_flash_drive.png`

Ces exemples montrent que MobileCLIP peut sélectionner une région pertinente même lorsque la classe brute proposée par YOLO ne porte pas le bon nom.

### Objet et couleur

- `hybrid_black_glass.png`
- `hybrid_green_hat.png`
- `hybrid_black_hat.png`

Ce mode combine la validation de couleur par OpenCV HSV et la sélection sémantique par YOLO/MobileCLIP.

## Vidéos

Les vidéos montrant les yeux et la tête de Reachy en mouvement peuvent être ajoutées dans `results/videos`. Lorsque les fichiers sont trop lourds pour GitHub, ajouter ici un lien vers un dossier partagé.
