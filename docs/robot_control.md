# Contrôle du robot Reachy 2019

## Vue d'ensemble

Le programme principal utilise `head_control.py` pour communiquer avec les servomoteurs Dynamixel responsables des yeux et de la tête.

Le contrôle est organisé en deux niveaux :

1. `main_reachy.py` calcule où se trouve la cible dans l'image ;
2. `head_control.py` transforme les commandes de haut niveau en écritures sur les moteurs Dynamixel.

## Acquisition du flux vidéo

La caméra des yeux est ouverte comme une webcam avec OpenCV :

```python
cap = cv2.VideoCapture(camera_id)
```

Le paramètre `--camera` permet de modifier l'index de la caméra.

## Calcul de l'erreur visuelle

Pour une image de largeur `W` et de hauteur `H`, le programme compare le centre de la cible au centre de l'image.

```text
erreur_x = centre_cible_x - W / 2
erreur_y = centre_cible_y - H / 2
```

- erreur horizontale positive : cible située à droite ;
- erreur horizontale négative : cible située à gauche ;
- erreur verticale positive : cible située vers le bas ;
- erreur verticale négative : cible située vers le haut.

## Stabilisation avec Kalman

Le centre brut fourni par la détection est stabilisé avec un filtre de Kalman. Le filtre réduit les variations rapides et fournit une courte prédiction lorsque la détection disparaît temporairement.

## Commande des yeux

Les fonctions principales sont :

```python
go_left(step)
go_right(step)
go_up(step)
go_down(step)
```

La taille du pas dépend de l'amplitude de l'erreur. Une zone morte évite de faire bouger les yeux lorsque la cible est déjà proche du centre.

Paramètres principaux dans `main_reachy.py` :

```python
EYE_KP_X
EYE_KP_Y
EYE_DEADZONE_X
EYE_DEADZONE_Y
EYE_MIN_STEP
EYE_MAX_STEP
EYE_CONTROL_COOLDOWN
```

## Assistance horizontale de la tête

La tête complète le mouvement horizontal des yeux avec une fréquence et des pas plus faibles.

Le mouvement vertical de la tête est volontairement désactivé :

```python
HEAD_VERTICAL_ENABLED = False
```

Paramètres principaux :

```python
HEAD_ASSIST_ENABLED
HEAD_SPEED
HEAD_KP_X
HEAD_MIN_STEP
HEAD_MAX_STEP
HEAD_DEADZONE_X
HEAD_TRACK_COOLDOWN
HEAD_H_LIMIT
```

## Scan automatique

Lorsqu'une cible est demandée mais n'est pas encore détectée :

1. les yeux effectuent un balayage horizontal puis vertical ;
2. la tête complète plus lentement le balayage horizontal ;
3. le pipeline continue à analyser le flux ;
4. le scan s'arrête dès que la cible est verrouillée.

Paramètres concernés :

```python
EYE_SCAN_ENABLED
EYE_SCAN_STEP
EYE_SCAN_COOLDOWN
EYE_SCAN_HORIZONTAL_COUNT
EYE_SCAN_VERTICAL_COUNT
HEAD_SCAN_STEP
HEAD_SCAN_COOLDOWN
```

## Recentrage

La touche `h` :

- efface la cible ;
- réinitialise le filtre de Kalman ;
- réinitialise la mémoire MobileCLIP ;
- replace les yeux à leur position mémorisée au démarrage ;
- replace l'axe horizontal de la tête à sa position initiale.

## Communication Dynamixel

Configuration connue :

```text
Port série : /dev/ttyUSB0
Baudrate : 1 000 000
Protocole : Dynamixel 1.0
```

Identifiants utilisés dans le fichier fourni :

```text
Tête horizontale : 0
Tête verticale   : 2
Œil vertical     : 5
Œil horizontal   : 6
```

Le fichier principal utilise notamment :

```python
Robot_Interface()
get_position()
set_position()
set_speed()
shutdown()
```

## Sécurité et limites

- Aucun bras ni aucune main n'est commandé.
- La tête verticale reste désactivée.
- Les pas, fréquences et amplitudes sont limités.
- Tout autre programme utilisant `/dev/ttyUSB0` doit être fermé avant le lancement.
- La caméra doit également être libérée par les autres logiciels.

## Statut du fichier de contrôle

Le pipeline principal et la logique de suivi ont été utilisés sur le robot au laboratoire. Le fichier `head_control.py` fourni dans ce dépôt est une reconstruction compatible fondée sur les constantes, les identifiants moteurs et les méthodes appelées par le programme principal.

L'ancien fichier exact n'étant plus disponible, ce fichier ne doit pas être présenté comme une copie bit à bit du module historique.
