# Installation de MobileCLIP

## Rôle de MobileCLIP dans le projet

MobileCLIP est utilisé pour rechercher des objets à partir d'une requête
textuelle libre lorsque le nom demandé ne correspond pas directement à une
classe connue par YOLO.

Exemples de requêtes :

- `pingouin`
- `manette ps4`
- `chapeau`
- `flash disque`

YOLO produit d'abord des régions et des masques candidats. MobileCLIP compare
ensuite ces régions avec la requête saisie par l'utilisateur afin de
sélectionner l'objet le plus pertinent.

## Modèle retenu

La version finale du projet utilise MobileCLIP-S2 :

```python
MOBILECLIP_MODEL = "mobileclip_s2"
MOBILECLIP_REPO = "apple/MobileCLIP-S2"
MOBILECLIP_FILENAME = "mobileclip_s2.pt"
MOBILECLIP_LOCAL_PATH = "ml-mobileclip/checkpoints/mobileclip_s2.pt"
```

MobileCLIP-S2 a été retenu comme compromis entre les performances sémantiques
et le temps d'inférence sur le matériel disponible.

## Organisation locale attendue

Le dépôt GitHub ne contient ni le dépôt externe MobileCLIP ni le checkpoint
du modèle. Ils doivent être installés séparément.

L'organisation locale attendue est la suivante :

```text
reachy-semantic-vision/
├── main_reachy.py
├── head_control.py
├── ml-mobileclip/
│   ├── mobileclip/
│   ├── setup.py
│   └── checkpoints/
│       └── mobileclip_s2.pt
└── ...
```

## Installation

Il est recommandé d'utiliser Python 3.10.

### 1. Cloner le dépôt officiel MobileCLIP

Depuis la racine de `reachy-semantic-vision` :

```bash
git clone https://github.com/apple/ml-mobileclip.git ml-mobileclip
```

### 2. Installer MobileCLIP

Sous Windows PowerShell :

```powershell
python -m pip install -e .\ml-mobileclip
```

Sous Linux :

```bash
python -m pip install -e ./ml-mobileclip
```

### 3. Installer Hugging Face Hub

```bash
python -m pip install -U huggingface_hub
```

### 4. Créer le dossier des checkpoints

Sous Windows PowerShell :

```powershell
New-Item -ItemType Directory -Force .\ml-mobileclip\checkpoints
```

Sous Linux :

```bash
mkdir -p ./ml-mobileclip/checkpoints
```

### 5. Télécharger MobileCLIP-S2

Sous Windows PowerShell :

```powershell
hf download apple/MobileCLIP-S2 mobileclip_s2.pt --local-dir .\ml-mobileclip\checkpoints
```

Sous Linux :

```bash
hf download apple/MobileCLIP-S2 mobileclip_s2.pt --local-dir ./ml-mobileclip/checkpoints
```

Le fichier final doit se trouver ici :

```text
ml-mobileclip/checkpoints/mobileclip_s2.pt
```

## Téléchargement automatique

Le programme contient également l'option :

```python
MOBILECLIP_AUTO_DOWNLOAD = True
```

Si le checkpoint local n'est pas trouvé et que `huggingface_hub` est installé,
le programme peut télécharger automatiquement le fichier depuis Hugging Face.

## Autres variantes étudiées

Plusieurs variantes de MobileCLIP ont été considérées pendant le projet :

- MobileCLIP-S0 : modèle léger ;
- MobileCLIP-S1 : modèle intermédiaire ;
- MobileCLIP-S2 : modèle retenu dans la version finale ;
- MobileCLIP-B : modèle plus lourd ;
- MobileCLIP-B (LT) : variante du modèle B entraînée plus longtemps.

La version publiée de `main_reachy.py` est configurée pour MobileCLIP-S2.

## Remarque

Les checkpoints `.pt` peuvent être volumineux. Ils ne sont donc pas ajoutés
au dépôt GitHub et doivent être téléchargés séparément.
