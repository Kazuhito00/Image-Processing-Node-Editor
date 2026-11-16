# Guide de Construction d'un Exécutable (.exe) pour CV_Studio

## Vue d'ensemble

Ce guide explique comment créer un fichier exécutable Windows (.exe) autonome pour CV_Studio qui inclut tous les nœuds, en particulier les nœuds de détection d'objets ONNX.

## 🎯 Objectif

Créer un fichier `.exe` qui :
- ✅ Fonctionne de manière autonome (pas besoin d'installer Python)
- ✅ Inclut tous les nœuds (Input, Process, DL, Audio, etc.)
- ✅ Contient tous les modèles ONNX pour la détection d'objets
- ✅ Embarque toutes les dépendances nécessaires
- ✅ Peut être distribué facilement

## 📋 Prérequis

### Logiciels requis

1. **Python 3.7 ou supérieur** (testé avec Python 3.12)
2. **Git** pour cloner le dépôt
3. **Visual C++ Redistributable** (pour l'exécution)

### Installation des dépendances

```bash
# Cloner le dépôt
git clone https://github.com/hackolite/CV_Studio.git
cd CV_Studio

# Installer les dépendances Python
pip install -r requirements.txt

# Installer PyInstaller (outil de construction)
pip install pyinstaller
```

## 🚀 Construction rapide

### Méthode 1 : Script automatique (RECOMMANDÉ)

La méthode la plus simple est d'utiliser le script de construction automatique :

```bash
# Construction standard
python build_exe.py

# Construction avec nettoyage préalable
python build_exe.py --clean

# Construction en mode fenêtré (sans console)
python build_exe.py --windowed

# Construction avec icône personnalisée
python build_exe.py --icon CV_Studio.ico
```

Le script va :
1. ✅ Vérifier les dépendances
2. ✅ Nettoyer les anciens builds (si --clean)
3. ✅ Configurer la construction
4. ✅ Compiler l'exécutable
5. ✅ Créer la documentation

### Méthode 2 : Construction manuelle avec PyInstaller

Si vous préférez plus de contrôle :

```bash
# Utiliser le fichier spec pré-configuré
pyinstaller CV_Studio.spec

# Ou construction directe (sans spec)
pyinstaller --name CV_Studio ^
            --add-data "node;node" ^
            --add-data "node_editor;node_editor" ^
            --add-data "src;src" ^
            --hidden-import dearpygui ^
            --hidden-import cv2 ^
            --hidden-import onnxruntime ^
            --collect-all mediapipe ^
            main.py
```

## 📂 Structure de sortie

Après la construction, vous obtiendrez :

```
dist/CV_Studio/
├── CV_Studio.exe           # Exécutable principal ← LANCEZ CECI
├── README.txt              # Documentation d'utilisation
├── node/                   # Tous les nœuds
│   ├── DLNode/            # Nœuds Deep Learning
│   │   └── object_detection/
│   │       ├── YOLOX/model/*.onnx      # Modèles YOLOX
│   │       ├── YOLO/model/*.onnx       # Modèles YOLO
│   │       ├── FreeYOLO/model/*.onnx   # Modèles FreeYOLO
│   │       └── ...
│   ├── InputNode/         # Nœuds d'entrée
│   ├── ProcessNode/       # Nœuds de traitement
│   ├── AudioProcessNode/  # Nœuds audio
│   └── ...
├── node_editor/           # Éditeur de nœuds
│   ├── font/             # Polices
│   └── setting/          # Fichiers de configuration
├── src/                   # Utilitaires source
└── _internal/            # Runtime Python et dépendances
```

## 🎮 Utilisation de l'exécutable

### Lancement simple

```bash
# Double-clic sur le fichier
CV_Studio.exe

# Ou depuis la ligne de commande
cd dist\CV_Studio
CV_Studio.exe
```

### Options de ligne de commande

```bash
# Avec fichier de configuration personnalisé
CV_Studio.exe --setting mon_config.json

# Mode debug
CV_Studio.exe --use_debug_print

# Désactiver le rendu asynchrone
CV_Studio.exe --unuse_async_draw

# Smoke test : démarrage complet, rendu de 30 frames, puis sortie (code 0 = OK).
# Utilisé par le workflow GitHub Actions pour valider chaque build.
CV_Studio.exe --smoke-test
```

## 🧪 Test de l'exécutable

### Vérification de base

1. **Lancer l'application**
   ```bash
   dist\CV_Studio\CV_Studio.exe
   ```

2. **Tester un nœud simple**
   - Ajouter un nœud "Image" (Input → Image)
   - Sélectionner une image
   - Ajouter un nœud "Result Image"
   - Connecter les deux nœuds

3. **Tester la détection d'objets ONNX**
   - Ajouter un nœud "Image" ou "WebCam"
   - Ajouter un nœud "Object Detection" (VisionModel → Object Detection)
   - Sélectionner un modèle (ex: YOLOX nano)
   - Ajouter un nœud "Draw Information"
   - Connecter : Input → Object Detection → Draw Information → Result Image

### Vérification des modèles ONNX

Les modèles suivants doivent être présents et fonctionnels :

```
node/DLNode/object_detection/
├── YOLOX/model/
│   ├── yolox_nano.onnx    ✅
│   ├── yolox_tiny.onnx    ✅
│   ├── yolox_s.onnx       ✅
│   └── yolo11_n.onnx      ✅
├── FreeYOLO/model/
│   └── freeyolo.onnx      ✅
└── TennisYOLO/model/
    └── tennis.onnx        ✅
```

## 🎨 Options de construction avancées

### Mode fenêtré (sans console)

Pour une application purement GUI sans fenêtre de console :

```bash
python build_exe.py --windowed
```

### Fichier unique (onefile) — RECOMMANDÉ pour la distribution

Crée un **seul `CV_Studio.exe`** autonome. À l'exécution, il s'extrait dans un
dossier temporaire (lecture seule). Toutes les données utilisateur sont stockées
dans un dossier `CV_Studio_data/` **à côté du .exe**.

```bash
# Compilation onefile avec le spec dédié
pyinstaller CV_Studio_onefile.spec
```

Le .exe résultant se trouve dans `dist/CV_Studio.exe`.

#### Arborescence du dossier de données (créé automatiquement)

```
MonDossier/
├── CV_Studio.exe                  ← l'exécutable unique
└── CV_Studio_data/                ← créé au premier lancement
    ├── _VideoWriter/              ← vidéos enregistrées & métadonnées
    ├── models/                    ← modèles ONNX uploadés par l'utilisateur
    │   ├── object_detection/
    │   ├── classification/
    │   ├── face_detection/
    │   ├── pose_estimation/
    │   ├── semantic_segmentation/
    │   ├── monocular_depth_estimation/
    │   └── audio/
    └── registries/                ← registres JSON persistants
        ├── od_custom_models_registry.json
        ├── cls_custom_models_registry.json
        ├── fd_custom_models_registry.json
        ├── pose_custom_models_registry.json
        ├── seg_custom_models_registry.json
        ├── depth_custom_models_registry.json
        └── audio_models_registry.json
```

#### Pourquoi cette organisation ?

| Problème | Solution |
|----------|----------|
| `_MEIPASS` de PyInstaller est temporaire et en lecture seule | Écriture à côté du .exe |
| VideoWriter nécessite des enregistrements persistants | `CV_Studio_data/_VideoWriter/` |
| Les modèles uploadés doivent survivre aux redémarrages | `CV_Studio_data/models/<type>/` |
| Les JSON de registre doivent être modifiables | `CV_Studio_data/registries/` |
| Modèles ONNX intégrés (bundled) | Inclus dans le .exe (lus depuis `_MEIPASS`) |
| Settings (`setting.json`) | Inclus dans le .exe (valeurs par défaut en lecture seule) |

#### Notes

- **Le premier démarrage est plus lent** (~5-15 sec) car le .exe extrait ~300 Mo vers un dossier temp.
- Le dossier `CV_Studio_data/` est portable — vous pouvez copier le exe + dossier ensemble.
- Les modèles intégrés (YOLOX, YAMNet, etc.) sont dans le .exe, pas besoin d'être dans `CV_Studio_data/`.
- Si vous supprimez `CV_Studio_data/`, il sera recréé (vide) au prochain lancement.

### Icône personnalisée

```bash
python build_exe.py --icon mon_icone.ico
```

### Build de debug

Pour le débogage :

```bash
python build_exe.py --debug
```

## 📦 Distribution

### Préparer la distribution

1. **Tester l'exécutable** sur votre machine
2. **Compresser le dossier**
   ```bash
   # Créer une archive ZIP
   cd dist
   tar -a -c -f CV_Studio_v1.0.zip CV_Studio
   ```

3. **Partager l'archive**
   - Uploader sur GitHub Releases
   - Partager via Google Drive / Dropbox
   - Distribuer directement

### Ce que les utilisateurs doivent faire

1. Télécharger l'archive ZIP
2. Extraire le dossier `CV_Studio`
3. Lancer `CV_Studio.exe`

**C'est tout !** Aucune installation Python requise.

### Taille approximative

- Build standard : ~800 MB - 1.5 GB
  - Python runtime : ~100 MB
  - OpenCV + dépendances : ~200 MB
  - ONNX Runtime : ~100 MB
  - Modèles ONNX : ~100-500 MB
  - Autres dépendances : ~300 MB

## 🔧 Dépannage

### Problème : PyInstaller non trouvé

```bash
pip install pyinstaller
```

### Problème : Dépendances manquantes (ModuleNotFoundError: No module named 'cv2')

Si vous rencontrez `ModuleNotFoundError` en exécutant `python build_exe.py`, cela signifie que les packages Python requis ne sont pas installés.

**Solution 1 : Laisser le script d'installation les installer automatiquement (Recommandé)**

Exécutez le script de construction et lorsque vous y êtes invité, sélectionnez l'option 1 :

```bash
python build_exe.py --clean

# Lorsqu'on vous le demande, choisissez l'option 1 pour installer les packages automatiquement
# (Le message s'affiche en anglais)
Choose option (1/2/3) [1]: 1
```

**Solution 2 : Installer manuellement d'abord**

```bash
# Installer toutes les dépendances avant la construction
pip install -r requirements.txt

# Puis exécuter le script de construction
python build_exe.py --clean
```

**Solution 3 : Ignorer la vérification des packages (environnements CI/CD)**

Si les packages sont déjà installés mais la vérification échoue, utilisez :

```bash
python build_exe.py --clean --skip-package-check
```

**Note** : La construction nécessite tous les packages de `requirements.txt` incluant :
- opencv-contrib-python (cv2)
- onnxruntime-gpu
- dearpygui
- numpy
- mediapipe
- Et beaucoup d'autres...

### Problème : Erreur d'importation onnxruntime à la ligne 26 (DLL load failed)

Si vous rencontrez une erreur comme celle-ci lors de l'exécution de `python build_exe.py --clean` :

```
File "C:\Users\...\site-packages\onnxruntime\__init__.py", line 26 in <module>
ImportError: DLL load failed while importing onnxruntime_pybind11_state
```

Cela indique qu'onnxruntime ne peut pas charger ses dépendances d'exécution C++ requises.

**Cause principale** : onnxruntime nécessite l'installation de Visual C++ Redistributable sur Windows. Sans cela, le package est installé mais ne peut pas charger ses fichiers DLL natifs.

**Solution 1 : Installer Visual C++ Redistributable (Recommandé)**

1. Téléchargez le Visual C++ Redistributable de Microsoft :
   - [VC++ Redistributable x64 (2015-2022)](https://aka.ms/vs/17/release/vc_redist.x64.exe)
   - Il s'agit du package Microsoft Visual C++ 2015-2022 Redistributable
   - Compatible avec Visual Studio 2015, 2017, 2019 et 2022
2. Exécutez l'installateur
3. Redémarrez votre terminal/invite de commande
4. Essayez de relancer la construction :
   ```bash
   python build_exe.py --clean
   ```

**Solution 2 : Utiliser le flag skip-package-check**

Si vous êtes certain que tous les packages sont correctement installés (par exemple, dans un environnement CI/CD), vous pouvez ignorer la vérification des packages :

```bash
python build_exe.py --clean --skip-package-check
```

**Note** : Cette erreur est spécifique à Windows. Sur Linux/macOS, onnxruntime fonctionne généralement sans dépendances supplémentaires.

### Problème : Erreur "module not found" dans l'exe

Ajouter le module manquant dans `CV_Studio.spec` :

```python
hiddenimports += [
    'nom_du_module_manquant',
]
```

Puis reconstruire :

```bash
pyinstaller CV_Studio.spec
```

### Problème : Modèles ONNX non trouvés

Vérifier que les modèles sont inclus dans `datas` dans le fichier spec :

```python
# Dans CV_Studio.spec
datas.append(('node/DLNode', 'node/DLNode'))
```

### Problème : FileNotFoundError pour setting.json

**Message d'erreur :**
```
FileNotFoundError: [Errno 2] No such file or directory: 
'C:\Users\...\AppData\Local\Temp\_MEI...\node_editor\setting\setting.json'
```

**Cause principale :** Cela se produit lorsque le fichier de configuration `setting.json` n'est pas correctement inclus dans l'exécutable, ou lorsqu'il y a des entrées de données conflictuelles dans le fichier spec de PyInstaller.

**Solution :**

1. **Vérifier le fichier spec** (`CV_Studio.spec`) a les bonnes entrées de données :

```python
# Correct - Ajouter les répertoires entiers une seule fois
datas.append(('node_editor', 'node_editor'))

# INCORRECT - Ne pas ajouter les sous-répertoires séparément car cela peut causer des conflits
# datas.append(('node_editor/setting', 'node_editor/setting'))  # Supprimer ceci !
# datas.append(('node_editor/font', 'node_editor/font'))        # Supprimer ceci !
```

2. **Reconstruire l'exécutable :**

```bash
# Nettoyer les constructions précédentes
python build_exe.py --clean

# Ou manuellement :
pyinstaller CV_Studio.spec --clean
```

3. **Activer les logs de débogage** pour diagnostiquer les problèmes de chemin :

```bash
dist\CV_Studio\CV_Studio.exe --use_debug_print
```

Cela affichera des informations détaillées sur la résolution des chemins incluant :
- Si l'exécution est en mode gelé (exe) ou script
- Chemin de base (emplacement _MEIPASS)
- Chemin résolu du fichier de configuration
- Si le fichier existe à cet emplacement

**Note :** Le correctif a été implémenté dans la dernière version. Si vous utilisez une version plus ancienne, mettez à jour le fichier `CV_Studio.spec` pour supprimer les entrées de données redondantes.

### Problème : L'exe ne démarre pas

1. **Tester depuis la ligne de commande** pour voir les erreurs :
   ```bash
   cd dist\CV_Studio
   CV_Studio.exe --use_debug_print
   ```

2. **Installer Visual C++ Redistributable** :
   - Télécharger : https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Installer et redémarrer

3. **Vérifier les permissions** :
   - Exécuter en tant qu'administrateur
   - Désactiver l'antivirus temporairement

### Problème : "Failed to execute script"

Reconstruire avec le mode debug pour voir les détails :

```bash
python build_exe.py --debug
```

### Problème : Performance faible

- Utiliser les modèles ONNX plus petits (nano, tiny)
- Désactiver l'accélération GPU si pas de GPU compatible
- Réduire la résolution de traitement

## 🌟 Fonctionnalités incluses

### Nœuds inclus dans l'exe

✅ **Input Nodes**
- Image, Video, WebCam, RTSP, Screen Capture
- Int Value, Float Value

✅ **Process Nodes**
- Blur, Brightness, Contrast, Canny
- Crop, Flip, Resize, Threshold, Grayscale
- Et plus...

✅ **Deep Learning Nodes**
- Object Detection (YOLOX, YOLO, FreeYOLO)
- Face Detection (YuNet, MediaPipe)
- Classification, Pose Estimation
- Semantic Segmentation
- Low-Light Enhancement, Depth Estimation

✅ **Audio Nodes**
- Audio processing and model nodes
- Spectrogram, ESC50 classification

✅ **Other Nodes**
- Tracking (MOT)
- Overlay (Draw, PutText, Image Concat)
- Visual (Result Image, RGB Histogram)
- Action (Video Writer, ON/OFF Switch)

### Modèles ONNX inclus

✅ **Object Detection**
- YOLOX (nano, tiny, small)
- YOLO11 (nano)
- FreeYOLO
- Tennis YOLO
- Lightweight Person Detector

✅ **Face Detection**
- YuNet

✅ **Classification**
- ResNet, MobileNet, EfficientNet

✅ **Autres**
- Depth estimation models
- Low-light enhancement models
- Segmentation models

## 📝 Personnalisation

### Modifier le fichier spec

Pour personnaliser la construction, éditez `CV_Studio.spec` :

```python
# Ajouter des modules cachés
hiddenimports += [
    'mon_module',
]

# Ajouter des fichiers de données
datas.append(('mon_dossier', 'mon_dossier'))

# Exclure des packages inutiles
excludes=[
    'package_a_exclure',
]

# Changer le nom de l'exe
name='MonApplication',

# Masquer la console
console=False,

# Ajouter une icône
icon='mon_icone.ico',
```

### Optimiser la taille

Pour réduire la taille de l'exe :

1. **Exclure des packages inutilisés** dans le spec
2. **Supprimer les modèles ONNX non utilisés**
3. **Utiliser UPX compression** (déjà activé)
4. **Nettoyer les fichiers de test/doc**

## 🔗 Liens utiles

- **PyInstaller Documentation** : https://pyinstaller.org/
- **CV_Studio GitHub** : https://github.com/hackolite/CV_Studio
- **ONNX Runtime** : https://onnxruntime.ai/
- **DearPyGUI** : https://github.com/hoffstadt/DearPyGui

## ✅ Checklist de construction

- [ ] Python 3.7+ installé
- [ ] Dépendances installées (`pip install -r requirements.txt`)
- [ ] PyInstaller installé (`pip install pyinstaller`)
- [ ] Exécuter `python build_exe.py`
- [ ] Tester `dist/CV_Studio/CV_Studio.exe`
- [ ] Vérifier que les nœuds ONNX fonctionnent
- [ ] Vérifier que tous les nœuds sont présents
- [ ] Créer l'archive ZIP pour distribution
- [ ] Tester sur une machine propre (sans Python)

## 🎓 Exemples d'utilisation

### Exemple 1 : Build standard

```bash
cd CV_Studio
python build_exe.py --clean
```

### Exemple 2 : Build pour distribution

```bash
# Build avec icône personnalisée et mode fenêtré
python build_exe.py --clean --windowed --icon logo.ico

# Tester
cd dist\CV_Studio
CV_Studio.exe

# Créer l'archive
cd dist
tar -a -c -f CV_Studio_Release_v1.0.zip CV_Studio
```

### Exemple 3 : Build de debug

```bash
# Build avec informations de debug
python build_exe.py --debug

# Lancer avec debug
dist\CV_Studio\CV_Studio.exe --use_debug_print
```

## 📞 Support

Pour toute question ou problème :

1. **Vérifier ce guide** en premier
2. **Consulter la documentation PyInstaller**
3. **Ouvrir une issue** sur GitHub : https://github.com/hackolite/CV_Studio/issues
4. **Vérifier les issues existantes** pour des problèmes similaires

---

**Bon build ! 🚀**
