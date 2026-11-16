# 📖 CV Studio — Documentation Exhaustive

> Application professionnelle de traitement d'image par nœuds (node-based) pour le développement, la vérification et la comparaison en vision par ordinateur.

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Installation](#2-installation)
3. [Démarrage rapide](#3-démarrage-rapide)
4. [Architecture du projet](#4-architecture-du-projet)
5. [Système de nœuds](#5-système-de-nœuds)
6. [Catalogue complet des nœuds](#6-catalogue-complet-des-nœuds)
7. [Éditeur de nœuds (Node Editor)](#7-éditeur-de-nœuds-node-editor)
8. [Système de files d'attente horodatées](#8-système-de-files-dattente-horodatées)
9. [Architecture src/ (moderne)](#9-architecture-src-moderne)
10. [Création de nœuds personnalisés](#10-création-de-nœuds-personnalisés)
11. [Configuration](#11-configuration)
12. [Tests](#12-tests)
13. [Construction d'exécutable (.exe)](#13-construction-dexécutable-exe)
14. [Intégrations externes](#14-intégrations-externes)
15. [Exemples et démos](#15-exemples-et-démos)
16. [Outils utilitaires](#16-outils-utilitaires)
17. [Dépannage](#17-dépannage)
18. [Contribution](#18-contribution)
19. [Licence](#19-licence)

---

## 1. Vue d'ensemble

### Qu'est-ce que CV Studio ?

CV Studio est une application avancée de traitement d'image basée sur un éditeur visuel de nœuds (nodes). Elle permet de créer des pipelines de vision par ordinateur par glisser-déposer, en connectant visuellement des blocs fonctionnels.

### Cas d'usage

| Domaine | Description |
|---------|-------------|
| **Prototypage** | Tester et comparer rapidement différents algorithmes CV |
| **Éducation** | Apprendre la vision par ordinateur de manière interactive |
| **Développement** | Construire et valider des pipelines avant la production |
| **Recherche** | Expérimenter avec des modèles ML et des techniques CV traditionnelles |
| **Surveillance** | Détection d'objets en temps réel, tracking, alertes |
| **Audio** | Classification audio, spectrogrammes, traitement du signal |

### Caractéristiques clés

- 🎨 **Éditeur visuel** — Interface drag-and-drop DearPyGUI avec zoom (10%–500%)
- 🔄 **Traitement temps réel** — Résultats instantanés lors de la construction du pipeline
- 🧩 **100+ nœuds intégrés** — Entrées, traitements, ML/DL, analyse, visualisation
- 🤖 **Intégration ML/DL** — ONNX, MediaPipe, modèles personnalisés
- 📹 **Sources multiples** — Webcam, vidéo, images, RTSP, capture d'écran, microphone
- 💾 **Import/Export** — Sauvegarde et chargement des graphes en JSON
- 🔌 **Extensible** — Ajout facile de nœuds personnalisés
- 🏗️ **Architecture moderne** — Gestion d'erreurs, logging, tests complets

---

## 2. Installation

### Prérequis

| Composant | Version minimale |
|-----------|-----------------|
| Python | 3.7+ |
| opencv-python | 4.5.5+ |
| onnxruntime | 1.16.0+ |
| dearpygui | 2.0.0+ |
| mediapipe | 0.10.0+ (optionnel) |
| librosa | 0.10.0+ (pour l'audio) |
| filterpy | 1.4.5+ (pour le tracking) |

### Méthode 1 : Installation directe (recommandée)

```bash
git clone https://github.com/hackolite/CV_Studio.git
cd CV_Studio
pip install -r requirements.txt
python main.py
```

### Méthode 2 : Environnement virtuel (développement)

```bash
python -m venv venv

# Activation
# Windows :
venv\Scripts\activate
# Linux/Mac :
source venv/bin/activate

pip install -r requirements.txt
python main.py
```

### Méthode 3 : Installation pip

```bash
pip install Cython numpy wheel
pip install git+https://github.com/hackolite/CV_Studio.git
ipn-editor
```

### Méthode 4 : Docker

Voir [docker/nvidia-gpu](https://github.com/Kazuhito00/Image-Processing-Node-Editor/tree/main/docker/nvidia-gpu) pour la configuration Docker avec support GPU.

### Dépendances complètes

Le fichier `requirements.txt` inclut :

| Catégorie | Packages |
|-----------|----------|
| **Vision** | opencv-contrib-python, Pillow |
| **ML/DL** | onnxruntime, onnx, mediapipe, openvino-dev, nncf, scikit-learn |
| **Audio** | librosa, soundfile, sounddevice |
| **Tracking** | filterpy, scipy, lap, motpy, norfair |
| **Vidéo** | ffmpeg-python, imageio-ffmpeg, pafy, yt-dlp, streamlink |
| **Réseau** | requests, websockets, pymongo, dnspython |
| **GUI** | dearpygui, matplotlib |
| **Système** | psutil, pyserial, python-dotenv |
| **Cartographie** | contextily, onvif-zeep, WSDiscovery |
| **Texte** | wordcloud |

---

## 3. Démarrage rapide

### Lancement

```bash
python main.py
```

### Options de ligne de commande

| Option | Description |
|--------|-------------|
| `--setting <chemin>` | Fichier de configuration personnalisé (défaut : `node_editor/setting/setting.json`) |
| `--unuse_async_draw` | Désactiver le dessin asynchrone (débogage) |
| `--use_debug_print` | Activer les logs de débogage |
| `--smoke-test [N]` | Démarrage complet, rendu de N frames (30 par défaut) puis sortie ; code 0 = OK (utilisé en CI) |

### Premiers pas

1. **Créer un nœud** — Sélectionner dans le menu, cliquer sur le canevas
2. **Connecter** — Glisser d'une sortie vers une entrée (même couleur = même type)
3. **Zoomer** — Molette de souris (10% à 500%)
4. **Supprimer** — Sélectionner + touche `Suppr`
5. **Exporter** — Menu Export pour sauvegarder le pipeline en JSON
6. **Importer** — Charger un pipeline précédemment sauvegardé

### Raccourcis clavier

| Action | Raccourci |
|--------|-----------|
| Ajouter un nœud | Clic menu + clic canevas |
| Supprimer | Sélection + `Suppr` |
| Panoramique | Clic molette ou `Ctrl` + clic gauche |
| Connecter | Glisser sortie → entrée |
| Déconnecter | Clic droit sur le lien |
| Multi-sélection | `Ctrl` + clic |

### Exemple : Pipeline basique

```
[Image] → [Blur] → [Canny] → [Result Image]
```

1. Ajouter un nœud **Image** (Input → Image)
2. Ajouter un nœud **Blur** (VisionProcess → Blur)
3. Ajouter un nœud **Canny** (VisionProcess → Canny)
4. Ajouter un nœud **Result Image** (Visual → Result Image)
5. Connecter les nœuds dans l'ordre
6. Charger une image et ajuster les paramètres

### Exemple : Détection d'objets webcam

```
[WebCam] → [Object Detection] → [Draw Information] → [Result Image]
```

---

## 4. Architecture du projet

### Arborescence principale

```
CV_Studio/
├── main.py                    # Point d'entrée principal
├── requirements.txt           # Dépendances Python
├── pyproject.toml             # Configuration build (setuptools)
├── setup.py                   # Installation pip
├── pytest.ini                 # Configuration pytest
│
├── node/                      # Implémentations des nœuds
│   ├── basenode.py            # Classe de base Node (1000+ lignes)
│   ├── node_abc.py            # Classe abstraite DpgNodeABC
│   ├── node_factory.py        # Pattern Factory
│   ├── timestamped_queue.py   # Système de files FIFO horodatées
│   ├── queue_adapter.py       # Adaptateur rétrocompatible
│   ├── InputNode/             # Nœuds d'entrée (11 fichiers)
│   ├── ProcessNode/           # Traitement d'image (24 fichiers)
│   ├── DLNode/                # Deep Learning (20+ fichiers)
│   ├── AudioModelNode/        # Classification audio
│   ├── AudioProcessNode/      # Traitement audio (9 fichiers)
│   ├── VideoNode/             # Composition vidéo (8 fichiers)
│   ├── OverlayNode/           # Superpositions visuelles (4 fichiers)
│   ├── VisualNode/            # Visualisations avancées (10 fichiers)
│   ├── TrackerNode/           # Suivi d'objets (2 fichiers)
│   ├── TriggerNode/           # Logique conditionnelle (7 fichiers)
│   ├── ActionNode/            # Actions/sorties (7+ fichiers)
│   ├── StatsNode/             # Statistiques (6 fichiers)
│   ├── RouterNode/            # Routage de données (1 fichier)
│   ├── NLPModelNode/          # Modèles NLP (1 fichier)
│   ├── TimeseriesNode/        # Séries temporelles (1 fichier)
│   └── SystemNode/            # Opérations système (5 fichiers)
│
├── node_editor/               # Cœur de l'éditeur GUI
│   ├── node_main.py           # Éditeur principal DpgNodeEditor
│   ├── util.py                # Utilitaires (caméra, serial)
│   ├── style.py               # Thèmes et couleurs
│   ├── font/                  # Polices personnalisées
│   ├── setting/               # Configuration par défaut
│   └── splash/                # Écran de démarrage
│
├── src/                       # Architecture moderne (refactorisée)
│   ├── core/                  # Logique métier
│   │   ├── nodes/             # BaseNode, EnhancedNode, Factory
│   │   ├── config/            # Gestion des paramètres
│   │   └── pipeline/          # Pipeline d'exécution
│   ├── nodes/                 # Adaptateurs de nœuds
│   ├── utils/                 # Utilitaires réutilisables
│   │   ├── exceptions.py      # Hiérarchie d'exceptions
│   │   ├── logging.py         # Logging centralisé
│   │   ├── gpu_utils.py       # Détection GPU
│   │   └── resource_manager.py # Gestion du cycle de vie
│   └── gui/                   # Composants GUI (futur)
│
├── tests/                     # Suite de tests (150+ fichiers)
├── examples/                  # Exemples exécutables
├── tools/                     # Outils utilitaires
├── docs/                      # Documentation supplémentaire
│
├── build_unified.py           # Build cross-platform
├── build_exe.py               # Build Windows (.exe)
├── build_windows.bat          # Script batch Windows
├── build_windows.ps1          # Script PowerShell Windows
└── build.sh                   # Script build Linux
```

### Flux de données

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  InputNode  │────▶│  ProcessNode /   │────▶│  VisualNode /   │
│  (source)   │     │  DLNode / etc.   │     │  ActionNode     │
└─────────────┘     └──────────────────┘     └─────────────────┘
       │                      │                        │
       ▼                      ▼                        ▼
┌─────────────────────────────────────────────────────────────┐
│            TimestampedQueue (FIFO horodaté)                  │
│   Chaque nœud publie ses données avec un timestamp.         │
│   Les nœuds consommateurs récupèrent en ordre FIFO.         │
└─────────────────────────────────────────────────────────────┘
```

### Types de données échangés

| Type | Constante | Description |
|------|-----------|-------------|
| Image | `TYPE_IMAGE` | Frame OpenCV (numpy array BGR) |
| Audio | `TYPE_AUDIO` | Chunk audio (dict avec samples, sr, etc.) |
| JSON | `TYPE_JSON` | Résultats structurés (détections, etc.) |
| Float | `TYPE_FLOAT` | Valeur flottante |
| Int | `TYPE_INT` | Valeur entière |
| Boolean | `TYPE_BOOLEAN` | Vrai/Faux |
| Time | `TYPE_TIME_MS` | Timestamp en millisecondes |

---

## 5. Système de nœuds

### Classe de base : `Node` (node/basenode.py)

Tous les nœuds héritent de la classe `Node` qui fournit :

#### Propriétés principales

```python
class Node:
    node_label = "BaseNode"      # Nom affiché dans l'interface
    node_tag = "BaseNode"        # Identifiant interne unique
    
    # Types de données
    TYPE_IMAGE = "IMAGE"
    TYPE_AUDIO = "AUDIO"
    TYPE_JSON = "JSON"
    TYPE_FLOAT = "FLOAT"
    TYPE_INT = "INT"
    TYPE_BOOLEAN = "BOOLEAN"
    TYPE_TIME_MS = "TIME_MS"
    
    # Directions de port
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
```

#### Méthodes essentielles

| Méthode | Description |
|---------|-------------|
| `__init__(node_id, connection_dict, opencv_setting_dict)` | Initialisation avec config |
| `update(node_id, connection_list, node_image_dict, node_result_dict)` | **Traitement principal** (appelé à chaque frame) |
| `close(node_id)` | Nettoyage des ressources |
| `convert_cv_to_dpg(image, w, h)` | Conversion OpenCV → texture DearPyGUI |
| `get_input_frame(connection_list, node_image_dict, node_audio_dict)` | Récupération de l'entrée |
| `get_setting_dict(node_id)` | Sérialisation des paramètres |
| `set_setting_dict(node_id, setting_dict)` | Restauration des paramètres |

#### Méthodes de visualisation

La classe `Node` fournit 20+ méthodes `draw_*_info()` pour dessiner :
- Boîtes englobantes (bounding boxes)
- Labels de classification
- Squelettes de pose
- Masques de segmentation
- Résultats de tracking
- QR codes détectés

### Classe abstraite : `DpgNodeABC` (node/node_abc.py)

```python
class DpgNodeABC(ABC):
    @abstractmethod
    def add_node(self, parent, node_id, pos, ...):
        """Crée le nœud dans l'interface DearPyGUI"""
        
    @abstractmethod
    def update(self, node_id, connection_list, node_image_dict, node_result_dict):
        """Logique de traitement"""
        
    @abstractmethod
    def close(self, node_id):
        """Libération des ressources"""
        
    @abstractmethod
    def get_setting_dict(self, node_id):
        """Export des paramètres"""
        
    @abstractmethod
    def set_setting_dict(self, node_id, setting_dict):
        """Import des paramètres"""
```

### Pattern Factory

Chaque fichier de nœud expose une classe `FactoryNode` :

```python
class FactoryNode:
    node_label = "Mon Nœud"     # Affiché dans le menu
    node_tag = "MonNoeud"       # Tag interne
    node_color = [R, G, B]      # Couleur du nœud
    
    def create(self, node_id, ...):
        """Instancie le nœud concret"""
        return MonNoeudImpl(node_id, ...)
```

---

## 6. Catalogue complet des nœuds

### 6.1 InputNode — Nœuds d'entrée

| Nœud | Fichier | Description |
|------|---------|-------------|
| **Image** | `node_image.py` | Lecture d'images statiques (bmp, jpg, png, gif) |
| **Video** | `node_video.py` | Lecture de vidéos (mp4, avi) avec boucle et skip |
| **WebCam** | `node_webcam.py` | Capture webcam en temps réel |
| **RTSP** | `node_rtsp.py` | Flux RTSP de caméras réseau |
| **YouTube** | `node_youtube.py` | Lecture de flux YouTube |
| **HLS** | `node_hls.py` | Flux HLS (HTTP Live Streaming) |
| **WebRTC** | `node_webrtc.py` | Streaming WebRTC |
| **Microphone** | `node_microphone.py` | Capture audio en temps réel (8–48 kHz) |
| **WebSocket** | `node_websocket.py` | Réception de données WebSocket |
| **API** | `node_api.py` | Entrée REST API |
| **MQTT** | `node_mqtt.py` | Messages MQTT |
| **Temperature** | `node_temperature.py` | Données de capteur de température |
| **Coordinate Examples** | `node_coordinate_examples.py` | Génération de coordonnées GPS |
| **JSON Boolean** | `node_json_boolean.py` | Valeur JSON/booléenne |
| **Int Value** | `_node_int_value.py` | Sortie d'un entier |
| **Float Value** | `_node_float_value.py` | Sortie d'un flottant |

### 6.2 ProcessNode — Traitement d'image

| Nœud | Fichier | Description |
|------|---------|-------------|
| **Blur** | `node_blur.py` | Flou gaussien |
| **Bilateral Filter** | `node_bilateral_filter.py` | Filtre bilatéral (préserve les bords) |
| **Brightness** | `node_brightness.py` | Ajustement de luminosité |
| **Contrast** | `node_contrast.py` | Ajustement de contraste |
| **Canny** | `node_canny.py` | Détection de contours Canny |
| **Threshold** | `node_threshold.py` | Seuillage simple |
| **Adaptive Threshold** | `node_adaptive_threshold.py` | Seuillage adaptatif |
| **Grayscale** | `node_grayscale.py` | Conversion en niveaux de gris |
| **Color Space** | `node_color_space.py` | Conversion d'espace couleur (HSV, LAB, etc.) |
| **Apply ColorMap** | `node_apply_color_map.py` | Application de pseudo-couleur |
| **Crop** | `node_crop.py` | Découpage d'image |
| **Resize** | `node_resize.py` | Redimensionnement |
| **Zoom** | `node_zoom.py` | Zoom sur une région |
| **Flip** | `node_flip.py` | Retournement (horizontal/vertical) |
| **Morphology** | `node_morphology.py` | Opérations morphologiques (érosion, dilatation, ouverture, fermeture) |
| **CLAHE** | `node_clahe.py` | Égalisation d'histogramme adaptative |
| **Equalize Hist** | `node_equalize_hist.py` | Égalisation d'histogramme |
| **Gamma Correction** | `node_gamma_correction.py` | Correction gamma |
| **Illumination Correct** | `node_illumination_correct.py` | Correction d'éclairage |
| **Alpha Blend** | `node_image_alpha_blend.py` | Mélange alpha de deux images |
| **Kernel Sharpen** | `node_kernel_sharpen.py` | Netteté par noyau |
| **NLM Denoise** | `node_nlm_denoise.py` | Débruitage Non-Local Means |
| **Unsharp Mask** | `node_unsharp_mask.py` | Masque de netteté |
| **Simple Filter** | `node_simple_filter.py` | Filtres de convolution personnalisés |
| **Omnidirectional Viewer** | `node_omnidirectional_viewer.py` | Visualisation 360° |
| **Crop Monitor** | `_node_crop_monitor.py` | Surveillance de zone recadrée |

### 6.3 DLNode — Deep Learning

| Nœud | Fichier | Description |
|------|---------|-------------|
| **Object Detection** | `node_object_detection.py` | Détection d'objets (YOLOX, YOLOv5, YOLOv8, FreeYOLO, ONNX personnalisé) |
| **Classification** | `node_classification.py` | Classification d'images (ResNet, EfficientNet, etc.) |
| **Face Detection** | `node_face_detection.py` | Détection de visages (YuNet, MediaPipe, MTCNN) |
| **Pose Estimation** | `node_pose_estimation.py` | Estimation de pose (OpenPose, MoveNet, MediaPipe) |
| **Semantic Segmentation** | `node_semantic_segmentation.py` | Segmentation sémantique (FCN, DeepLab) |
| **Monocular Depth** | `node_monocular_depth_estimation.py` | Estimation de profondeur (MiDaS) |
| **Low Light Enhancement** | `node_low_light_image_enhancement.py` | Amélioration d'images sombres |
| **Online Training** | `node_online_training.py` | Entraînement en temps réel (distillation teacher-student) — [📖 Documentation détaillée](../docs/DISTILLATION.md) |

#### Système de registre de modèles

Chaque catégorie DL possède son propre registre :
- `object_detection/custom_models_registry.json` — Modèles de détection
- `audio_models_registry.json` — Modèles audio
- Upload ONNX via interface graphique (bouton jaune "Add Model")
- Inspection automatique des métadonnées ONNX

#### Format d'entrée/sortie Object Detection

```python
# Entrée
input_image: np.ndarray  # BGR, n'importe quelle taille

# Sortie (JSON)
{
    "bboxes": [[x1, y1, x2, y2], ...],
    "scores": [0.95, 0.87, ...],
    "class_ids": [0, 1, ...],
    "class_names": ["person", "car", ...]
}
```

### 6.4 AudioModelNode — Classification audio

| Nœud | Fichier | Description |
|------|---------|-------------|
| **Audio Classification** | `node_audio_classification.py` | Classification de sons (YAMNet Qualcomm, modèles ONNX personnalisés) |

**Spécifications YAMNet :**
- Entrée : `[1, 1, 96, 64]` (time-major)
- Sample rate cible : 16 000 Hz
- Mel : 64 bins, FFT 512, hop 160, win 400
- Activation de sortie : sigmoid
- Seuil de silence RMS : 1e-3

### 6.5 AudioProcessNode — Traitement audio

| Nœud | Fichier | Description |
|------|---------|-------------|
| **Spectrogram** | `node_spectrogram.py` | Spectrogramme temps-fréquence (STFT, Mel) |
| **Equalizer** | `node_equalizer.py` | Égaliseur multi-bandes |
| **Compressor** | `node_compressor.py` | Compression dynamique |
| **Noise Gate** | `node_noise_gate.py` | Porte de bruit |
| **Bandpass Filter** | `node_bandpass_filter.py` | Filtre passe-bande |
| **Resample** | `node_resample.py` | Rééchantillonnage |
| **Decibel** | `node_decibel.py` | Conversion en dB |
| **Normalize** | `node_normalize.py` | Normalisation audio |

### 6.6 VideoNode — Composition vidéo

| Nœud | Fichier | Description |
|------|---------|-------------|
| **Video Writer** | `node_video_writer.py` | Écriture vidéo (MP4, AVI) avec synchronisation A/V |
| **Image Concat** | `node_image_concat.py` | Concaténation d'images (grille, côte à côte) |
| **Dynamic Play** | `node_dynamic_play.py` | Lecture dynamique avec contrôle |
| **Screen Capture** | `node_screen_capture.py` | Capture d'écran |

**Synchronisation A/V :**
- `sync.py` — SyncVideoWriter, FramePacket, AVDriftDetector
- `av_encoder.py` — PyAVEncoder pour l'encodage
- Déduplication par `chunk_index` et trim temporel par `pts_ms`

### 6.7 OverlayNode — Superpositions

| Nœud | Fichier | Description |
|------|---------|-------------|
| **Draw Information** | `node_draw_information.py` | Dessine bboxes, labels, scores sur l'image |
| **Overlay** | `node_overlay.py` | Superposition générique |
| **Overlay Image** | `node_overlay_image.py` | Superposition d'image (logo, watermark) |
| **PutText** | `node_puttext.py` | Ajout de texte sur l'image |

### 6.8 VisualNode — Visualisations avancées

| Nœud | Fichier | Description |
|------|---------|-------------|
| **Heatmap** | `node_heatmap.py` | Carte de chaleur 2D |
| **ObjHeatmap** | `node_obj_heatmap.py` | Carte de chaleur basée objets |
| **ObjChart** | `node_obj_chart.py` | Graphique statistique d'objets (24h) |
| **Map** | `node_map.py` | Visualisation cartographique (OpenStreetMap) |
| **Tennis Court** | `node_tennis_court.py` | Overlay terrain de tennis |
| **Vigilance Gauge** | `node_vigilance_gauge.py` | Jauge de vigilance |
| **Word Cloud** | `node_word_cloud.py` | Nuage de mots |

### 6.9 TrackerNode — Suivi d'objets

| Nœud | Fichier | Description |
|------|---------|-------------|
| **MOT** | `node_mot.py` | Multi-Object Tracking (SORT, ByteTrack, BOTSort, OC-SORT, MotPy, IOU, Norfair, CenterTrack) |
| **ReID** | `node_reid.py` | Ré-identification (apparence + mouvement) |

**Algorithmes de tracking disponibles :**
- SORT — Simple Online Realtime Tracking
- ByteTrack — Association par score bas/haut
- BOTSort — BoT-SORT amélioré
- OC-SORT — Observation-Centric SORT
- MotPy — Filtre de Kalman simple
- IOU Tracker — Association par IoU
- Norfair — Tracking flexible
- CenterTrack — Suivi par centre

### 6.10 TriggerNode — Logique conditionnelle

| Nœud | Fichier | Description |
|------|---------|-------------|
| **Trigger** | `node_trigger.py` | Déclencheur générique |
| **ON/OFF Switch** | `node_on_off_switch.py` | Interrupteur booléen |
| **Boolean Inverter** | `node_boolean_inverter.py` | Inverseur logique (NOT) |
| **ObjDetCount** | `node_objdetcount.py` | Déclencheur sur comptage d'objets |
| **DbDetCount** | `node_dbdetcount.py` | Comptage avec seuil dB |
| **Keypoint Deviation** | `node_trigger_keypoint_deviation.py` | Détection de déviation de pose |

### 6.11 ActionNode — Actions et sorties

| Nœud | Fichier | Description |
|------|---------|-------------|
| **Video Recorder** | `node_video_recorder.py` | Enregistrement vidéo conditionnel |
| **Buzzer** | `node_buzzer.py` | Alerte sonore |
| **MongoDB** | `node_mongodb.py` | Stockage en base MongoDB |
| **PTZ** | `node_ptz.py` | Contrôle caméra PTZ (ONVIF) |
| **VLM** | `node_vlm.py` | Vision-Language Model (inférence texte+image) |
| **REST API** | `restapi/` | Envoi de données vers API |
| **RabbitMQ** | `rabbitmq/` | Publication de messages |
| **Streaming** | `streaming/` | Diffusion HLS/RTMP |

### 6.12 StatsNode — Statistiques

| Nœud | Fichier | Description |
|------|---------|-------------|
| **Histogram** | `node_histo.py` | Histogramme d'image |
| **Bar Chart** | `node_bar.py` | Graphique à barres |
| **Homography** | `node_homography.py` | Calcul de matrice d'homographie |
| **Distance Tracker** | `node_distance_tracker.py` | Calcul de distances |
| **Keypoints Processing** | `node_dataprocessing_keypoints.py` | Analyse de points clés |
| **BRISQUE** | `BRISQUE/` | Score de qualité d'image |

### 6.13 RouterNode — Routage

| Nœud | Fichier | Description |
|------|---------|-------------|
| **Simple Router** | `node_simple_router.py` | Routage conditionnel de données |

### 6.14 NLPModelNode — Traitement du langage

| Nœud | Fichier | Description |
|------|---------|-------------|
| **TinyBERT Vigilance** | `node_tinybert_vigilance.py` | Détection de vigilance par texte |

### 6.15 TimeseriesNode — Séries temporelles

| Nœud | Fichier | Description |
|------|---------|-------------|
| **Position Prediction** | `node_position_prediction.py` | Prédiction de position (filtre de Kalman) |

### 6.16 SystemNode — Système

| Nœud | Fichier | Description |
|------|---------|-------------|
| **Sync Queue** | `node_sync_queue.py` | Synchronisation de files d'attente |
| **Deploy** | `node_deploy.py` | Export de schéma/déploiement |
| **OnvifScan** | `node_onvif_scan.py` | Scan de périphériques réseau |
| **System Resource** | `node_system_resource.py` | Monitoring CPU/GPU/RAM |

---

## 7. Éditeur de nœuds (Node Editor)

### Classe principale : `DpgNodeEditor`

Fichier : `node_editor/node_main.py` (~795 lignes)

#### Responsabilités

1. **Gestion des nœuds** — Chargement dynamique, instanciation, positionnement
2. **Interface graphique** — Viewport DearPyGUI, menu, thèmes
3. **Flux de données** — Connexions, tri topologique, exécution séquentielle
4. **Fichiers** — Export/import JSON des pipelines

#### Méthodes clés

| Méthode | Rôle |
|---------|------|
| `__init__(width, height, opencv_setting_dict, menu_dict, node_dir)` | Initialise l'éditeur |
| `_load_node_factory()` | Charge dynamiquement les nœuds depuis le filesystem |
| `_callback_add_node()` | Ajoute un nœud au canevas |
| `_callback_delete_node()` | Supprime un nœud |
| `_callback_create_link()` | Crée une connexion |
| `_callback_file_export()` / `_callback_file_import()` | Sauvegarde/chargement JSON |
| `get_sorted_node_connection()` | Tri topologique pour l'ordre d'exécution |
| `set_terminate_flag()` | Signal d'arrêt propre |

#### Système de thèmes

Chaque catégorie de nœud a sa propre couleur (définie dans `node_editor/style.py`) :
- Les nœuds sélectionnés ont une saturation/luminosité augmentée
- Les contrôles (sliders, boutons) adoptent la couleur du nœud
- Palette cohérente via le système de thèmes DearPyGUI

#### Fichier de configuration par défaut

`node_editor/setting/setting.json` contient :
- Résolution webcam (`webcam_width`, `webcam_height`)
- Résolution de traitement (`process_width`, `process_height`)
- Taille de la fenêtre (`editor_width`, `editor_height`)
- Activation GPU (`use_gpu`)
- Compteur de performance (`use_pref_counter`)
- Port série (`serial_port`, `enable_serial`)

---

## 8. Système de files d'attente horodatées

### Vue d'ensemble

Le système de files FIFO horodatées (`node/timestamped_queue.py`) assure :
- ✅ Récupération FIFO — Les données les plus anciennes sont traitées en premier
- ✅ Horodatage automatique — Chaque donnée reçoit un timestamp à la création
- ✅ Thread-safe — Accès concurrent sécurisé entre nœuds
- ✅ Rétrocompatibilité — Les nœuds existants fonctionnent sans modification
- ✅ Gestion mémoire — Limite automatique de taille (défaut : 10 éléments)

### Classes principales

#### `TimestampedData`

```python
@dataclass
class TimestampedData:
    data: Any           # Payload (image, audio, json...)
    timestamp: float    # Unix timestamp
    node_id: str        # ID du nœud source
```

#### `TimestampedQueue`

```python
class TimestampedQueue:
    def put(self, data, timestamp=None)    # Ajouter avec horodatage
    def get_latest(self)                    # Obtenir le plus récent (sans retirer)
    def get_oldest(self)                    # Obtenir le plus ancien (sans retirer)
    def pop_oldest(self)                    # FIFO pop (retirer le plus ancien)
    def clear(self)                         # Vider la file
    def size(self)                          # Nombre d'éléments
    def is_empty(self)                      # File vide ?
    def get_all(self)                       # Tous les éléments
```

#### `NodeDataQueueManager`

```python
class NodeDataQueueManager:
    def get_queue(self, node_id_name, data_type)     # Obtenir/créer une file
    def put_data(self, node_id_name, data_type, value, timestamp=None)
    def get_latest(self, node_id_name, data_type)    # Dernière donnée
```

#### `QueueBackedDict` (Adaptateur rétrocompatible)

```python
class QueueBackedDict:
    # Interface dictionnaire standard
    dict[node_id] = value            # Écriture avec auto-timestamp
    value = dict[node_id]            # Lecture du plus récent
    node_id in dict                  # Test d'existence
    dict.get(node_id, default)       # Get avec défaut
    
    # Méthodes étendues
    dict.set_with_timestamp(node_id, value, timestamp)  # Timestamp explicite
    dict.get_timestamp(node_id)                          # Récupérer le timestamp
```

---

## 9. Architecture src/ (moderne)

Le répertoire `src/` introduit une architecture professionnelle optionnelle, 100% rétrocompatible.

### Hiérarchie d'exceptions

```python
from src.utils.exceptions import (
    CVStudioError,           # Base
    NodeExecutionError,      # Erreur d'exécution de nœud
    NodeConfigurationError,  # Erreur de configuration
    ResourceError,           # Ressource indisponible
)

raise NodeExecutionError(node_id="123", message="Traitement échoué", cause=original_exc)
```

### Logging centralisé

```python
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)
logger.info("Traitement du nœud...")
logger.error("Échec", exc_info=True)
```

### Gestion des ressources

```python
from src.utils.resource_manager import get_resource_manager

manager = get_resource_manager()
manager.register('video_capture', video_cap, cleanup_func=lambda v: v.release())
# Nettoyage automatique à la fermeture
```

### Gestion de configuration

```python
from src.core.config import Settings

settings = Settings('config.json')
width = settings.get('webcam_width', 640)
settings.set('use_gpu', True)
settings.save()
```

### Nœud amélioré (EnhancedNode)

```python
from src.core.nodes import EnhancedNode

class MonNoeud(EnhancedNode):
    node_label = 'Mon Nœud'
    node_tag = 'MonNoeud'
    
    def update(self, node_id, connection_list, node_image_dict, node_result_dict):
        # Logging, gestion d'erreurs et ressources intégrés
        result = self.safe_execute(self.process_image, node_image_dict)
        return {"image": result, "json": None}
```

### Détection GPU

```python
from src.utils.gpu_utils import log_gpu_info, get_onnx_providers

log_gpu_info()  # Affiche les GPU disponibles
providers = get_onnx_providers(use_gpu=True)  # ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

---

## 10. Création de nœuds personnalisés

### Étape 1 : Créer le fichier

Créer un fichier dans la catégorie appropriée, par exemple `node/ProcessNode/node_mon_filtre.py`.

### Étape 2 : Implémenter le nœud

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import dearpygui.dearpygui as dpg
from node.node_abc import DpgNodeABC


class FactoryNode:
    """Classe Factory exposée à l'éditeur."""
    node_label = "Mon Filtre"          # Nom dans le menu
    node_tag = "MonFiltre"             # ID interne unique
    node_color = [100, 200, 150]       # Couleur RGB du nœud
    
    def __init__(self):
        pass
    
    def create(self, node_id, parent, pos, opencv_setting_dict):
        return MonFiltreNode(node_id, parent, pos, opencv_setting_dict)


class MonFiltreNode(DpgNodeABC):
    """Implémentation du nœud."""
    
    _ver = "0.0.1"
    node_label = "Mon Filtre"
    node_tag = "MonFiltre"
    
    def __init__(self, node_id, parent, pos, opencv_setting_dict):
        self.node_id = node_id
        self.parent = parent
        self.pos = pos
        self._opencv_setting_dict = opencv_setting_dict
        
        # Tags pour les widgets DearPyGUI
        self.tag_node_name = f"{node_id}:{self.node_tag}"
        self.tag_node_input = f"{self.tag_node_name}:IMAGE:Input0"
        self.tag_node_output = f"{self.tag_node_name}:IMAGE:Output0"
        self.tag_slider = f"{self.tag_node_name}:slider_value"
        
    def add_node(self, parent, node_id, pos, opencv_setting_dict, **kwargs):
        """Construit l'interface du nœud dans DearPyGUI."""
        with dpg.node(
            tag=self.tag_node_name,
            parent=parent,
            label=self.node_label,
            pos=pos,
        ):
            # Port d'entrée IMAGE
            with dpg.node_attribute(
                tag=self.tag_node_input,
                attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text("Image In")
            
            # Paramètre (slider)
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_slider_int(
                    tag=self.tag_slider,
                    label="Intensité",
                    default_value=50,
                    min_value=0,
                    max_value=100,
                    width=150,
                )
            
            # Port de sortie IMAGE
            with dpg.node_attribute(
                tag=self.tag_node_output,
                attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_text("Image Out")
    
    def update(self, node_id, connection_list, node_image_dict, node_result_dict):
        """Appelé à chaque frame. Traite l'image."""
        # Récupérer l'image d'entrée
        input_image = None
        for connection_info in connection_list:
            connection_type = connection_info[0].split(":")[2]
            if connection_type == "IMAGE":
                src_tag = ":".join(connection_info[0].split(":")[:2])
                input_image = node_image_dict.get(src_tag, None)
                break
        
        if input_image is None:
            return {"image": None, "json": None}
        
        # Lire la valeur du slider
        intensity = dpg.get_value(self.tag_slider)
        
        # Appliquer le traitement
        output_image = self._apply_filter(input_image, intensity)
        
        return {"image": output_image, "json": None}
    
    def _apply_filter(self, image, intensity):
        """Logique de traitement personnalisée."""
        # Exemple : ajustement de luminosité
        factor = intensity / 50.0
        result = np.clip(image * factor, 0, 255).astype(np.uint8)
        return result
    
    def close(self, node_id):
        """Nettoyage."""
        pass
    
    def get_setting_dict(self, node_id):
        """Sérialise les paramètres."""
        return {
            "ver": self._ver,
            "pos": dpg.get_item_pos(self.tag_node_name),
            "slider_value": dpg.get_value(self.tag_slider),
        }
    
    def set_setting_dict(self, node_id, setting_dict):
        """Restaure les paramètres."""
        if "slider_value" in setting_dict:
            dpg.set_value(self.tag_slider, setting_dict["slider_value"])
```

### Étape 3 : Le nœud apparaît automatiquement

L'éditeur charge dynamiquement tous les fichiers `node_*.py` dans chaque dossier de catégorie. Votre nœud apparaîtra dans le menu correspondant au prochain lancement.

### Bonnes pratiques

- Nommer le fichier `node_<nom>.py`
- Utiliser des tags uniques basés sur `node_id` et `node_tag`
- Toujours gérer le cas où l'entrée est `None`
- Nettoyer les ressources dans `close()`
- Sérialiser tous les paramètres utilisateur dans `get_setting_dict()`

---

## 11. Configuration

### Fichier principal : `node_editor/setting/setting.json`

```json
{
    "webcam_width": 960,
    "webcam_height": 540,
    "process_width": 640,
    "process_height": 480,
    "editor_width": 1280,
    "editor_height": 720,
    "use_gpu": false,
    "use_pref_counter": false,
    "serial_port": "",
    "enable_serial": false,
    "camera_list": [0, 1]
}
```

### Variables d'environnement

Fichier `.env` (voir `.env.example`) :

```env
# API Keys (ne jamais commiter !)
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...

# MongoDB
MONGODB_URI=mongodb://localhost:27017

# MQTT
MQTT_BROKER=localhost
MQTT_PORT=1883
```

### Personnalisation

```bash
# Copier et modifier la config
cp node_editor/setting/setting.json ma_config.json

# Lancer avec la config personnalisée
python main.py --setting ma_config.json
```

---

## 12. Tests

### Lancement des tests

```bash
# Tous les tests
python -m pytest tests/ -v

# Tests spécifiques
python -m pytest tests/test_core/ -v
python -m pytest tests/test_utils/ -v

# Tests du système de files
python -m pytest tests/test_timestamped_queue.py tests/test_queue_adapter.py -v

# Avec couverture
python -m pytest tests/ --cov=src --cov=node --cov-report=html

# Tests marqués
python -m pytest tests/ -m "not slow"    # Exclure les tests lents
python -m pytest tests/ -m "unit"         # Seulement les tests unitaires
```

### Structure des tests

```
tests/
├── test_core/                  # Tests architecture core
├── test_utils/                 # Tests utilitaires
├── test_*_node.py              # Tests individuels par nœud
├── test_*_integration.py       # Tests d'intégration pipeline
├── demo_*.py                   # Démonstrations exécutables
├── dummy_servers/              # Serveurs mock pour tests réseau
└── validate_*.py               # Scripts de validation
```

### Couverture

| Module | Tests |
|--------|-------|
| Architecture core (BaseNode, EnhancedNode, Factory) | 60+ |
| Utilitaires (exceptions, logging, resources, GPU) | 28+ |
| Système de files horodatées | 35 |
| Nœuds d'intégration | 150+ |
| **Total** | **250+** |

### Écrire un test

```python
# tests/test_mon_filtre.py
import numpy as np
import pytest


def test_mon_filtre_basic():
    """Test basique du filtre."""
    from node.ProcessNode.node_mon_filtre import MonFiltreNode
    
    # Créer une image test
    image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    # Appliquer le filtre
    node = MonFiltreNode(node_id=1, parent=None, pos=[0, 0], opencv_setting_dict={})
    result = node._apply_filter(image, intensity=75)
    
    assert result is not None
    assert result.shape == image.shape
    assert result.dtype == np.uint8
```

---

## 13. Construction d'exécutable (.exe)

### Build automatique (GitHub Actions)

1. Aller dans l'onglet **Actions** du dépôt
2. Cliquer sur **"Build Windows Executable"**
3. Cliquer **"Run workflow"** → Sélectionner la branche
4. Attendre 10-15 minutes
5. Télécharger `CV_Studio-Windows-Executable.zip`

### Build unifié (cross-platform)

```bash
# GPU
python build_unified.py --clean

# CPU seulement
python build_unified.py --clean --cpu
```

### Build Windows local

```bash
# Script batch (double-clic)
build_windows.bat

# PowerShell
powershell -ExecutionPolicy Bypass -File build_windows.ps1

# Manuel
pip install -r requirements-build.txt
python build_exe.py --clean
```

### Options de build

| Option | Description |
|--------|-------------|
| `--clean` | Nettoyer les artefacts précédents |
| `--windowed` | Sans console (GUI uniquement) |
| `--debug` | Logs détaillés |
| `--icon <fichier.ico>` | Icône personnalisée |
| `--cpu` | Build sans CUDA |

### Résultat

```
dist/CV_Studio/
├── CV_Studio.exe      # Exécutable principal
├── node/              # Nœuds et modèles ONNX
├── node_editor/       # Éditeur et paramètres
├── src/               # Utilitaires source
└── _internal/         # Runtime Python et dépendances
```

Taille approximative : 800 Mo – 1,5 Go.

---

## 14. Intégrations externes

### Bases de données

| Système | Nœud | Usage |
|---------|------|-------|
| **MongoDB** | `ActionNode/node_mongodb.py` | Stockage de résultats de détection |
| **RabbitMQ** | `ActionNode/rabbitmq/` | File de messages asynchrone |

### Protocoles réseau

| Protocole | Nœud | Usage |
|-----------|------|-------|
| **RTSP** | `InputNode/node_rtsp.py` | Caméras IP |
| **WebSocket** | `InputNode/node_websocket.py` | Données temps réel |
| **WebRTC** | `InputNode/node_webrtc.py` | Streaming pair-à-pair |
| **HLS** | `InputNode/node_hls.py` | HTTP Live Streaming |
| **MQTT** | `InputNode/node_mqtt.py` | IoT messaging |
| **REST API** | `InputNode/node_api.py` | Endpoints HTTP |
| **ONVIF** | `ActionNode/node_ptz.py` | Contrôle caméra PTZ |

### Matériel

| Appareil | Support |
|----------|---------|
| **Webcam** | Détection automatique multi-caméras |
| **Microphone** | Capture audio temps réel (sounddevice) |
| **Arduino/Serial** | Communication série (pyserial) |
| **GPU NVIDIA** | ONNX Runtime CUDA, OpenVINO |

### Services cloud

| Service | Intégration |
|---------|-------------|
| **OpenAI** | VLM (Vision-Language Model) |
| **Google** | APIs diverses |
| **AIS** | Données maritimes (Automatic Identification System) |

---

## 15. Exemples et démos

### Répertoire `examples/`

| Fichier | Description |
|---------|-------------|
| `demo_object_detection_homography.py` | Détection + homographie |
| `demo_tennis_court.py` | Visualisation terrain de tennis |
| `demo_map_visualization.py` | Cartographie GPS |
| `demo_enhanced_osm_features.py` | Tuiles OpenStreetMap avancées |
| `demo_averaging_and_persistence.py` | Lissage temporel |
| `demo_two_boats_with_pan_zoom.py` | Pan/zoom avec deux bateaux |
| `example_ais_stream.py` | Streaming de données AIS |
| `zoomable_node_editor.py` | Éditeur avec zoom avancé |
| `dearpygui_node_editor_colored_combo_example.py` | Exemple de combo coloré |

### Pipelines exemples

#### Détection d'objets en temps réel

```
[WebCam] → [Object Detection (YOLOv8)] → [MOT (ByteTrack)] → [Draw Information] → [Result Image]
                                                    ↓
                                              [ObjDetCount] → [Buzzer]
```

#### Analyse audio

```
[Microphone] → [Spectrogram] → [Audio Classification (YAMNet)] → [ObjChart]
                     ↓
              [Result Image]
```

#### Surveillance avec enregistrement conditionnel

```
[RTSP] → [Object Detection] → [ObjDetCount (seuil=3)] → [Video Recorder]
                  ↓                                              ↓
          [Draw Information]                              [MongoDB (log)]
                  ↓
          [Result Image]
```

#### Analyse sportive

```
[Video] → [Pose Estimation] → [Tennis Court] → [Heatmap] → [Image Concat] → [Video Writer]
                                     ↓
                           [Position Prediction]
```

---

## 16. Outils utilitaires

### `tools/set_onnx_audio_metadata.py`

Ajoute des métadonnées à un modèle ONNX audio :
- Noms de classes
- Sample rate cible
- Configuration mel-spectrogram

```bash
python tools/set_onnx_audio_metadata.py model.onnx --classes "classe1,classe2" --sr 16000
```

### `tools/deepstream_engine/`

Convertit un export JSON CvStudio (*File > Export*) en projet NVIDIA DeepStream
prêt à déployer (serveur GPU ou Jetson) : configuration `deepstream-app`,
configurations `nvinfer`, librairie de parsing des boîtes, Dockerfile,
`docker-compose.yml`, `Makefile` et `run.sh`.

```bash
python -m tools.deepstream_engine mon_workflow.json -o mon_projet/
# Fichier d'un nœud Video / périphérique d'un nœud Webcam (absents de l'export) :
python -m tools.deepstream_engine mon_workflow.json --source 1=/data/video.mp4
cd mon_projet && make run
```

Le nœud **Deploy** fait la même conversion depuis l'éditeur et transmet
automatiquement le fichier vidéo et la caméra sélectionnés.

| Nœud CvStudio | Élément DeepStream |
|---------------|--------------------|
| RTSP, Video, HLS, YouTube, Webcam | groupes `[sourceN]` (décodage matériel) |
| ObjectDetection (YOLO11/YOLOv8, YOLOX, FreeYOLO) | `nvinfer` primaire (TensorRT) |
| Classification | `nvinfer` secondaire sur les objets détectés |
| MultiObjectTracking | `nvtracker` (IOU, NvSORT ou NvDCF) |
| VideoWriter, VideoRecorder | enregistrement MP4 H.264 |
| RTMPOutput, Streaming, ou aucune sortie | serveur RTSP `rtsp://<hôte>:8554/ds-test` |

Les nœuds sans équivalent DeepStream (traitement d'image OpenCV, pose,
segmentation, audio, actions…) sont ignorés et listés dans le résumé de
conversion et dans le `README.md` du projet généré, avec les points à
vérifier avant la mise en production.

### `verify_dependencies.py`

Vérifie que toutes les dépendances sont installées :

```bash
python verify_dependencies.py
```

---

## 17. Dépannage

### Problèmes courants

| Problème | Solution |
|----------|----------|
| Crash au démarrage | `pip install -r requirements.txt` puis `python main.py --unuse_async_draw` |
| Webcam non détectée | Fermer les autres applications utilisant la caméra |
| Impossible de connecter deux nœuds | Vérifier que les types (couleurs) correspondent |
| Modèle DL introuvable | Vérifier le chemin du modèle ONNX |
| FPS faible | Ajouter un nœud Resize en amont, activer le GPU |
| Export/Import échoue | Vérifier les permissions d'écriture |
| Paramètres ne se mettent pas à jour | Reconnecter les nœuds ou redémarrer |
| `ModuleNotFoundError` | Vérifier l'environnement virtuel activé |
| Erreur CUDA | Installer `onnxruntime-gpu` au lieu de `onnxruntime` |
| Erreur MediaPipe | `pip install mediapipe protobuf>=3.20.0,<4.0.0` |

### Logs de débogage

```bash
# Mode debug complet
python main.py --use_debug_print

# Depuis l'exécutable
CV_Studio.exe --use_debug_print
```

### Performance

- **Resize tôt** — Réduire la résolution en début de pipeline
- **Skip frames** — Ajuster le taux de saut dans les nœuds Video
- **GPU** — Activer dans les nœuds DL quand disponible
- **Désactiver les nœuds** — Utiliser ON/OFF Switch pour les opérations coûteuses
- **Monitoring** — Utiliser le nœud System Resource pour surveiller CPU/RAM/GPU

---

## 18. Contribution

### Structure d'un commit

```
type(scope): description courte

Corps optionnel avec plus de détails.
```

Types : `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### Ajouter un nœud

1. Créer le fichier dans le dossier de catégorie approprié
2. Implémenter `FactoryNode` et la classe du nœud
3. Ajouter des tests dans `tests/`
4. Documenter dans le README de la catégorie si existant

### Lancer les vérifications

```bash
# Tests
python -m pytest tests/ -v

# Vérifier les dépendances
python verify_dependencies.py
```

### Conventions

- Nommer les fichiers : `node_<nom_en_snake_case>.py`
- Tags uniques : `{node_id}:{node_tag}:{TYPE}:{Input|Output}{index}`
- Toujours gérer `None` en entrée
- Libérer les ressources dans `close()`
- Sérialiser/désérialiser les paramètres utilisateur

---

## 19. Licence

Ce projet est sous licence **Apache 2.0**. Voir le fichier [LICENSE](LICENSE) pour les détails.

---

## Annexe : Diagramme d'architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              CV Studio                                       │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐     ┌──────────────────────────────────────────────┐       │
│  │   main.py   │────▶│           DpgNodeEditor                       │       │
│  │ (entry point)│     │  - Menu construction                         │       │
│  └─────────────┘     │  - Node instantiation                        │       │
│                       │  - Connection management                     │       │
│                       │  - Topological sort & execution              │       │
│                       │  - Export/Import JSON                         │       │
│                       └──────────────────────────────────────────────┘       │
│                                        │                                     │
│                        ┌───────────────┼───────────────┐                     │
│                        ▼               ▼               ▼                     │
│               ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│               │  InputNode   │ │ ProcessNode  │ │   DLNode     │            │
│               │  VideoNode   │ │ OverlayNode  │ │  AudioModel  │            │
│               │  SystemNode  │ │  StatsNode   │ │  TrackerNode │            │
│               └──────────────┘ └──────────────┘ └──────────────┘            │
│                        │               │               │                     │
│                        ▼               ▼               ▼                     │
│               ┌─────────────────────────────────────────────────┐            │
│               │         TimestampedQueue / QueueBackedDict       │            │
│               │    (Communication inter-nœuds FIFO horodatée)    │            │
│               └─────────────────────────────────────────────────┘            │
│                                        │                                     │
│                        ┌───────────────┼───────────────┐                     │
│                        ▼               ▼               ▼                     │
│               ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│               │  VisualNode  │ │  ActionNode  │ │  TriggerNode │            │
│               │  (affichage) │ │  (sorties)   │ │  (logique)   │            │
│               └──────────────┘ └──────────────┘ └──────────────┘            │
│                                                                              │
├────────────────────────────────────────────────────────────────────────────┤
│  Technologies : DearPyGUI │ OpenCV │ ONNX Runtime │ NumPy │ librosa         │
└────────────────────────────────────────────────────────────────────────────┘
```

---

*Documentation générée le 2 juin 2026. Pour toute question, ouvrir une issue sur le dépôt GitHub.*
