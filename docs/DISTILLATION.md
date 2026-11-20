# 🧠 Distillation de connaissances — Nœud OnlineTraining

> Entraînement en temps réel par distillation teacher-student pour la détection d'objets.

> ⚠️ Pour distiller un gros YOLOX vers un petit YOLOX, utilisez le nœud
> **YoloxDistillation** : voir [DISTILLATION_YOLOX.md](DISTILLATION_YOLOX.md).
> Mesuré sur des images jamais utilisées pour l'entraînement, le nœud
> OnlineTraining décrit ci-dessous ne fait pas progresser le student.

---

## Table des matières

1. [Concept](#concept)
2. [Architecture](#architecture)
3. [Utilisation](#utilisation)
4. [Score de distillation](#score-de-distillation)
5. [Interface utilisateur](#interface-utilisateur)
6. [Fichiers source](#fichiers-source)
7. [Prérequis](#prérequis)
8. [FAQ](#faq)

---

## Concept

La **distillation de connaissances** (knowledge distillation) est une technique où un petit modèle (l'**élève** / student) apprend à reproduire les prédictions d'un modèle plus gros et plus performant (le **professeur** / teacher).

Dans CV Studio, le nœud **OnlineTraining** implémente cette technique en temps réel :

```
┌─────────────────┐       ┌──────────────────┐       ┌─────────────────┐
│   Image source  │──────▶│ ObjectDetection  │──────▶│ OnlineTraining  │
│  (Video/Caméra) │       │  (Professeur)    │       │   (Élève)       │
└─────────────────┘       └──────────────────┘       └─────────────────┘
         │                                                     │
         └──────────────── IMAGE ─────────────────────────────▶│
                                    JSON (prédictions prof.) ──▶│
                                                               │
                                              IMAGE (annotée) ◀─┘
                                              JSON (prédictions élève) ◀─┘
```

**Objectif** : L'élève converge progressivement vers les performances du professeur tout en restant plus léger et rapide.

---

## Architecture

### Composants principaux

| Fichier | Rôle |
|---------|------|
| `node/DLNode/node_online_training.py` | Nœud principal (UI DearPyGUI + logique de pipeline) |
| `node/DLNode/online_training/student_trainer.py` | Gestionnaire du cycle de vie de l'élève (inférence, scoring, entraînement) |
| `node/DLNode/online_training/torch_student.py` | Élève PyTorch : conversion ONNX→PyTorch (`onnx2torch`) + **vraie rétropropagation** à travers la backbone et/ou les têtes |
| `node/DLNode/online_training/online_adapter.py` | Tête de correction affine entraînée par gradient sur la loss demandée (repli quand PyTorch est indisponible) |
| `node/DLNode/online_training/distillation_loss.py` | Fonctions de perte et score de distillation (IoU, matching, F1) |
| `node/DLNode/online_training/models/` | Répertoire de stockage des modèles élèves |

### Flux de données par frame

```
1. Réception de l'image (input IMAGE)
2. Réception du JSON du professeur (input JSON) : {bboxes, scores, class_ids}
3. Inférence de l'élève sur l'image → prédictions student
   • Mode PyTorch : forward du réseau entraîné (poids à jour)
   • Mode repli   : forward onnxruntime + tête de correction affine apprise
4. Calcul du score + de la loss de distillation demandée (teacher vs student)
5. [Si Training Active] Rétropropagation de la loss demandée :
   • Mode PyTorch : backward + optimizer.step() à travers backbone/têtes
   • Mode repli   : descente de (sous-)gradient sur la tête affine
6. Affichage : bounding boxes élève (vert) + professeur (bleu) + score/loss/amélioration/mode
7. Sortie : IMAGE annotée + JSON des prédictions élève
```

### Modes de fonctionnement

| Mode | Condition | Comportement |
|------|-----------|--------------|
| **Rétropropagation réseau (PyTorch)** (préféré) | `torch` **et** `onnx2torch` installés **et** conversion réussie | Le modèle ONNX de l'élève est chargé en PyTorch ; la loss demandée est **réellement rétropropagée** à travers les têtes (`train_scope='head'`, défaut) et/ou la backbone (`train_scope='all'`) via un `optimizer.step()`. L'inférence utilise les **poids mis à jour** → amélioration observable. `backprop_mode = pytorch-head`/`pytorch-all`. |
| **Adaptation en ligne (tête affine)** (repli) | `onnxruntime` standard seul, ou format de décodage non supporté par PyTorch | Une tête de correction affine `(sx, sy, tx, ty)` est entraînée par descente de gradient sur la **loss demandée**. `backprop_mode = affine-head`. |

> ℹ️ **PyTorch d'abord, tête affine en repli.** Le post-traitement de l'élève
> (décodage des boxes + NMS) est en NumPy : il ne fait pas partie d'un graphe
> ONNX différentiable, donc `onnxruntime` standard ne permet pas la
> rétropropagation. Pour entraîner **réellement** le réseau, `torch_student.py`
> convertit le graphe ONNX en `torch.nn.Module` (`onnx2torch`), rend les têtes
> (et optionnellement la backbone) entraînables, décode les sorties brutes de
> façon différentiable (yolo11 / yolox / nanodet), apparie les boxes professeur↔élève
> (matching sans gradient) puis rétropropage la loss demandée (box L1 + (1−IoU)
> + classification) avec un `optimizer.step()`. Quand `torch`/`onnx2torch` sont
> absents ou que la conversion échoue, on retombe sur la **tête affine**
> `(sx, sy, tx, ty)` (voir `online_adapter.py`), qui démarre à l'identité puis
> rapproche les boxes de l'élève de celles du professeur. Dans les deux cas
> l'amélioration est visible via la loss décroissante et le champ `Improv:` de
> l'overlay.

---

## Utilisation

### Pipeline minimal

1. **Ajouter un nœud source** (Video ou Caméra)
2. **Ajouter un nœud ObjectDetection** (professeur) — chargez un modèle performant (ex: YOLO11-L)
3. **Ajouter un nœud OnlineTraining**
4. **Connecter** :
   - Sortie IMAGE de la source → Entrée IMAGE de OnlineTraining
   - Sortie JSON de ObjectDetection → Entrée "Teacher JSON" de OnlineTraining
5. **Choisir un modèle élève** dans la liste du nœud, puis cliquer **Start**
   (voir ci-dessous ; "Load Student ONNX" ajoute un autre fichier à la liste)
6. **Lancer** le pipeline — le score de distillation s'affiche en temps réel

### Modèles élèves fournis

Quatre élèves sont proposés d'office dans la liste, du moins coûteux au plus
coûteux. Tous détectent les 80 classes COCO.

| Élève | Entrée | Paramètres | Entraînement | F1 face à YOLOX-s | Inférence |
|---|---|---|---|---|---|
| `yolox_nano` | 416 × 416 | 0,9 M | rétropropagation réseau | 0,59 | 37 ms |
| `yolox_tiny` | 416 × 416 | 5,1 M | rétropropagation réseau | 0,69 | 69 ms |
| `nanodet-plus-m_416` | 416 × 416 | 1,2 M | rétropropagation réseau | 0,56 | 48 ms |
| `yolo11_n` | 608 × 416 | 2,6 M | tête affine seulement | 0,73 | 28 ms |

Les deux dernières colonnes sont indicatives : élève d'origine, sans
entraînement, sur 10 images de la vidéo de rue du dépôt, CPU 2 cœurs.

- `yolo11_n` utilise un opérateur (`Split`, opset 18) que `onnx2torch` ne
  convertit pas : le réseau n'est pas entraîné, seule la tête de correction
  affine l'est.
- `nanodet-plus-m_416` est lu à la racine du dépôt ; il n'est pas embarqué
  dans l'exécutable, où il n'apparaît donc pas.
- Ces élèves ne peuvent pas être supprimés de la liste. Les modèles ajoutés
  par "Load Student ONNX" suivent, et restent supprimables.
- `nanodet-m_320_int8` (présent dans `online_training/models/`) n'est pas
  proposé : sa sortie quantifiée est mal décodée (des milliers de boîtes par
  image).

### Exporter le modèle entraîné

Cliquez **"Export Student ONNX"** pour sauvegarder l'état courant du modèle élève. Ce fichier ONNX peut ensuite être utilisé directement dans le nœud ObjectDetection.

### Réinitialiser

Cliquez **"Reset Student"** pour restaurer les poids originaux du modèle élève.

---

## Score de distillation

Le score mesure la similarité **ensembliste** entre les prédictions du
professeur et celles de l'élève. Il est calculé à chaque frame et ne dépend pas
d'un appariement strict 1-à-1.

### Métriques retournées

| Métrique | Description |
|----------|-------------|
| `score` | Score global [0, 1] — 1.0 = correspondance parfaite (plus haut = mieux) |
| `class_similarity` | Similarité cosinus des histogrammes de classes |
| `count_ratio` | Ratio `min/max` du nombre de détections |
| `confidence_alignment` | Similarité des profils de confiance |
| `spatial_coverage` | IoU des masques de couverture spatiale agrégés |
| `loss` | Loss de distillation set-based demandée (plus bas = mieux) |
| `teacher_count` / `student_count` | Nombre de détections de chaque côté |

### Ce que vaut l'élève : accord avec le professeur

Les métriques ci-dessus sont calculées image par image, sans seuil d'IoU :
elles bougent beaucoup d'une image à l'autre. Le JSON de sortie donne aussi,
en tête de `distillation_losses`, des mesures cumulées sur les **100
dernières images** :

| Métrique | Description |
|----------|-------------|
| `f1`, `precision`, `recall` | Accord avec le professeur : une détection compte si elle recouvre une boîte du professeur de la même classe (IoU ≥ 0,5) |
| `mean_iou` | Qualité de localisation des détections qui concordent |
| `baseline_f1` | Le même F1 pour l'élève **d'origine**, sur les mêmes images |
| `f1_gain` | `f1 − baseline_f1` sur ces images : ce que l'entraînement a apporté. Négatif s'il a dégradé l'élève |
| `inference_ms`, `train_ms` | Durée de l'inférence de l'élève et de l'étape d'entraînement pour la dernière image |

Chaque image est notée **avant** de servir à l'entraînement : aucune de ces
mesures ne porte sur une image déjà apprise. Avec la rétropropagation réseau,
l'élève d'origine est rejoué sur une image sur cinq pour `baseline_f1`. Une
valeur est `null` tant qu'elle n'a pas été mesurée ; une image sans résultat
du professeur (non connecté, ou trop ancien) ne compte pas.

Dans le nœud **Chart**, `f1` est la série affichée par défaut ; les autres se
choisissent dans la liste. `f1_gain` est la courbe à regarder pour savoir si
l'entraînement sert à quelque chose : avec les réglages par défaut, sur la
vidéo de rue du dépôt, il reste nul (à 0,01 près) pour les quatre élèves
fournis (voir l'avertissement en tête de ce document) ; dans un essai où
toutes les couches de `yolox_nano` étaient entraînées, il devenait négatif.

### Algorithme de calcul (score `[0, 1]`)

Le score est une combinaison pondérée de quatre composantes ensemblistes,
robustes à un nombre de boxes différent :

```python
score = 0.30 * class_similarity
      + 0.35 * spatial_coverage
      + 0.15 * count_ratio
      + 0.20 * confidence_alignment
```

### `score`, `loss` et `best` — que représentent-ils ?

| Grandeur | Sens | Direction | Source |
|----------|------|-----------|--------|
| `score` | Accord global élève↔professeur (composantes ci-dessus) | **plus haut = mieux** | `compute_distillation_score` |
| `loss` | Loss set-based demandée (DETR : box L1 + 1-IoU + classe + cardinalité + FP/FN) | **plus bas = mieux** | `compute_set_distillation_loss` |
| `best_score` | Meilleur (max) `score` observé depuis le dernier reset | plus haut = mieux | `StudentTrainer.best_score` |
| `best_loss` | Meilleure (min) `loss` demandée observée depuis le dernier reset | plus bas = mieux | `StudentTrainer.best_loss` |
| `improvement` / `improvement_pct` | Réduction de la loss depuis la 1ʳᵉ frame (absolue / %) — c'est l'amélioration visible de l'élève | plus haut = mieux | `StudentTrainer.improvement` |

> La **loss demandée** (set-based, hongroise) est l'unique source de vérité :
> c'est exactement la même valeur que celle affichée par les nœuds **IoU** /
> **Chart** et celle utilisée comme signal d'entraînement quand la
> rétropropagation est active.

### Loss de distillation *set-based* (DETR-style)

En complément du score `[0, 1]`, une **loss de distillation set-based** est
calculée : matching hongrois (`cost = (1-IoU) + class_cost`), puis box
regression (L1 + 1-IoU), distillation de classe, cardinalité `|N_s - N_t|`,
pénalités faux positifs / faux négatifs et mismatch de classe. Elle est exposée
sous `distillation_losses` (affichable dans le nœud **Chart**) et sert de signal
d'entraînement quand la rétropropagation est active. Voir
**[`distillation_loss.md`](distillation_loss.md)** pour la description complète
de la méthode et des métriques.

---

## Interface utilisateur

Le nœud OnlineTraining expose les contrôles suivants :

| Contrôle | Description |
|----------|-------------|
| **score_th** (slider) | Seuil de confiance minimum pour les prédictions de l'élève (0.0–1.0) |
| **learning_rate** (slider) | Taux d'apprentissage (0.00001–0.01) |
| **Training Active** (checkbox) | Active/désactive l'entraînement (l'inférence continue) |
| **Score display** | Affiche : score courant, loss courante, meilleur score, meilleure loss |
| **Stats display** | Affiche : nombre de frames traitées, état de l'entraînement |
| **Liste des élèves** | Élèves fournis, puis modèles ajoutés |
| **Start** (bouton vert) | Charge l'élève sélectionné et active l'entraînement |
| **Load Student ONNX** (bouton jaune) | Ajouter un modèle ONNX élève à la liste |
| **Export Student ONNX** (bouton vert) | Exporter le modèle courant |
| **Reset Student** (bouton) | Réinitialiser aux poids originaux |

### Visualisation

- **Vert** : Bounding boxes de l'élève (avec label et score)
- **Bleu** : Bounding boxes du professeur (référence)
- **Jaune (overlay)** : Score de distillation + statistiques

---

## Fichiers source

```
node/DLNode/
├── node_online_training.py          # Nœud principal
└── online_training/
    ├── __init__.py
    ├── student_trainer.py           # Classe StudentTrainer
    ├── distillation_loss.py         # Calcul du score de distillation
    └── models/                      # Modèles élèves stockés
```

### StudentTrainer — API principale

```python
class StudentTrainer:
    def __init__(self, model_path, input_width, input_height, output_format, num_classes, learning_rate, score_threshold, providers)
    def infer(self, frame) -> (bboxes, scores, class_ids)
    def train_step(self, frame, teacher_bboxes, teacher_scores, teacher_class_ids, score_threshold) -> dict
    def reset(self)              # Restaure les poids originaux
    def export_onnx(self, path)  # Exporte le modèle courant
    def get_stats(self) -> dict  # Statistiques d'entraînement
```

### distillation_loss — Fonctions

```python
def compute_iou(box_a, box_b) -> float
def match_detections(teacher_bboxes, teacher_scores, student_bboxes, student_scores, iou_threshold=0.5)
def compute_distillation_score(teacher_bboxes, teacher_scores, teacher_class_ids, student_bboxes, student_scores, student_class_ids, iou_threshold=0.5) -> dict
```

---

## Prérequis

### Minimum (inférence seule)

```bash
pip install onnxruntime numpy opencv-python
```

### Complet (vraie rétropropagation réseau — recommandé)

```bash
pip install torch onnx2torch numpy opencv-python
```

> **Recommandé** : avec `torch` + `onnx2torch`, l'élève est converti en PyTorch
> et la loss demandée est **réellement rétropropagée** à travers les têtes
> (et/ou la backbone). C'est le mode qui entraîne effectivement le réseau.

> **Note** : Sans `torch`/`onnx2torch`, le nœud retombe sur la **tête de
> correction affine** — l'élève s'améliore toujours sur la loss demandée, mais
> seuls les paramètres `(sx, sy, tx, ty)` sont appris, pas les poids internes du
> réseau.

### Modèles compatibles

Le modèle élève doit être au format ONNX avec un format de sortie supporté :
- **yolo11** (YOLO v11 / Ultralytics)
- **yolox** (YOLOX)
- **nanodet** (NanoDet / NanoDet-Plus, sortie unique GFL/DFL) — décodage DFL
  différentiable (distances de bord softmax×stride, géométrie letterbox uniforme)

Le modèle doit contenir les métadonnées suivantes (inspectées automatiquement) :
- Dimensions d'entrée (input_width × input_height)
- Nombre de classes
- Noms de classes (optionnel, sinon `class_0`, `class_1`, etc.)

---

## FAQ

### Comment choisir le professeur et l'élève ?

| Critère | Professeur | Élève |
|---------|-----------|-------|
| **Taille** | Grand (YOLO11-L, YOLO11-X) | Petit (YOLO11-N, YOLO11-S) |
| **Précision** | Haute | En cours d'apprentissage |
| **Vitesse** | Lente (acceptable) | Rapide (objectif) |

### Que signifie un score de 0.85 ?

Un score de 0.85 signifie que l'élève reproduit 85% de la qualité des détections du professeur, en tenant compte du rappel, de la précision et de la qualité des localisations.

### L'entraînement fonctionne-t-il sans GPU ?

Oui, le mode CPU est supporté. L'entraînement sera plus lent mais fonctionnel. Le provider par défaut est `CPUExecutionProvider`.

### Puis-je utiliser le modèle exporté dans ObjectDetection ?

Oui ! Le modèle exporté est un fichier ONNX standard. Il peut être importé directement via le bouton "Add Model" du nœud ObjectDetection.

### Que se passe-t-il si je ne connecte pas le JSON du professeur ?

Le nœud détecte l'absence de JSON et l'élève infère seul sans calcul de score ni entraînement. Les prédictions sont tout de même affichées.
