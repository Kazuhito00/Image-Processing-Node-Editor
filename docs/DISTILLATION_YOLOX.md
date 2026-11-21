# Distillation YOLOX → YOLOX en direct — nœud `YoloxDistillation`

Un gros modèle YOLOX (le *teacher*) entraîne un petit modèle YOLOX (le
*student*) sur les images qui passent réellement dans le pipeline, pendant que
le student continue de détecter en temps réel. Le student se spécialise ainsi
sur **votre** scène (caméra, angle, lumière, objets) et se rapproche du
teacher, tout en gardant sa vitesse.

- Fonctionne **avec ou sans GPU** (détection automatique).
- Le student en production n'est remplacé que par un student **mesuré
  meilleur** sur des images récentes qu'aucun des deux n'a apprises.
- Le résultat est un fichier ONNX YOLOX standard, utilisable partout.

> Ce nœud remplace, pour YOLOX, l'entraînement du nœud `OnlineTraining` : mesuré
> sur des images jamais utilisées pour l'entraînement, celui-ci ne faisait pas
> progresser le student, ou le détruisait quand toutes les couches étaient
> entraînées.

---

## 1. Utilisation

> Manuel d'utilisation complet (mise en place, suivi avec le nœud Chart,
> exploitation, diagnostic) : `docs/MANUEL_YOLOX_DISTILLATION.pdf`, généré par
> `python tools/make_distillation_manual.py [--run WORKDIR]`.

### Dans l'application

1. Menu **VisionModel → YoloxDistillation**, relier une source d'images.
2. Choisir le **teacher** (le plus gros) et le **student** dans le catalogue
   (voir ci-dessous).
3. Cocher **Distill (background)**.

Le nœud affiche et sort les détections du student (image + JSON au format
`ObjectDetection`). Le texte d'état indique :

```
student v3 - training (cpu)        version déployée, activité, matériel
samples: 412 train / 96 val        échantillons collectés
val loss 0.871  AP50 0.76          mesure du student déployé (validation)
```

- **Export ONNX** écrit le student déployé dans `_Distillation/exports/`.
- **Reset student** revient au modèle d'origine (les échantillons sont gardés).
- Décocher *Distill* met la collecte et l'entraînement en pause ; la détection
  continue.

### Métriques dans le JSON, et dans le nœud Chart

Le JSON de sortie contient les mesures de la distillation sous
`distillation` et, tant que **Metrics to Chart** est coché, sous
`distillation_losses` : c'est la clé que le nœud **Chart** trace dans le
temps. Reliez la sortie JSON à un nœud Chart ; `val_f1` s'affiche par défaut,
les autres séries se choisissent dans sa liste.

| Métrique | Signification |
|---|---|
| `val_f1`, `val_ap50`, `val_precision`, `val_recall`, `val_loss` | Student **déployé** face au teacher, sur la fenêtre de validation (images récentes jamais entraînées) |
| `original_val_f1`, `original_val_ap50` | Student **d'origine**, sur les mêmes images |
| `f1_gain` | `val_f1 − original_val_f1` : ce que la distillation a apporté jusqu'ici |
| `candidate_val_loss`, `candidate_val_ap50` | Student issu du dernier cycle, déployé ou non |
| `train_loss` | Perte d'entraînement en cours |
| `disagreement` | Écart teacher / student déployé sur la dernière image montrée au teacher (0 = identiques) |
| `version`, `rounds` | Students déployés, cycles effectués |
| `train_samples`, `val_samples` | Échantillons collectés |
| `detections`, `inference_ms` | Détections et durée d'inférence du student sur l'image courante |

- Une valeur est `null` tant qu'elle n'a pas été mesurée (les `val_*` avant
  la fin du premier cycle) ; le Chart laisse alors un vide plutôt qu'un zéro.
- La fenêtre de validation avance avec les images : `val_f1` varie donc aussi
  parce que les images changent. `f1_gain` compare les deux students sur les
  **mêmes** images et ne dépend pas de leur difficulté.
- Mesurer le student d'origine ajoute une évaluation par cycle (environ 3 s
  pour 70 images sur 2 cœurs, soit 15 % d'un cycle) ; la détection n'est pas
  ralentie.
- **Metrics to Chart** décoché, un Chart relié au nœud compte les classes
  détectées, comme pour `ObjectDetection` ; les mesures restent sous
  `distillation`.

Tout est conservé dans `_Distillation/<student>__from__<teacher>/`
(`CV_Studio_data/distillation/` pour l'exécutable) : au redémarrage, le nœud
reprend avec le dernier student déployé et les échantillons déjà collectés.

### Les students proposés

Quatre petits students prêts à l'emploi sont proposés dans la liste
**student**, du moins cher au plus précis. Ils partent tous de poids
pré-entraînés COCO (80 classes) :

| Clé | Réseau | Entrée | Paramètres | Quand le choisir |
|---|---|---|---|---|
| `yolox_nano_320` | YOLOX-Nano | 320×320 | 0,9 M | CPU faible, objets grands ou proches ; le plus rapide |
| `yolox_nano` | YOLOX-Nano | 416×416 | 0,9 M | **défaut** : le plus petit YOLOX à sa résolution native |
| `yolox_tiny_320` | YOLOX-Tiny | 320×320 | 5,1 M | plus de capacité que nano, pour le même coût par image que nano 416 |
| `yolox_tiny` | YOLOX-Tiny | 416×416 | 5,1 M | le plus précis des petits ; entraînement plus lourd (GPU ou CPU solide) |

Les variantes 320 px sont dérivées des fichiers ONNX fournis : le graphe YOLOX
est entièrement convolutif et n'a de résolution fixée que dans ses déclarations
d'entrée et de sortie. Elles sont générées à la première sélection, vérifiées
avec onnxruntime, et conservées dans `_Distillation/students/`. 320 px est dans
la plage multi-échelle (320 à 640) utilisée par l'entraînement officiel de ces
poids. Une variante 320 coûte environ 40 % de moins par image que la même
architecture en 416 ; en contrepartie elle voit moins bien les petits objets.

Le catalogue est défini dans `node/DLNode/online_training/distill/students.py`
(`STUDENTS`) ; les fichiers `yolox*.onnx` qui n'en font pas partie restent
sélectionnables par leur nom.

### Ajouter un teacher plus gros

Tout fichier `yolox*.onnx` déposé dans
`node/DLNode/object_detection/YOLOX/model/` apparaît dans les listes. Les
modèles doivent être des YOLOX exportés **sans décodage des boîtes**
(`decode_in_inference=False`, l'export ONNX officiel), avec **les mêmes
classes**, et l'entrée du teacher doit être au moins aussi grande que celle du
student. Fournis : `yolox_nano` (416), `yolox_tiny` (416), `yolox_s` (640).

### Hors ligne, sur une vidéo enregistrée

```bash
python -m node.DLNode.online_training.distill.cli \
    --teacher yolox_s --student yolox_nano_320 \
    --source camera.mp4 --workdir _Distillation/essai --rounds 30
```

`--teacher` et `--student` acceptent un fichier ONNX, le nom d'un modèle des
répertoires de modèles ou, pour le student, une clé du catalogue
(`--list-students` les affiche).

La vidéo est rejouée comme un flux : le début alimente le moteur, la fin
(`--holdout`, 30 % par défaut) ne lui est jamais montrée et sert à mesurer le
student avant et après. C'est le moyen de **vérifier sur vos propres images**
que la distillation apporte quelque chose, ou d'entraîner sur une machine
équipée d'un GPU.

### Dépendances

L'inférence ne demande que `onnxruntime`. L'entraînement demande en plus
`torch`, `torchvision` et `onnx2torch` (dans `requirements.txt`, inclus dans
l'exécutable). S'ils manquent, le nœud détecte normalement et indique que
l'entraînement est indisponible.

---

## 2. Fonctionnement

### Trois activités indépendantes

```
 images ──► INFÉRENCE (pipeline)        student déployé, onnxruntime
   │           n'attend jamais les deux autres
   │
   └─ 1 image / 0,5 s ──► COLLECTE (thread de fond)
                             le teacher regarde l'image ;
                             image + prédictions du teacher → disque
                                        │
                                        ▼
                          ENTRAÎNEMENT (thread de fond, PyTorch)
                             par cycles : entraîner → valider → déployer
                                        │ seulement si meilleur
                                        ▼
                             nouveau student ONNX, remplacé à chaud
```

Le teacher ne tourne donc jamais dans le pipeline, et un cycle d'entraînement
ne ralentit pas la détection : les threads de fond ont une priorité inférieure
à celle de l'interface et du pipeline.

### Ce que le student apprend : distillation dense

Un YOLOX prédit, pour chaque case de trois grilles (pas de 8, 16 et 32
pixels), une boîte, une probabilité « il y a un objet » (*objectness*) et une
probabilité par classe. Teacher et student ont la même structure de sortie.

On donne au teacher **exactement les pixels vus par le student**, placés dans
le coin de son entrée plus grande. Comme les deux réseaux ont les mêmes pas,
chaque case du student correspond alors à une case du teacher qui regarde le
même endroit à la même échelle. Le student peut donc imiter le teacher **case
par case**, sans aucun appariement de boîtes :

| Sortie | Ce que le student imite | Poids |
|---|---|---|
| objectness | la probabilité du teacher, sur **toutes** les cases | 1 |
| classes | les probabilités du teacher, pondérées par son objectness | 1 |
| boîte | la boîte du teacher de la même case (1 − GIoU), pondérée par son score | 5 |

Pourquoi c'est une vraie distillation, et pas un simple pseudo-étiquetage :

- le student reçoit les **probabilités** du teacher (0,62 « personne »,
  0,08 « vélo »…), pas une liste de boîtes seuillées. Les hésitations du
  teacher, ses quasi-détections et ses confusions entre classes sont de
  l'information ;
- le fond est appris partout, y compris sur les images où le teacher ne voit
  rien : c'est ce qui supprime les faux positifs du student ;
- les classes ne comptent que là où le teacher voit un objet (ailleurs sa
  sortie « classes » n'a pas de sens) — *objectness-scaled distillation*,
  Mehta & Ozturk, 2018 ;
- l'entropie croisée est multipliée par `|teacher − student|²`
  (*quality focal loss*, Li et al., 2020) : les milliers de cases de fond
  déjà correctes ne noient pas les quelques cases qui comptent. Conséquence :
  un student identique à son teacher a une perte nulle et n'est pas modifié.

Option `teacher_view="full"` : le teacher regarde l'image dans sa propre
résolution, plus élevée, et ses prédictions sont projetées sur la grille du
student. Il voit mieux les petits objets, mais une partie de ce qu'il trouve
est hors de portée du student à sa résolution. Sur nos essais ce mode n'a pas
fait mieux ; il reste disponible pour les scènes à très petits objets avec
beaucoup de données.

### Entraîner le fichier ONNX lui-même

Le student déployé est un fichier ONNX. C'est **ce graphe-là** qui est
entraîné (converti en PyTorch par `onnx2torch`), et les poids appris sont
réécrits dans le graphe d'origine. Le modèle déployé est donc exactement celui
qui a été entraîné et validé : mêmes opérateurs, même entrée, même sortie.

Trois précautions, vérifiées numériquement à chaque chargement et à chaque
export (le moteur refuse de continuer en cas d'écart) :

- les sigmoïdes de la tête de détection sont retirées pour entraîner sur les
  *logits* (entraîner sur des probabilités est numériquement mauvais) ; elles
  sont toujours dans le fichier déployé ;
- le graphe, exporté pour une image à la fois, est adapté pour traiter des
  lots ;
- les normalisations de lot sont déjà fusionnées dans les convolutions d'un
  modèle exporté. C'est un avantage ici : aucune statistique ne dérive quand
  on apprend sur quelques images très corrélées.

### Quelles images sont gardées

Une image toutes les 0,5 s est proposée au teacher. Elle n'est stockée que si
elle apprend quelque chose :

- la scène a changé depuis la dernière image stockée, **ou**
- teacher et student ne sont pas d'accord dessus.

Un quart des images « d'accord » est gardé comme exemples faciles. Les images
difficiles sont rejouées plus souvent à l'entraînement. Le stock est borné
(4 000 échantillons) ; quand il est plein, un ancien échantillon tiré au
hasard est remplacé : le stock suit la scène tout en gardant de la mémoire.

Pendant l'entraînement, les images sont retournées horizontalement et
modifiées en luminosité, contraste, teinte, bruit et flou ; les prédictions du
teacher restent valables pour ces variantes.

### Le garde-fou : validation « vers l'avant »

Les échantillons sont gardés dans l'ordre chronologique. Les **96 plus
récents** ne servent jamais à l'entraînement en cours : ils forment la fenêtre
de validation. Après chaque cycle :

1. le student candidat (moyenne glissante des poids, *EMA*) et le student
   déployé sont évalués sur cette fenêtre ;
2. le candidat est déployé seulement si sa perte de validation est meilleure
   d'au moins 1 % **et** si son AP50 face aux détections du teacher ne baisse
   pas (tolérance 0,02, le bruit de mesure sur une centaine d'images) ;
3. sinon rien ne change en production. Après 3 cycles sans progrès de la
   perte, ou dès qu'un cycle est nettement moins bon (+10 %) ou diverge,
   l'entraînement repart des poids déployés avec un pas d'apprentissage
   divisé par deux.

La question posée est donc toujours : *un student entraîné sur le passé
est-il meilleur sur ce qui est venu ensuite ?* C'est volontairement sévère.
Les images d'une vidéo se ressemblent d'une seconde à l'autre ; une validation
tirée au milieu des images d'entraînement répondrait « oui » beaucoup trop
facilement (nous l'avons mesuré, voir § 4). Quand de nouvelles images
arrivent, les plus anciennes de la fenêtre rejoignent l'entraînement.

Les 5 derniers students déployés sont conservés dans `checkpoints/`, et
l'historique des décisions dans `state.json`.

### Avec ou sans GPU

| | GPU (CUDA) | CPU |
|---|---|---|
| Lot | 16 images | 4 images |
| Pas par cycle | 200 | 40 |
| Couches entraînées | les 30 % les plus proches de la sortie | idem |
| Teacher (onnxruntime) | GPU si disponible | CPU |

Le code est le même ; seuls le débit et la taille des cycles changent. Sur
CPU (2 cœurs), un cycle de `yolox_nano` dure 10 à 20 s.

**Quelles couches sont entraînées.** Par défaut seuls les 30 % de tenseurs les
plus proches de la sortie le sont (la tête de détection et la fin du cou,
environ 11 % des poids de `yolox_nano`). Mesuré sur des images postérieures à
celles de l'entraînement, c'est ce qui généralise le mieux quand on a peu
d'images : la progression est régulière dès le premier cycle. Entraîner tout
le réseau (`trainable_fraction=1.0`) commence par dégrader le student avant
de le rattraper ; cela ne devient intéressant qu'avec beaucoup d'échantillons
(un millier et plus).

---

## 3. Réglages (`DistillConfig`)

Les valeurs par défaut conviennent ; les principales :

| Paramètre | Défaut | Rôle |
|---|---|---|
| `device` | `auto` | `cuda` si disponible, sinon `cpu` |
| `sample_interval` | 0,5 s | délai entre deux images montrées au teacher |
| `teacher_view` | `student` | ce que regarde le teacher (voir § 2) |
| `max_train` / `val_window` | 4000 / 96 | taille du stock et de la fenêtre de validation |
| `min_train` / `min_val` | 48 / 12 | échantillons requis avant le premier cycle |
| `learning_rate` | 5e-5 | pas d'apprentissage (AdamW) |
| `trainable_fraction` | 0,3 | part des couches entraînées, depuis la sortie |
| `focal_gamma` | 2 | 0 = entropie croisée simple |
| `ema_decay` | 0,99 | lissage des poids du candidat |
| `min_gain` | 0,01 | gain relatif de perte exigé pour déployer |
| `patience` | 3 | cycles sans déploiement avant de repartir des poids déployés |

---

## 4. Résultats mesurés

Protocole : `yolox_s` (640) distillé vers `yolox_nano` (416), **sur CPU
(2 cœurs)**, avec les réglages par défaut. Chaque vidéo est rejouée comme un
flux ; les 70 % du début alimentent le moteur (30 cycles, 5 à 6 minutes
d'entraînement), les **30 % de la fin ne sont jamais montrés au moteur** et
servent à la mesure. Référence : les détections du teacher en pleine
résolution (score ≥ 0,3, IoU ≥ 0,5, même classe).

| Vidéo | Échantillons | AP50 avant → après | F1 avant → après |
|---|---|---|---|
| Caméra fixe, passage de personnes (139 s) | 879 | 0,990 → **0,994** | 0,730 → **0,989** |
| Caméra fixe, allée de magasin (65 s) | 590 | 0,237 → **0,392** | 0,383 → **0,543** |
| Personne filmée de près (36 s utilisées) | 270 | 0,553 → 0,573 | 0,606 → 0,604 |
| Caméra embarquée sur route (36 s utilisées) | 270 | 0,373 → 0,385 | 0,526 → 0,517 |

Ce qu'il faut en retenir :

- **Caméra fixe et quelques centaines d'échantillons : le gain est net**, sur
  des images postérieures que le student n'a jamais vues. Sur la première
  vidéo, le student d'origine produisait un faux positif pour presque chaque
  vraie détection (précision 0,58) ; après distillation la précision est de
  0,98 pour le même rappel (0,99).
- **Clip court ou scène qui change sans cesse : gain à peu près nul.** Avec
  270 échantillons tirés de 36 secondes, il n'y a pas assez de variété pour
  apprendre quelque chose qui vaille pour la suite. Le garde-fou fait alors
  son travail : le student n'est pas dégradé.
- Mesuré contre un détecteur tiers (YOLOv8m) au lieu du teacher, le F1 passe
  de 0,718 à 0,937 sur la première vidéo ; sur l'allée de magasin l'AP50
  monte (0,318 → 0,364) mais le F1 ne bouge pas (0,478 → 0,465) : le student
  se rapproche de **son** teacher, y compris là où celui-ci et YOLOv8m ne
  sont pas d'accord.

Ces chiffres viennent de quatre vidéos courtes (deux du dépôt, deux vidéos
publiques d'exemple) : ils indiquent un ordre de grandeur, pas une garantie.
**Mesurez sur vos propres images** avec l'outil en ligne de commande (§ 1)
avant de vous fier au résultat.

Pourquoi la validation « vers l'avant ». Sur ces mêmes vidéos, une validation
tirée au milieu des images d'entraînement annonçait un AP50 passant de 0,62 à
0,75 sur le clip de 36 s, alors que sur les images suivantes il ne passait
que de 0,55 à 0,56. La validation vers l'avant ne commet pas cette erreur. Elle a aussi
montré qu'entraîner toutes les couches dégrade d'abord le student, d'où le
réglage par défaut (§ 2).

Interface pendant l'entraînement (machine à 2 cœurs, rendu logiciel, donc un
cas défavorable) : 42 images/s sans distillation, 37 avec.

Pour comparaison, dans un essai du même type (teacher YOLOv8m, images de fin
de vidéo jamais entraînées), l'entraînement du nœud `OnlineTraining` laissait le F1 du student inchangé (0,70 → 0,70) avec ses
réglages par défaut et le ramenait à 0 quand toutes les couches étaient
entraînées.

---

## 5. Limites

- **YOLOX uniquement**, teacher et student avec les mêmes classes. Les autres
  formats (YOLOv8/11, NanoDet) ne sont pas pris en charge par ce nœud.
- Le student ne dépassera pas le teacher : il s'en rapproche. Ses erreurs
  sont apprises avec le reste.
- La spécialisation se fait sur la scène observée. Un student distillé sur
  une caméra n'est pas fait pour une autre ; le modèle d'origine reste
  disponible (*Reset student*).
- Seules les sorties sont distillées. La distillation des cartes de
  caractéristiques intermédiaires (FPN) n'est pas implémentée.
- Le chemin GPU utilise le même code que le chemin CPU mais n'a pas pu être
  exécuté sur une machine avec GPU lors du développement.
- Sur une machine dont tous les cœurs sont occupés, l'entraînement avance
  lentement : il ne prend que le temps processeur laissé libre.

## 6. Références

- Hinton, Vinyals, Dean — *Distilling the Knowledge in a Neural Network*, 2015.
- Mehta, Ozturk — *Object detection at 200 Frames Per Second*, 2018
  (distillation pondérée par l'objectness).
- Li et al. — *Generalized Focal Loss*, 2020 (quality focal loss).
- Zhou et al. — *Dense Teacher*, 2022 (pseudo-étiquettes denses).
- Tarvainen, Valpola — *Mean teachers are better role models*, 2017 (EMA).
- Ge et al. — *YOLOX: Exceeding YOLO Series in 2021*.

## 7. Code

| Fichier | Rôle |
|---|---|
| `node/DLNode/node_yolox_distillation.py` | le nœud |
| `node/DLNode/online_training/distill/engine.py` | collecte, entraînement, validation, déploiement |
| `…/distill/yolox_torch.py` | student entraînable depuis l'ONNX, perte de distillation |
| `…/distill/yolox_ops.py` | géométrie YOLOX, détections, mesure (NumPy) |
| `…/distill/yolox_onnx.py` | teacher et student déployé (onnxruntime) |
| `…/distill/students.py` | catalogue des students, variantes redimensionnées |
| `…/distill/store.py` | stock d'échantillons sur disque |
| `…/distill/cli.py` | outil en ligne de commande |
| `tests/test_yolox_distillation.py` | tests |
