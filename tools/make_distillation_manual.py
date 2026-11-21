#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the PDF manual of the YoloxDistillation node.

    python tools/make_distillation_manual.py [--run WORKDIR] [--out PDF]

The figures of section 5 are drawn from the ``state.json`` history of a
distillation run (``--run``, a workdir of the engine or of the command line
tool).  Without it, the history shipped in this script is used.

Needs reportlab and matplotlib (``pip install reportlab matplotlib``).
"""
import argparse
import io
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_OUT = os.path.join(_ROOT, "docs", "MANUEL_YOLOX_DISTILLATION.pdf")

# yolox_s -> yolox_nano, road_sample.mp4, CPU, default settings (12 rounds)
_DEFAULT_HISTORY = [
    {"round": 1, "candidate": {"loss": 1.0313, "ap50": 0.542}, "deployed": {"loss": 1.0947, "ap50": 0.490}, "deployed_now": True, "rolled_back": False, "train_loss": 1.2759},
    {"round": 2, "candidate": {"loss": 1.0091, "ap50": 0.595}, "deployed": {"loss": 1.0313, "ap50": 0.542}, "deployed_now": True, "rolled_back": False, "train_loss": 1.2172},
    {"round": 3, "candidate": {"loss": 0.9984, "ap50": 0.622}, "deployed": {"loss": 1.0091, "ap50": 0.595}, "deployed_now": True, "rolled_back": False, "train_loss": 1.2230},
    {"round": 4, "candidate": {"loss": 0.9866, "ap50": 0.624}, "deployed": {"loss": 0.9984, "ap50": 0.622}, "deployed_now": True, "rolled_back": False, "train_loss": 1.1969},
]

BLUE = colors.HexColor("#1f4e79")
LIGHT = colors.HexColor("#dce6f1")
GREY = colors.HexColor("#f2f2f2")


# ---------------------------------------------------------------------- #
# Styles


def styles():
    base = getSampleStyleSheet()
    s = {
        "title": ParagraphStyle("title", parent=base["Title"], fontSize=26, leading=32, textColor=BLUE, spaceAfter=18),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontSize=13, leading=18, alignment=TA_CENTER, textColor=colors.HexColor("#444444")),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=17, leading=22, textColor=BLUE, spaceBefore=14, spaceAfter=8),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=13, leading=17, textColor=BLUE, spaceBefore=10, spaceAfter=5),
        "h3": ParagraphStyle("h3", parent=base["Heading3"], fontSize=11, leading=14, textColor=colors.black, spaceBefore=8, spaceAfter=3),
        "body": ParagraphStyle("body", parent=base["Normal"], fontSize=9.5, leading=13.5, spaceAfter=5),
        "bullet": ParagraphStyle("bullet", parent=base["Normal"], fontSize=9.5, leading=13.5, leftIndent=12, bulletIndent=2, spaceAfter=2),
        "code": ParagraphStyle("code", parent=base["Code"], fontSize=8, leading=10.5, backColor=GREY, borderPadding=5, leftIndent=4, spaceBefore=4, spaceAfter=8),
        "caption": ParagraphStyle("caption", parent=base["Normal"], fontSize=8.5, leading=11, alignment=TA_CENTER, textColor=colors.HexColor("#555555"), spaceAfter=10),
        "note": ParagraphStyle("note", parent=base["Normal"], fontSize=9, leading=12.5, backColor=LIGHT, borderPadding=6, leftIndent=4, spaceBefore=4, spaceAfter=10),
        "cell": ParagraphStyle("cell", parent=base["Normal"], fontSize=8.5, leading=11),
        "cellb": ParagraphStyle("cellb", parent=base["Normal"], fontSize=8.5, leading=11, fontName="Helvetica-Bold", textColor=colors.white),
    }
    return s


S = styles()


def P(text, style="body"):
    return Paragraph(text, S[style])


def B(text):
    return Paragraph(text, S["bullet"], bulletText="\u2022")


def code(text):
    return Preformatted(text.strip("\n"), S["code"])


def table(header, rows, widths=None):
    data = [[Paragraph(h, S["cellb"]) for h in header]]
    data += [[Paragraph(str(c), S["cell"]) for c in row] for row in rows]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GREY]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def figure(fig, width=16 * cm, caption=None):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = width, width * ratio
    items = [img]
    if caption:
        items.append(P(caption, "caption"))
    return KeepTogether(items)


# ---------------------------------------------------------------------- #
# Figures


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    def box(x, y, w, h, text, color="#dce6f1", fs=9):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#1f4e79", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, text=None, style="-|>", ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color="#1f4e79", lw=1.2, ls=ls))
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, text, ha="center", fontsize=7.5, color="#333333")

    box(0.2, 2.2, 1.6, 0.9, "Source\n(Webcam, RTSP,\nVideo, YouTube)")
    box(2.6, 2.2, 2.4, 0.9, "YoloxDistillation\nstudent en production", "#c6e0b4")
    box(5.8, 2.2, 1.7, 0.9, "Chart\nmétriques")
    box(8.1, 2.2, 1.7, 0.9, "Mongodb /\nVideoRecorder")
    box(5.8, 0.9, 1.7, 0.9, "Overlay /\nPreview")
    box(2.6, 0.3, 2.4, 1.1, "Moteur (fils d'arrière-plan)\nteacher ➜ échantillons ➜\nentraînement ➜ validation", "#fff2cc", 8)

    arrow(1.8, 2.65, 2.6, 2.65, "image")
    arrow(5.0, 2.65, 5.8, 2.65, "JSON")
    arrow(7.5, 2.65, 8.1, 2.65, "JSON")
    arrow(5.0, 2.45, 5.8, 1.35, "image")
    arrow(3.8, 2.2, 3.8, 1.4, "1 image / 0,5 s", ls="--")
    arrow(4.6, 1.4, 4.6, 2.2, "student v+1\n(si meilleur)", ls="--")
    return fig


def fig_rounds(history):
    rounds = [r["round"] for r in history]
    cand = [r["candidate"]["loss"] for r in history]
    dep = [r["deployed"]["loss"] for r in history]
    cand_ap = [r["candidate"]["ap50"] for r in history]
    dep_ap = [r["deployed"]["ap50"] for r in history]
    deployed = [r["round"] for r in history if r["deployed_now"]]
    rolled = [r["round"] for r in history if r["rolled_back"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    ax.plot(rounds, dep, "o-", color="#1f4e79", label="student déployé (val_loss)")
    ax.plot(rounds, cand, "s--", color="#c55a11", label="candidat (candidate_val_loss)")
    for r in deployed:
        ax.axvline(r, color="#70ad47", alpha=0.25, lw=6)
    for r in rolled:
        ax.axvline(r, color="#ff0000", alpha=0.25, lw=6)
    ax.set_xlabel("cycle d'entraînement")
    ax.set_ylabel("perte de validation")
    ax.set_title("Perte de distillation (plus bas = mieux)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)

    ax = axes[1]
    ax.plot(rounds, dep_ap, "o-", color="#1f4e79", label="student déployé (val_ap50)")
    ax.plot(rounds, cand_ap, "s--", color="#c55a11", label="candidat (candidate_val_ap50)")
    ax.set_xlabel("cycle d'entraînement")
    ax.set_ylabel("AP50 face au teacher")
    ax.set_title("Accord avec le teacher (plus haut = mieux)", fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    return fig


def fig_chart_example(history):
    """What the Chart node shows over a session: step-like deployed metrics."""
    import numpy as np

    rng = np.random.default_rng(1)
    minutes = np.arange(0, 40)
    dep_ap = [r["deployed"]["ap50"] for r in history] + [history[-1]["candidate"]["ap50"]]
    dep_f1 = [0.7 * a + 0.1 for a in dep_ap]
    original = dep_f1[0]
    # One round every ~3 minutes after a 6-minute collection
    f1 = np.array([dep_f1[min(len(dep_f1) - 1, max(0, (m - 6) // 3))] for m in minutes])
    gain = f1 - original
    train = np.where(minutes < 6, np.nan, 1.25 - 0.004 * (minutes - 6) + rng.normal(0, 0.012, len(minutes)))
    disagreement = 0.12 - 0.3 * gain + rng.normal(0, 0.004, len(minutes))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
    axes[0].plot(minutes, f1, color="#1f4e79", label="val_f1")
    axes[0].axhline(original, color="#7f7f7f", ls="--", label="original_val_f1")
    axes[0].set_title("Panneau A : qualité du student", fontsize=9.5)
    axes[0].set_ylim(0, 1)
    axes[1].plot(minutes, train, color="#c55a11", label="train_loss")
    axes[1].plot(minutes, disagreement * 8, color="#7030a0", label="disagreement (x8)")
    axes[1].set_title("Panneau B : activité d'apprentissage", fontsize=9.5)
    axes[2].plot(minutes, gain, color="#70ad47", label="f1_gain")
    axes[2].axhline(0, color="#7f7f7f", lw=0.8)
    axes[2].set_title("Panneau C : apport de la distillation", fontsize=9.5)
    for ax in axes:
        ax.set_xlabel("minutes (Time Unit = minute)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    return fig


def fig_decision():
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.6)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#1f4e79", lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.3)

    def arrow(x1, y1, x2, y2, text=None):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#1f4e79"))
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.1, text, ha="center", fontsize=7.5)

    box(0.2, 1.0, 2.0, 0.8, "candidat évalué sur\nla fenêtre de validation", "#dce6f1")
    box(2.9, 1.0, 2.3, 0.8, "loss ≤ déployé × 0,99\net AP50 ≥ déployé − 0,02 ?", "#fff2cc")
    box(6.1, 1.8, 1.9, 0.7, "DÉPLOYÉ\nversion +1", "#c6e0b4")
    box(6.1, 0.9, 1.9, 0.7, "KEPT\non continue", "#dce6f1")
    box(6.1, 0.0, 1.9, 0.7, "ROLLED BACK\nlr ÷ 2", "#f8cbad")
    box(8.4, 0.0, 1.5, 2.5, "Chart :\nversion ↑\nval_loss ↓\nval_ap50 ↑\n\ncandidate_* ≠\nval_* pendant\n3 cycles ⇒\nrollback", "#f2f2f2")
    arrow(2.2, 1.4, 2.9, 1.4)
    arrow(5.2, 1.5, 6.1, 2.15, "oui")
    arrow(5.2, 1.4, 6.1, 1.25, "non, perte en baisse")
    arrow(5.2, 1.3, 6.1, 0.35, "non, 3 cycles / +10 % / NaN")
    return fig


# ---------------------------------------------------------------------- #
# Content


def build(history, out):
    doc = SimpleDocTemplate(
        out, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="CV Studio - Manuel du nœud YoloxDistillation", author="CV Studio",
    )
    story = []

    # Cover ------------------------------------------------------------ #
    story += [
        Spacer(1, 4 * cm),
        Paragraph("CV Studio", S["subtitle"]),
        Spacer(1, 0.4 * cm),
        Paragraph("Nœud <b>YoloxDistillation</b>", ParagraphStyle("t", parent=S["title"], alignment=TA_CENTER)),
        Paragraph("Manuel d'utilisation professionnelle<br/>distillation en direct d'un détecteur YOLOX et suivi par le nœud Chart", S["subtitle"]),
        Spacer(1, 1.5 * cm),
        figure(fig_pipeline(), 15 * cm, "Le pipeline de référence : détection, apprentissage en arrière-plan, suivi des métriques et archivage."),
        Spacer(1, 1 * cm),
        P("Ce manuel s'adresse aux intégrateurs qui déploient CV Studio sur des caméras fixes (commerce, "
          "espaces urbains, sites industriels) et veulent qu'un petit détecteur s'améliore sur la scène "
          "observée, sans annotation, tout en gardant la maîtrise de ce qui tourne en production. "
          "Il couvre la mise en place, la lecture des métriques dans le nœud <b>Chart</b>, l'exploitation "
          "(export, retour arrière, reprise) et le diagnostic.", "body"),
        PageBreak(),
    ]

    # TOC -------------------------------------------------------------- #
    story += [P("Sommaire", "h1")]
    for n, t in [
        (1, "Principe et cas d'usage"),
        (2, "Prérequis et installation"),
        (3, "Mise en place pas à pas"),
        (4, "Choisir teacher et student"),
        (5, "Suivre la distillation avec le nœud Chart"),
        (6, "Lire les métriques : référence complète"),
        (7, "Exploitation : export, versions, retour arrière, reprise"),
        (8, "Procédure de qualification hors ligne (CLI)"),
        (9, "Réglages avancés (DistillConfig)"),
        (10, "Diagnostic et problèmes fréquents"),
        (11, "Bonnes pratiques par métier"),
        (12, "Annexes : JSON de sortie, fichiers, glossaire"),
    ]:
        story.append(P(f"{n}.&nbsp;&nbsp;{t}", "bullet"))
    story.append(PageBreak())

    # 1 ------------------------------------------------------------------ #
    story += [
        P("1. Principe et cas d'usage", "h1"),
        P("Un détecteur léger (le <b>student</b>, YOLOX-Nano ou YOLOX-Tiny) tourne en temps réel dans le pipeline. "
          "En arrière-plan, un détecteur lourd (le <b>teacher</b>, YOLOX-S ou plus gros) regarde une image sur deux "
          "secondes environ et ses prédictions servent de cibles pour entraîner le student. Le student apprend ainsi "
          "à imiter le teacher <i>sur votre scène</i> : angle de vue, éclairage, objets typiques, arrière-plan."),
        P("Trois propriétés rendent ce mécanisme exploitable en production :"),
        B("<b>La détection ne ralentit jamais.</b> Seul le student tourne dans le pipeline ; la collecte et "
          "l'entraînement utilisent des fils de priorité basse et le temps processeur laissé libre."),
        B("<b>Rien n'est déployé sans preuve.</b> Un student entraîné ne remplace celui en production que s'il est "
          "meilleur sur des images <i>postérieures</i> à celles de son entraînement (validation « vers l'avant »)."),
        B("<b>Tout est réversible et traçable.</b> Chaque version déployée est conservée, l'historique des décisions "
          "est écrit sur disque, et le modèle d'origine reste disponible en un clic."),
        P("Cas d'usage typiques", "h2"),
        table(["Contexte", "Objectif", "Ce que la distillation apporte"], [
            ["Comptage de passage en magasin", "fiabiliser le comptage « personne » sur une caméra plafonnière",
             "le student apprend les silhouettes vues de haut, que les poids COCO génériques détectent mal ; "
             "suppression des faux positifs récurrents (mannequins, reflets)"],
            ["Occupation d'un espace public", "véhicules / piétons / vélos sur un carrefour",
             "adaptation à l'angle, à la distance et à l'éclairage nocturne de la caméra"],
            ["Zone industrielle, EPI et engins", "détecter des classes présentes dans COCO (person, truck…) sur site",
             "un student 320 px suffisant pour un CPU embarqué, qui rattrape une partie de l'écart avec le teacher"],
            ["Préparation d'un déploiement edge", "obtenir un ONNX spécialisé à embarquer",
             "la CLI (§ 8) distille sur enregistrements, mesure le gain et exporte le modèle"],
        ], [4 * cm, 5 * cm, 8 * cm]),
        Spacer(1, 6),
        P("<b>Ce que la distillation ne fait pas.</b> Elle ne crée pas de classe nouvelle (teacher et student "
          "partagent les 80 classes COCO) et le student ne dépasse pas le teacher : il s'en rapproche. Pour "
          "apprendre des classes spécifiques, utilisez le nœud <b>OnlineTraining</b> ou un entraînement classique.", "note"),
    ]

    # 2 ------------------------------------------------------------------ #
    story += [
        P("2. Prérequis et installation", "h1"),
        table(["Composant", "Requis pour", "Remarque"], [
            ["onnxruntime", "détection (student) et teacher", "inclus ; GPU utilisé si CUDAExecutionProvider est disponible"],
            ["torch, torchvision, onnx2torch", "entraînement du student", "dans requirements.txt et l'exécutable ; s'ils manquent, le nœud détecte et affiche « unavailable »"],
            ["Modèles yolox_nano / yolox_tiny / yolox_s", "students et teacher fournis", "node/DLNode/object_detection/YOLOX/model/"],
            ["Disque", "échantillons et versions", "compter ~150 Mo pour 4 000 échantillons + checkpoints (≈ 5 × taille du student)"],
            ["CPU / GPU", "vitesse d'apprentissage", "CPU 2 cœurs : un cycle yolox_nano ≈ 10-20 s ; GPU : lots et cycles 5× plus grands"],
        ], [5 * cm, 4.5 * cm, 7.5 * cm]),
        Spacer(1, 6),
        P("Vérification rapide depuis les sources :"),
        code("""
python -c "from node.DLNode.online_training.distill import yolox_torch; print(yolox_torch.backend_status())"
# (True, '')  -> l'entraînement est possible
python -m node.DLNode.online_training.distill.cli --list-students
"""),
        P("Au lancement, <code>main.py --smoke-test</code> convertit un student en PyTorch et le réexporte : c'est "
          "le test à faire après toute mise à jour de l'environnement Python."),
    ]

    # 3 ------------------------------------------------------------------ #
    story += [
        P("3. Mise en place pas à pas", "h1"),
        P("3.1 Le pipeline minimal", "h2"),
        P("Menu <b>Input</b> → une source (Webcam, RTSP, Video, YouTube…) ; menu <b>VisionModel</b> → "
          "<b>YoloxDistillation</b> ; relier la sortie image de la source à l'entrée <i>Image</i> du nœud."),
        table(["Contrôle du nœud", "Rôle"], [
            ["teacher", "modèle lourd qui enseigne (par défaut le plus gros fichier yolox*.onnx trouvé)"],
            ["student", "modèle léger en production, choisi dans le catalogue (§ 4) ; une ligne décrit le choix"],
            ["Distill (background)", "<b>décoché au départ</b>. Coché : collecte + entraînement. Décoché : pause (la détection continue)"],
            ["Metrics to Chart", "coché (défaut) : le JSON porte <code>distillation_losses</code>, que le nœud Chart trace en séries temporelles. "
                                 "Décoché : un Chart connecté compte les classes détectées, comme pour ObjectDetection"],
            ["score", "seuil de score des détections affichées et sorties (0,3 par défaut)"],
            ["Texte d'état", "version déployée, phase, matériel, échantillons, dernière mesure"],
            ["Export ONNX", "écrit le student déployé dans <code>_Distillation/exports/</code>"],
            ["Reset student", "revient au modèle d'origine ; les échantillons sont conservés"],
        ], [4.5 * cm, 12.5 * cm]),
        Spacer(1, 6),
        P("3.2 Lire le texte d'état", "h2"),
        code("""
student v3 - training (cpu)      <- version déployée (v0 = modèle d'origine), phase, matériel
samples: 412 train / 96 val      <- échantillons d'entraînement / fenêtre de validation
val loss 0.871  AP50 0.76        <- mesure du student déployé sur la fenêtre de validation
"""),
        table(["Phase", "Signification", "Action"], [
            ["paused", "Distill décoché", "aucune : la détection tourne"],
            ["collecting", "le teacher enregistre des échantillons ; pas encore 48 train / 12 val", "attendre (≈ 1 min à 0,5 s/image sur une scène qui bouge)"],
            ["training", "un cycle d'entraînement est en cours", "—"],
            ["evaluating", "candidat et déployé mesurés sur la fenêtre de validation", "—"],
            ["idle", "entre deux cycles, en attente de nouveaux échantillons", "—"],
            ["error", "message affiché (modèles incompatibles, disque…)", "voir § 10"],
            ["unavailable", "PyTorch / onnx2torch manquants", "installer les dépendances (§ 2)"],
        ], [3 * cm, 8 * cm, 6 * cm]),
        Spacer(1, 6),
        P("3.3 Ordre de démarrage conseillé", "h2"),
        B("Lancer le pipeline avec <i>Distill</i> décoché ; vérifier que les détections du student sont plausibles et que le débit convient."),
        B("Connecter un nœud <b>Chart</b> sur la sortie JSON (§ 5) <i>avant</i> de cocher <i>Distill</i> : la courbe de référence "
          "<code>original_val_f1</code> et la première mesure du déployé seront visibles dès le premier cycle."),
        B("Cocher <i>Distill (background)</i>. Laisser tourner au moins 30 minutes sur une scène représentative avant de juger."),
        B("Sauvegarder le pipeline : teacher, student, état de la case et seuil sont persistés ; au redémarrage le nœud "
          "reprend avec le dernier student déployé et le stock d'échantillons."),
    ]

    # 4 ------------------------------------------------------------------ #
    story += [
        P("4. Choisir teacher et student", "h1"),
        P("4.1 Le catalogue de students", "h2"),
        table(["Clé", "Réseau", "Entrée", "Paramètres", "Coût relatif / image", "Choisir quand"], [
            ["yolox_nano_320", "YOLOX-Nano", "320×320", "0,9 M", "≈ 0,6", "CPU faible ou très chargé ; objets grands ou proches (caméra à hauteur d'homme, comptoir)"],
            ["yolox_nano", "YOLOX-Nano", "416×416", "0,9 M", "1 (référence)", "<b>défaut</b> : bon compromis pour une caméra de magasin ou de rue sur CPU"],
            ["yolox_tiny_320", "YOLOX-Tiny", "320×320", "5,1 M", "≈ 2,3", "plus de capacité d'apprentissage que nano pour un coût d'inférence proche ; scènes denses"],
            ["yolox_tiny", "YOLOX-Tiny", "416×416", "5,1 M", "≈ 4", "le plus précis des petits ; entraînement plus lourd (GPU ou CPU 4 cœurs+), petits objets"],
        ], [2.7 * cm, 2.2 * cm, 1.6 * cm, 1.9 * cm, 2.6 * cm, 6 * cm]),
        Spacer(1, 6),
        P("Les variantes 320 px sont dérivées des fichiers fournis à la première sélection (le graphe YOLOX est "
          "entièrement convolutif), vérifiées avec onnxruntime et conservées dans <code>_Distillation/students/</code>. "
          "320 px reste dans la plage multi-échelle (320-640) de l'entraînement officiel. Elles voient moins bien les "
          "petits objets : préférez 416 si vos cibles font moins de ~5 % de la largeur de l'image."),
        P("4.2 Le teacher", "h2"),
        P("Plus le teacher est bon sur votre scène, plus le student a à apprendre. <code>yolox_s</code> (640) est fourni. "
          "Tout fichier <code>yolox*.onnx</code> déposé dans le répertoire des modèles apparaît dans la liste : "
          "<code>yolox_m</code>, <code>yolox_l</code>, <code>yolox_x</code> exportés avec l'export ONNX officiel "
          "(<code>decode_in_inference=False</code>). Contraintes : mêmes classes que le student, entrée au moins aussi grande."),
        P("<b>Coût du teacher.</b> Il ne voit qu'une image toutes les 0,5 s, sur un fil de priorité basse. Sur CPU, "
          "<code>yolox_s</code> prend ≈ 150-300 ms par image, <code>yolox_m</code> 2 à 3 fois plus : compter cela dans la "
          "charge si la machine est partagée. Si le teacher est trop lent, augmentez <code>sample_interval</code> (§ 9).", "note"),
        P("4.3 Règle de décision rapide", "h2"),
        table(["Situation", "Teacher", "Student"], [
            ["CPU 2 cœurs partagé, caméra de magasin, personnes à moins de 10 m", "yolox_s", "yolox_nano_320 ou yolox_nano"],
            ["CPU 4-8 cœurs dédié, carrefour urbain, véhicules et piétons", "yolox_s (ou yolox_m ajouté)", "yolox_tiny_320"],
            ["GPU disponible, scène dense ou petits objets", "yolox_m / yolox_l ajouté", "yolox_tiny"],
            ["Qualification d'un modèle pour une cible edge précise", "le plus gros disponible (CLI, § 8)", "la taille acceptée par la cible"],
        ], [7 * cm, 4.5 * cm, 5.5 * cm]),
    ]

    # 5 ------------------------------------------------------------------ #
    story += [
        PageBreak(),
        P("5. Suivre la distillation avec le nœud Chart", "h1"),
        P("Le texte d'état du nœud donne l'instant présent ; le nœud <b>Chart</b> (menu <b>Visual</b>) donne la "
          "tendance, qui seule permet de décider : la distillation progresse-t-elle ? S'est-elle stabilisée ? Le student "
          "déployé vaut-il mieux que l'original ?"),
        P("5.1 Branchement", "h2"),
        B("Relier la sortie <b>JSON</b> de YoloxDistillation à l'entrée <i>Input detection JSON</i> du Chart."),
        B("Laisser <b>Metrics to Chart</b> coché sur YoloxDistillation. Le Chart reconnaît la clé "
          "<code>distillation_losses</code> et bascule en mode « séries temporelles » : chaque métrique devient une courbe, "
          "moyennée par intervalle de temps."),
        B("<b>Time Unit</b> : <i>second</i> pour une démonstration ou un débogage, <i>minute</i> pour une session de travail, "
          "<i>hour</i> pour une supervision sur plusieurs jours (le Chart conserve 24 h glissantes)."),
        B("<b>Chart Type</b> : <i>line</i> est le bon choix pour des métriques ; <i>area</i> ou <i>bar</i> pour comparer "
          "<code>train_samples</code> / <code>val_samples</code>."),
        B("<b>Class 1</b> et <i>Add Class Slot</i> : chaque emplacement choisit une métrique à tracer. Les listes se remplissent "
          "avec les clés de <code>distillation_losses</code> dès la première image."),
        B("<b>Download Chart Image</b> enregistre le graphique courant (PNG) ; <b>Flush</b> efface l'historique du Chart "
          "(utile après un <i>Reset student</i>)."),
        P("<b>Un Chart par échelle.</b> Les métriques n'ont pas la même échelle : F1 et AP50 dans [0, 1], perte autour de 1, "
          "<code>inference_ms</code> en millisecondes, <code>train_samples</code> en milliers. Utilisez plusieurs nœuds Chart "
          "connectés au même JSON, un par famille (ci-dessous), plutôt qu'un seul Chart avec tout.", "note"),
        P("5.2 Les trois panneaux recommandés", "h2"),
        figure(fig_chart_example(history), 17 * cm,
               "Figure 2 - Illustration (construite à partir de l'historique d'un run) d'une session de 40 minutes, Time Unit = minute, Chart Type = line. "
               "Les paliers des courbes « val_* » correspondent aux déploiements ; la ligne pointillée est le student d'origine."),
        table(["Panneau", "Métriques (Class slots)", "Ce qu'on y lit"], [
            ["A - Qualité du student déployé", "val_f1, original_val_f1<br/>(option : val_ap50, val_precision, val_recall)",
             "La courbe val_f1 doit se détacher vers le haut d'original_val_f1 et se stabiliser. Précision qui chute + rappel "
             "qui monte = le student détecte plus mais se trompe plus : vérifier visuellement."],
            ["B - Activité d'apprentissage", "train_loss, disagreement<br/>(option : candidate_val_loss, val_loss)",
             "train_loss décroît puis oscille : normal. disagreement (écart teacher/student sur la dernière image collectée) "
             "doit baisser lentement. S'il remonte durablement, la scène a changé (§ 10)."],
            ["C - Apport et santé", "f1_gain, version<br/>(option : rounds, inference_ms, detections)",
             "f1_gain > 0 = la distillation a apporté quelque chose, mesuré sur des images jamais vues à l'entraînement. "
             "version en escalier ; inference_ms stable = la détection n'est pas pénalisée."],
        ], [3.8 * cm, 5.2 * cm, 8 * cm]),
        Spacer(1, 6),
        P("5.3 Interpréter les formes de courbes", "h2"),
        table(["Forme observée", "Interprétation", "Décision"], [
            ["val_f1 monte par paliers, f1_gain > 0,05 après 30 min", "distillation efficace sur cette scène", "laisser tourner ; exporter une version (§ 7) quand le plateau est atteint"],
            ["version reste à 0 ; candidate_val_loss ≈ val_loss", "le student est déjà d'accord avec le teacher, ou la scène est trop pauvre", "normal sur une scène vide ; vérifier disagreement (< 0,03 = rien à apprendre)"],
            ["candidate_val_loss < val_loss mais pas de déploiement", "gain de perte < 1 % ou AP50 en baisse (garde-fou)", "attendre ; si cela dure plus de 3 cycles, un rollback se produit automatiquement"],
            ["version monte puis val_f1 baisse légèrement", "dérive de la scène (la fenêtre de validation a changé)", "lecture : les val_* sont mesurées sur les images récentes ; comparer à original_val_f1, pas à l'ancien val_f1"],
            ["train_loss en dents de scie croissantes, puis chute brutale", "divergence puis rollback (lr ÷ 2)", "aucune action ; si récurrent, baisser learning_rate (§ 9)"],
            ["inference_ms grimpe pendant « training »", "machine saturée : le pipeline attend le processeur", "augmenter sample_interval, réduire cpu_threads, ou passer à un student 320"],
            ["disagreement remonte durablement", "changement de scène : éclairage, caméra déplacée, foule", "normal : le moteur va réapprendre ; surveiller version et f1_gain"],
        ], [5.3 * cm, 6 * cm, 5.7 * cm]),
        Spacer(1, 6),
        P("5.4 Au-delà du Chart : archiver et alerter", "h2"),
        B("<b>Chart → Mongodb</b> : la sortie JSON du Chart donne la dernière valeur de chaque métrique sélectionnée ; "
          "un nœud <b>Mongodb</b> (menu Action) en fait un historique interrogeable (une ligne par image traitée : filtrer en amont ou "
          "utiliser Time Unit = minute pour lisser)."),
        B("<b>Chart → VideoRecorder</b> : la sortie image du Chart peut être enregistrée pour joindre la courbe au compte-rendu d'un essai."),
        B("<b>YoloxDistillation → Operator → Trigger</b> : le nœud <b>Operator</b> (DataProcess) combine des valeurs JSON ; "
          "il permet par exemple de calculer <code>val_f1 − original_val_f1</code> ou de seuiller <code>inference_ms</code> pour "
          "déclencher une action."),
        B("<b>Rapport d'essai</b> : <i>Download Chart Image</i> des panneaux A et C + le fichier <code>state.json</code> "
          "(historique des cycles) constituent la preuve de ce qui a été déployé et pourquoi."),
    ]

    # 6 ------------------------------------------------------------------ #
    story += [
        PageBreak(),
        P("6. Lire les métriques : référence complète", "h1"),
        P("Toutes les métriques sont disponibles dans <code>distillation_losses</code> (pour le Chart) et dans "
          "<code>distillation</code> (avec la phase) du JSON de sortie. Une valeur non encore mesurée vaut <code>null</code> "
          "et n'est pas tracée. <b>val_*</b> et <b>original_val_*</b> sont mesurés sur la <b>fenêtre de validation</b> : les "
          "96 échantillons les plus récents, jamais utilisés pour l'entraînement en cours, avec pour référence les détections du "
          "teacher (score ≥ 0,3, IoU ≥ 0,5, même classe)."),
        table(["Clé", "Plage", "Définition", "Bonne valeur"], [
            ["val_f1", "0-1", "F1 du student déployé face au teacher sur la fenêtre de validation", "↑, au-dessus de original_val_f1"],
            ["val_ap50", "0-1", "AP à IoU 0,5 du student déployé face au teacher", "↑"],
            ["val_precision", "0-1", "part des détections du student confirmées par le teacher", "↑ ; une baisse = faux positifs"],
            ["val_recall", "0-1", "part des détections du teacher retrouvées par le student", "↑ ; une baisse = objets manqués"],
            ["val_loss", "≈ 0,5-2", "perte de distillation du déployé (objectness + classes + boîtes, toutes les ancres)", "↓ ; critère principal de déploiement"],
            ["original_val_f1", "0-1", "F1 du student <i>d'origine</i> sur les mêmes images", "référence fixe pour la scène courante"],
            ["original_val_ap50", "0-1", "AP50 du student d'origine sur les mêmes images", "référence"],
            ["f1_gain", "−1-1", "val_f1 − original_val_f1 : l'apport net de la distillation", "> 0 ; > 0,05 = gain net"],
            ["candidate_val_loss", "≈ 0,5-2", "perte du student entraîné au dernier cycle, déployé ou non", "< val_loss × 0,99 pour déployer"],
            ["candidate_val_ap50", "0-1", "AP50 du candidat", "≥ val_ap50 − 0,02 pour déployer"],
            ["train_loss", "≈ 0,5-2", "perte moyenne des 20 derniers pas d'entraînement (données augmentées : plus haute que val_loss)", "↓ puis stable"],
            ["disagreement", "0-1", "écart teacher/student sur la dernière image montrée au teacher (0 = identiques)", "↓ ; < 0,03 = rien à apprendre"],
            ["version", "entier", "nombre de students déployés depuis l'original", "escalier croissant"],
            ["rounds", "entier", "cycles d'entraînement effectués (déployés ou non)", "—"],
            ["train_samples", "0-4000", "échantillons disponibles pour l'entraînement", "≥ 48 pour démarrer ; quelques centaines pour un gain"],
            ["val_samples", "0-96", "taille de la fenêtre de validation", "96 en régime établi"],
            ["detections", "entier", "détections du student sur l'image courante", "—"],
            ["inference_ms", "ms", "durée de la détection du student sur l'image courante", "stable ; nano 416 ≈ 10-30 ms CPU"],
        ], [3.2 * cm, 1.6 * cm, 8.2 * cm, 4 * cm]),
        Spacer(1, 6),
        P("<b>Pourquoi deux pertes.</b> <code>train_loss</code> est calculée sur des images retournées, recolorées, bruitées : "
          "elle est structurellement plus élevée que <code>val_loss</code> et n'est pas comparable. C'est <code>val_loss</code> "
          "(et <code>candidate_val_loss</code>) qui décide du déploiement.", "note"),
    ]

    # 7 ------------------------------------------------------------------ #
    story += [
        P("7. Exploitation : export, versions, retour arrière, reprise", "h1"),
        figure(fig_decision(), 17 * cm, "Figure 3 - Décision prise à la fin de chaque cycle et sa trace dans le Chart."),
        P("7.1 Arborescence de travail", "h2"),
        code("""
_Distillation/                                  (CV_Studio_data/distillation/ pour l'exécutable)
  students/yolox_nano_320.onnx                  variantes 320 px générées
  yolox_nano__from__yolox_s/                    un répertoire par couple student/teacher
    deployed.onnx                               student en production (rechargé au démarrage)
    state.json                                  version + historique des 100 derniers cycles
    checkpoints/v00001.onnx ... v00005.onnx     5 dernières versions déployées
    samples/                                    échantillons (JPEG + cibles du teacher), bornés à 4 000
  exports/yolox_nano_distilled_v7.onnx          exports manuels
"""),
        P("7.2 Exporter un student", "h2"),
        P("<b>Export ONNX</b> écrit le student déployé, prêt à l'emploi : même graphe, mêmes entrées/sorties que le fichier "
          "d'origine (80 classes COCO, décodage des boîtes à faire comme pour tout YOLOX). Il se charge dans le nœud "
          "<b>ObjectDetection</b> (format YOLOX) ou dans tout runtime ONNX. Nommez et archivez l'export avec la capture du "
          "Chart et le <code>state.json</code> du moment : vous aurez la version, la date et les mesures qui ont justifié le déploiement."),
        P("7.3 Revenir en arrière", "h2"),
        B("<b>Reset student</b> : retour immédiat au modèle d'origine (v0). Les échantillons sont conservés : le réapprentissage reprend de zéro mais vite."),
        B("<b>Revenir à une version précise</b> : arrêter le pipeline, copier <code>checkpoints/vNNNNN.onnx</code> sur "
          "<code>deployed.onnx</code>, relancer. Le moteur vérifie la cohérence du fichier au chargement."),
        B("<b>Figer la production</b> : décocher <i>Distill</i>. Le student déployé reste actif, plus rien ne change."),
        P("7.4 Reprise et persistance", "h2"),
        P("Au redémarrage, le nœud recharge <code>deployed.onnx</code>, l'historique et le stock. Les échantillons portent une "
          "signature (teacher, taille d'entrée, classes) : changer de teacher ou de student crée un nouveau répertoire et repart "
          "d'un stock vide. Deux nœuds utilisant le même couple partagent un seul moteur."),
        P("7.5 Charge machine", "h2"),
        table(["Ressource", "Consommation", "Levier"], [
            ["CPU pipeline", "détection du student uniquement (nano 416 : 10-30 ms)", "student 320 ; seuil de score"],
            ["CPU arrière-plan", "teacher 1 image / 0,5 s + entraînement (moitié des cœurs par défaut)", "cpu_threads, sample_interval (§ 9) ; fils en priorité basse"],
            ["GPU", "teacher et student onnxruntime si CUDA ; entraînement torch", "device = cuda ; use_gpu_inference"],
            ["Disque", "≈ 30-40 Ko par échantillon, 4 000 max + 5 checkpoints", "max_train ; keep_checkpoints"],
            ["Mémoire", "≈ 1-2 Go avec PyTorch chargé (CPU)", "un seul moteur par couple de modèles"],
        ], [3.5 * cm, 7.5 * cm, 6 * cm]),
    ]

    # 8 ------------------------------------------------------------------ #
    story += [
        PageBreak(),
        P("8. Procédure de qualification hors ligne (CLI)", "h1"),
        P("Avant d'activer la distillation sur un site, ou pour produire un modèle à embarquer, rejouez un enregistrement "
          "de la caméra avec l'outil en ligne de commande. Il reproduit exactement le moteur du nœud et ajoute une mesure "
          "honnête : la fin de la vidéo (30 % par défaut) n'est <b>jamais</b> montrée au moteur et sert à comparer le student "
          "avant/après, face au teacher en pleine résolution."),
        code("""
python -m node.DLNode.online_training.distill.cli \\
    --teacher yolox_s --student yolox_nano \\
    --source enregistrement_camera_12.mp4 --workdir _Distillation/qualif_cam12 \\
    --rounds 30 --every 3

# options utiles
#   --list-students          catalogue des students
#   --device cuda            entraîner sur GPU
#   --holdout 0.3            part finale réservée à la mesure
#   --teacher-view full      le teacher regarde l'image en pleine résolution (voir doc technique)
"""),
        P("Sortie type :"),
        code("""
stream: 270 frames -> 203 training samples, 67 most recent ones for validation; held out: 116 frames
round  steps  train-loss  val-loss cand/deployed   val-AP50 cand/deployed  decision
    1     40      1.2759     1.0313 / 1.0947       0.542 / 0.490     deployed v1  (5.8s)
    2     80      1.2172     1.0091 / 1.0313       0.595 / 0.542     deployed v2  (5.9s)
  ...
held-out frames (never used), reference = teacher at full resolution
  original student   AP50=0.271 F1=0.423 precision=0.768 recall=0.292
  distilled student  AP50=0.309 F1=0.451 precision=0.626 recall=0.352
deployed student: _Distillation/qualif_cam12/deployed.onnx
"""),
        figure(fig_rounds(history), 16.5 * cm,
               f"Figure 4 - Historique réel d'un run CLI (yolox_s → yolox_nano, caméra embarquée, CPU, {len(history)} cycles) : "
               "zones vertes = déploiements, rouges = rollbacks. Ce sont les mêmes grandeurs que val_loss / candidate_val_loss "
               "et val_ap50 / candidate_val_ap50 dans le Chart."),
        P("8.1 Protocole recommandé", "h2"),
        B("Enregistrer 10 à 30 minutes représentatives (heures d'affluence et heures creuses, jour et nuit si pertinent)."),
        B("Lancer la CLI avec les réglages par défaut ; noter AP50/F1 avant/après sur la partie réservée."),
        B("Accepter la mise en production si F1 progresse et que la précision ne s'effondre pas ; sinon essayer un student plus grand "
          "(tiny) ou plus de cycles."),
        B("Le <code>deployed.onnx</code> obtenu peut être copié dans le répertoire de travail du nœud pour démarrer la production avec un student déjà spécialisé."),
        P("<b>Lecture des résultats mesurés dans la documentation technique</b> (docs/DISTILLATION_YOLOX.md, § 4) : caméra fixe et "
          "quelques centaines d'échantillons donnent un gain net (F1 0,73 → 0,99 ; 0,38 → 0,54) ; caméra mobile ou scène très "
          "changeante donnent un gain faible mais jamais une dégradation, grâce au garde-fou de validation.", "note"),
    ]

    # 9 ------------------------------------------------------------------ #
    story += [
        P("9. Réglages avancés (DistillConfig)", "h1"),
        P("Le nœud utilise les valeurs par défaut, qui ont été mesurées comme les plus sûres. La CLI expose les principales "
          "(<code>--lr</code>, <code>--steps</code>, <code>--batch</code>, <code>--trainable</code>, <code>--teacher-view</code>) ; "
          "le reste se règle dans <code>DistillConfig</code> (node/DLNode/online_training/distill/engine.py)."),
        table(["Paramètre", "Défaut", "Effet", "Quand le changer"], [
            ["device", "auto", "cuda si disponible, sinon cpu", "forcer cpu pour laisser le GPU à d'autres modèles"],
            ["cpu_threads", "moitié des cœurs", "threads du teacher et de l'entraînement", "machine partagée : réduire"],
            ["sample_interval", "0,5 s", "délai entre deux images montrées au teacher", "teacher lourd ou CPU saturé : 1-2 s ; scène rapide : 0,25 s"],
            ["min_disagreement / keep_easy", "0,03 / 0,25", "images ignorées si le student est déjà d'accord, sauf 25 % gardées", "scène très stable : baisser keep_easy pour économiser le disque"],
            ["max_train / val_window", "4000 / 96", "taille du stock et de la fenêtre de validation", "longue durée : augmenter val_window (mesures plus stables)"],
            ["min_train / min_val", "48 / 12", "échantillons requis avant le premier cycle", "démonstration : baisser ; production : laisser"],
            ["learning_rate", "5e-5", "pas d'apprentissage AdamW", "rollbacks fréquents : diviser par 2"],
            ["trainable_fraction", "0,3", "part des couches entraînées, depuis la sortie", "> 1 000 échantillons et GPU : 0,5-1,0"],
            ["steps_per_round / batch_size", "40 / 4 (CPU), 200 / 16 (GPU)", "longueur d'un cycle", "cycles plus longs = décisions moins fréquentes mais plus sûres"],
            ["min_gain / ap_tolerance", "0,01 / 0,02", "garde-fou de déploiement", "ne pas assouplir en production"],
            ["patience", "3", "cycles sans progrès avant rollback", "—"],
            ["keep_checkpoints", "5", "versions conservées", "traçabilité longue : augmenter"],
        ], [3.6 * cm, 3.2 * cm, 5.2 * cm, 5 * cm]),
    ]

    # 10 ----------------------------------------------------------------- #
    story += [
        PageBreak(),
        P("10. Diagnostic et problèmes fréquents", "h1"),
        table(["Symptôme", "Cause probable", "Résolution"], [
            ["État « unavailable »", "torch / onnx2torch absents ou cassés", "pip install torch torchvision onnx2torch ; relancer main.py --smoke-test"],
            ["État « error : teacher has N classes, student M »", "modèles de classes différentes", "teacher et student doivent partager les classes (COCO 80 pour les fichiers fournis)"],
            ["« the teacher input must be at least as large… »", "teacher plus petit que le student", "choisir un teacher ≥ 416 (yolox_s = 640)"],
            ["« select a teacher and a student model »", "liste vide : aucun yolox*.onnx trouvé", "vérifier node/DLNode/object_detection/YOLOX/model/"],
            ["Reste en « collecting » très longtemps", "scène immobile : images identiques ignorées ; ou disagreement < 0,03", "normal sur scène vide ; attendre du mouvement ; vérifier disagreement dans le Chart"],
            ["version reste à 0 après 10+ cycles", "candidat jamais meilleur de 1 % avec AP50 maintenu", "regarder candidate_val_loss vs val_loss ; student déjà proche du teacher, ou student trop petit → tiny"],
            ["Rollbacks répétés (train_loss en dents de scie)", "pas d'apprentissage trop grand pour la scène", "baisser learning_rate (CLI --lr 2.5e-5)"],
            ["inference_ms double pendant l'entraînement", "CPU saturé malgré la priorité basse", "cpu_threads plus bas ; sample_interval plus long ; student 320"],
            ["Le Chart compte des classes au lieu de tracer des métriques", "Metrics to Chart décoché", "cocher la case ; Flush sur le Chart"],
            ["Le Chart trace une courbe plate à 0", "métrique encore null (pas de cycle terminé)", "attendre le premier cycle ; depuis #894 les null ne sont plus tracés"],
            ["Les échantillons ont disparu après changement de teacher", "signature différente → nouveau répertoire", "attendu ; l'ancien répertoire est intact"],
            ["« exported ONNX model does not match » à l'export", "poids non finis après divergence (rare)", "Reset student, baisser learning_rate"],
        ], [5 * cm, 5.5 * cm, 6.5 * cm]),
        Spacer(1, 6),
        P("Journal : les messages du moteur sont préfixés <code>distillation:</code> et ceux du nœud "
          "<code>[YoloxDistillation]</code> dans les logs de CV Studio."),
    ]

    # 11 ----------------------------------------------------------------- #
    story += [
        P("11. Bonnes pratiques par métier", "h1"),
        P("11.1 Commerce (comptage, files, rayons)", "h2"),
        B("Caméra fixe, student <code>yolox_nano</code>, teacher <code>yolox_s</code> ; laisser apprendre sur une journée complète d'ouverture."),
        B("Chart A : <code>val_f1</code>, <code>original_val_f1</code> en Time Unit = hour, pour suivre le gain jour après jour."),
        B("Figer (<i>Distill</i> décoché) une fois le plateau atteint et exporter : un modèle stable est préférable à un modèle qui change pendant une campagne de mesure."),
        B("Relancer la distillation après un réagencement du magasin ou un changement d'éclairage (disagreement remonte)."),
        P("11.2 Urbanisme et mobilité", "h2"),
        B("Scènes larges, objets petits : <code>yolox_tiny</code> (416) si la machine le permet ; teacher <code>yolox_m</code> ajouté au répertoire si GPU."),
        B("Surveiller <code>val_recall</code> : c'est le rappel qui souffre le plus sur les petits objets distants."),
        B("Jour/nuit : le stock borné (4 000) suit la scène ; sur 24 h le student apprend les deux régimes. Si la nuit domine, augmenter <code>max_train</code>."),
        P("11.3 Spatial intelligence / site multi-caméras", "h2"),
        B("Un moteur par caméra (répertoires séparés : préfixer le workdir ou utiliser un pipeline par caméra). Un student distillé sur une caméra n'est pas fait pour une autre."),
        B("Centraliser les métriques : Chart → Mongodb par caméra, ou exporter périodiquement <code>state.json</code>."),
        B("Qualifier d'abord en CLI sur des enregistrements de chaque caméra (§ 8) et déployer les <code>deployed.onnx</code> obtenus comme point de départ."),
    ]

    # 12 ----------------------------------------------------------------- #
    story += [
        PageBreak(),
        P("12. Annexes", "h1"),
        P("12.1 JSON de sortie du nœud", "h2"),
        code("""
{
  "bboxes": [[x1, y1, x2, y2], ...],          "scores": [0.87, ...],
  "class_ids": [0, 2, ...],                    "class_names": {"0": "person", ...},
  "score_th": 0.3,
  "distillation": {
    "phase": "training",
    "val_f1": 0.81, "val_ap50": 0.76, "val_precision": 0.84, "val_recall": 0.78, "val_loss": 0.871,
    "original_val_f1": 0.69, "original_val_ap50": 0.61, "f1_gain": 0.12,
    "candidate_val_loss": 0.864, "candidate_val_ap50": 0.77,
    "train_loss": 1.02, "disagreement": 0.041,
    "version": 3, "rounds": 7, "train_samples": 412, "val_samples": 96,
    "detections": 4, "inference_ms": 18.2
  },
  "distillation_losses": { ...mêmes clés sans "phase", présent si « Metrics to Chart » est coché... }
}
"""),
        P("Le bloc <code>bboxes / scores / class_ids / class_names</code> est au format du nœud ObjectDetection : tous les nœuds "
          "aval (Overlay, Heatmap, Tracking, Trigger ObjDetCount…) fonctionnent sans changement."),
        P("12.2 Fichiers du code", "h2"),
        table(["Fichier", "Rôle"], [
            ["node/DLNode/node_yolox_distillation.py", "le nœud, le texte d'état, les métriques pour le Chart"],
            ["node/DLNode/online_training/distill/engine.py", "collecte, entraînement, validation, déploiement, DistillConfig"],
            ["…/distill/students.py", "catalogue des students, variantes 320 px"],
            ["…/distill/yolox_torch.py", "student entraînable depuis l'ONNX, perte de distillation"],
            ["…/distill/yolox_onnx.py", "teacher et student déployé (onnxruntime)"],
            ["…/distill/store.py", "stock d'échantillons sur disque"],
            ["…/distill/cli.py", "outil de qualification hors ligne"],
            ["node/VisualNode/node_obj_chart.py", "le nœud Chart"],
            ["docs/DISTILLATION_YOLOX.md", "documentation technique : méthode, résultats mesurés, références"],
            ["tools/make_distillation_manual.py", "génère ce manuel"],
        ], [7.5 * cm, 9.5 * cm]),
        Spacer(1, 6),
        P("12.3 Glossaire", "h2"),
        table(["Terme", "Définition"], [
            ["Teacher / Student", "modèle lourd qui enseigne / modèle léger en production qui apprend"],
            ["Distillation dense", "le student imite le teacher sur chaque case de la grille de détection (objectness, classes, boîte), pas seulement sur les objets finaux"],
            ["Fenêtre de validation", "les 96 échantillons les plus récents, exclus de l'entraînement, sur lesquels candidat et déployé sont comparés"],
            ["Candidat", "student entraîné au dernier cycle (moyenne EMA des poids), pas encore déployé"],
            ["Déployé", "student qui tourne dans le pipeline ; version n"],
            ["Cycle (round)", "40 pas (CPU) ou 200 pas (GPU) d'entraînement suivis d'une évaluation et d'une décision"],
            ["Rollback", "retour des poids d'entraînement à ceux du déployé, pas d'apprentissage divisé par 2"],
            ["AP50", "précision moyenne à IoU ≥ 0,5, ici face aux détections du teacher"],
            ["F1", "moyenne harmonique de la précision et du rappel"],
            ["disagreement", "écart normalisé des scores teacher/student sur toutes les cases d'une image ; 0 = identiques"],
            ["EMA", "moyenne exponentielle des poids pendant l'entraînement : lisse le candidat"],
        ], [4 * cm, 13 * cm]),
    ]

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(2 * cm, 1.1 * cm, "CV Studio - Manuel du nœud YoloxDistillation")
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=lambda c, d: None, onLaterPages=footer)


def load_history(run):
    if not run:
        return _DEFAULT_HISTORY
    with open(os.path.join(run, "state.json")) as fp:
        history = json.load(fp)["history"]
    return history or _DEFAULT_HISTORY


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", help="workdir of a distillation run (state.json) for the figures")
    parser.add_argument("--out", default=_DEFAULT_OUT)
    args = parser.parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    build(load_history(args.run), args.out)
    print(args.out)


if __name__ == "__main__":
    main()
