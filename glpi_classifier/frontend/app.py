import sys
from pathlib import Path
import torch

import streamlit as st
from transformers import (
    CamembertTokenizerFast,
    AutoModelForSequenceClassification
)


# ============================================================
# CHEMINS DU PROJET
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

MODEL_PATH = "Hanane23/Odoo_glpi_classifier_model"


# ============================================================
# BACKEND
# ============================================================

from backend.processor import GLPITicketProcessor


# ============================================================
# CONFIGURATION
# ============================================================

MAX_LENGTH = 267

ID2LABEL = {
    0: "Administration et Support",
    1: "BI (Business Intelligence)",
    2: "Recherche et Développement"
}


# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================

@st.cache_resource
def load_model():

    tokenizer = CamembertTokenizerFast.from_pretrained(
        MODEL_PATH, subfolder="model"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, subfolder="model"
    )

    model.eval()

    processor = GLPITicketProcessor()

    return tokenizer, model, processor


tokenizer, model, processor = load_model()


# ============================================================
# FONCTION DE PRÉDICTION
# ============================================================

def predict_category(title, content):

    # On combine titre + contenu
    text = f"{title}\n{content}"

    # Même preprocessing que pendant l'entraînement
    cleaned_text = processor.clean_text(text)

    # Tokenisation
    inputs = tokenizer(
        cleaned_text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    # Inférence
    with torch.no_grad():

        outputs = model(**inputs)

    # Probabilités
    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    # Classe prédite
    predicted_id = torch.argmax(
        probabilities
    ).item()

    confidence = probabilities[
        predicted_id
    ].item()

    probability_dict = {
        ID2LABEL[i]: probabilities[i].item()
        for i in range(len(ID2LABEL))
    }

    return (
        ID2LABEL[predicted_id],
        confidence,
        probability_dict
    )


# ============================================================
# INTERFACE
# ============================================================

st.set_page_config(
    page_title="GLPI Ticket Classifier",
    page_icon="🎫",
    layout="centered"
)


st.title("🎫 Classification automatique des tickets GLPI")

st.write(
    "Entrez le titre et le contenu du ticket "
    "pour prédire automatiquement sa catégorie."
)


# ============================================================
# FORMULAIRE
# ============================================================

title = st.text_input(
    "Titre du ticket",
    placeholder="Exemple : Problème d'accès au module RH"
)


content = st.text_area(
    "Contenu du ticket",
    placeholder=(
        "Décrivez le problème rencontré..."
    ),
    height=200
)


classify_button = st.button(
    "🔍 Classifier le ticket",
    type="primary",
    use_container_width=True
)


# ============================================================
# PRÉDICTION
# ============================================================

if classify_button:

    if not title.strip() and not content.strip():

        st.warning(
            "Veuillez saisir au moins un titre ou un contenu."
        )

    else:

        with st.spinner("Classification en cours..."):

            predicted_label, confidence, probabilities = (
                predict_category(
                    title,
                    content
                )
            )


        st.success("Classification terminée")


        # ----------------------------------------------------
        # Résultat principal
        # ----------------------------------------------------

        st.subheader("Catégorie prédite")

        st.markdown(
            f"## {predicted_label}"
        )

        st.metric(
            "Confiance",
            f"{confidence * 100:.2f}%"
        )


        # ----------------------------------------------------
        # Probabilités
        # ----------------------------------------------------

        st.subheader("Probabilités par catégorie")

        for label, probability in probabilities.items():

            st.write(
                f"**{label}** — "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                probability
            )
