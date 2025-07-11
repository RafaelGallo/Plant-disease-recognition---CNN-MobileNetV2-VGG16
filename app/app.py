import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download

# ------------------------------
# Configurações da Página
# ------------------------------
st.set_page_config(page_title="🌿 Classificador de Doenças em Folhas", layout="centered")
st.title("🌿 Classificador de Doenças em Folhas (VGG16)")
st.write("Classifique imagens de folhas em: **Healthy**, **Powdery** ou **Rust**.")

# ------------------------------
# Caminho do Modelo
# ------------------------------
MODEL_REPO = "rafaelgallods/vgg16-plant-disease"
MODEL_FILENAME = "vgg16_plant_disease_tf14.keras"
MODEL_PATH = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)

# ------------------------------
# Função para Carregar o Modelo
# ------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ------------------------------
# Função de Preprocessamento
# ------------------------------
def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ------------------------------
# Upload da Imagem
# ------------------------------
uploaded_file = st.file_uploader("\U0001F4F7 Envie uma imagem de folha", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagem enviada", use_column_width=True)

    # Preprocessamento
    input_image = preprocess_image(image)

    # Predição
    pred = model.predict(input_image)
    class_names = ['Healthy', 'Powdery', 'Rust']
    predicted_class = class_names[np.argmax(pred)]

    st.success(f"✅ Previsão: **{predicted_class}**")
