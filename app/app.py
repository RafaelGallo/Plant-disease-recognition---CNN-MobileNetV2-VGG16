import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download

# ============== CONFIGURAÇÃO ==============
st.set_page_config(page_title="🌿 Classificador de Doenças em Folhas", layout="centered")
st.title("🌿 Classificador de Doenças em Folhas (VGG16)")
st.caption("Classifique imagens de folhas em: Healthy, Powdery ou Rust")

# ============== CLASSES ====================
CLASS_NAMES = ['Healthy', 'Powdery', 'Rust']

# ============== FUNÇÃO DE DOWNLOAD DO MODELO ==========
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="rafaelgallogg/plant-disease-vgg16",
        filename="VGG16_model.keras",
        repo_type="model"
    )
    return tf.keras.models.load_model(model_path)

model = load_model()

# ============== FUNÇÃO DE PRÉ-PROCESSAMENTO ===========
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))  # VGG16 espera 224x224
    image = np.array(image) / 255.0   # Normaliza
    if image.shape[-1] == 4:  # Remove alpha channel, se houver
        image = image[:, :, :3]
    image = np.expand_dims(image, axis=0)  # Adiciona batch dimension
    return image

# ============== INTERFACE ==================
uploaded_file = st.file_uploader("📤 Envie a imagem da folha...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Imagem carregada", use_column_width=True)

    input_image = preprocess_image(image)
    prediction = model.predict(input_image)[0]
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.markdown("---")
    st.subheader("🔍 Resultado")
    st.write(f"**Classe prevista:** `{predicted_class}`")
    st.write(f"**Confiança:** `{confidence:.2f}%`")
