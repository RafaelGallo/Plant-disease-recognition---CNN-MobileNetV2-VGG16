import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2

# === Configuração inicial ===
st.set_page_config(page_title="Classificação de Folhas", layout="centered")
st.title("📷 Classificador de Doenças em Folhas")
st.markdown("Envie uma imagem de **uma folha** para identificar possíveis doenças.")

# === Carregar modelo ===
@st.cache_resource
def load_model_cached():
    return load_model("MobileNetV2_model.h5")

model = load_model_cached()
class_names = ['Healthy', 'Powdery', 'Rust']

# === Captura ou upload ===
img_file = st.camera_input("📸 Tire uma foto da folha")  # Ativa a câmera
if not img_file:
    img_file = st.file_uploader("📁 Ou envie uma imagem da folha", type=["jpg", "png", "jpeg"])

# === Processamento ===
if img_file:
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="📷 Imagem carregada", use_column_width=True)

    # Redimensionar e preparar para o modelo
    img_array = np.array(image.resize((224, 224)))
    img_preprocessed = preprocess_input(img_array.astype(np.float32))
    img_batch = np.expand_dims(img_preprocessed, axis=0)

    # Inferência
    prediction = model.predict(img_batch)[0]
    confidence = float(np.max(prediction)) * 100
    predicted_class = class_names[np.argmax(prediction)]

    # === Lógica para detectar imagem inválida ===
    if confidence < 60:  # Ajustável
        st.error("❌ A imagem enviada não parece conter uma folha. Por favor, envie uma imagem clara de uma folha.")
    else:
        st.markdown(f"🧠 **Previsão:** `{predicted_class}`")
        st.markdown(f"📊 **Confiabilidade:** `{confidence:.2f}%`")

        # Mostrar detalhes
        st.markdown("📌 **Detalhes da previsão:**")
        for cls, score in zip(class_names, prediction):
            st.markdown(f"- {cls}: `{score * 100:.2f}%`")
