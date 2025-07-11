import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests

# Configuração da página
st.set_page_config(page_title="Classificador de Doenças em Folhas", layout="centered")
st.title("🌿 Classificador de Doenças em Folhas (VGG16)")
st.write("Classifique imagens de folhas em: **Healthy**, **Powdery** ou **Rust**.")

# URL do modelo no Hugging Face (corrigido)
MODEL_URL = "https://huggingface.co/Gallorafael2222/plantdiseasecnn/resolve/main/models/VGG16_model.keras"
MODEL_PATH = "VGG16_model.keras"

# Download e cache do modelo
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🔄 Baixando modelo..."):
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, 'wb') as f:
                f.write(r.content)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Função de preprocessamento
def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]  # remove alpha se houver
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Labels
class_names = ['Healthy', 'Powdery', 'Rust']

# Upload da imagem
uploaded_file = st.file_uploader("📤 Faça upload de uma imagem da folha", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Imagem carregada", use_column_width=True)

    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.markdown(f"### 🧠 Previsão: `{predicted_class}`")
    st.write(f"📊 Confiabilidade: `{confidence:.2%}`")

    st.subheader("📌 Detalhes da previsão:")
    for i, class_name in enumerate(class_names):
        st.write(f"- {class_name}: {prediction[i]:.2%}")
