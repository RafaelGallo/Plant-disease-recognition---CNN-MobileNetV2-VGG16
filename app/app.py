import streamlit as st
import keras
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# Configuração do app
st.set_page_config(page_title="🌿 Classificador de Doenças em Folhas", layout="centered")
st.title("🌿 Classificador de Doenças em Folhas (VGG16)")
st.write("Classifique imagens de folhas em: **Healthy**, **Powdery** ou **Rust**.")

# Carregamento do modelo do Hugging Face com Keras
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Gallorafael2222/plantdiseasecnn",      # Seu repo no HF
        filename="models/VGG16_model.keras",            # Caminho dentro do repo
        repo_type="model"
    )
    return keras.models.load_model(model_path)

model = load_model()

# Labels de saída
class_names = ['Healthy', 'Powdery', 'Rust']

# Preprocessamento da imagem
def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]  # Remove alpha se houver
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Upload da imagem pelo usuário
uploaded_file = st.file_uploader("📤 Faça upload de uma imagem da folha", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Imagem carregada", use_column_width=True)

    # Previsão
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Resultado
    st.markdown(f"### 🧠 Previsão: `{predicted_class}`")
    st.write(f"📊 Confiabilidade: `{confidence:.2%}`")

    # Detalhes por classe
    st.subheader("📌 Probabilidades por classe")
    for i, class_name in enumerate(class_names):
        st.write(f"- {class_name}: {prediction[i]:.2%}")
