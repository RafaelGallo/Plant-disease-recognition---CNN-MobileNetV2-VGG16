import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# Configuração da página
st.set_page_config(page_title="Classificador de Doenças em Folhas", layout="centered")
st.title("🌿 Classificador de Doenças em Folhas (MobileNetV2)")
st.write("Classifique imagens de folhas em: **Healthy**, **Powdery** ou **Rust**.")

# Baixar modelo treinado do Hugging Face (formato .h5)
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Gallorafael2222/plantdiseasecnn",
        filename="models/MobileNetV2_model.h5",
        repo_type="model"
    )
    return tf.keras.models.load_model(model_path)

model = load_model()

# Classes do modelo
class_names = ['Healthy', 'Powdery', 'Rust']

# Função: Verifica se a imagem é uma folha (baseado em pixels verdes)
def is_leaf_image(img, threshold=0.2):
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]  # remove canal alpha
    img_array = img_array / 255.0
    r, g, b = img_array[..., 0], img_array[..., 1], img_array[..., 2]
    green_pixels = (g > r) & (g > b)
    proportion_green = np.sum(green_pixels) / green_pixels.size
    return proportion_green > threshold

# Pré-processamento da imagem
def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# Upload de imagem
uploaded_file = st.file_uploader("📤 Faça upload de uma imagem da folha", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Imagem carregada", use_container_width=True)

    if not is_leaf_image(image):
        st.error("❌ A imagem enviada não parece conter uma folha. Por favor, envie uma imagem clara de uma folha.")
    else:
        processed_img = preprocess_image(image)
        prediction = model.predict(processed_img)[0]
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        st.markdown(f"### 🧠 Previsão: :green[`{predicted_class}`]")
        st.write(f"📊 Confiabilidade: `{confidence:.2%}`")

        st.subheader("📌 Detalhes da previsão:")
        for i, class_name in enumerate(class_names):
            st.write(f"- {class_name}: {prediction[i]:.2%}")
