import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# Configuração da página
st.set_page_config(page_title="Classificador de Doenças em Folhas", layout="centered")
st.title("🌿 Classificador de Doenças em Folhas (MobileNetV2)")
st.write("Envie uma imagem de uma folha para classificar como: **Healthy**, **Powdery** ou **Rust**.")

# Carregamento do modelo a partir do Hugging Face
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Gallorafael2222/plantdiseasecnn",
        filename="models/MobileNetV2_model.h5",
        repo_type="model"
    )
    return tf.keras.models.load_model(model_path)

model = load_model()
class_names = ['Healthy', 'Powdery', 'Rust']

# Pré-processamento da imagem
def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]  # Remove canal alpha se existir
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# Verifica se a imagem parece uma folha válida com base na confiança geral
def is_valid_leaf(prediction, threshold=0.70):
    return np.max(prediction) >= threshold

# Interface de entrada: upload ou câmera
option = st.radio("📷 Escolha o modo de envio da imagem:", ["Upload de imagem", "Usar câmera"])

uploaded_file = None
if option == "Upload de imagem":
    uploaded_file = st.file_uploader("📤 Envie uma imagem da folha", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("📸 Tire uma foto da folha")

# Processamento da imagem
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Imagem carregada", use_container_width=True)

    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)[0]

    if is_valid_leaf(prediction):
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        st.markdown(f"### 🧠 Previsão: `{predicted_class}`")
        st.write(f"📊 Confiabilidade: `{confidence:.2%}`")

        st.subheader("📌 Detalhes da previsão:")
        for i, class_name in enumerate(class_names):
            st.write(f"- {class_name}: {prediction[i]:.2%}")
    else:
        st.error("❌ A imagem enviada **não parece conter uma folha**. Por favor, envie uma imagem clara de uma folha.")
