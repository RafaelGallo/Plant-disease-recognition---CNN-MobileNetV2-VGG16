import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Configuração da página do Streamlit
st.set_page_config(page_title="🌿 Classificador de Doenças em Folhas", layout="centered")

# Título e descrição
st.title("🌱 Classificador de Folhas de Plantas (VGG16)")
st.write("Este aplicativo classifica uma imagem de folha como: **Healthy**, **Powdery** ou **Rust**.")

# Caminho do modelo treinado
MODEL_PATH = r"C:\Users\rafae.RAFAEL_NOTEBOOK\Downloads\github_projetos_deeplearning_CNN\Plant-disease-recognition---CNN-MobileNetV2-VGG16\models\VGG16_model.keras"

# Carregamento do modelo com cache
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# Lista de classes
class_names = ['Healthy', 'Powdery', 'Rust']

# Preprocessamento da imagem
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    img_array = np.array(image)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]  # Remove canal alfa se existir
    img_array = img_array / 255.0  # Normaliza
    img_array = np.expand_dims(img_array, axis=0)  # Adiciona batch dimension
    return img_array

# Upload da imagem
uploaded_file = st.file_uploader("📤 Faça upload de uma imagem de folha", type=["jpg", "jpeg", "png"])

# Previsão
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Imagem carregada", use_column_width=True)

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0]

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[predicted_index]

    # Resultado
    st.markdown(f"### 🧠 Previsão: `{predicted_class}`")
    st.markdown(f"📊 Confiabilidade: `{confidence:.2%}`")

    # Probabilidades
    st.subheader("🔍 Probabilidades por Classe")
    for i, class_name in enumerate(class_names):
        st.write(f"- {class_name}: {prediction[i]:.2%}")
