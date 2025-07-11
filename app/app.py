import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

# Configuração da página
st.set_page_config(page_title="Classificador de Doenças em Folhas", layout="centered")
st.title("🌿 Classificador de Doenças em Folhas (Rede Neural Convolucional CNN - MobileNetV2)")
st.write("Envie uma imagem de uma folha para classificar como: **Healthy**, **Powdery** ou **Rust**.")

# Carregamento do modelo da Hugging Face
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

# Função de pré-processamento
def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]  # remove canal alfa
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# Função para verificar se parece ser uma folha
def is_valid_leaf(prediction, threshold=0.70):
    return np.max(prediction) >= threshold

# Interface de envio
option = st.radio("📷 Escolha o modo de envio da imagem:", ["Upload de imagem", "Usar câmera"])
uploaded_file = st.file_uploader("📤 Envie uma imagem da folha", type=["jpg", "jpeg", "png"]) if option == "Upload de imagem" else st.camera_input("📸 Tire uma foto da folha")

# Processa e exibe resultado
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

        st.subheader("📌 Detalhes da previsão por classe:")
        for i, class_name in enumerate(class_names):
            st.write(f"- {class_name}: {prediction[i]:.2%}")

        # 🎯 Gráfico de barras
        fig, ax = plt.subplots()
        ax.bar(class_names, prediction, color='green')
        ax.set_ylabel("Confiança")
        ax.set_ylim(0, 1)
        ax.set_title("Distribuição da Confiança")
        st.pyplot(fig)

    else:
        st.error("❌ A imagem enviada **não parece conter uma folha**. Por favor, envie uma imagem clara de uma folha.")

# Bloco com métricas do modelo
with st.expander("📊 Métricas do Modelo (MobileNetV2)", expanded=False):
    st.markdown("""
    | Métrica | Valor |
    |--------|--------|
    | **Épocas Treinadas** | 22 |
    | **Acurácia de Validação Final** | 96.67% |
    | **Melhor Acurácia de Validação** | 98.33% |
    | **Loss de Validação Final** | 0.0948 |
    | **Acurácia de Treinamento Final** | 97.50% |
    | **Loss de Treinamento Final** | 0.0834 |
    """)
    st.info("Essas métricas foram obtidas com MobileNetV2 treinado em três classes: Healthy, Powdery e Rust.")
