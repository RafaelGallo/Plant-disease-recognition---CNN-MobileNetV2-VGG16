import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from huggingface_hub import hf_hub_download

# Configuração da página
st.set_page_config(page_title="Classificador de Doenças em Folhas", layout="wide")
st.title("🌿 Classificador de Doenças em Folhas (Rede Neural Convolucional CNN - MobileNetV2)")
st.write("Envie uma imagem de uma folha para classificar como: **Healthy**, **Powdery** ou **Rust**.")

# Carrega modelo .h5 do Hugging Face
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
        img_array = img_array[..., :3]
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# Valida se parece uma folha
def is_valid_leaf(prediction, threshold=0.70):
    return np.max(prediction) >= threshold

# Interface com abas
tab1, tab2 = st.tabs(["📸 Classificador", "📊 Métricas dos Modelos"])

# ================================
# 📸 Aba 1: Classificador
# ================================
with tab1:
    option = st.radio("Escolha o modo de envio da imagem:", ["Upload de imagem", "Usar câmera"])

    uploaded_file = None
    if option == "Upload de imagem":
        uploaded_file = st.file_uploader("📤 Envie uma imagem da folha", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("📸 Tire uma foto da folha")

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

# ================================
# 📊 Aba 2: Métricas dos Modelos
# ================================
with tab2:
    st.subheader("📈 Acurácia dos Modelos Testados")
    
    data = {
        "Modelo": ["DenseNet121", "InceptionV3", "MobileNetV2", "VGG16", "ResNet50", "EfficientNetB0"],
        "Acurácia Validação": [0.983333, 1.0, 0.966667, 0.916667, 0.616667, 0.333333],
        "Acurácia Treino": [0.978064, 0.947806, 0.975038, 0.906959, 0.453858, 0.320726]
    }
    df = pd.DataFrame(data)

    st.dataframe(df, use_container_width=True)

    # Gráfico de comparação
    st.markdown("### 📊 Comparação Gráfica de Acurácias")
    fig, ax = plt.subplots(figsize=(10, 4))
    bar_width = 0.35
    index = np.arange(len(df))

    ax.bar(index, df["Acurácia Validação"], bar_width, label='Validação', color='skyblue')
    ax.bar(index + bar_width, df["Acurácia Treino"], bar_width, label='Treino', color='lightgreen')

    ax.set_xlabel('Modelos')
    ax.set_ylabel('Acurácia')
    ax.set_title('Acurácia de Treino vs Validação')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(df["Modelo"], rotation=45)
    ax.legend()
    st.pyplot(fig)
