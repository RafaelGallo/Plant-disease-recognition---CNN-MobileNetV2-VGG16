import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from huggingface_hub import hf_hub_download

# Configuração da página
st.set_page_config(page_title="Classificador de Doenças em Folhas", layout="wide")
st.title("🌿 Classificador de Doenças em Folhas com MobileNetV2")

# Carregamento do modelo
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

# Verifica se parece uma folha
def is_valid_leaf(prediction, threshold=0.70):
    return np.max(prediction) >= threshold

# Interface com abas
tab1, tab2, tab3 = st.tabs(["📸 Classificador", "📊 Métricas dos Modelos", "🧠 Sobre os Modelos CNN"])

# =============================
# 📸 Aba 1 - Classificador
# =============================
with tab1:
    st.subheader("Reconhecimento de Doenças: Healthy, Powdery, Rust")
    option = st.radio("Modo de envio da imagem:", ["Upload de imagem", "Usar câmera"])

    uploaded_file = st.file_uploader("📤 Envie uma imagem da folha", type=["jpg", "jpeg", "png"]) if option == "Upload de imagem" else st.camera_input("📸 Tire uma foto da folha")

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

            st.subheader("📌 Confiança para cada classe:")
            df_plot = pd.DataFrame({
                'Classe': class_names,
                'Probabilidade': prediction
            })

            palette = {
                'Healthy': '#2ecc71',
                'Powdery': '#f39c12',
                'Rust': '#e74c3c'
            }

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=df_plot, x='Classe', y='Probabilidade', palette=palette, ax=ax)
            for i, row in df_plot.iterrows():
                ax.text(i, row['Probabilidade'] + 0.02, f"{row['Probabilidade']:.2%}", ha='center', va='bottom', fontsize=12, weight='bold')
            ax.set_ylim(0, 1.1)
            ax.set_title("Distribuição da Confiança", fontsize=14, weight='bold')
            ax.set_ylabel("Probabilidade")
            ax.set_xlabel("")
            st.pyplot(fig)

        else:
            st.error("❌ A imagem enviada **não parece conter uma folha**. Por favor, envie uma imagem clara de uma folha.")

# =============================
# 📊 Aba 2 - Métricas dos Modelos
# =============================
with tab2:
    st.subheader("📈 Acurácia dos Modelos Testados")

    data = {
        "Modelo": ["DenseNet121", "InceptionV3", "MobileNetV2", "VGG16", "ResNet50", "EfficientNetB0"],
        "Acurácia Validação": [0.983, 1.0, 0.967, 0.917, 0.617, 0.333],
        "Acurácia Treino": [0.978, 0.948, 0.975, 0.907, 0.454, 0.321]
    }
    df = pd.DataFrame(data)

    st.dataframe(df, use_container_width=True)

    st.markdown("### 📊 Comparativo Gráfico de Acurácias")
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

# =============================
# 🧠 Aba 3 - Sobre os Modelos
# =============================
with tab3:
    st.subheader("🧠 O que são os modelos CNN utilizados?")
    st.markdown("""
As **Redes Neurais Convolucionais (CNNs)** são arquiteturas de deep learning eficazes para o reconhecimento de padrões em imagens.

**Modelos utilizados:**

- **MobileNetV2**: Leve e eficiente, ideal para dispositivos móveis.
- **DenseNet121**: Cada camada conecta-se a todas as anteriores.
- **InceptionV3**: Usa múltiplos tamanhos de filtros simultaneamente.
- **VGG16**: Camadas simples de 3x3, muito utilizada para transfer learning.
- **ResNet50**: Usa conexões residuais para evitar perda de gradiente.
- **EfficientNetB0**: Otimiza profundidade, largura e resolução de forma balanceada.

Todos foram treinados para detectar **Healthy**, **Powdery** e **Rust**.
    """)
