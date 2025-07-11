import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(page_title="Classificador de Doenças em Folhas", layout="centered")

# Título
st.title("🌿 Classificador de Doenças em Folhas (MobileNetV2)")
st.write("Envie uma imagem de uma folha para classificar como: **Healthy**, **Powdery** ou **Rust**.")

# ============
# Modelo CNN
# ============
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

# ============
# Funções
# ============
def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

def is_valid_leaf(prediction, threshold=0.70):
    return np.max(prediction) >= threshold

# ============
# Tabs
# ============
aba1, aba2, aba3 = st.tabs(["📸 Classificação", "📊 Métricas do Modelo", "🧠 Modelos Utilizados"])

# ============
# ABA 1 - Classificação
# ============
with aba1:
    st.subheader("Envie uma imagem de folha")

    # Interface de envio
    option = st.radio("Escolha o modo de envio:", ["Upload de imagem", "Usar câmera"])
    uploaded_file = st.file_uploader("📤 Upload", type=["jpg", "jpeg", "png"]) if option == "Upload de imagem" else st.camera_input("📸 Tire uma foto")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imagem carregada", use_container_width=True)

        processed_img = preprocess_image(image)
        prediction = model.predict(processed_img)[0]

        if is_valid_leaf(prediction):
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)

            st.success(f"🧠 Previsão: **{predicted_class}**")
            st.write(f"📊 Confiabilidade: `{confidence:.2%}`")

            # Detalhamento
            st.subheader("Detalhes por classe:")
            for i, class_name in enumerate(class_names):
                st.write(f"- {class_name}: {prediction[i]:.2%}")
        else:
            st.error("❌ A imagem enviada **não parece conter uma folha**. Por favor, envie uma imagem clara.")

# ============
# ABA 2 - Métricas
# ============
with aba2:
    st.header("📊 Métricas do Modelo MobileNetV2")

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

    # Gráfico comparativo
    modelos = ['MobileNetV2', 'VGG16', 'CNN Simples']
    acuracias = [98.33, 96.25, 92.80]
    perdas = [0.0948, 0.125, 0.158]

    fig, ax = plt.subplots(figsize=(8, 5))
    largura = 0.35
    x = np.arange(len(modelos))

    ax.bar(x - largura/2, acuracias, largura, label='Acurácia (%)', color='green')
    ax.bar(x + largura/2, perdas, largura, label='Loss', color='red')

    ax.set_ylabel('Valor (%)')
    ax.set_title('Comparativo de Acurácia e Loss')
    ax.set_xticks(x)
    ax.set_xticklabels(modelos)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)

    st.pyplot(fig)

# ============
# ABA 3 - Descrição dos Modelos
# ============
with aba3:
    st.header("📚 Modelos CNN Utilizados")

    st.markdown("""
    ### 1. MobileNetV2
    - ✅ Rápido e leve
    - ✅ Excelente para dispositivos móveis
    - ✅ Acurácia: **98.33%**

    ### 2. VGG16
    - 🧱 Rede mais profunda
    - 🔎 Requer mais memória
    - Acurácia: **96.25%**

    ### 3. CNN Simples
    - 🔧 Modelo customizado leve
    - ✅ Fácil de treinar e interpretar
    - Acurácia: **92.80%**
    """)
