import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tempfile
from PIL import Image
from huggingface_hub import hf_hub_download

# Configuração da página
st.set_page_config(page_title="Classificador de Doenças em Folhas", layout="wide")
st.title("🌿 Classificador de Doenças em Folhas com MobileNetV2")

# Carrega modelo da Hugging Face
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

# Preprocessamento da imagem
def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# Verificação de folha válida
def is_valid_leaf(prediction, threshold=0.70):
    return np.max(prediction) >= threshold

# Tabs de navegação
tab1, tab2, tab3 = st.tabs(["📸 Classificador", "📊 Métricas dos Modelos", "🧠 Sobre os Modelos CNN"])

# ===============================
# 📸 Aba 1 - Classificador
# ===============================
with tab1:
    st.subheader("📷 Envie uma imagem de uma folha")

    option = st.radio("Modo de envio:", ["Upload de imagem", "Usar câmera"])
    uploaded_file = st.file_uploader("📤 Upload", type=["jpg", "jpeg", "png"]) if option == "Upload de imagem" else st.camera_input("📸 Câmera")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Imagem carregada", use_container_width=True)

        processed_img = preprocess_image(image)
        prediction = model.predict(processed_img)[0]

        if is_valid_leaf(prediction):
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)

            st.markdown(f"### 🧠 Previsão: `{predicted_class}`")
            st.success(f"📊 Confiabilidade: `{confidence:.2%}`")

            st.subheader("📌 Detalhes da previsão por classe:")
            for i, class_name in enumerate(class_names):
                st.write(f"- {class_name}: `{prediction[i]:.2%}`")

            # OpenCV: Desenhar caixa + texto
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_file.read())
            temp_file.flush()

            cv_img = cv2.imread(temp_file.name)
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            (h, w) = cv_img.shape[:2]
            start_x, start_y = int(w * 0.2), int(h * 0.2)
            end_x, end_y = int(w * 0.8), int(h * 0.8)

            colors = {'Rust': (255, 0, 0), 'Healthy': (0, 255, 0), 'Powdery': (255, 165, 0)}
            color = colors.get(predicted_class, (0, 0, 0))

            cv2.rectangle(cv_img, (start_x, start_y), (end_x, end_y), color, 3)
            cv2.putText(cv_img, predicted_class, (start_x, start_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            st.image(cv_img, caption="🖼️ Detecção com OpenCV", use_container_width=True)

            # Seaborn: gráfico de barras
            st.markdown("### 📊 Distribuição de Confiança")
            fig, ax = plt.subplots()
            sns.barplot(x=class_names, y=prediction, palette=['green', 'orange', 'red'], ax=ax)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probabilidade")
            ax.set_title("Confiança por Classe")
            for i, v in enumerate(prediction):
                ax.text(i, v + 0.01, f"{v:.2%}", ha='center', fontsize=10)
            st.pyplot(fig)

        else:
            st.error("❌ A imagem enviada não parece conter uma folha. Tente novamente com outra imagem.")

# ===============================
# 📊 Aba 2 - Métricas dos Modelos
# ===============================
with tab2:
    st.subheader("📈 Acurácia dos Modelos CNN Testados")

    data = {
        "Modelo": ["DenseNet121", "InceptionV3", "MobileNetV2", "VGG16", "ResNet50", "EfficientNetB0"],
        "Acurácia Validação": [0.983, 1.0, 0.967, 0.917, 0.617, 0.333],
        "Acurácia Treino": [0.978, 0.948, 0.975, 0.907, 0.454, 0.321]
    }
    df = pd.DataFrame(data)

    st.dataframe(df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    bar_width = 0.35
    index = np.arange(len(df))

    ax.bar(index, df["Acurácia Validação"], bar_width, label='Validação', color='skyblue')
    ax.bar(index + bar_width, df["Acurácia Treino"], bar_width, label='Treino', color='lightgreen')

    ax.set_xlabel('Modelos')
    ax.set_ylabel('Acurácia')
    ax.set_title('Comparação de Acurácia - Treino vs Validação')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(df["Modelo"], rotation=45)
    ax.legend()
    st.pyplot(fig)

# ===============================
# 🧠 Aba 3 - Sobre os Modelos CNN
# ===============================
with tab3:
    st.subheader("🧠 O que são os modelos CNN utilizados?")
    st.markdown("""
As **Redes Neurais Convolucionais (CNNs)** são altamente eficazes para o reconhecimento de padrões visuais.

### Modelos utilizados:

- **MobileNetV2**: Leve, rápido, ideal para dispositivos móveis.
- **DenseNet121**: Conexões densas entre camadas. Melhora o fluxo de gradientes.
- **InceptionV3**: Vários filtros em paralelo. Captura padrões multi-escala.
- **VGG16**: Simples, camadas 3x3 empilhadas. Boa base para transfer learning.
- **ResNet50**: Introduz conexões residuais (atalhos). Resolve gradiente desaparecendo.
- **EfficientNetB0**: Otimiza profundidade, largura e resolução. Muito eficiente.

Todos foram treinados para classificar folhas em: `Healthy`, `Powdery` e `Rust`.
""")
