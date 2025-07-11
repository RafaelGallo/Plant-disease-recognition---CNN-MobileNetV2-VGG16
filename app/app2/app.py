import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tempfile
import os
from PIL import Image
from io import BytesIO
from huggingface_hub import hf_hub_download 
from ultralytics import YOLO

# Configuração da página
st.set_page_config(page_title="Classificador de Doenças em Folhas", layout="wide")
st.title("🌿 Rede Neural Convolucional  MobileNetV2 + YOLO  - Classificador de Doenças em Folhas")

# Carrega os modelos CNN e YOLOv8
@st.cache_resource
def load_models():
    cnn_model_path = hf_hub_download(
        repo_id="Gallorafael2222/plantdiseasecnn",
        filename="models/MobileNetV2_model.h5",
        repo_type="model"
    )
    cnn_model = tf.keras.models.load_model(cnn_model_path)
    yolo_model = YOLO("yolov8n.pt")
    return cnn_model, yolo_model

cnn_model, yolo_model = load_models()
class_names = ['Healthy', 'Powdery', 'Rust']

# Pré-processamento da imagem para CNN
def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

def is_valid_leaf(prediction, threshold=0.70):
    return np.max(prediction) >= threshold

# Tabs
tab1, tab2, tab3 = st.tabs(["📸 Classificador", "📊 Métricas dos Modelos", "🧠 Sobre os Modelos CNN"])

# ===============================
# 📸 Aba 1 - Classificador Visual
# ===============================
with tab1:
    st.subheader("📷 Envie uma imagem de uma folha")

    option = st.radio("Modo de envio:", ["Upload de imagem", "Usar câmera"])
    uploaded_file = st.file_uploader("📤 Upload", type=["jpg", "jpeg", "png"]) if option == "Upload de imagem" else st.camera_input("📸 Câmera")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Imagem original", use_container_width=True)

        file_bytes = uploaded_file.getvalue()
        temp_path = os.path.join(tempfile.gettempdir(), "uploaded_image.jpg")
        image.save(temp_path)

        results = yolo_model(temp_path)
        img_cv = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        detectou = False
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped = img_cv[y1:y2, x1:x2]
                input_img = preprocess_image(Image.fromarray(cropped))

                prediction = cnn_model.predict(input_img)[0]
                if is_valid_leaf(prediction):
                    detectou = True
                    label = class_names[np.argmax(prediction)]
                    confidence = np.max(prediction)

                    color_map = {'Rust': (255, 0, 0), 'Healthy': (0, 255, 0), 'Powdery': (255, 165, 0)}
                    color = color_map.get(label, (0, 0, 0))
                    label_text = f"{label} ({confidence:.2%})"

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.9
                    thickness = 2
                    margin = 8
                    (w, h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

                    # fundo escuro
                    cv2.rectangle(img_cv, (x1, y1 - h - 2 * margin), (x1 + w + 2 * margin, y1), (0, 0, 0), -1)
                    # texto com sombra branca
                    cv2.putText(img_cv, label_text, (x1 + margin, y1 - margin), font, font_scale, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                    # texto principal
                    cv2.putText(img_cv, label_text, (x1 + margin, y1 - margin), font, font_scale, color, thickness + 1, lineType=cv2.LINE_AA)

                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 3)

                    st.image(img_cv, caption="🖼️ Detecção YOLO + Classificação CNN", width=900)
                    st.markdown(f"### ✅ Previsão: `{label}`")
                    st.success(f"📊 Confiabilidade: `{confidence:.2%}`")

                    st.markdown("### 📊 Distribuição de Confiança")
                    fig, ax = plt.subplots()
                    sns.barplot(x=class_names, y=prediction, palette='viridis', ax=ax)
                    ax.set_ylim(0, 1)
                    for i, v in enumerate(prediction):
                        ax.text(i, v + 0.01, f"{v:.2%}", ha='center', fontsize=10)
                    st.pyplot(fig)

        # 🔁 Fallback automático
        if not detectou:
            st.warning("⚠️ Nenhuma folha detectada pela YOLO. Usando classificação direta pela CNN.")
            input_img = preprocess_image(image)
            prediction = cnn_model.predict(input_img)[0]
            label = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)

            st.markdown(f"### ✅ Previsão (sem detecção): `{label}`")
            st.success(f"📊 Confiabilidade: `{confidence:.2%}`")

            fig, ax = plt.subplots()
            sns.barplot(x=class_names, y=prediction, palette='viridis', ax=ax)
            ax.set_ylim(0, 1)
            for i, v in enumerate(prediction):
                ax.text(i, v + 0.01, f"{v:.2%}", ha='center', fontsize=10)
            st.pyplot(fig)

            st.image(image, caption="🖼️ Classificação direta pela CNN (imagem completa)", use_container_width=True)

# ===========================
# 📊 Aba 2 - Métricas CNN
# ===========================
with tab2:
    st.subheader("📈 Acurácia dos Modelos CNN")

    data = {
        "Modelo": ["DenseNet121", "InceptionV3", "MobileNetV2", "VGG16", "ResNet50", "EfficientNetB0"],
        "Validação": [0.983, 1.0, 0.967, 0.917, 0.617, 0.333],
        "Treino": [0.978, 0.948, 0.975, 0.907, 0.454, 0.321]
    }
    df = pd.DataFrame(data)

    st.dataframe(df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    index = np.arange(len(df))
    bar_width = 0.35

    ax.bar(index, df["Validação"], bar_width, label="Validação", color="skyblue")
    ax.bar(index + bar_width, df["Treino"], bar_width, label="Treino", color="lightgreen")

    ax.set_xlabel("Modelos")
    ax.set_ylabel("Acurácia")
    ax.set_title("Acurácia dos Modelos CNN")
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(df["Modelo"], rotation=45)
    ax.legend()
    st.pyplot(fig)

# ===========================
# 🧠 Aba 3 - Explicação CNN
# ===========================
with tab3:
    st.subheader("📚 O que são os Modelos CNN?")
    st.markdown("""
As **Redes Neurais Convolucionais (CNNs)** são muito eficazes para processamento de imagens.

### Modelos utilizados:

- **MobileNetV2**: Leve, eficiente e ideal para dispositivos móveis.
- **DenseNet121**: Conecta cada camada a todas as anteriores.
- **InceptionV3**: Arquitetura que combina filtros de diferentes tamanhos.
- **VGG16**: Muito usado para transfer learning.
- **ResNet50**: Introduz conexões residuais para redes profundas.
- **EfficientNetB0**: Alta acurácia com menos parâmetros.

Todos os modelos foram treinados para classificar folhas como: **Healthy**, **Powdery** ou **Rust**.
    """)
