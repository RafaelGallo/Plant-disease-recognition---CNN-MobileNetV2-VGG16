import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# Configuração da página
st.set_page_config(page_title="Classificador de Doenças em Folhas", layout="centered")
st.title("🌿 Classificador de Doenças em Folhas (MobileNetV2)")
st.write("Classifique imagens de folhas em: **Healthy**, **Powdery** ou **Rust**.")

# Carregar modelo do Hugging Face
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

# Upload ou câmera
upload_option = st.radio("📷 Como deseja enviar a imagem?", ("Upload de Arquivo", "Câmera"))

image = None
if upload_option == "Upload de Arquivo":
    uploaded_file = st.file_uploader("📤 Envie uma imagem da folha", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
elif upload_option == "Câmera":
    camera_image = st.camera_input("📸 Tire uma foto com a câmera")
    if camera_image:
        image = Image.open(camera_image).convert("RGB")

# Previsão
if image:
    st.image(image, caption="📷 Imagem carregada", use_column_width=True)

    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Lógica de verificação adicional (caso necessário)
    if confidence < 0.60:
        st.error("❌ A imagem enviada não parece conter uma folha. Por favor, envie uma imagem clara de uma folha.")
    else:
        st.markdown(f"### 🧠 Previsão: :green[{predicted_class}]")
        st.markdown(f"📊 **Confiabilidade:** `{confidence:.2%}`")
        st.subheader("📌 Detalhes da previsão:")
        for i, class_name in enumerate(class_names):
            st.write(f"- {class_name}: {prediction[i]:.2%}")
