import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ==========================
# Konfigurasi Halaman
# ==========================
st.set_page_config(
    page_title="Deteksi Penyakit Daun Anggur",
    page_icon="🍇",
    layout="centered"
)

st.title("🍇 Deteksi Penyakit Daun Anggur")
st.write("Upload gambar daun anggur untuk melakukan prediksi penyakit menggunakan MobileNetV2.")

# ==========================
# Nama kelas
# ==========================
CLASS_NAMES = [
    "Black Measles",
    "Black Rot",
    "Healthy",
    "Leaf Blight"
]

# ==========================
# Load Model
# ==========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model.h5")
    return model

model = load_model()

# ==========================
# Preprocessing
# ==========================
IMG_SIZE = (224, 224)

def preprocess_image(image):

    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    img = np.array(image)

    img = img.astype(np.float32) / 255.0

    img = np.expand_dims(img, axis=0)

    return img

# ==========================
# Upload Gambar
# ==========================
uploaded_file = st.file_uploader(
    "Upload Gambar",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    if st.button("Prediksi"):

        input_image = preprocess_image(image)

        prediction = model.predict(input_image)

        predicted_class = np.argmax(prediction)

        confidence = np.max(prediction)

        st.success(f"Hasil Prediksi : **{CLASS_NAMES[predicted_class]}**")

        st.info(f"Tingkat Keyakinan : **{confidence*100:.2f}%**")

        st.subheader("Probabilitas")

        for i, cls in enumerate(CLASS_NAMES):
            st.write(f"{cls} : {prediction[0][i]*100:.2f}%")
