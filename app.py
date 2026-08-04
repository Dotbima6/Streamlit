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

st.image("assets/logo.png", width=120)

st.title("🍇 Deteksi Penyakit Daun Anggur")

st.markdown("""
Aplikasi ini menggunakan model **MobileNetV2**
untuk mengidentifikasi penyakit pada daun anggur.
""")

# ==========================
# Nama kelas
# ==========================
with open("labels.txt", "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

# ==========================
# Load Model
# ==========================
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("best_model.h5")
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

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

        with st.spinner("Sedang melakukan prediksi..."):
    prediction = model.predict(input_image, verbose=0)

        predicted_class = np.argmax(prediction)

        confidence = np.max(prediction)

        st.success(f"Hasil Prediksi : **{CLASS_NAMES[predicted_class]}**")

        st.info(f"Tingkat Keyakinan : **{confidence*100:.2f}%**")

        st.subheader("Probabilitas")

        st.subheader("Probabilitas Tiap Kelas")

        for i, cls in enumerate(CLASS_NAMES):
            prob = float(prediction[0][i])
            st.write(f"{cls} : {prob*100:.2f}%")
            st.progress(prob)

st.markdown("---")
st.caption(
    "Deteksi Penyakit Daun Anggur menggunakan MobileNetV2 | "
    "Universitas Dian Nuswantoro"
)
