# Streamlit
Grape Leaf Disease Detection using MobileNetV2

# 🍇 Grape Leaf Disease Detection

Aplikasi berbasis **Streamlit** untuk mendeteksi penyakit pada daun anggur menggunakan model **MobileNetV2** dengan visualisasi interpretabilitas **Grad-CAM** dan **LIME**.

---

## 📖 Deskripsi

Aplikasi ini dikembangkan sebagai implementasi penelitian:

> **Deteksi Penyakit Daun Anggur Menggunakan MobileNetV2 dengan Interpretabilitas LIME dan Grad-CAM**

Model mampu mengklasifikasikan gambar daun anggur ke dalam empat kategori penyakit.

---

## 📂 Dataset

Dataset terdiri dari empat kelas:

- Black Measles
- Black Rot
- Healthy
- Leaf Blight

---

## 🧠 Model

- MobileNetV2
- Transfer Learning
- TensorFlow / Keras

Input gambar:

```
224 × 224 pixel
```

---

## 📊 Fitur

- Upload gambar daun anggur
- Prediksi penyakit
- Confidence Score
- Visualisasi Grad-CAM *(opsional)*
- Visualisasi LIME *(opsional)*

---

## 🚀 Menjalankan Secara Lokal

Clone repository

```bash
git clone https://github.com/username/grape-disease-app.git
```

Masuk ke folder

```bash
cd grape-disease-app
```

Install library

```bash
pip install -r requirements.txt
```

Jalankan aplikasi

```bash
streamlit run app.py
```

---

## ☁️ Deploy

Aplikasi dapat di-deploy menggunakan

- Streamlit Community Cloud

---

## 📁 Struktur Project

```
grape-disease-app/
│
├── app.py
├── best_model.h5
├── requirements.txt
├── README.md
├── labels.txt
├── assets/
└── images/
```

---

## 👨‍💻 Author

Bima Nur Abdillah

Universitas Dian Nuswantoro

Teknik Informatika
