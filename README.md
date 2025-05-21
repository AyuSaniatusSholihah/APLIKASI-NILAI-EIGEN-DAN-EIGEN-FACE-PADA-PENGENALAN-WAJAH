# APLIKASI-NILAI-EIGEN-DAN-EIGEN-FACE-PADA-PENGENALAN-WAJAH
PROJECT BASED LEARNING 1 ALJABAR LINEAR


Aplikasi pengenalan wajah berbasis Python yang menggunakan metode **Euclidean Distance** dan **threshold adaptif** untuk menemukan kemiripan wajah dalam dataset.

## 📌 Konsep Utama

- Awal dimulai dari threshold = 500
- Sistem membandingkan wajah input dengan data latih (dataset)
- Jika tidak ditemukan wajah dengan jarak di bawah threshold, maka threshold dinaikkan secara bertahap (misal, +500)
- Proses berulang hingga wajah yang paling mirip ditemukan, atau hingga threshold maksimum tercapai

## 🧠 Teknologi yang Digunakan

- Python
- OpenCV
- NumPy
- scikit-learn (PCA opsional)
- Streamlit (untuk antarmuka pengguna)

## 🗂️ Struktur Folder
├── app.py # Program utama (Streamlit)
├── config.py # Konfigurasi nilai threshold, step, dan maksimum
├── utils/
│ └── distance.py # Fungsi Euclidean distance
├── dataset/
│ ├── Orang1/
│ │ └── img1.jpg
│ └── Orang2/
│ └── img2.jpg
├── screenshots/
│ └── hasil_deteksi.png
├── requirements.txt # Daftar dependensi Python
└── README.md # Dokumentasi proyek

## 🚀 Cara Menjalankan
1. Clone repository ini:
   ```bash
   git clone https://github.com/AyuSaniatusSholihah/APLIKASI-NILAI-EIGEN-DAN-EIGEN-FACE-PADA-PENGENALAN-WAJAH.git

2. Masuk ke 
    cd AyuSaniatusSholihah
3. Masuk ke
    cd /APLIKASI-NILAI-EIGEN-DAN-EIGEN-FACE-PADA-PENGENALAN-WAJAH/
4. Jalankan aplikasi Streamlit: 
    streamlit run app.py
