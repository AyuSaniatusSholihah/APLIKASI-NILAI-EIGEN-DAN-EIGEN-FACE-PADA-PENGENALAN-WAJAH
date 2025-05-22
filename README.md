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
├── __pycache__/
├── assets/
├── dataset/
├── images/
│   └── sample_result.jpg
├── dataset_loader.py
├── eigen_utils.py
├── face_recognition.py
├── gui.py
├── helpers.py
├── main.py
└── README.md


## 🚀 Cara Menjalankan
1. Clone repository ini:
   ```bash
   git clone https://github.com/AyuSaniatusSholihah/APLIKASI-NILAI-EIGEN-DAN-EIGEN-FACE-PADA-PENGENALAN-WAJAH.git

2. Masuk ke 
    cd AyuSaniatusSholihah
3. Masuk ke
    cd /APLIKASI-NILAI-EIGEN-DAN-EIGEN-FACE-PADA-PENGENALAN-WAJAH/
4. Jalankan main Streamlit: 
    streamlit run main.py

## 👌Dokumentasi Program
## Input (masukkan address folder dataset dan upload gambar)
![dd](img/2.png)

## Jika foto tidak dikenali, tambahkan threshold
![dd](img/3.png)

## Output 
![dd](img/1.png)

## Output setelah menambah threshold menunjukkan foto memiliki kemiripan dengan sesuai jarak euklidean 
![dd](img/4.png)

    