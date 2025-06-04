import streamlit as st
import os
from PIL import Image
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score

from dataset_loader import load_dataset
from helpers import read_upimg
from face_recognition import recogface
from eigen_utils import meancenter, h_eigenvekt

def evaluate_model(dataset_images, labels, filenames, threshold=5000):
    true_labels = []
    pred_labels = []

    labeluniq = sorted(set(labels))

    centered_data, mean_face = meancenter(dataset_images)
    eigenfaces, _ = h_eigenvekt(centered_data)

    proyeksi = np.dot(centered_data, eigenfaces)

    n = len(dataset_images)

    for i in range(n):
        tes_proyeksi = proyeksi[i]
        tes_label = labels[i]

        temp_proyeksi = np.delete(proyeksi, i, axis=0)
        temp_label = np.delete(labels, i)

        distances = np.linalg.norm(temp_proyeksi - tes_proyeksi, axis=1)
        min_index = np.argmin(distances)
        min_distance = distances[min_index]

        pred_label = temp_label[min_index] if min_distance < threshold else "Tidak Dikenal"

        true_labels.append(tes_label)
        pred_labels.append(pred_label)
    return true_labels, pred_labels, labeluniq

def start():
    st.set_page_config(page_title="Face Recognition with Eigenfaces", layout="centered")
    st.markdown("""
        <style>
            .main {
                background-color: transparent;
            }
            h1, h4 {
                text-align: center;
            }
            .stMarkdown, .stText, .stImage {
                margin: auto;
            }
            .stImage img {
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.2);
            }
            .result-box {
                background-color: rgba(0,0,0,0.05);
                padding: 1rem;
                border-radius: 8px;
                margin-top: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("## Aplikasi Pengenalan Wajah")
    st.markdown("### Menggunakan PCA dan Eigenfaces")

    st.write("---")

    st.sidebar.header("Pengaturan Input")
    dataset_path = st.sidebar.text_input("Masukkan path ke folder dataset:")
    uploaded_file = st.sidebar.file_uploader("Upload Gambar untuk Dikenali", type=["jpg", "png", "jpeg"])

    threshold = st.sidebar.slider("Ambang Batas Threshold (Euclidean Distance)", 1000, 10000, 5000, step=100)

    dataset_valid = False
    if dataset_path:
        if os.path.isdir(dataset_path):
            st.sidebar.success("Folder dataset ditemukan.")
            dataset_valid = True
        else:
            st.sidebar.error("Folder dataset tidak ditemukan.")

    img_up = None
    img_vector = None
    if uploaded_file:
        img_up, img_vector = read_upimg(uploaded_file)
        if img_up:
            st.image(img_up, caption='Gambar Input', use_container_width=True)
        else:
            st.error("Gagal memproses gambar yang diupload.")

    if st.sidebar.button("Mulai Pengenalan Wajah"):
        if not dataset_valid or not uploaded_file:
            st.warning("Mohon isi semua input terlebih dahulu.")
        else:
            st.info("Sedang memproses gambar dan dataset...")

            dataset_images, labels, filenames = load_dataset(dataset_path)
            if dataset_images.size == 0:
                st.error("Dataset tidak valid atau kosong.")
                return

            filename_match, label_match, distance, eigenfaces, mean_face = recogface(
                img_vector, dataset_images, labels, filenames, threshold=threshold, return_pca=True
            )

            st.write("---")
            st.subheader("Hasil Pengenalan Wajah")

            if filename_match:
                matched_image_path = ""
                for root, _, files in os.walk(dataset_path):
                    if filename_match in files:
                        matched_image_path = os.path.join(root, filename_match)
                        break

                col1, col2 = st.columns([1, 2])

                with col1:
                    try:
                        matched_image = Image.open(matched_image_path)
                        st.image(matched_image, caption="Gambar Paling Mirip", use_container_width=True)
                    except Exception as e:
                        st.error(f"Gagal menampilkan gambar hasil: {e}")

                with col2:
                    st.markdown(f"""
                        <div class="result-box">
                            <p><b>Nama Berkas:</b> {filename_match}</p>
                            <p><b>Label Folder:</b> {label_match}</p>
                            <p><b>Jarak Euclidean:</b> {distance:.2f}</p>
                            <p><b>Keterangan:</b> Gambar paling mirip ditemukan pada dataset.</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.success("Pengenalan wajah berhasil dilakukan!")

            else:
                st.warning(f"Tidak ditemukan kecocokan wajah. Jarak: {distance:.2f} melebihi threshold ({threshold}).")

            st.write("---")
            st.subheader("Visualisasi Komponen PCA (Eigenfaces)")

            jumlahTampil = 5
            eigenf_img = eigenfaces.T[:jumlahTampil]

            cols = st.columns(jumlahTampil)
            for i in range(jumlahTampil):
                face = eigenf_img[i].reshape(img_up.size[::-1])
                face_min, face_max = np.min(face), np.max(face)
                norm_face = ((face - face_min) / (face_max - face_min) * 255).astype(np.uint8)
                img = Image.fromarray(norm_face)
                cols[i].image(img, caption=f"Eigenface #{i+1}", use_container_width=True)

            st.write("---")
            st.subheader("Grafik Nilai Eigen dan Vektor Eigen")

            #st.markdown("**Visualisasi Nilai Eigen**")

            eigen_norms = np.linalg.norm(eigenfaces, axis=0)
            x_vals = list(range(1, len(eigen_norms) + 1))

            fig_val = go.Figure()
            fig_val.add_trace(go.Scatter(
                x=x_vals,
                y=eigen_norms,
                mode='lines+markers',
                name='Norm Eigenface',
                line=dict(color='royalblue', width=3),
                marker=dict(size=8, color='royalblue', line=dict(width=1, color='white')),
                hovertemplate='Komponen ke-%{x}<br>Norm: %{y:.2f}'
            ))
            fig_val.add_trace(go.Scatter(
                x=x_vals,
                y=[threshold] * len(eigen_norms),
                mode='lines',
                name=f'Threshold ({threshold})',
                line=dict(dash='dash', color='red'),
                hoverinfo='skip'
            ))
            fig_val.update_layout(
                title='Distribusi Norm Vektor Eigenface',
                xaxis_title='Komponen Eigen ke-',
                yaxis_title='Norm Vektor',
                template='plotly_white',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                margin=dict(l=40, r=40, t=60, b=40),
                height=450
            )
            st.plotly_chart(fig_val, use_container_width=True)
            st.markdown("""
            <div style="text-align: justify; font-size: 0.9rem;">
                <b>Penjelasan:</b> Grafik di atas menunjukkan <i>norm</i> dari masing-masing komponen eigenface yang diperoleh dari hasil PCA.
                Setiap titik merepresentasikan kekuatan atau kontribusi relatif dari suatu komponen eigen terhadap representasi wajah.
                Komponen dengan norm lebih tinggi umumnya membawa informasi yang lebih penting, sedangkan komponen dengan norm rendah dapat dianggap kurang signifikan.
                Threshold (garis putus-putus merah) digunakan untuk menentukan apakah hasil proyeksi wajah yang diuji cukup dekat (mirip) dengan wajah dalam dataset.
            </div>
            """, unsafe_allow_html=True)

            #st.markdown("** Visualisasi Vektor Eigen Pertama (1D)**")

            fig_vec = go.Figure()
            fig_vec.add_trace(go.Scatter(
                y=eigenfaces[:, 0],
                mode='lines',
                line=dict(color='orange', width=2),
                name='Vektor Eigen #1',
                hovertemplate='Index: %{x}<br>Nilai: %{y:.2f}'
            ))
            fig_vec.update_layout(
                title='Plot Vektor Eigenface Pertama',
                xaxis_title='Index Piksel',
                yaxis_title='Nilai',
                template='plotly_white',
                height=400,
                margin=dict(l=40, r=40, t=50, b=30)
            )
            st.plotly_chart(fig_vec, use_container_width=True)

            st.write("---")
            st.markdown("""
            <div style="text-align: justify; font-size: 0.9rem;">
                <b>Penjelasan:</b> Vektor eigen pertama menggambarkan pola variasi wajah paling dominan dalam dataset.
                Grafik ini memperlihatkan distribusi nilai dari komponen pertama yang dihasilkan PCA untuk setiap piksel pada gambar wajah.
                Nilai-nilai ini mencerminkan seberapa besar pengaruh piksel tersebut dalam membentuk representasi wajah utama.
                Pola ini bersifat numerik dan tidak langsung menyerupai wajah, namun sangat penting dalam proses pengenalan.
            </div>
            """, unsafe_allow_html=True)



    if st.sidebar.button("Evaluasi Akurasi Model"):
        if not dataset_valid:
            st.warning("Mohon masukkan folder dataset terlebih dahulu.")
        else:
            st.info("Melakukan evaluasi akurasi...")

            dataset_images, labels, filenames = load_dataset(dataset_path)
            true_labels, pred_labels, class_labels = evaluate_model(dataset_images, labels, filenames, threshold)

            acc = accuracy_score(true_labels, pred_labels)
            cm = confusion_matrix(true_labels, pred_labels, labels=class_labels)

            st.success(f"Akurasi Model: {acc * 100:.2f}%")

            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=class_labels,
                y=class_labels,
                colorscale='Blues',
                hovertemplate='Prediksi: %{x}<br>Asli: %{y}<br>Jumlah: %{z}<extra></extra>',
                zmin=0
            ))
            fig_cm.update_layout(
                title='Confusion Matrix',
                xaxis_title='Predicted Label',
                yaxis_title='True Label',
                template='plotly_white',
                height=450
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
            st.write("---")
            st.subheader("Penjelasan")

            st.markdown("""
            Evaluasi ini dilakukan dengan metode **Leave-One-Out Cross Validation**, yang artinya:

            Setiap gambar dalam dataset diuji satu per satu.
            
            Saat satu gambar diuji, gambar tersebut sementara **dikeluarkan dari data latih**.
            
            Model mencoba mengenali gambar tersebut tanpa melihatnya sebelumnya.

            ### 🔢 Akurasi
            Akurasi dihitung sebagai:

            **(Jumlah Prediksi Benar / Total Gambar) × 100%**

            Contoh:
            Jika dari 100 gambar, 85 dikenali dengan benar, maka akurasi = **85%**.

            ### 📊 Confusion Matrix
            Matriks ini menunjukkan **jumlah prediksi untuk tiap label**:

            **Baris = Label Asli (Ground Truth)**
            
            **Kolom = Label Prediksi dari Model**
            
            **Diagonal** menunjukkan jumlah yang dikenali dengan benar.
            
            **Di luar diagonal** menunjukkan kesalahan (misalnya wajah “Ayu” dikenali sebagai “Fadhil”)
            .
            “Tidak Dikenal” menunjukkan bahwa gambar tidak cocok dengan siapa pun karena **jarak terlalu jauh (melebihi threshold).**

            Semakin besar nilai di diagonal utama, semakin baik model pengenalan wajahmu.

            """)


    st.write("---")
    st.markdown("""
        <div style='text-align: center; font-size: small; color: gray; padding-top: 20px;'>
            Kelompok 3 - Aplikasi Pengenalan Wajah dengan Eigenface<br>
            Anggota: Ayu Saniatus Sholihah (L0124005), Fadhil Rusadi (L0124013), Muhamad Nabil Fannani (L0124135)
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    start()
