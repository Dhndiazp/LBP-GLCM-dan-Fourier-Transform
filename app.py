import streamlit as st
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import matplotlib.pyplot as plt

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Analisis Tekstur Citra", layout="wide")

st.write(" By Dhani ")
st.title("🔍 Analisis Tekstur Citra Digital")
st.markdown("""
Aplikasi ini mendeteksi tekstur menggunakan tiga metode populer:
1.  **LBP (Local Binary Pattern)**
2.  **GLCM (Gray Level Co-occurrence Matrix)**
3.  **Fourier Transform**
""")

# --- Upload Gambar ---
uploaded_file = st.sidebar.file_uploader("Upload Citra (JPG/PNG)", type=["jpg", "jpeg", "png"])

def load_image(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

if uploaded_file is not None:
    # 1. Tampilkan Citra Asli
    image = load_image(uploaded_file)
    gray_image = to_grayscale(image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Citra Asli", use_container_width=True)
    with col2:
        st.image(gray_image, caption="Citra Grayscale", use_container_width=True, clamp=True)
        with st.expander("Lihat Matriks Citra Grayscale (Snippet)"):
            st.write(gray_image[:20, :20]) # Menampilkan sebagian kecil agar tidak berat

    st.divider()

    # --- Tabs untuk Metode ---
    tab1, tab2, tab3 = st.tabs(["1. Local Binary Pattern (LBP)", "2. GLCM", "3. Fourier Transform"])

    # === TAB 1: LBP ===
    with tab1:
        st.header("Local Binary Pattern (LBP)")
        st.info("LBP membandingkan setiap piksel dengan tetangganya untuk membentuk pola biner.")

        # Parameter LBP
        radius = st.slider("Radius LBP", 1, 5, 1)
        n_points = 8 * radius
        
        # Kalkulasi LBP
        lbp = local_binary_pattern(gray_image, n_points, radius, method="uniform")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.image(lbp, caption=f"Hasil Visualisasi LBP (R={radius})", clamp=True, use_container_width=True)
        with c2:
            st.subheader("Output Matriks LBP")
            st.write("Matriks ini merepresentasikan kode tekstur untuk setiap piksel.")
            st.dataframe(lbp)
            
            # Histogram LBP
            fig, ax = plt.subplots()
            ax.hist(lbp.ravel(), bins=int(lbp.max() + 1), range=(0, int(lbp.max() + 1)), density=True, color='green', alpha=0.7)
            ax.set_title("Histogram LBP")
            st.pyplot(fig)

    # === TAB 2: GLCM ===
    with tab2:
        st.header("Gray Level Co-occurrence Matrix (GLCM)")
        st.info("GLCM menghitung seberapa sering pasangan piksel dengan nilai tertentu muncul bersebelahan.")

        # Parameter GLCM
        dist = st.slider("Jarak Piksel (Distance)", 1, 10, 1)
        angle_dict = {"0°": 0, "45°": np.pi/4, "90°": np.pi/2, "135°": 3*np.pi/4}
        angle_label = st.selectbox("Sudut (Angle)", list(angle_dict.keys()))
        angle = angle_dict[angle_label]

        # Kalkulasi GLCM
        # Kita binning gambar ke 256 level (standard uint8)
        glcm = graycomatrix(gray_image, distances=[dist], angles=[angle], levels=256, symmetric=True, normed=True)
        
        # Ekstraksi Fitur Tekstur
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]

        st.subheader("Fitur Statistik GLCM")
        metrics_df = {
            "Fitur": ["Contrast", "Dissimilarity", "Homogeneity", "Energy", "Correlation"],
            "Nilai": [contrast, dissimilarity, homogeneity, energy, correlation]
        }
        st.table(metrics_df)

        st.subheader("Output Matriks GLCM (Snippet)")
        st.write(f"Ukuran Matriks Penuh: {glcm.shape[0]}x{glcm.shape[1]}")
        st.write("Menampilkan pojok kiri atas (20x20) dari matriks co-occurrence:")
        
        # GLCM adalah 4D array (levels, levels, distances, angles), kita ambil 2D nya
        glcm_matrix_2d = glcm[:, :, 0, 0]
        st.dataframe(glcm_matrix_2d[:20, :20]) # Menampilkan sebagian

        # Heatmap GLCM
        if st.checkbox("Tampilkan Heatmap GLCM (Mungkin lambat untuk matriks besar)"):
            fig2, ax2 = plt.subplots()
            cax = ax2.matshow(np.log(glcm_matrix_2d + 1e-6), cmap='viridis') # Log scale agar terlihat
            fig2.colorbar(cax)
            ax2.set_title("GLCM Heatmap (Log Scale)")
            st.pyplot(fig2)

    # === TAB 3: FOURIER TRANSFORM ===
    with tab3:
        st.header("Fourier Transform (Frequency Domain)")
        st.info("Mengubah citra dari domain spasial ke domain frekuensi untuk melihat pola periodik.")

        # Kalkulasi FFT
        f = np.fft.fft2(gray_image)
        fshift = np.fft.fftshift(f) # Geser frekuensi nol ke tengah
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6) # Skala Logaritma untuk visualisasi

        c1, c2 = st.columns([1, 2])
        with c1:
            st.image(magnitude_spectrum, caption="Magnitude Spectrum", clamp=True, use_container_width=True)
        
        with c2:
            st.subheader("Output Matriks Spektrum")
            st.write("Nilai Magnitude Spectrum (dalam skala Log):")
            st.dataframe(magnitude_spectrum)
            
            st.write("Data Raw (Kompleks - 5x5 tengah):")
            center_y, center_x = fshift.shape[0]//2, fshift.shape[1]//2
            st.write(fshift[center_y-2:center_y+3, center_x-2:center_x+3])

else:
    st.warning("Silakan upload gambar di sidebar sebelah kiri untuk memulai.")