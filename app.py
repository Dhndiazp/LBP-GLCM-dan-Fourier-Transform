import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Analisis Tekstur Citra", layout="wide")

st.title("🔍 Analisis Tekstur Citra Digital")
st.markdown("""
Aplikasi ini mendeteksi tekstur menggunakan tiga metode:
1.  **LBP (Local Binary Pattern)**
2.  **GLCM (Gray Level Co-occurrence Matrix)**
3.  **Fourier Transform (FFT)**
""")

# --- Fungsi Helper ---
def load_image(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# --- Upload Gambar ---
uploaded_file = st.sidebar.file_uploader("Upload Citra (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Tampilkan Citra Asli & Grayscale
    image = load_image(uploaded_file)
    gray_image = to_grayscale(image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Citra Asli", use_container_width=True)
    with col2:
        st.image(gray_image, caption="Citra Grayscale", use_container_width=True)

    st.divider()

    # --- Tabs untuk Metode ---
    tab1, tab2, tab3 = st.tabs(["1. Local Binary Pattern (LBP)", "2. GLCM", "3. Fourier Transform"])

    # ==========================
    # TAB 1: Local Binary Pattern
    # ==========================
    with tab1:
        st.header("Local Binary Pattern (LBP)")
        
        # Parameter
        radius = st.slider("Radius LBP", 1, 5, 1)
        n_points = 8 * radius
        
        # Kalkulasi
        lbp = local_binary_pattern(gray_image, n_points, radius, method="uniform")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            # Normalisasi visualisasi LBP agar kontras terlihat jelas
            lbp_vis = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            st.image(lbp_vis, caption=f"Visualisasi LBP (R={radius})", use_container_width=True)
        
        with c2:
            st.subheader("Output Matriks LBP")
            st.write("Matriks kode tekstur (Pojok Kiri Atas 10x10):")
            st.dataframe(lbp[:10, :10]) # Snippet
            
            # Histogram
            fig, ax = plt.subplots()
            ax.hist(lbp.ravel(), bins=int(lbp.max() + 1), range=(0, int(lbp.max() + 1)), density=True, color='green', alpha=0.7)
            ax.set_title("Histogram Distribusi Pola LBP")
            st.pyplot(fig)

    # ==========================
    # TAB 2: GLCM
    # ==========================
    with tab2:
        st.header("Gray Level Co-occurrence Matrix (GLCM)")
        
        # Parameter
        dist = st.slider("Jarak Piksel (Distance)", 1, 10, 1)
        angle_dict = {"0°": 0, "45°": np.pi/4, "90°": np.pi/2, "135°": 3*np.pi/4}
        angle_label = st.selectbox("Sudut (Angle)", list(angle_dict.keys()))
        angle = angle_dict[angle_label]

        # Kalkulasi
        glcm = graycomatrix(gray_image, distances=[dist], angles=[angle], levels=256, symmetric=True, normed=True)
        
        # Ambil matriks 2D untuk ditampilkan
        glcm_matrix_2d = glcm[:, :, 0, 0]

        # Fitur Statistik
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]

        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.subheader("Statistik Tekstur")
            metrics = {
                "Contrast": contrast,
                "Dissimilarity": dissimilarity,
                "Homogeneity": homogeneity,
                "Energy": energy,
                "Correlation": correlation
            }
            st.table(metrics)

        with c2:
            st.subheader("Matriks GLCM (Snippet)")
            st.write("Pojok Kiri Atas (10x10):")
            st.dataframe(glcm_matrix_2d[:10, :10])
            
            if st.checkbox("Tampilkan Visualisasi Heatmap GLCM"):
                fig2, ax2 = plt.subplots()
                # Gunakan Log agar pola terlihat (karena banyak nilai kecil)
                cax = ax2.matshow(np.log(glcm_matrix_2d + 1e-6), cmap='viridis') 
                fig2.colorbar(cax)
                ax2.set_title("GLCM Heatmap (Log Scale)")
                st.pyplot(fig2)

    # ==========================
    # TAB 3: Fourier Transform
    # ==========================
    with tab3:
        st.header("Fourier Transform (FFT)")
        
        # 1. Kalkulasi FFT
        f = np.fft.fft2(gray_image)
        fshift = np.fft.fftshift(f) # Geser frekuensi rendah ke tengah
        
        # 2. Magnitude Spectrum (Log Scale)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)
        
        # 3. NORMALISASI (PENTING AGAR GAMBAR TIDAK PUTIH POLOS)
        # Mengubah rentang angka float berapapun menjadi 0-255 (integer)
        norm_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
        img_spectrum = np.uint8(norm_spectrum)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.image(img_spectrum, caption="Magnitude Spectrum (Ternormalisasi)", use_container_width=True)
        
        with c2:
            st.subheader("Data Matriks Spektrum")
            st.write("Nilai Frekuensi (Log Scale, Belum dinormalisasi):")
            st.dataframe(magnitude_spectrum)
            
            st.info("Pusat matriks (tengah) mewakili frekuensi rendah, pinggir mewakili frekuensi tinggi.")

else:
    st.info("Silakan upload gambar melalui sidebar di sebelah kiri.")
