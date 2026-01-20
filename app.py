import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Analisis Tekstur Citra", layout="wide")

st.title("🔍 Analisis Tekstur Citra Digital")
st.markdown("""
Aplikasi ini mendeteksi tekstur menggunakan tiga metode utama:
1.  **LBP (Local Binary Pattern):** Menghasilkan *citra tekstur* baru.
2.  **GLCM (Gray Level Co-occurrence Matrix):** Menghasilkan *angka statistik* (tidak ada citra output).
3.  **Fourier Transform (FFT):** Menghasilkan *citra spektrum frekuensi*.
""")

# --- Fungsi Helper ---
def load_image(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# --- Sidebar Upload ---
uploaded_file = st.sidebar.file_uploader("Upload Citra (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Tampilkan Citra Asli & Grayscale
    image = load_image(uploaded_file)
    gray_image = to_grayscale(image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Citra Asli", use_container_width=True)
    with col2:
        st.image(gray_image, caption="Citra Grayscale (Input Analisis)", use_container_width=True)

    st.divider()

    # --- Tabs untuk Metode ---
    tab1, tab2, tab3 = st.tabs(["1. Local Binary Pattern (LBP)", "2. GLCM (Statistik)", "3. Fourier Transform (FFT)"])

    # ==========================
    # TAB 1: Local Binary Pattern
    # ==========================
    with tab1:
        st.header("Local Binary Pattern (LBP)")
        st.info("Metode ini menghasilkan CITRA BARU yang menonjolkan garis tekstur.")
        
        # Parameter
        radius = st.slider("Radius LBP", 1, 5, 1, key='lbp_rad')
        n_points = 8 * radius
        
        # Kalkulasi
        lbp = local_binary_pattern(gray_image, n_points, radius, method="uniform")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            # Normalisasi visualisasi LBP agar kontras terlihat jelas (0-255)
            lbp_vis = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            st.image(lbp_vis, caption=f"Visualisasi LBP (R={radius})", use_container_width=True)
        
        with c2:
            st.subheader("Output Matriks LBP")
            st.write("Setiap angka merepresentasikan pola biner piksel. (Snippet 10x10):")
            st.dataframe(lbp[:10, :10]) 
            
            # Histogram
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(lbp.ravel(), bins=int(lbp.max() + 1), range=(0, int(lbp.max() + 1)), density=True, color='green', alpha=0.7)
            ax.set_title("Histogram Distribusi Pola")
            st.pyplot(fig)

    # ==========================
    # TAB 2: GLCM
    # ==========================
    with tab2:
        st.header("Gray Level Co-occurrence Matrix (GLCM)")
        st.warning("ℹ️ GLCM adalah metode statistik. Output utamanya adalah ANGKA, bukan gambar baru.")
        
        # Parameter
        c_param1, c_param2 = st.columns(2)
        with c_param1:
            dist = st.slider("Jarak Piksel", 1, 10, 1, key='glcm_dist')
        with c_param2:
            angle_dict = {"0°": 0, "45°": np.pi/4, "90°": np.pi/2, "135°": 3*np.pi/4}
            angle_label = st.selectbox("Sudut", list(angle_dict.keys()), key='glcm_ang')
            angle = angle_dict[angle_label]

        # Kalkulasi GLCM
        glcm = graycomatrix(gray_image, distances=[dist], angles=[angle], levels=256, symmetric=True, normed=True)
        
        # Fitur Statistik
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]

        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.subheader("Hasil Ekstraksi Fitur")
            metrics = {
                "Contrast": f"{contrast:.4f}",
                "Dissimilarity": f"{dissimilarity:.4f}",
                "Homogeneity": f"{homogeneity:.4f}",
                "Energy": f"{energy:.4f}",
                "Correlation": f"{correlation:.4f}"
            }
            st.table(metrics)

        with c2:
            st.subheader("Matriks GLCM (Data)")
            st.write("Matriks probabilitas kemunculan pasangan piksel (Snippet 10x10):")
            glcm_matrix_2d = glcm[:, :, 0, 0]
            st.dataframe(glcm_matrix_2d[:10, :10])
            
            if st.checkbox("Lihat Visualisasi Heatmap GLCM"):
                fig2, ax2 = plt.subplots()
                # Log scale agar terlihat warnanya
                cax = ax2.matshow(np.log(glcm_matrix_2d + 1e-6), cmap='viridis') 
                fig2.colorbar(cax)
                ax2.set_title("GLCM Heatmap (Log Scale)")
                st.pyplot(fig2)

    # ==========================
    # TAB 3: Fourier Transform
    # ==========================
    with tab3:
        st.header("Fourier Transform (FFT)")
        st.info("Metode ini mengubah citra ke domain frekuensi (Spektrum).")
        
        # 1. Kalkulasi FFT
        f = np.fft.fft2(gray_image)
        fshift = np.fft.fftshift(f) # Geser frekuensi rendah ke tengah
        
        # 2. Magnitude Spectrum (Log Scale)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)
        
        # 3. NORMALISASI (FIX: Agar gambar tidak putih polos)
        # Mengubah rentang angka menjadi 0 - 255
        norm_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
        img_spectrum = np.uint8(norm_spectrum)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.image(img_spectrum, caption="Magnitude Spectrum (Normalized)", use_container_width=True)
            st.caption("*Titik terang di tengah = frekuensi rendah (rata-rata intensitas).*")
        
        with c2:
            st.subheader("Data Matriks Spektrum")
            st.write("Nilai Magnitude (Log Scale, Raw):")
            st.dataframe(magnitude_spectrum)

else:
    st.info("Silakan upload gambar melalui sidebar di sebelah kiri.")
