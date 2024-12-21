import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# Load model XGBoost
xgb_model = joblib.load("best_model.pkl")

# Load data
data_file = 'data/Data Dashboard.xlsx'
data = pd.read_excel(data_file)

# Fungsi untuk menghitung rasio keuangan
def calculate_ratios_from_input(inputs):
    A = inputs['aktiva_lancar'] / inputs['liabilitas_lancar']
    B = inputs['laba_bersih'] / inputs['total_aktiva']  
    C = inputs['total_liabilitas'] / inputs['total_aktiva']
    X1 = inputs['kas'] / inputs['liabilitas_lancar']
    X2 = inputs['piutang'] / (inputs['penjualan'] / 365)
    X3 = inputs['aktiva_tetap'] / inputs['penjualan']
    X4 = inputs['total_aktiva'] / inputs['penjualan']
    X5 = inputs['laba_bersih'] / inputs['total_ekuitas']
    X6 = inputs['laba_bersih'] / inputs['penjualan']
    X7 = inputs['laba_bruto'] / inputs['penjualan']
    X8 = inputs['ebit'] / inputs['beban_bunga']
    X9 = inputs['total_liabilitas'] / inputs['total_ekuitas']
    X10 = inputs['harga_saham'] / inputs['laba_saham']
    X11 = inputs['harga_saham'] / (inputs['laba_bersih'] / inputs['saham_beredar'])

    return [A, B, C, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11]

# Initialize SHAP explainer
explainer = shap.Explainer(xgb_model)

# Fungsi untuk menampilkan bar chart SHAP
def generate_shap_barchart(shap_values, feature_names):
    fig, ax = plt.subplots(figsize=(10, 6))
    shap_values = shap_values[0]  # SHAP values untuk instance pertama
    sorted_indices = np.argsort(shap_values)  # Urutkan SHAP values
    sorted_shap_values = shap_values[sorted_indices]
    sorted_feature_names = np.array(feature_names)[sorted_indices]

    ax.barh(sorted_feature_names, sorted_shap_values, color='#2980b9')
    ax.set_title("SHAP Bar Chart untuk Prediksi Individual")
    ax.set_xlabel("Nilai SHAP")
    ax.set_ylabel("Fitur")
    plt.tight_layout()
    return fig

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Dashboard Financial Distress", layout="centered", initial_sidebar_state="expanded")

# Title Dashboard
st.markdown("""
    <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            color: white;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .stButton button {
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            padding: 12px 24px;
            font-size: 16px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stButton button:hover {
            background-color: #2980b9;
        }
        .stSelectbox, .stTextInput {
            width: 100%;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stDataFrame {
            margin-top: 20px;
        }
        .footer {
            font-size: 14px;
            color: #2c3e50; 
            text-align: center;
            margin-top: 20px;
        }
        .alert {
            font-size: 18px;
            padding: 10px;
            border-radius: 5px;
            font-weight: bold;
        }
        .alert-success {
            background-color: #2ecc71; /* Green */
            color: white;
        }
        .alert-danger {
            background-color: #e74c3c; /* Red */
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>MODEL DETEKSI DINI FINANCIAL DISTRESS PADA PERUSAHAAN SEKTOR TRANSPORTASI DAN LOGISTIK DI INDONESIA</div>", unsafe_allow_html=True)

# Tab navigasi dengan tambahan tab Cara Penggunaan dan Profil Pembuat
tabs = st.tabs(["Tentang Dashboard", "Panduan", "Analisis & Prediksi", "Input Data", "Profil"])

# Tab Tentang Dashboard
with tabs[0]:
    st.title("Tentang Dashboard")
    st.write("""
    **Selamat datang di Dashboard Deteksi Dini Financial Distress!**

    Dashboard ini dirancang untuk membantu menganalisis kondisi keuangan perusahaan-perusahaan yang terdaftar di sektor transportasi dan logistik Indonesia. Dengan menggunakan model machine learning berbasis XGBoost, dashboard ini dapat memprediksi kemungkinan **financial distress** pada perusahaan-perusahaan tersebut berdasarkan rasio-rasio keuangan penting yang dihitung dari data yang diberikan.
    Dengan menggunakan analisis rasio keuangan dan prediksi berdasarkan model, diharapkan dapat membantu para analis, investor, dan manajemen perusahaan dalam mengambil keputusan strategis terkait kelangsungan dan kesehatan keuangan perusahaan.
    """)
    st.image("image/image 1.png", use_container_width=True)
    st.write("""
             
    **Fitur Utama :**
    - **Analisis Rasio Keuangan:** Melihat rasio keuangan dari kombinasi saham dan tahun yang dipilih, yang akan dihitung dan ditampilkan secara otomatis.
    - **Prediksi Financial Distress:** Memasukkan data keuangan perusahaan untuk memprediksi apakah perusahaan mengalami **financial distress** atau tidak, dengan memberikan probabilitas dari model yang sudah dilatih.

    **Tentang Model :**
    Model prediksi yang digunakan dalam model ini menggunakan metode machine learning yaitu Extreme Gradient Boosting (XGBoost) dengan tingkat akurasi mencapai 95,45%.
    Variabel yang digunakan dalam melatih model terdiri dari 11 rasio keuangan yaitu Cash Ratio, Day's Sales Outstanding, Fixed Assets Turn Over, Total Assets Turn Over, Return on Equity, Net Profit Margin, Gross Profit Margin, Interests Coverage Ratio, Debt to Equity Ratio, Price Earning Ratio, dan Market to Book Ratio.
    Status Financial Distress dideteksi menggunakan metode terkemuka oleh Mark E. Zmijewski (1984) yang mengukur financial distress berdasarkan pendapatan, hutang, serta aset dari perusahaan dengan rasio Return on Assets, Debt to Assets Ratio, serta Current Ratio sebagai variabel pembangun modelnya.         
    """)

# Tab Cara Penggunaan
with tabs[1]:
    st.title("Panduan Penggunaan")
    st.write("""
    **Panduan Menggunakan Dashboard Deteksi Financial Distress :**

    1. **Tab Analisis & Prediksi :**
        - Pilih kombinasi kode saham dan tahun yang ingin dianalisis.
        - Jika diperlukan, pilih kode saham kedua untuk membandingkan kondisi keuangan antara dua perusahaan.
        - Anda dapat melihat hasil perhitungan rasio keuangan dari data yang dipilih pada tabel yang tersedia. Rasio yang disajikan dalam bentuk Desimal.
        - Klik tombol **Prediksi** untuk melihat hasil apakah perusahaan berada dalam kondisi financial distress atau tidak. Hasil prediksi akan memberikan informasi berupa Status: **Financial Distress** atau **Tidak Financial Distress** dan Probabilitas prediksi dalam bentuk persentase akurasi.
        - Pada Grafik Variable Importance, urutan variabel disusun berurut sesuai dengan tingkat kepentingan tertinggi hingga terendah. Jika SHAP Value bernilai positif menandakan variabel yang berpengaruh positif atas terjadinya financial distress pada perusahaan
             
    2. **Tab Input Data :**
        - Input data keuangan sesuai dengan form yang tertera dalam satuan **mata uang yang sama.**
        - Klik Tombol **Analisis Rasio Keuangan** untuk menampilkan hasil perhitungan rasio keuangan berdasarkan data keuangan yang anda input.
        - Klik Tombol **Prediksi Financial Distress** untuk menampilkan hasil prediksi status financial distress beserta probabilitas financial distress berdasarkan data keuangan yang anda input.
        - Pada Grafik Variable Importance, urutan variabel disusun berurut sesuai dengan tingkat kepentingan tertinggi hingga terendah. Jika SHAP Value bernilai positif menandakan variabel yang berpengaruh positif atas terjadinya financial distress pada perusahaan
             
    3. Gunakan informasi ini untuk analisis resiko keuangan perusahaan.
    """)
import streamlit as st
import numpy as np

# Tab Analisis dan Prediksi
with tabs[2]:
    st.title("Analisis dan Prediksi Financial Distress")
    st.write("Pilih kombinasi saham dan tahun untuk melihat informasi rasio keuangan dan memprediksi kemungkinan financial distress.")

    # Card layout untuk informasi
    with st.container():
        st.write("### Pilih Kombinasi Saham dan Tahun untuk Rasio Keuangan dan Prediksi")
        col1, col2 = st.columns(2)

        with col1:
            kode_saham_1 = st.selectbox("**Pilih Kode Saham (1)**", data['kode'].unique(), key="kode_saham_1")
            tahun_1 = st.selectbox("**Pilih Tahun (1)**", data[data['kode'] == kode_saham_1]['tahun'].unique(), key="tahun_1")

        with col2:
            kode_saham_2 = st.selectbox("**Pilih Kode Saham (2)**", ["Kosongkan"] + data['kode'].unique().tolist(), key="kode_saham_2")
            if kode_saham_2 != "Kosongkan":
                tahun_2 = st.selectbox("**Pilih Tahun (2)**", data[data['kode'] == kode_saham_2]['tahun'].unique(), key="tahun_2")
            else:
                tahun_2 = None

    # Filter data berdasarkan kode saham dan tahun untuk kombinasi pertama dan kedua
    filtered_data_1 = data[(data['kode'] == kode_saham_1) & (data['tahun'] == tahun_1)]
    
    if kode_saham_2 != "Kosongkan":
        filtered_data_2 = data[(data['kode'] == kode_saham_2) & (data['tahun'] == tahun_2)]
    else:
        filtered_data_2 = None

    if not filtered_data_1.empty:
        # Hitung rasio keuangan untuk kombinasi pertama
        rasio_keuangan_1 = calculate_ratios_from_input(filtered_data_1.iloc[0])
        
        if filtered_data_2 is not None and not filtered_data_2.empty:
            # Hitung rasio keuangan untuk kombinasi kedua
            rasio_keuangan_2 = calculate_ratios_from_input(filtered_data_2.iloc[0])

            # Buat DataFrame untuk rasio keuangan
            rasio_df = pd.DataFrame({
                "Rasio": [
                    "Current Ratio", "Return on Assets", "Debt to Assets Ratio", "Cash Ratio", "Day's Sales Outstanding", "Fixed Assets Turn Over", "Total Assets Turn Over",
                    "Return on Equity", "Net Profit Margin", "Gross Profit Margin", "Interest Coverage Ratio", "Debt to Equity Ratio", "Price to Earnings Ratio", "Price to Book Ratio"
                ],
                f'{kode_saham_1} ({tahun_1})': [f"{x:.4f}" for x in rasio_keuangan_1],
                f'{kode_saham_2} ({tahun_2})': [f"{x:.4f}" for x in rasio_keuangan_2]
            })
        else:
            rasio_df = pd.DataFrame({
                "Rasio": [
                    "Current Ratio", "Return on Assets", "Debt to Assets Ratio", "Cash Ratio", "Day's Sales Outstanding", "Fixed Assets Turn Over", "Total Assets Turn Over",
                    "Return on Equity", "Net Profit Margin", "Gross Profit Margin", "Interest Coverage Ratio", "Debt to Equity Ratio", "Price to Earnings Ratio", "Price to Book Ratio"
                ],
                f'{kode_saham_1} ({tahun_1})': [f"{x:.4f}" for x in rasio_keuangan_1]
            })

        # Tampilkan dataframe rasio keuangan
        st.dataframe(rasio_df, use_container_width=True)

        # Prediksi Financial Distress untuk kombinasi 1 saham dan tahun
        rasio_keuangan_1 = np.array(rasio_keuangan_1[3:]).reshape(1, -1)

        # Prediksi Financial Distress untuk kombinasi 1 saham dan tahun
        if st.button("Prediksi Financial Distress Kombinasi 1"):
            prediction = xgb_model.predict(rasio_keuangan_1)
            probability = xgb_model.predict_proba(rasio_keuangan_1)[:, 1]
    
            # Kalkulasi SHAP values
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(rasio_keuangan_1)
            feature_names = [
                "Cash Ratio", "Day's Sales Outstanding", "Fixed Assets Turn Over", "Total Assets Turn Over",
                "Return on Equity", "Net Profit Margin", "Gross Profit Margin", "Interest Coverage Ratio",
                "Debt to Equity Ratio", "Price to Earnings Ratio", "Price to Book Ratio"
            ]
    
            st.subheader("Hasil Prediksi")
            if prediction[0] == 1:
                st.markdown(f"<div class='alert alert-danger'>Financial Distress<br>Probabilitas: {probability[0]:.4f}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='alert alert-success'>Tidak Financial Distress<br>Probabilitas: {probability[0]:.4f}</div>", unsafe_allow_html=True)
    
            # Tampilkan bar chart SHAP
            st.write("### Tingkat Pengaruh Variabel")
            st.write()
            shap_fig = generate_shap_barchart(shap_values, feature_names)
            st.pyplot(shap_fig)

        # Prediksi Financial Distress untuk kombinasi 2 saham dan tahun
        if filtered_data_2 is not None and not filtered_data_2.empty:
            rasio_keuangan_2 = np.array(rasio_keuangan_2[3:]).reshape(1, -1)

            if st.button("Prediksi Financial Distress Kombinasi 2"):
                prediction = xgb_model.predict(rasio_keuangan_2)
                probability = xgb_model.predict_proba(rasio_keuangan_2)[:, 1]
                # Kalkulasi SHAP values
                explainer = shap.TreeExplainer(xgb_model)
                shap_values = explainer.shap_values(rasio_keuangan_2)
                feature_names = [
                    "Cash Ratio", "Day's Sales Outstanding", "Fixed Assets Turn Over", "Total Assets Turn Over",
                    "Return on Equity", "Net Profit Margin", "Gross Profit Margin", "Interest Coverage Ratio",
                    "Debt to Equity Ratio", "Price to Earnings Ratio", "Price to Book Ratio"
                ]
                st.subheader("Hasil Prediksi")

                if prediction[0] == 1:
                    st.markdown(f"<div class='alert alert-danger'>Financial Distress<br>Probabilitas: {probability[0]:.4f}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='alert alert-success'>Tidak Financial Distress<br>Probabilitas: {probability[0]:.4f}</div>", unsafe_allow_html=True)
                # Tampilkan bar chart SHAP
                st.write("### Tingkat Kepentingan Variabel")
                st.write()
                shap_fig = generate_shap_barchart(shap_values, feature_names)
                st.pyplot(shap_fig)
        else:
            st.write(" ")
    else:
        st.write("Data tidak ditemukan untuk kombinasi saham dan tahun yang dipilih.")

# Tab Analisis dan Prediksi
with tabs[3]:
    st.title("Analisis dan Prediksi Financial Distress")
    st.write("Input Variabel Keuangan Lainnya")
    
    # Input untuk variabel keuangan
    inputs = {
        'laba_bersih': st.number_input("Laba Bersih", min_value=0.0),
        'total_aktiva': st.number_input("Total Aktiva", min_value=0.0),
        'total_liabilitas': st.number_input("Total Liabilitas", min_value=0.0),
        'aktiva_lancar': st.number_input("Aktiva Lancar", min_value=0.0),
        'liabilitas_lancar': st.number_input("Liabilitas Lancar", min_value=0.0),
        'kas': st.number_input("Kas", min_value=0.0),
        'penjualan': st.number_input("Penjualan", min_value=0.0),
        'piutang': st.number_input("Piutang", min_value=0.0),
        'aktiva_tetap': st.number_input("Aktiva Tetap", min_value=0.0),
        'total_ekuitas': st.number_input("Total Ekuitas", min_value=0.0),
        'laba_bruto': st.number_input("Laba Bruto", min_value=0.0),
        'ebit': st.number_input("EBIT", min_value=0.0),
        'beban_bunga': st.number_input("Beban Bunga", min_value=0.0),
        'harga_saham': st.number_input("Harga Saham", min_value=0.0),
        'laba_saham': st.number_input("Laba Saham", min_value=0.0),
        'saham_beredar': st.number_input("Saham Beredar", min_value=0.0)
    }

    # Analisis Rasio Keuangan
    if st.button("Analisis Rasio Keuangan"):
        # Menghitung rasio keuangan dari input manual
        rasio_keuangan_manual = calculate_ratios_from_input(inputs)

        # Buat DataFrame untuk menampilkan rasio keuangan
        rasio_df_manual = pd.DataFrame({
            "Rasio": [
                "Current Ratio", "Return on Assets", "Debt to Assets Ratio", "Cash Ratio", 
                "Day's Sales Outstanding", "Fixed Assets Turn Over", "Total Assets Turn Over", 
                "Return on Equity", "Net Profit Margin", "Gross Profit Margin", 
                "Interest Coverage Ratio", "Debt to Equity Ratio", "Price to Earnings Ratio", 
                "Price to Book Ratio"
            ],
            "Nilai": [f"{x:.4f}" for x in rasio_keuangan_manual]
        })

        # Menampilkan DataFrame rasio keuangan yang telah dihitung
        st.dataframe(rasio_df_manual, use_container_width=True)

    # Prediksi Financial Distress
    if st.button("Prediksi Financial Distress"):
        # Menghitung rasio keuangan dari input manual
        rasio_keuangan_manual = calculate_ratios_from_input(inputs)
        rasio_keuangan_manual = np.array(rasio_keuangan_manual[3:]).reshape(1, -1)
    
        # Prediksi menggunakan model
        prediction_manual = xgb_model.predict(rasio_keuangan_manual)
        probability_manual = xgb_model.predict_proba(rasio_keuangan_manual)[:, 1]
    
        # Kalkulasi SHAP values
        explainer = shap.TreeExplainer(xgb_model)
        shap_values_manual = explainer.shap_values(rasio_keuangan_manual)
        feature_names = [
            "Cash Ratio", "Day's Sales Outstanding", "Fixed Assets Turn Over", "Total Assets Turn Over",
            "Return on Equity", "Net Profit Margin", "Gross Profit Margin", "Interest Coverage Ratio",
            "Debt to Equity Ratio", "Price to Earnings Ratio", "Price to Book Ratio"
        ]

        # Menampilkan hasil prediksi
        st.subheader("Hasil Prediksi")
        if prediction_manual[0] == 1:
            st.markdown(
                f"<div class='alert alert-danger'>Financial Distress<br>Probabilitas: {probability_manual[0]:.4f}</div>", 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='alert alert-success'>Tidak Financial Distress<br>Probabilitas: {probability_manual[0]:.4f}</div>", 
                unsafe_allow_html=True
            )
    
        # Tampilkan bar chart SHAP
        st.write("### Tingkat Kepentingan Variabel")
        shap_fig_manual = generate_shap_barchart(shap_values_manual, feature_names)
        st.pyplot(shap_fig_manual)

# Tab Profil Pembuat
with tabs[4]:
    st.title("Profil Pembuat")
    st.write("""
    **Profil Pembuat Dashboard:**

    - **Nama:** Ezar Alvah Rayhan
    - **Asal Universitas:** Institut Teknologi Sepuluh Nopember
    - **Program Studi:** D4 Statistika Bisnis
    - **NRP:** 2043211090
    - **Kontak:**
        - Email: ezarrayhan2@gmail.com
        - LinkedIn: www.linkedin.com/in/ezar-alvah-rayhan

    Dashboard ini dikembangkan sebagai bagian dari tugas akhir/skripsi untuk membantu menganalisis resiko keuangan pada perusahaan sektor transportasi dan logistik di Indonesia.
    """)

# Footer
st.markdown("""
    <div class='footer'>
        <p>© 2024 - Model Deteksi Dini Financial Distress</p>
    </div>
""", unsafe_allow_html=True)