import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date
import time
import json
import streamlit as st
from streamlit_lottie import st_lottie

current_year = date.today().year
years = list(range(1997, current_year + 1))

path = "Car.json"
with open(path,"r") as file:
    url = json.load(file)

# --- 1. Konfigurasi dan Pemuatan Model ---
@st.cache_resource # Memuat model hanya sekali untuk efisiensi
def load_ml_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Error: File model tidak ditemukan di '{model_path}'.")
        st.stop()
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        st.stop()

# Panggil fungsi pemuatan model
model_path = 'prediksi_mobil.pkl' # GANTI INI!
model = load_ml_model(model_path)

# Definisi Kolom (Harus cocok dengan urutan X_train!)
kolom_model_mobil = [
    'model_ Auris', 'model_ Avensis', 'model_ Aygo', 'model_ C-HR',
    'model_ Camry', 'model_ Corolla', 'model_ GT86', 'model_ Hilux',
    'model_ IQ', 'model_ Land Cruiser', 'model_ PROACE VERSO', 'model_ Prius',
    'model_ RAV4', 'model_ Supra', 'model_ Urban Cruiser', 'model_ Verso',
    'model_ Verso-S', 'model_ Yaris' 
]

kolom_transmisi = ['transmission_Automatic', 'transmission_Manual', 'transmission_Other', 'transmission_Semi-Auto']
kolom_bahan_bakar = ['fuelType_Diesel', 'fuelType_Hybrid', 'fuelType_Other', 'fuelType_Petrol']

semua_kolom = (
    ['year', 'mileage', 'tax', 'mpg', 'engineSize'] +
    kolom_model_mobil +
    kolom_transmisi +
    kolom_bahan_bakar
)
# --- End of Pemuatan Model & Konfigurasi ---
st.set_page_config(page_title="Prediksi Harga Mobil", layout="wide")
st.title("🚗 Prediksi Harga Mobil Bekas Toyota")
tab1, tab2 = st.tabs(["Beranda", "Prediksi"])
with tab1:
    st.header('Selamat Datang Di Prediksi Mobil Toyota Bekas')
    st.write('Dataset berasal dari :')
    st.link_button("Dataset", "https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes?resource=download&select=toyota.csv")
    df = pd.read_csv('toyota.csv')
    st.dataframe(df)
    kol_1, kol_2, kol_3 = st.columns([1, 1, 1])
    with kol_1:
        st.subheader("Rata-Rata Harga Berdasarkan Model Mobil")
        average_price_by_model = df.groupby('model')['price'].mean().sort_values(ascending=False)
        st.dataframe(average_price_by_model)
    with kol_2:
        st.subheader("Rata-Rata Harga Berdasarkan Bahan Bakar")
        average_price_by_fuel = df.groupby('fuelType')['price'].mean().sort_values(ascending=False)
        st.dataframe(average_price_by_fuel)
    with kol_3:
        st.subheader("Rata-Rata Harga Berdasarkan Transmisi")
        average_price_by_trans = df.groupby('transmission')['price'].mean().sort_values(ascending=False)
        st.dataframe(average_price_by_trans)

with tab2: 
    st.markdown("Masukkan spesifikasi mobil untuk mendapatkan estimasi harga")

    # Tiga kolom untuk layout input
    col1, col2, col3 = st.columns(3)

    # Kolom 1: Input Numerik Utama
    with col1:
        st.header("Detail Dasar")
        year = st.selectbox(
        label='Pilih Tahun',
        options=years,
        index=20
        )
        mileage = st.number_input("Jarak Tempuh (mileage, mil)", min_value=0, value=2000, step=100)
        tax = st.number_input("Pajak (tax, £)", min_value=0, value=200, step=10)

    # Kolom 2: Input Numerik Lanjutan & Transmisi
    with col2:
        st.header("Spesifikasi Teknis")
        mpg = st.number_input("MPG (Miles per Gallon)", min_value=0.0, value=36.2, step=0.1, format="%.1f")
        engineSize = st.number_input("Ukuran Mesin (liter)", min_value=0.5, max_value=6.0, value=2.0, step=0.1, format="%.1f")
        
        # Input Transmisi (Menggunakan nama kolom tanpa prefix untuk display)
        st.subheader("Transmisi")
        transmisi_display = [kol.replace('transmission_', '') for kol in kolom_transmisi]
        transmisi_pilihan = st.radio("Pilih Transmisi", transmisi_display, index=0, key='radio_transmisi')

    # Kolom 3: Input Kategorikal (Model dan Bahan Bakar)
    with col3:
        st.header("Kategori Mobil")
        
        # Input Model Mobil
        st.subheader("Model Mobil")
        model_display = [kol.replace('model_', '') for kol in kolom_model_mobil]
        model_pilihan_display = st.selectbox("Pilih Model Mobil", model_display, index=model_display.index(' Urban Cruiser'), key='select_model')
        
        # Input Bahan Bakar
        st.subheader("Jenis Bahan Bakar")
        fuel_display = [kol.replace('fuelType_', '') for kol in kolom_bahan_bakar]
        fuel_pilihan_display = st.radio("Pilih Jenis Bahan Bakar", fuel_display, index=fuel_display.index('Petrol'), key='radio_fuel')


    # --- 3. Logika Prediksi ---
    if st.button("💰 Prediksi Harga Mobil"):
        # 1. Inisialisasi dictionary dengan semua kolom bernilai 0
        new_data_dict = {kol: [0] for kol in semua_kolom}
        
        # 2. Isi nilai numerik
        new_data_dict['year'] = [year]
        new_data_dict['mileage'] = [mileage]
        new_data_dict['tax'] = [tax]
        new_data_dict['mpg'] = [mpg]
        new_data_dict['engineSize'] = [engineSize]
        
        # 3. Isi nilai kategorikal (one-hot encoding = 1)
        
        # Model Mobil
        model_kolom = 'model_' + model_pilihan_display
        new_data_dict[model_kolom] = [1]
            
        # Transmisi
        transmisi_kolom = 'transmission_' + transmisi_pilihan
        new_data_dict[transmisi_kolom] = [1]

        # Bahan Bakar
        fuel_kolom = 'fuelType_' + fuel_pilihan_display
        new_data_dict[fuel_kolom] = [1]

        # 4. Konversi ke DataFrame dan pastikan urutan kolom sesuai
        new_data_df = pd.DataFrame(new_data_dict)
        new_data_df = new_data_df[semua_kolom] 
        
        try:
            # 5. Lakukan Prediksi
            # Asumsi model.predict mengembalikan array [harga]
            predicted_price = model.predict(new_data_df)[0]
            # Jika menggunakan dummy: predicted_price = model(new_data_df)[0] 
            print(predicted_price)
            

            with st.spinner("Wait for it...", show_time=True):
                time.sleep(2)
            # 6. Tampilkan Hasil
            st.success("✅ Prediksi Harga Berhasil Dibuat!")
            
            # Format harga ke mata uang (contoh: Pound Sterling £)
            formatted_price = f"£ {predicted_price:,.0f}"
            formatted_price_rupiah = f"Rp {predicted_price*21000:,.0f}"
            
            st.metric(label="Harga Prediksi Dalam Pounds 💵", value=formatted_price)
            st.metric(label="Harga Prediksi Dalan Rupiah 💵", value=formatted_price_rupiah)
            st_lottie(url,
                reverse=True,
                height=300,
                width=300,
                speed=1,
                loop=True,
                quality='high',
                key='Car'
            )
            st.write(new_data_df)
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat membuat prediksi: {e}")
            st.caption("Pastikan Anda telah memuat model dengan benar dan urutan kolom (`semua_kolom`) sudah sesuai dengan data pelatihan.")

