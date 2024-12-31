import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import streamlit as st

# Path to dataset
file_path = 'data/Smartphone_Usage_Behavior_Dataset.csv'

# Load dataset
data = pd.read_csv(file_path)

# Separate features and target
X = data[[
    "Social_Media_Usage_Hours",
    "Productivity_App_Usage_Hours",
    "Gaming_App_Usage_Hours",
    "Total_App_Usage_Hours"
]]
y = data["Daily_Screen_Time_Hours"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Find the best K
best_k = None
best_r2 = float('-inf')
k_values = range(1, 21)
r2_scores = []
for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, scoring='r2', cv=5)
    mean_score = scores.mean()
    r2_scores.append(mean_score)
    if mean_score > best_r2:
        best_k = k
        best_r2 = mean_score

# Train final model
knn_regressor = KNeighborsRegressor(n_neighbors=best_k)
knn_regressor.fit(X_train, y_train)

# Predictions
y_pred = knn_regressor.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Streamlit UI
st.set_page_config(page_title="KNN Screen Time Dashboard", layout="wide")

# GUI 1: Dashboard Utama
def dashboard():
    st.title("Dashboard Utama")

    st.subheader("Evaluasi Model")
    st.write(f"Mean Squared Error (MSE): {mse:.4f}")
    st.write(f"R-squared (R\u00b2): {r2:.4f}")

    st.subheader("Prediksi vs Nilai Sebenarnya")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.set_xlabel("Nilai Sebenarnya")
    ax.set_ylabel("Prediksi")
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
    st.pyplot(fig)

# GUI 2: Data Preprocessing
def preprocessing():
    st.title("Data Preprocessing")

    st.subheader("Deskripsi Data")
    st.write(data.describe())

    st.subheader("Distribusi Fitur")
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].hist(data["Social_Media_Usage_Hours"], bins=30)
    axes[0, 0].set_title("Social Media Usage Hours")
    axes[0, 1].hist(data["Productivity_App_Usage_Hours"], bins=30)
    axes[0, 1].set_title("Productivity App Usage Hours")
    axes[1, 0].hist(data["Gaming_App_Usage_Hours"], bins=30)
    axes[1, 0].set_title("Gaming App Usage Hours")
    axes[1, 1].hist(data["Total_App_Usage_Hours"], bins=30)
    axes[1, 1].set_title("Total App Usage Hours")
    st.pyplot(fig)

# GUI 3: Evaluasi Model KNN
def evaluation():
    st.title("Evaluasi Model KNN")

    st.subheader("Perbandingan K dan R\u00b2")
    st.write(f"Nilai K terbaik: {best_k} dengan R\u00b2 rata-rata: {best_r2:.4f}")

    st.subheader("Grafik R\u00b2 untuk Berbagai Nilai K")
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, r2_scores, marker="o")
    plt.xlabel("Nilai K")
    plt.ylabel("R-squared (R\u00b2)")
    st.pyplot()

# GUI 4: Prediksi Waktu Layar Aktif
def prediction():
    st.title("Prediksi Waktu Layar Aktif")

    social_media_usage = st.number_input("Jam Penggunaan Media Sosial", min_value=0.0, max_value=24.0, step=0.1)
    productivity_usage = st.number_input("Jam Penggunaan Aplikasi Produktivitas", min_value=0.0, max_value=24.0, step=0.1)
    gaming_usage = st.number_input("Jam Penggunaan Aplikasi Gaming", min_value=0.0, max_value=24.0, step=0.1)
    total_usage = st.number_input("Jam Penggunaan Aplikasi Total", min_value=0.0, max_value=24.0, step=0.1)

    if st.button("Prediksi"):
        input_data = [[social_media_usage, productivity_usage, gaming_usage, total_usage]]
        input_data_scaled = scaler.transform(input_data)
        prediction = knn_regressor.predict(input_data_scaled)
        st.write(f"Prediksi Waktu Layar Aktif: {prediction[0]:.2f} jam")

# Navigation between GUIs
pages = {
    "Dashboard Utama": dashboard,
    "Data Preprocessing": preprocessing,
    "Evaluasi Model KNN": evaluation,
    "Prediksi Waktu Layar Aktif": prediction
}

st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", list(pages.keys()))
pages[page]()
