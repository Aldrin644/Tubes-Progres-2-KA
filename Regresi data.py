import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Path ke file data Anda
file_path = '/content/drive/MyDrive/Data Tugas KA/Smartphone Usage and Behavioral Dataset - mobile_usage_behavioral_analysis.csv'

# Membaca data
data = pd.read_csv(file_path)

# Menampilkan beberapa baris pertama data
print(data.head())

# Memisahkan fitur (X) dan target (y)
X = data[[
    "Social_Media_Usage_Hours",
    "Productivity_App_Usage_Hours",
    "Gaming_App_Usage_Hours",
    "Total_App_Usage_Hours"
]]
y = data["Daily_Screen_Time_Hours"]  # Target: waktu layar aktif

# Membagi data menjadi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi data (standarisasi)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Uji beberapa nilai K
best_k = None
best_r2 = float('-inf')
for k in range(1, 21):
    knn = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, scoring='r2', cv=5)
    mean_score = scores.mean()
    if mean_score > best_r2:
        best_k = k
        best_r2 = mean_score

print(f"Nilai K terbaik: {best_k} dengan R² rata-rata: {best_r2:.4f}")

# Membuat model dengan K terbaik
knn_regressor = KNeighborsRegressor(n_neighbors=best_k)
knn_regressor.fit(X_train, y_train)

# Memprediksi data uji
y_pred = knn_regressor.predict(X_test)

# Evaluasi
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)

# Visualisasi hasil prediksi vs nilai sebenarnya
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.title("Prediksi vs Nilai Sebenarnya")
plt.xlabel("Nilai Sebenarnya")
plt.ylabel("Prediksi")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
plt.grid()
plt.show()
