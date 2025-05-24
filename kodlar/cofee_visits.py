import pandas as pd
import numpy as np
import glob
from sklearn.neighbors import BallTree
from tqdm import tqdm

# 1. 📥 Cihaz CSV dosyalarını oku ve birleştir
device_csv_folder = "../devices/*.csv"
all_files = glob.glob(device_csv_folder)

df_list = []
for file in tqdm(all_files, desc="Cihaz verileri yükleniyor"):
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Etc/GMT-3')
    df_list.append(df)

mobility_df = pd.concat(df_list, ignore_index=True)

# 2. 📍 Kahve zinciri verisini oku
coffee_df = pd.read_csv("../kahveci2.csv")  # Kolonlar: name, lat, lng

# 3. 🧭 BallTree ile yakın kahveci kontrolü
# BallTree 0.1km (100 metre) içinde olan kahvecileri bulur
RADIUS_KM = 0.1
EARTH_RADIUS_KM = 6371.0

# a. Kahveci koordinatlarını radyana çevir
coffee_coords = np.radians(coffee_df[['latitude', 'longitude']].values)
tree = BallTree(coffee_coords, metric='haversine')

# b. Cihaz koordinatlarını radyana çevir
user_coords = np.radians(mobility_df[['latitude', 'longitude']].values)

# c. Yakın kahvecileri bul (BallTree sorgusu)
indices = tree.query_radius(user_coords, r=RADIUS_KM / EARTH_RADIUS_KM)

# d. Eşleşmeleri tabloya yaz
matches = []
for idx, cafe_indices in enumerate(indices):
    if len(cafe_indices) > 0:
        matched_names = [coffee_df.iloc[i]['name'] for i in cafe_indices]
        matches.append(matched_names[0])  # İlkini alıyoruz
    else:
        matches.append(None)

mobility_df['near_coffee'] = matches

# 4. 📊 Kahveci ziyaret sayısı hesapla
coffee_visits = (
    mobility_df.dropna(subset=['near_coffee'])
    .groupby('device_aid')['near_coffee']
    .count()
    .reset_index()
    .rename(columns={'near_coffee': 'coffee_visit_count'})
)

# 5. 💾 Sonucu CSV olarak kaydet
coffee_visits.to_csv("coffee_visits_balltree.csv", index=False)
print("Ziyaret analizi tamamlandı: coffee_visits_balltree.csv")
