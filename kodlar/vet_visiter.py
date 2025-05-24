import pandas as pd
import numpy as np
import glob
from sklearn.neighbors import BallTree
from tqdm import tqdm
from collections import Counter

# 1. 📥 Cihaz CSV dosyalarını oku ve birleştir
device_csv_folder = "devices/*.csv"
all_files = glob.glob(device_csv_folder)

if not all_files:
    raise FileNotFoundError("Devices klasöründe CSV dosyası bulunamadı!")

df_list = []
for file in tqdm(all_files, desc="Cihaz verileri yükleniyor"):
    try:
        df = pd.read_csv(file)
        # Gerekli kolonların varlığını kontrol et
        required_cols = ['timestamp', 'latitude', 'longitude', 'device_aid']
        if not all(col in df.columns for col in required_cols):
            print(f"Uyarı: {file} dosyasında gerekli kolonlar eksik")
            continue
            
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Etc/GMT-3')
        df_list.append(df)
    except Exception as e:
        print(f"Hata: {file} dosyası okunamadı - {e}")

if not df_list:
    raise ValueError("Hiçbir dosya başarıyla okunamadı!")

mobility_df = pd.concat(df_list, ignore_index=True)

# Veri temizleme
mobility_df = mobility_df.dropna(subset=['latitude', 'longitude'])
print(f"Toplam kayıt sayısı: {len(mobility_df):,}")

# 2. 🏥 Veteriner verisini oku
try:
    vet_df = pd.read_csv("veterinerlerson.csv")
    # Kolon isimlerini standartlaştır
    if 'lat' in vet_df.columns and 'lng' in vet_df.columns:
        vet_df = vet_df.rename(columns={'lat': 'latitude', 'lng': 'longitude'})
    
    vet_df = vet_df.dropna(subset=['latitude', 'longitude'])
    print(f"Toplam veteriner sayısı: {len(vet_df):,}")
except FileNotFoundError:
    raise FileNotFoundError("veterinerlerson.csv dosyası bulunamadı!")

# 3. 🧭 BallTree ile yakın veteriner kontrolü
RADIUS_KM = 0.02  # 20 metre
EARTH_RADIUS_KM = 6371.0

# Veteriner koordinatlarını radyana çevir
vet_coords = np.radians(vet_df[['latitude', 'longitude']].values)
tree = BallTree(vet_coords, metric='haversine')

# Cihaz koordinatlarını radyana çevir
user_coords = np.radians(mobility_df[['latitude', 'longitude']].values)

# Yakın veterinerleri bul
print("Yakın veterinerler aranıyor...")
indices = tree.query_radius(user_coords, r=RADIUS_KM / EARTH_RADIUS_KM)

# Eşleşmeleri tabloya yaz (tüm eşleşmeleri kaydet)
matches = []
vet_names = []

for idx, vet_indices in enumerate(tqdm(indices, desc="Eşleşmeler işleniyor")):
    if len(vet_indices) > 0:
        # En yakın veterineri bul
        distances = tree.query([user_coords[idx]], k=len(vet_indices), return_distance=True)[0][0]
        closest_idx = vet_indices[np.argmin(distances)]
        closest_name = vet_df.iloc[closest_idx]['name']
        
        matches.append(closest_name)
        vet_names.append(closest_name)
    else:
        matches.append(None)

mobility_df['near_vet'] = matches

# 4. 📊 Cihaz bazında veteriner ziyaret sayısı
vet_visits = (
    mobility_df.dropna(subset=['near_vet'])
    .groupby('device_aid')['near_vet']
    .count()
    .reset_index()
    .rename(columns={'near_vet': 'vet_visit_count'})
)

# Temel istatistikler ekle
vet_visits['avg_visits_per_day'] = vet_visits['vet_visit_count'] / (
    (mobility_df['timestamp'].max() - mobility_df['timestamp'].min()).days + 1
)

# 5. 🏥 Veteriner klinikleri hakkında detaylı analiz
print("Veteriner klinikleri analizi yapılıyor...")

# Ziyaret edilen veterinerler
visited_vet_df = mobility_df.dropna(subset=['near_vet']).copy()

# Her veteriner için analiz
vet_analysis = []

for vet_name in visited_vet_df['near_vet'].unique():
    vet_visits_data = visited_vet_df[visited_vet_df['near_vet'] == vet_name]
    
    # Temel istatistikler
    total_visits = len(vet_visits_data)
    unique_devices = vet_visits_data['device_aid'].nunique()
    
    # Zaman analizi
    visits_by_hour = vet_visits_data['timestamp'].dt.hour.value_counts().sort_index()
    peak_hour = visits_by_hour.idxmax()
    
    visits_by_day = vet_visits_data['timestamp'].dt.day_name().value_counts()
    peak_day = visits_by_day.idxmax()
    
    # Tarih aralığı
    first_visit = vet_visits_data['timestamp'].min()
    last_visit = vet_visits_data['timestamp'].max()
    
    # En sık ziyaret eden cihaz
    top_visitor = vet_visits_data['device_aid'].value_counts().iloc[0]
    top_visitor_device = vet_visits_data['device_aid'].value_counts().index[0]
    
    vet_analysis.append({
        'veteriner_name': vet_name,
        'total_visits': total_visits,
        'unique_visitors': unique_devices,
        'avg_visits_per_visitor': round(total_visits / unique_devices, 2),
        'peak_hour': peak_hour,
        'peak_day': peak_day,
        'first_visit_date': first_visit.strftime('%Y-%m-%d %H:%M'),
        'last_visit_date': last_visit.strftime('%Y-%m-%d %H:%M'),
        'most_frequent_visitor_device': top_visitor_device,
        'most_frequent_visitor_count': top_visitor,
        'visits_per_day': round(total_visits / ((last_visit - first_visit).days + 1), 2)
    })

vet_analysis_df = pd.DataFrame(vet_analysis)

# Popülerlik sıralaması
vet_analysis_df = vet_analysis_df.sort_values('total_visits', ascending=False)

# 6. 📈 Zaman bazında analiz
hourly_visits = visited_vet_df.groupby(visited_vet_df['timestamp'].dt.hour).size()
daily_visits = visited_vet_df.groupby(visited_vet_df['timestamp'].dt.date).size()

# 7. 💾 Sonuçları kaydet
vet_visits.to_csv("vet_visits_balltree.csv", index=False)
vet_analysis_df.to_csv("veteriner_detailed_analysis.csv", index=False)

# Özet istatistikler
summary_stats = {
    'total_vet_visits': len(visited_vet_df),
    'unique_veteriner_visited': visited_vet_df['near_vet'].nunique(),
    'unique_devices_visiting_vet': visited_vet_df['device_aid'].nunique(),
    'most_popular_veteriner': vet_analysis_df.iloc[0]['veteriner_name'],
    'most_popular_visits': vet_analysis_df.iloc[0]['total_visits'],
    'avg_visits_per_veteriner': round(vet_analysis_df['total_visits'].mean(), 2),
    'peak_hour_overall': hourly_visits.idxmax(),
    'analysis_date_range': f"{mobility_df['timestamp'].min().strftime('%Y-%m-%d')} - {mobility_df['timestamp'].max().strftime('%Y-%m-%d')}"
}

# Özet dosyası
summary_df = pd.DataFrame([summary_stats])
summary_df.to_csv("veteriner_analysis_summary.csv", index=False)

# 8. 📋 Sonuçları yazdır
print("\n" + "="*50)
print("🏥 VETERİNER KLİNİKLERİ ANALİZ SONUÇLARI")
print("="*50)
print(f"✅ Toplam veteriner ziyareti: {summary_stats['total_vet_visits']:,}")
print(f"🏥 Ziyaret edilen farklı veteriner: {summary_stats['unique_veteriner_visited']}")
print(f"📱 Veteriner ziyaret eden cihaz: {summary_stats['unique_devices_visiting_vet']}")
print(f"🥇 En popüler veteriner: {summary_stats['most_popular_veteriner']} ({summary_stats['most_popular_visits']} ziyaret)")
print(f"⏰ En yoğun saat: {summary_stats['peak_hour_overall']}:00")
print(f"📅 Analiz dönemi: {summary_stats['analysis_date_range']}")

print("\n📁 Oluşturulan dosyalar:")
print("- vet_visits_balltree.csv (cihaz bazında ziyaret sayıları)")
print("- veteriner_detailed_analysis.csv (veteriner bazında detaylı analiz)")
print("- veteriner_analysis_summary.csv (genel özet)")

print(f"\n🔝 En popüler 5 veteriner:")
for i, row in vet_analysis_df.head().iterrows():
    print(f"{i+1}. {row['veteriner_name']}: {row['total_visits']} ziyaret ({row['unique_visitors']} farklı ziyaretçi)")

print("\nAnaliz tamamlandı! 🎉")