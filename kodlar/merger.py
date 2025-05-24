import pandas as pd

# Veri setlerini okuyun (dosya isimlerini kendi dosyalarınızla değiştirin)
df1 = pd.read_csv('datalar/coffee_visits_balltree.csv')  # coffee visit data
df2 = pd.read_csv('datalar/venue_analysis_results.csv')  # venue scores data  
df3 = pd.read_csv('datalar/kullanici_profilleri_fast.csv')  # comprehensive visit data
df4 = pd.read_csv('datalar/vet_visits_balltree.csv')  # vet visit data



# Veri setlerini device_aid üzerinden birleştir
# Ana veri seti olarak 2. veri setini kullanıyoruz (Kahve_Puani ve Veteriner_Puani çıkarılmış hali)
merged_df =df2

# Diğer veri setlerini sırasıyla birleştir
merged_df = merged_df.merge(df1, on='device_aid', how='left')
merged_df = merged_df.merge(df3, on='device_aid', how='left')  
merged_df = merged_df.merge(df4, on='device_aid', how='left')

# Sonucu kontrol et
print(f"Birleştirilmiş veri setinin boyutu: {merged_df.shape}")
print(f"Toplam sütun sayısı: {len(merged_df.columns)}")
print("\nSütun isimleri:")
print(merged_df.columns.tolist())

# İlk birkaç satırı göster
print("\nİlk 5 satır:")
print(merged_df.head())

# Eksik değerleri kontrol et
print(f"\nEksik değer sayıları:")
print(merged_df.isnull().sum())

# Birleştirilmiş veri setini kaydet
merged_df.to_csv('merged_dataset.csv', index=False)
print(f"\nBirleştirilmiş veri seti 'merged_dataset.csv' olarak kaydedildi.")

# Opsiyonel: Sütun isimlerini düzenle (aynı isimli sütunlar varsa)
# Örneğin avg_visits_per_day sütunu hem 1. hem 4. veri setinde var
# Bu durumda pandas otomatik olarak _x, _y eklikleri yapar
if 'avg_visits_per_day_x' in merged_df.columns:
    merged_df = merged_df.rename(columns={
        'avg_visits_per_day_x': 'coffee_avg_visits_per_day',
        'avg_visits_per_day_y': 'vet_avg_visits_per_day'
    })
    print("Sütun isimleri düzenlendi.")