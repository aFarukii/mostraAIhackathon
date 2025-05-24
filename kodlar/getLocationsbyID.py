import pyarrow.dataset as ds
import pandas as pd
import os
from collections import defaultdict

# Dosya yolları
parquet_filepath = "MobilityDataMay2024.paraquet"
device_ids_file = 'unique_device_ids.csv'

# İstenen device_id'leri oku
target_device_ids = set(pd.read_csv(device_ids_file)["device_aid"].dropna().unique())
print(f"Toplam hedef device_id sayısı: {len(target_device_ids)}")

# Dataset'i oluştur
dataset = ds.dataset(parquet_filepath, format="parquet")
scanner = dataset.scanner(batch_size=1_000_000)  # Batch size'ı artırdık

# Her device_id için veri toplama
device_data = defaultdict(list)
processed_rows = 0
found_device_count = 0

print("Veri işleniyor...")

# Tek seferde tüm veriyi oku ve gruplandır
for batch_num, record_batch in enumerate(scanner.to_batches()):
    df_chunk = record_batch.to_pandas()
    processed_rows += len(df_chunk)
    
    # Sadece hedef device_id'leri filtrele
    filtered_chunk = df_chunk[df_chunk["device_aid"].isin(target_device_ids)]
    
    if not filtered_chunk.empty:
        # Device_id'lere göre gruplandır
        for device_id, group in filtered_chunk.groupby("device_aid"):
            device_data[device_id].append(group)
    
    # İlerleme raporu
    if batch_num % 10 == 0:
        print(f"İşlenen batch: {batch_num + 1}, İşlenen satır: {processed_rows:,}")

print(f"Toplam işlenen satır: {processed_rows:,}")
print(f"Veri bulunan device_id sayısı: {len(device_data)}")

# Her device_id için dosya oluştur
print("CSV dosyaları oluşturuluyor...")
saved_count = 0

for device_id, data_chunks in device_data.items():
    if data_chunks:
        # Tüm chunk'ları birleştir
        result_df = pd.concat(data_chunks, ignore_index=True)
        
        # CSV olarak kaydet
        filename = f"{device_id}_data.csv"
        result_df.to_csv(filename, index=False)
        
        saved_count += 1
        print(f"Kaydedildi: {filename} ({len(result_df)} satır)")

print(f"\nToplam {saved_count} dosya oluşturuldu.")

# Veri bulunamayan device_id'leri göster
missing_devices = target_device_ids - set(device_data.keys())
if missing_devices:
    print(f"\nVeri bulunamayan {len(missing_devices)} device_id:")
    for device_id in list(missing_devices)[:10]:  # İlk 10'unu göster
        print(f"  - {device_id}")
    if len(missing_devices) > 10:
        print(f"  ... ve {len(missing_devices) - 10} tane daha")