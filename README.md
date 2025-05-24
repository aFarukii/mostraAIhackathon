# Mobility Data Analysis Scripts

Bu proje, mobil cihaz konum verilerini analiz ederek kullanıcıların kahveci ve veteriner kliniklerine ziyaretlerini tespit eden Python scriptlerini içermektedir. Spatial indexing ve machine learning teknikleri kullanılarak büyük veri setleri üzerinde hızlı ve etkili analizler yapılmaktadır.

## 📋 İçindekiler

- [Gereksinimler](#gereksinimler)
- [Kurulum](#kurulum)
- [Scriptler](#scriptler)
  - [1. getunique_ids.py](#1-getunique_idspy)
  - [2. getLocationsbyID.py](#2-getlocationsbyidpy)
  - [3. matcher.py](#3-matcherpy)
  - [4. cofee_visits.py](#4-cofee_visitspy)
  - [5. vet_visiter.py](#5-vet_visiterpy)
  - [6. merger.py](#6-mergerpy)
- [Veri Formatları](#veri-formatları)
- [Kullanım Örnekleri](#kullanım-örnekleri)
- [Performans ve Optimizasyon](#performans-ve-optimizasyon)

## 🔧 Gereksinimler

```bash
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
pyarrow>=10.0.0
tqdm>=4.60.0
```

## 📦 Kurulum

```bash
# Gerekli kütüphaneleri yükleyin
pip install pandas numpy scikit-learn pyarrow tqdm

# Projeyi klonlayın
git clone [repository-url]
cd mobility-data-analysis
```

## 📊 Scriptler

### 1. getunique_ids.py

**Amaç:** Parquet dosyasından benzersiz cihaz ID'lerini çıkarır ve CSV formatında kaydeder.

**Özellikler:**
- Büyük parquet dosyalarını batch'ler halinde işler
- Memory-efficient okuma
- Benzersiz device_aid değerlerini toplar

**Kullanım:**
```bash
python getunique_ids.py
```

**Girdi:**
- `MobilityDataMay2024.paraquet` - Ana mobility veri dosyası

**Çıktı:**
- `unique_device_ids.csv` - Benzersiz cihaz ID'leri listesi

**Kod Yapısı:**
```python
# Ana parquet dosyasını okur
dataset = ds.dataset(parafilepath, format="parquet")
scanner = dataset.scanner(batch_size=100_000)

# Her batch'i işleyerek unique ID'leri toplar
for record_batch in scanner.to_batches():
    df_chunk = record_batch.to_pandas()
    unique_ids.update(df_chunk["device_aid"].dropna().unique())
```

---

### 2. getLocationsbyID.py

**Amaç:** Belirli cihaz ID'leri için konum verilerini çıkarır ve her cihaz için ayrı CSV dosyası oluşturur.

**Özellikler:**
- Hedef cihaz ID'lerini filtreler
- Her cihaz için ayrı dosya oluşturur
- Batch processing ile memory optimization
- İlerleme takibi

**Kullanım:**
```bash
python getLocationsbyID.py
```

**Girdi:**
- `MobilityDataMay2024.paraquet` - Ana mobility veri dosyası
- `unique_device_ids.csv` - Hedef cihaz ID'leri

**Çıktı:**
- `{device_id}_data.csv` - Her cihaz için ayrı konum veri dosyası

**Algoritma:**
1. Hedef device_id'leri yükler
2. Parquet dosyasını batch'ler halinde okur
3. Her batch'i filtreler
4. Cihaz bazında gruplar
5. Her cihaz için ayrı CSV oluşturur

---

### 3. matcher.py

**Amaç:** BallTree algoritması kullanarak cihaz konumları ile mekan koordinatlarını eşleştirir.

**Özellikler:**
- Spatial indexing ile hızlı konum eşleştirme
- Haversine distance metriği
- Configurable distance threshold
- Toplu dosya işleme

**Kullanım:**
```python
from matcher import toplu_eslestirme

toplu_eslestirme(
    mekanlar_dosyasi="clean_places.csv",
    devices_klasoru="devices",
    cikti_dosyasi="eslesmeler.csv",
    esik_metre=30
)
```

**Algoritma Detayları:**
```python
def balltree_eslestirme(hareketler_df, mekanlar_df, esik_metre=50):
    # Koordinatları radyana çevir
    mekan_coords = np.radians(mekanlar_df[['lat', 'lng']].values)
    hareket_coords = np.radians(hareketler_df[['latitude', 'longitude']].values)
    
    # BallTree oluştur
    tree = BallTree(mekan_coords, metric='haversine')
    
    # Yakın mekanları bul
    indices, distances = tree.query_radius(hareket_coords, r=esik_radyan, return_distance=True)
```

**Performans:**
- O(log n) arama kompleksitesi
- Binlerce noktayı saniyeler içinde işler
- Memory-efficient spatial indexing

---

### 4. cofee_visits.py

**Amaç:** Kullanıcıların kahveci ziyaretlerini tespit eder ve analiz eder.

**Özellikler:**
- 100 metre yarıçapında kahveci tespiti
- Zaman damgası dönüştürme (UTC → GMT+3)
- Cihaz bazında ziyaret sayısı hesaplama
- BallTree ile hızlı spatial analysis

**Kullanım:**
```bash
python cofee_visits.py
```

**Girdi:**
- `devices/*.csv` - Cihaz konum dosyaları
- `kahveci2.csv` - Kahveci koordinatları (name, lat, lng)

**Çıktı:**
- `coffee_visits_balltree.csv` - Cihaz bazında kahveci ziyaret sayıları

**Analiz Süreci:**
1. Tüm cihaz verilerini birleştirir
2. Kahveci koordinatlarını yükler
3. BallTree ile 100m yarıçapında eşleştirme
4. Cihaz başına ziyaret sayısı hesaplama

**Veri İşleme:**
```python
# Timestamp dönüştürme
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Etc/GMT-3')

# BallTree ile yakınlık analizi
RADIUS_KM = 0.1  # 100 metre
tree = BallTree(coffee_coords, metric='haversine')
indices = tree.query_radius(user_coords, r=RADIUS_KM / EARTH_RADIUS_KM)
```

---

### 5. vet_visiter.py

**Amaç:** Kullanıcıların veteriner klinikleri ziyaretlerini kapsamlı analiz eder.

**Özellikler:**
- 20 metre hassasiyetle veteriner tespiti
- Detaylı zaman analizi (saatlik, günlük)
- Veteriner bazında popülerlik analizi
- Çoklu çıktı dosyası
- Comprehensive error handling

**Kullanım:**
```bash
python vet_visiter.py
```

**Girdi:**
- `devices/*.csv` - Cihaz konum dosyaları
- `veterinerlerson.csv` - Veteriner koordinatları

**Çıktı:**
- `vet_visits_balltree.csv` - Cihaz bazında veteriner ziyaretleri
- `veteriner_detailed_analysis.csv` - Veteriner bazında detaylı analiz
- `veteriner_analysis_summary.csv` - Genel özet istatistikleri

**Analiz Boyutları:**
```python
# Her veteriner için analiz
vet_analysis = {
    'total_visits': total_visits,
    'unique_visitors': unique_devices,
    'avg_visits_per_visitor': total_visits / unique_devices,
    'peak_hour': visits_by_hour.idxmax(),
    'peak_day': visits_by_day.idxmax(),
    'visits_per_day': total_visits / date_range_days
}
```

**İstatistiksel Çıktılar:**
- Toplam ziyaret sayısı
- Benzersiz ziyaretçi sayısı
- En yoğun saat/gün analizi
- Popülerlik sıralaması
- Zaman serisi analizi

---

### 6. merger.py

**Amaç:** Farklı analizlerden gelen sonuçları birleştirir ve master dataset oluşturur.

**Özellikler:**
- Multiple dataset merging
- Left join stratejisi
- Column name conflict resolution
- Data quality checks

**Kullanım:**
```bash
python merger.py
```

**Birleştirilen Veri Setleri:**
- `coffee_visits_balltree.csv` - Kahveci ziyaret verileri
- `venue_analysis_results.csv` - Mekan puanlama verileri
- `kullanici_profilleri_fast.csv` - Kapsamlı ziyaret profilleri
- `vet_visits_balltree.csv` - Veteriner ziyaret verileri

**Çıktı:**
- `merged_dataset.csv` - Birleştirilmiş master dataset

**Merge Stratejisi:**
```python
# Sıralı left join işlemleri
merged_df = df2  # Base dataset
merged_df = merged_df.merge(df1, on='device_aid', how='left')
merged_df = merged_df.merge(df3, on='device_aid', how='left')
merged_df = merged_df.merge(df4, on='device_aid', how='left')

# Column name conflict resolution
if 'avg_visits_per_day_x' in merged_df.columns:
    merged_df = merged_df.rename(columns={
        'avg_visits_per_day_x': 'coffee_avg_visits_per_day',
        'avg_visits_per_day_y': 'vet_avg_visits_per_day'
    })
```

## 📁 Veri Formatları

### Girdi Veri Formatları

**Mobility Data (Parquet/CSV):**
```
device_aid,timestamp,latitude,longitude
ABC123,1684567890,41.0082,28.9784
DEF456,1684567920,41.0085,28.9787
```

**Coffee Shops (kahveci2.csv):**
```
name,lat,lng
Starbucks Taksim,41.0370,28.9857
Kahve Dünyası Kadıköy,40.9903,29.0275
```

**Veterinary Clinics (veterinerlerson.csv):**
```
name,latitude,longitude
Pet Vet Clinic,41.0123,28.9654
Animal Hospital Kadıköy,40.9876,29.0123
```

### Çıktı Veri Formatları

**Coffee Visits:**
```
device_aid,coffee_visit_count
ABC123,5
DEF456,12
```

**Veterinary Analysis:**
```
veteriner_name,total_visits,unique_visitors,peak_hour,peak_day
Pet Vet Clinic,45,12,14,Monday
Animal Hospital,67,23,16,Wednesday
```

## 🚀 Kullanım Örnekleri

### Tam Analiz Pipeline

```bash
# 1. Benzersiz ID'leri çıkar
python getunique_ids.py

# 2. Cihaz verilerini ayır
python getLocationsbyID.py

# 3. Kahveci analizini çalıştır
python cofee_visits.py

# 4. Veteriner analizini çalıştır
python vet_visiter.py

# 5. Tüm sonuçları birleştir
python merger.py
```

### Özelleştirilmiş Analiz

```python
# Mesafe threshold'unu değiştir
# cofee_visits.py içinde:
RADIUS_KM = 0.05  # 50 metre

# vet_visiter.py içinde:
RADIUS_KM = 0.03  # 30 metre
```

## ⚡ Performans ve Optimizasyon

### Memory Optimization
- Batch processing ile memory kullanımını kontrol eder
- Chunk-based okuma ile büyük dosyaları işler
- Garbage collection ile bellek temizliği

### Spatial Indexing
- BallTree algoritması O(log n) complexity
- Haversine distance metric için optimize edilmiş
- Vectorized operations ile hızlandırılmış

### Processing Speed
- Parallel processing capabilities
- Optimized pandas operations
- Efficient coordinate transformations

### Örnek Performans Metrikleri:
- 1M konum noktası: ~30 saniye
- 10K mekan noktası: ~2 saniye
- Memory usage: ~500MB peak

## 🐛 Troubleshooting

### Yaygın Hatalar

1. **FileNotFoundError**: Girdi dosyalarının varlığını kontrol edin
2. **MemoryError**: Batch size'ı küçültün
3. **KeyError**: CSV column isimlerini kontrol edin
4. **Coordinate Issues**: Lat/lng değerlerinin geçerliliğini kontrol edin

### Debug Modu
```python
# Verbose output için
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 🤝 Katkıda Bulunma

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📞 İletişim

Herhangi bir sorunuz için issue açabilir veya [maintainer-email] adresinden iletişime geçebilirsiniz.
