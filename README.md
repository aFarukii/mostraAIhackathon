# Mobility & Spatial Analysis Suite 🚀

Kapsamlı mobil veri analizi ve mekan-kullanıcı etkileşim analizi için geliştirilmiş entegre Python paket suiti. Büyük veri setleriyle çalışırken performans odaklı çözümler sunar ve kullanıcı davranış profillerini detaylı analiz eder.

## 📋 İçindekiler

- [Genel Bakış](#genel-bakış)
- [Kurulum](#kurulum)
- [Ana Modüller](#ana-modüller)
  - [1. Mobility Data Analysis Scripts](#1-mobility-data-analysis-scripts)
  - [2. Fast Spatial Social Analyzer](#2-fast-spatial-social-analyzer)
  - [3. Venue Analysis & Rich Score Calculator](#3-venue-analysis--rich-score-calculator)
- [Veri Formatları](#veri-formatları)
- [Tam Analiz Pipeline](#tam-analiz-pipeline)
- [Performans ve Optimizasyon](#performans-ve-optimizasyon)
- [İş Uygulamaları](#iş-uygulamaları)

## 🎯 Genel Bakış

Bu suite üç ana bileşenden oluşmaktadır:

1. **Mobility Data Processing**: Ham mobil veri işleme ve mekan eşleştirme
2. **Spatial Social Analysis**: Hızlı kullanıcı segmentasyonu ve hotspot analizi
3. **Rich Score Calculation**: Zenginlik skorları ve lifestyle persona analizi

### Temel Özellikler

- **🚀 Yüksek Performans**: Vectorized işlemler ve spatial indexing
- **📊 Kapsamlı Analiz**: Kullanıcı profilleri, mekan analizi, zaman desenleri
- **🎯 Segmentasyon**: Machine learning ile otomatik kullanıcı grupları
- **💰 Zenginlik Skoru**: Mekan tercihlerine göre gelir seviyesi tahmini
- **🗺️ Görselleştirme**: İnteraktif haritalar ve comprehensive raporlar
- **⚡ Memory Efficient**: Büyük veri setleri için optimize edilmiş

## 🔧 Kurulum

### Gerekli Kütüphaneler
```bash
pip install pandas numpy matplotlib seaborn scikit-learn folium tqdm pyarrow
```

### Dosya Yapısı
```
project/
├── # Mobility Data Analysis
├── getunique_ids.py
├── getLocationsbyID.py
├── matcher.py
├── cofee_visits.py
├── vet_visiter.py
├── merger.py
├── # Spatial Social Analysis
├── user_profile_analyzer.py
├── # Venue Analysis & Rich Score
├── venue_analyzer.py
├── richscorecalc.py
├── # Veri Dosyaları
├── MobilityDataMay2024.parquet
├── eslesmeler.csv
├── maindataplaces.csv
├── Hackathon_MainData.xlsx
├── modified_places.csv
├── kahveci2.csv
├── veterinerlerson.csv
└── output/
    ├── analysis_results/
    └── visualizations/
```

## 📊 Ana Modüller

## 1. Mobility Data Analysis Scripts

### Amaç
Ham mobil cihaz konum verilerini işleyerek kullanıcıların özel mekan türlerine (kahveci, veteriner) ziyaretlerini tespit eder.

### Ana Scriptler

#### 1.1 `getunique_ids.py`
**Amaç:** Parquet dosyasından benzersiz cihaz ID'lerini çıkarır.

```bash
python getunique_ids.py
```

**Özellikler:**
- Büyük parquet dosyalarını batch'ler halinde işler
- Memory-efficient okuma
- Benzersiz device_aid değerlerini toplar

#### 1.2 `getLocationsbyID.py`
**Amaç:** Belirli cihaz ID'leri için konum verilerini çıkarır.

```bash
python getLocationsbyID.py
```

**Çıktı:** Her cihaz için ayrı `{device_id}_data.csv` dosyası

#### 1.3 `matcher.py`
**Amaç:** BallTree algoritması ile cihaz konumları ve mekan koordinatlarını eşleştirir.

```python
from matcher import toplu_eslestirme

toplu_eslestirme(
    mekanlar_dosyasi="clean_places.csv",
    devices_klasoru="devices",
    cikti_dosyasi="eslesmeler.csv",
    esik_metre=30
)
```

**Algoritma Avantajları:**
- O(log n) arama kompleksitesi
- Haversine distance metriği
- Spatial indexing ile hızlı konum eşleştirme

#### 1.4 `cofee_visits.py`
**Amaç:** Kullanıcıların kahveci ziyaretlerini tespit eder.

```bash
python cofee_visits.py
```

**Özellikler:**
- 100 metre yarıçapında kahveci tespiti
- Zaman damgası dönüştürme (UTC → GMT+3)
- BallTree ile hızlı spatial analysis

#### 1.5 `vet_visiter.py`
**Amaç:** Veteriner klinikleri ziyaretlerini kapsamlı analiz eder.

```bash
python vet_visiter.py
```

**Çıktılar:**
- `vet_visits_balltree.csv` - Cihaz bazında veteriner ziyaretleri
- `veteriner_detailed_analysis.csv` - Veteriner bazında detaylı analiz
- `veteriner_analysis_summary.csv` - Genel özet istatistikleri

#### 1.6 `merger.py`
**Amaç:** Farklı analizlerden gelen sonuçları birleştirir.

```bash
python merger.py
```

## 2. Fast Spatial Social Analyzer

### Amaç
Mekan-kullanıcı etkileşim verilerini hızlı ve etkili analiz ederek kullanıcı davranış profillerini çıkarır.

### Temel Özellikler

#### 📊 Kullanıcı Profil Analizi
- **Hızlı Profil Oluşturma**: Vektörize edilmiş işlemlerle kullanıcı davranış profilleri
- **Zenginlik Skoru**: Mekan tercihlerine göre gelir seviyesi tahmini
- **Aktivite Desenleri**: Zaman bazlı davranış analizi
- **Seyahat Davranışı**: Kullanıcıların hareket alanı analizi

#### 🎯 Kullanıcı Segmentasyonu
**K-Means Clustering ile 5 Ana Segment:**
- 🍽️ **Casual Diners**: Gündelik yemek sevenler
- 🍺 **Bar Hoppers**: Bar ve pub müdavimleri  
- 🥂 **Fine Diners**: Kaliteli restoran tercih edenler
- 🏨 **Hotel Guests**: Otel konaklayanlar
- 🗺️ **Local Explorers**: Yerel keşifçiler

### Kullanım

#### Hızlı Başlangıç
```python
from user_profile_analyzer import run_fast_analysis

# Tam analizi çalıştır
results = run_fast_analysis("eslesmeler.csv", "maindataplaces.csv")
```

#### Manuel Kullanım
```python
from user_profile_analyzer import FastSpatialSocialAnalyzer
import pandas as pd

# Veri yükleme
eslesmeler = pd.read_csv("eslesmeler.csv")
mekanlar = pd.read_csv("maindataplaces.csv")

# Analyzer oluşturma
analyzer = FastSpatialSocialAnalyzer(eslesmeler, mekanlar)

# Analizler
user_profiles = analyzer.create_user_profiles_fast()
segments = analyzer.segment_users_fast()
venue_analysis = analyzer.analyze_venues_fast()
time_patterns = analyzer.analyze_time_patterns_fast()
hotspots = analyzer.find_hotspots_fast()

# İnteraktif harita
wealth_map = analyzer.create_wealth_map_fast()
wealth_map.save("wealth_map.html")
```

### MapinSegment Kodlaması
- **D**: Bar/Pub (örn: D3-A)
- **R**: Restoran (örn: R2-B)  
- **H**: Otel (örn: H4-A)
- **Sayı**: Seviye (1-5)
- **Harf**: Kalite (A/B/C)

## 3. Venue Analysis & Rich Score Calculator

### Amaç
Kullanıcıların mekan ziyaret verilerini analiz ederek zenginlik skorları ve lifestyle persona'larını hesaplar.

### 3.1 Venue Analyzer (`venue_analyzer.py`)

#### Puanlama Sistemi
```python
# Segment bazlı ağırlıklar
segment_weights = {
    'Otel':     {'H0': 3, 'H1': 5, 'H2': 8},
    'Restoran': {'A': 7, 'B': 4, 'C': 2},
    'Bar':      {'A': 6, 'B': 3, 'C': 2}
}

# Harcama bonusları
spending_bonus_config = {
    'Otel':     {'0-499': 0, '500-999': 1, '1000-1999': 2, '2000+': 4},
    'Restoran': {'0-499': 0, '500-999': 0.5, '1000-1999': 1.5, '2000+': 3},
    'Bar':      {'0-499': 0, '500-999': 0.5, '1000-1999': 1, '2000+': 2}
}
```

### 3.2 Rich Score Calculator (`richscorecalc.py`)

#### Yeni Zenginlik Skoru Formülü (Kahve & Pet Odaklı)

**Ana Sürücüler (%40 ağırlık):**
- Kahve ziyaretleri: %60
- Günlük kahve frekansı: %10
- Veteriner ziyaretleri: %15
- Günlük pet bakım frekansı: %2

**Geleneksel Zenginlik Göstergeleri (%35):**
- Genel zenginlik skoru: %15
- Toplam mekan puanı: %10
- High-level venue ziyaretleri: %10

#### Persona Kategorileri

**1. Kahve Persona:**
- **Kahve Bağımlısı**: 25+ ziyaret veya günlük 0.7+
- **Kahve Sever**: 15+ ziyaret veya günlük 0.4+
- **Ara Sıra İçer**: 5+ ziyaret
- **Kahve İçmez**: < 5 ziyaret

**2. Hayvan Persona:**
- **Hayvan Tutkunu**: 15+ vet ziyareti veya günlük 0.12+
- **Aktif Pet Sahibi**: 8+ vet ziyareti
- **Pet Sahibi**: 3+ vet ziyareti
- **Pet Sahibi Değil**: < 3 vet ziyareti

**3. Zenginlik Segmenti:**
- **Lüks Segment**: 75+ puan
- **Üst Segment**: 55-74 puan
- **Orta Segment**: 35-54 puan
- **Ekonomik Segment**: < 35 puan

## 📁 Veri Formatları

### Girdi Veri Formatları

**Mobility Data (Parquet/CSV):**
```
device_aid,timestamp,latitude,longitude
ABC123,1684567890,41.0082,28.9784
DEF456,1684567920,41.0085,28.9787
```

**eslesmeler.csv (Kullanıcı-Mekan Eşleşmeleri):**
```
device_aid,gidilen_mekan,timestamp,mesafe_m
ABC123,Starbucks Taksim,1684567890,25
```

**maindataplaces.csv (Mekan Bilgileri):**
```
MusteriTabelaAdi,MapinSegment,lat,lng
Starbucks Taksim,R3-A,41.0370,28.9857
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

### Çıktı Dosyaları

**Ana Çıktılar:**
- `kullanici_profilleri_fast.csv` - Detaylı kullanıcı profilleri
- `venue_analysis_results.csv` - Mekan puanlaması sonuçları  
- `optimized_device_personas.csv` - Kullanıcı persona'ları
- `coffee_visits_balltree.csv` - Kahveci ziyaret analizi
- `vet_visits_balltree.csv` - Veteriner ziyaret analizi
- `merged_dataset.csv` - Birleştirilmiş master dataset
- `zenginlik_haritasi_fast.html` - İnteraktif zenginlik haritası

## 🚀 Tam Analiz Pipeline

### Komple Analiz Süreci
```bash
# 1. Ham mobility verisini işle
python getunique_ids.py          # Benzersiz ID'leri çıkar  
python getLocationsbyID.py       # Cihaz verilerini ayır
python matcher.py                # Mekan eşleştirme yap

# 2. Özel mekan analizleri
python cofee_visits.py           # Kahveci ziyaret analizi
python vet_visiter.py            # Veteriner ziyaret analizi

# 3. Venue puanlama ve zenginlik skoru
python venue_analyzer.py         # Mekan puanlama sistemi
python richscorecalc.py          # Zenginlik skorları ve persona'lar

# 4. Veri birleştirme
python merger.py                 # Tüm sonuçları birleştir

# 5. Hızlı spatial analiz
python -c "from user_profile_analyzer import run_fast_analysis; run_fast_analysis('eslesmeler.csv', 'maindataplaces.csv')"
```

### Python API Kullanımı
```python
# Tam entegre analiz
from user_profile_analyzer import run_fast_analysis
from venue_analyzer import analyze_venues
from richscorecalc import calculate_rich_scores

# 1. Spatial analiz
spatial_results = run_fast_analysis("eslesmeler.csv", "maindataplaces.csv")

# 2. Venue analizi
venue_results = analyze_venues("eslesmeler.csv", "Hackathon_MainData.xlsx")

# 3. Rich score hesaplama
rich_scores = calculate_rich_scores("merged_dataset.csv")

print("✅ Tam analiz tamamlandı!")
```

## ⚡ Performans ve Optimizasyon

### Memory Optimization
- **Batch Processing**: Büyük dosyalar için chunk-based okuma
- **Vectorized Operations**: Pandas/NumPy optimizasyonları
- **Efficient Data Types**: Memory footprint minimize etme
- **Garbage Collection**: Otomatik bellek temizliği

### Spatial Indexing
- **BallTree Algorithm**: O(log n) complexity
- **Haversine Distance**: GPS koordinatları için optimize
- **Radius Queries**: Çoklu nokta sorgulama
- **Parallel Processing**: Multi-core utilization

### Performans Metrikleri
```
📊 Benchmark Sonuçları:
├── 1M konum noktası: ~30 saniye
├── 10K mekan noktası: ~2 saniye  
├── 100K kullanıcı profili: ~45 saniye
├── Hotspot detection: ~15 saniye
└── Memory usage: ~500MB peak
```

### Konfigürasyon Seçenekleri
```python
# Segment sayısı ayarlama
segments = analyzer.segment_users_fast(n_clusters=7)  # Varsayılan: 5

# Hotspot yarıçapı
hotspots = analyzer.find_hotspots_fast(radius_km=1.0)  # Varsayılan: 0.5

# Harita sample boyutu
wealth_map = analyzer.create_wealth_map_fast(sample_size=1000)  # Varsayılan: 500

# Mesafe threshold'u
RADIUS_KM = 0.05  # cofee_visits.py için 50 metre
RADIUS_KM = 0.03  # vet_visiter.py için 30 metre
```

## 🎯 İş Uygulamaları

### Pazarlama Segmentasyonu
```python
import pandas as pd

# Master dataset yükle
df = pd.read_csv("optimized_device_personas.csv")

# Zengin kahve bağımlıları (Premium kahve markaları için)
rich_coffee_addicts = df[
    (df['simple_kahve_persona'] == 'Kahve Bağımlısı') & 
    (df['simple_zenginlik_segmenti'].isin(['Lüks Segment', 'Üst Segment']))
]

# Kahve + Pet kombinasyonu (Premium pet ürünleri + kahve için)
coffee_pet_combo = df[
    (df['simple_kahve_persona'].isin(['Kahve Bağımlısı', 'Kahve Sever'])) & 
    (df['simple_hayvan_persona'].isin(['Hayvan Tutkunu', 'Aktif Pet Sahibi']))
]

# Keşifçi gurmeler (Yeni restoran önerileri için)
explorer_gourmets = df[
    (df['simple_keşif_persona'] == 'Mega Kaşif') & 
    (df['simple_yaşam_tarzı_persona'] == 'Gurme')
]
```

### Reklam Hedefleme Stratejileri

**1. Premium Kahve Markaları:**
- Target: Lüks Segment + Kahve Bağımlısı
- Mesaj: Exclusive coffee experience
- Kanal: High-end venue'larda digital advertising

**2. Pet Ürünleri:**
- Target: Hayvan Tutkunu + Üst Segment  
- Mesaj: Premium pet care solutions
- Kanal: Veteriner kliniklerinde partnership

**3. Restoran Önerileri:**
- Target: Mega Kaşif + Gurme
- Mesaj: Discover new culinary experiences
- Kanal: Food & travel apps

**4. Otel & Turizm:**
- Target: Hotel Guests + Fine Diners
- Mesaj: Luxury travel packages
- Kanal: Travel booking platforms

## 📊 Analiz Özet Raporu

### Örnek Çıktı
```
⚡ KAPSAMLI ANALİZ ÖZET RAPORU
============================================================
⏱️  Toplam İşlem Süresi: 8 dakika 45 saniye
👥 Toplam Kullanıcı: 45,231
📍 Toplam Ziyaret: 128,567  
🏢 Benzersiz Mekan: 3,421
☕ Kahveci Ziyareti: 23,456
🐕 Veteriner Ziyareti: 8,234
💰 Ortalama Zenginlik Skoru: 42.3
⏰ En Popüler Saat: 19:00
🔥 Hotspot Sayısı: 23

🎯 Kullanıcı Segmentasyonu:
  • Casual Diners: 18,492 (%40.9)
  • Bar Hoppers: 11,305 (%25.0)  
  • Fine Diners: 7,234 (%16.0)
  • Hotel Guests: 4,523 (%10.0)
  • Local Explorers: 3,677 (%8.1)

☕ Kahve Persona Dağılımı:
  • Kahve Bağımlısı: 8,234 (%18.2)
  • Kahve Sever: 12,456 (%27.5)
  • Ara Sıra İçer: 15,678 (%34.7)
  • Kahve İçmez: 8,863 (%19.6)

🐕 Pet Sahibi Dağılımı:
  • Hayvan Tutkunu: 2,341 (%5.2)
  • Aktif Pet Sahibi: 4,567 (%10.1)
  • Pet Sahibi: 8,234 (%18.2)
  • Pet Sahibi Değil: 30,089 (%66.5)

💰 Zenginlik Segmenti:
  • Lüks Segment: 3,456 (%7.6)
  • Üst Segment: 8,923 (%19.7)
  • Orta Segment: 18,567 (%41.1)
  • Ekonomik Segment: 14,285 (%31.6)
```

## 🔐 Veri Gizliliği ve Güvenlik

### Anonimizasyon
- Tüm veriler anonim `device_aid` ile işlenir
- Kişisel kimlik bilgisi saklanmaz
- Lokasyon verileri sadece istatistiksel analiz için kullanılır

### Veri Güvenilirlik Skoru
```python
# Güvenilirlik hesaplama
confidence_metrics = {
    'primary_fields': 0.70,    # Temel aktivite verileri
    'secondary_fields': 0.30,  # Ek bilgiler
    'thresholds': {
        'high': 80,      # Yüksek güvenilir
        'medium': 60,    # Orta güvenilir  
        'low': 40        # Düşük güvenilir
    }
}
```

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakınız.

## 📞 İletişim ve Destek

### Destek Kanalları
- **GitHub Issues**: Bug reports ve feature requests
- **Discussions**: Genel sorular ve topluluk desteği
- **Wiki**: Detaylı dokümantasyon ve örnekler
