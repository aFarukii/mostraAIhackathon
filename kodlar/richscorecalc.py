import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Veri setini oku
df = pd.read_csv('merged_dataset.csv')
df = df.fillna(0)  # Eksik değerleri 0 ile doldur

# Tüm sayısal sütunları skalanacak sütunlara dahil et
columns_to_scale = [
    'total_visits', 'unique_places', 'avg_distance', 'weekend_ratio',
    'morning_visits', 'high_level_venues', 'quality_A_visits', 'quality_B_visits',
    'bar_pub', 'restaurant', 'hotel', 'afternoon_visits', 'evening_visits',
    'travel_radius', 'venue_type_diversity', 'wealth_score',
    'coffee_visit_count', 'avg_visits_per_day_coffee', 'Otel_Puani',
    'Restoran_Puani', 'Bar_Puani', 'Toplam_Puan', 'Ziyaret_Edilen_Mekan_Sayisi',
    'Eslesme_Sayisi', 'Eslesme_Orani', 'vet_visit_count', 'avg_visits_per_day_vet'
]

# Sadece mevcut sütunları kullan
available_columns = [col for col in columns_to_scale if col in df.columns]
print(f"Skallanacak sütunlar: {available_columns}")

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[available_columns]), 
                        columns=[f"{col}_scaled" for col in available_columns])
df = pd.concat([df, df_scaled], axis=1)

# GELİŞMİŞ zenginlik skoru hesaplama - KAHVE VE PET ağırlığı artırıldı
def create_comprehensive_rich_score(row):
    score = 0
    
    # KAHVE VE PET - Ana sürücüler (Ağırlık: %40)
    score += row.get('coffee_visit_count_scaled', 0) * 0.6  # Kahve ağırlığı artırıldı
    score += row.get('avg_visits_per_day_coffee_scaled', 0) * 0.10  # Günlük kahve ağırlığı
    score += row.get('vet_visit_count_scaled', 0) * 0.15  # Pet ağırlığı artırıldı
    score += row.get('avg_visits_per_day_vet_scaled', 0) * 0.02  # Günlük pet
    
    # Geleneksel zenginlik göstergeleri (Ağırlık: %35)
    score += row.get('wealth_score_scaled', 0) * 0.15
    score += row.get('Toplam_Puan_scaled', 0) * 0.10
    score += row.get('high_level_venues_scaled', 0) * 0.10
    
    # Venue kalitesi ve puanları (Ağırlık: %15)
    score += row.get('Otel_Puani_scaled', 0) * 0.05
    score += row.get('Restoran_Puani_scaled', 0) * 0.05
    score += row.get('Bar_Puani_scaled', 0) * 0.03
    score += row.get('quality_A_visits_scaled', 0) * 0.02
    
    # Aktivite çeşitliliği ve mobilite (Ağırlık: %10)
    score += row.get('venue_type_diversity_scaled', 0) * 0.04
    score += row.get('travel_radius_scaled', 0) * 0.03
    score += row.get('unique_places_scaled', 0) * 0.02
    score += row.get('avg_distance_scaled', 0) * 0.01
    
    return round(min(score * 100, 100), 2)

# Gelişmiş veri güvenilirliği skoru
def calculate_comprehensive_data_confidence(row):
    # Tüm önemli alanları kontrol et
    primary_fields = [
        'total_visits', 'unique_places', 'coffee_visit_count', 'bar_pub',
        'restaurant', 'hotel', 'vet_visit_count', 'venue_type_diversity',
        'travel_radius', 'high_level_venues', 'quality_A_visits'
    ]
    
    secondary_fields = [
        'Otel_Puani', 'Restoran_Puani', 'Bar_Puani', 'wealth_score',
        'weekend_ratio', 'morning_visits', 'afternoon_visits', 'evening_visits'
    ]
    
    # Mevcut alanları kontrol et
    available_primary = [f for f in primary_fields if f in row.index]
    available_secondary = [f for f in secondary_fields if f in row.index]
    
    primary_filled = sum(row[f] > 0 for f in available_primary)
    secondary_filled = sum(row[f] > 0 for f in available_secondary)
    
    # Ağırlıklı güvenilirlik skoru
    primary_score = (primary_filled / len(available_primary)) * 0.7 if available_primary else 0
    secondary_score = (secondary_filled / len(available_secondary)) * 0.3 if available_secondary else 0
    
    return round((primary_score + secondary_score) * 100, 2)

# SADELEŞTIRILMIŞ persona fonksiyonları - MAX 4 KATEGORI
def determine_simple_coffee_persona(row):
    visits = row.get('coffee_visit_count', 0)
    avg_daily = row.get('avg_visits_per_day_coffee', 0)
    
    if visits >= 25 or avg_daily >= 0.7:
        return "Kahve Bağımlısı"
    elif visits >= 15 or avg_daily >= 0.4:
        return "Kahve Sever"
    elif visits >= 5:
        return "Ara Sıra İçer"
    else:
        return "Kahve İçmez"

def determine_simple_bar_persona(row):
    bar_visits = row.get('bar_pub', 0)
    bar_score = row.get('Bar_Puani', 0)
    evening_visits = row.get('evening_visits', 0)
    
    if bar_visits >= 15 and bar_score >= 3:
        return "Gece Hayatı Uzmanı"
    elif bar_visits >= 8:
        return "Bar Sever"
    elif bar_visits >= 3:
        return "Sosyal İçici"
    else:
        return "Bar Gitmez"

def determine_simple_explorer_persona(row):
    unique_places = row.get('unique_places', 0)
    diversity = row.get('venue_type_diversity', 0)
    radius = row.get('travel_radius', 0)
    
    if unique_places >= 40 and diversity >= 6:
        return "Mega Kaşif"
    elif unique_places >= 20 and diversity >= 4:
        return "Aktif Kaşif"
    elif unique_places >= 10:
        return "Keşif Sever"
    else:
        return "Rutin Sever"

def determine_simple_animal_lover_persona(row):
    vet_visits = row.get('vet_visit_count', 0)
    avg_daily_vet = row.get('avg_visits_per_day_vet', 0)
    
    if vet_visits >= 15 or avg_daily_vet >= 0.12:
        return "Hayvan Tutkunu"
    elif vet_visits >= 8:
        return "Aktif Pet Sahibi"
    elif vet_visits >= 3:
        return "Pet Sahibi"
    else:
        return "Pet Sahibi Değil"

def determine_simple_lifestyle_persona(row):
    restaurant_visits = row.get('restaurant', 0)
    hotel_visits = row.get('hotel', 0)
    evening_visits = row.get('evening_visits', 0)
    total_visits = row.get('total_visits', 1)
    evening_ratio = evening_visits / max(total_visits, 1)
    
    if hotel_visits >= 8 and restaurant_visits >= 15:
        return "Lüks Yaşam"
    elif restaurant_visits >= 15:
        return "Gurme"
    elif evening_ratio >= 0.4:
        return "Gece Kuşu"
    else:
        return "Dengeli Yaşam"

def determine_simple_wealth_segment(score):
    if score >= 75:
        return "Lüks Segment"
    elif score >= 55:
        return "Üst Segment"
    elif score >= 35:
        return "Orta Segment"
    else:
        return "Ekonomik Segment"

# Tüm hesaplamaları uygula
df['comprehensive_rich_score'] = df.apply(create_comprehensive_rich_score, axis=1)
df['comprehensive_data_confidence'] = df.apply(calculate_comprehensive_data_confidence, axis=1)
df['simple_kahve_persona'] = df.apply(determine_simple_coffee_persona, axis=1)
df['simple_bar_persona'] = df.apply(determine_simple_bar_persona, axis=1)
df['simple_keşif_persona'] = df.apply(determine_simple_explorer_persona, axis=1)
df['simple_hayvan_persona'] = df.apply(determine_simple_animal_lover_persona, axis=1)
df['simple_yaşam_tarzı_persona'] = df.apply(determine_simple_lifestyle_persona, axis=1)
df['simple_zenginlik_segmenti'] = df['comprehensive_rich_score'].apply(determine_simple_wealth_segment)

# Kapsamlı çıktı tablosu
comprehensive_persona_df = df[[
    'device_aid', 'comprehensive_rich_score', 'simple_zenginlik_segmenti', 
    'comprehensive_data_confidence', 'simple_kahve_persona', 'simple_bar_persona',
    'simple_keşif_persona', 'simple_hayvan_persona', 'simple_yaşam_tarzı_persona',
    'total_visits', 'unique_places', 'coffee_visit_count', 'bar_pub', 'restaurant',
    'hotel', 'vet_visit_count', 'wealth_score', 'venue_type_diversity', 
    'travel_radius', 'Toplam_Puan', 'high_level_venues'
]].round(2)

# Sonuçları kaydet
comprehensive_persona_df.to_csv('optimized_device_personas.csv', index=False)

# Detaylı özet raporu
print("=== OPTİMİZE EDİLMİŞ PERSONA ANALİZİ TAMAMLANDI ===")
print(f"Toplam cihaz sayısı: {len(comprehensive_persona_df)}")
print(f"Yüksek güvenilir cihaz (confidence > 80): {(comprehensive_persona_df['comprehensive_data_confidence'] > 80).sum()}")
print(f"Orta güvenilir cihaz (confidence 60-80): {((comprehensive_persona_df['comprehensive_data_confidence'] >= 60) & (comprehensive_persona_df['comprehensive_data_confidence'] <= 80)).sum()}")

print(f"\n=== KAHVE VE PET ODAKLI ZENGİNLİK SEGMENTİ DAĞILIMI ===")
wealth_dist = comprehensive_persona_df['simple_zenginlik_segmenti'].value_counts()
for segment, count in wealth_dist.items():
    percentage = (count / len(comprehensive_persona_df)) * 100
    print(f"{segment}: {count} ({percentage:.1f}%)")

print(f"\n=== SADELEŞTİRİLMİŞ PERSONA DAĞILIMLARI (MAX 4 KATEGORİ) ===")
personas = ['simple_kahve_persona', 'simple_bar_persona', 'simple_keşif_persona', 
           'simple_hayvan_persona', 'simple_yaşam_tarzı_persona']

for persona_col in personas:
    print(f"\n{persona_col.replace('simple_', '').replace('_persona', '').title()} Persona:")
    persona_dist = comprehensive_persona_df[persona_col].value_counts()
    for persona_type, count in persona_dist.items():  # Tüm kategorileri göster (max 4 olduğu için)
        percentage = (count / len(comprehensive_persona_df)) * 100
        print(f"  {persona_type}: {count} ({percentage:.1f}%)")

print(f"\n=== İSTATİSTİKLER (KAHVE VE PET ODAKLI) ===")
print(f"Ortalama zenginlik skoru: {comprehensive_persona_df['comprehensive_rich_score'].mean():.2f}")
print(f"En yüksek zenginlik skoru: {comprehensive_persona_df['comprehensive_rich_score'].max():.2f}")
print(f"Ortalama güvenilirlik skoru: {comprehensive_persona_df['comprehensive_data_confidence'].mean():.2f}")

print(f"\n=== KAHVE VE PET ODAKLI KOMBİNASYONLAR ===")
# Kahve bağımlısı zenginler
rich_coffee_addicts = len(comprehensive_persona_df[
    (comprehensive_persona_df['simple_kahve_persona'] == 'Kahve Bağımlısı') & 
    (comprehensive_persona_df['simple_zenginlik_segmenti'].isin(['Lüks Segment', 'Üst Segment']))
])
print(f"Zengin kahve bağımlıları: {rich_coffee_addicts}")

# Hayvan tutkunu zenginler
rich_pet_lovers = len(comprehensive_persona_df[
    (comprehensive_persona_df['simple_hayvan_persona'] == 'Hayvan Tutkunu') & 
    (comprehensive_persona_df['comprehensive_rich_score'] >= 55)
])
print(f"Zengin hayvan tutkunu: {rich_pet_lovers}")

# Kahve + Pet kombinasyonu
coffee_pet_combo = len(comprehensive_persona_df[
    (comprehensive_persona_df['simple_kahve_persona'].isin(['Kahve Bağımlısı', 'Kahve Sever'])) & 
    (comprehensive_persona_df['simple_hayvan_persona'].isin(['Hayvan Tutkunu', 'Aktif Pet Sahibi']))
])
print(f"Kahve sever + Pet sahibi: {coffee_pet_combo}")

# Lüks yaşam + Kahve bağımlısı
luxury_coffee = len(comprehensive_persona_df[
    (comprehensive_persona_df['simple_yaşam_tarzı_persona'] == 'Lüks Yaşam') & 
    (comprehensive_persona_df['simple_kahve_persona'] == 'Kahve Bağımlısı')
])
print(f"Lüks yaşam + Kahve bağımlısı: {luxury_coffee}")

# En yüksek skora sahip kahve severlerin istatistikleri
top_coffee_lovers = comprehensive_persona_df[
    comprehensive_persona_df['simple_kahve_persona'].isin(['Kahve Bağımlısı', 'Kahve Sever'])
]
print(f"\nKahve severler arasında ortalama zenginlik skoru: {top_coffee_lovers['comprehensive_rich_score'].mean():.2f}")

# En yüksek skora sahip pet sahiplerinin istatistikleri
top_pet_owners = comprehensive_persona_df[
    comprehensive_persona_df['simple_hayvan_persona'].isin(['Hayvan Tutkunu', 'Aktif Pet Sahibi'])
]
print(f"Pet sahipleri arasında ortalama zenginlik skoru: {top_pet_owners['comprehensive_rich_score'].mean():.2f}")

# Detaylı özet CSV'si oluştur
summary_data = []
summary_data.append(["Genel İstatistikler (Kahve & Pet Odaklı)", ""])
summary_data.append(["Toplam cihaz sayısı", len(comprehensive_persona_df)])
summary_data.append(["Yüksek güvenilir cihaz (>80)", (comprehensive_persona_df['comprehensive_data_confidence'] > 80).sum()])
summary_data.append(["Ortalama zenginlik skoru", round(comprehensive_persona_df['comprehensive_rich_score'].mean(), 2)])
summary_data.append(["En yüksek zenginlik skoru", comprehensive_persona_df['comprehensive_rich_score'].max()])

# Segment dağılımları
summary_data.append(["", ""])
summary_data.append(["Zenginlik Segmenti Dağılımı (Kahve & Pet Odaklı)", ""])
for segment, count in comprehensive_persona_df['simple_zenginlik_segmenti'].value_counts().items():
    percentage = (count / len(comprehensive_persona_df)) * 100
    summary_data.append([segment, f"{count} ({percentage:.1f}%)"])

# Her persona için dağılım (sadeleştirilmiş)
for persona_col in personas:
    summary_data.append(["", ""])
    persona_name = persona_col.replace('simple_', '').replace('_persona', '').title()
    summary_data.append([f"{persona_name} Persona Dağılımı", ""])
    for persona_type, count in comprehensive_persona_df[persona_col].value_counts().items():
        percentage = (count / len(comprehensive_persona_df)) * 100
        summary_data.append([persona_type, f"{count} ({percentage:.1f}%)"])

# Özel kombinasyonlar (Kahve & Pet odaklı)
summary_data.append(["", ""])
summary_data.append(["Kahve & Pet Odaklı Kombinasyonlar", ""])
summary_data.append(["Zengin kahve bağımlıları", rich_coffee_addicts])
summary_data.append(["Zengin hayvan tutkunu", rich_pet_lovers])
summary_data.append(["Kahve sever + Pet sahibi", coffee_pet_combo])
summary_data.append(["Lüks yaşam + Kahve bağımlısı", luxury_coffee])
summary_data.append(["Kahve severler ort. zenginlik skoru", round(top_coffee_lovers['comprehensive_rich_score'].mean(), 2)])
summary_data.append(["Pet sahipleri ort. zenginlik skoru", round(top_pet_owners['comprehensive_rich_score'].mean(), 2)])

# Özet CSV'sini kaydet
summary_df = pd.DataFrame(summary_data, columns=['Kategori', 'Değer'])
summary_df.to_csv("optimized_summary_stats.csv", index=False)

print(f"\n=== DOSYALAR KAYDEDILDI ===")
print("- optimized_device_personas.csv: Optimize edilmiş persona dosyası")
print("- optimized_summary_stats.csv: Kahve & Pet odaklı özet istatistikleri")
print("\nAnaliz tamamlandı! Kahve ve pet ziyaretleri zenginlik skorunda ana sürücü haline getirildi.")
print("Her persona kategorisi maksimum 4 alt kategoriye düşürüldü.")