import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm

# --- Ayarlar Başlangıcı ---

# 0. VERİ DOSYALARININ BULUNDUĞU TEMEL KLASÖR YOLU
data_base_path = ""

# 1. TÜM KULLANICI AKTİVİTELERİNİ İÇEREN BİRLEŞİK CSV DOSYASI
consolidated_activity_file_name = "eslesmeler.csv"
consolidated_activity_file_path = os.path.join(data_base_path, consolidated_activity_file_name)

# 2. BİRLEŞİK AKTİVİTE DOSYASINDAKİ SÜTUN ADLARI
device_id_col_in_activities = 'device_aid'
musteri_tabela_adi_aktivite_col = 'gidilen_mekan'

# 3. Ana Mekan Veri Dosyası ve İlgili Sütun Adları
main_venue_file_name = "Hackathon_MainData.xlsx"
main_venue_data_config = {
    'path': os.path.join(data_base_path, main_venue_file_name),
    'mekan_adi_col': 'MusteriTabelaAdi',
    'segment_col': 'Mapin Segment',
    'harcama_col': 'OrtalamaHarcamaTutari'
}

# 4. İkincil Mekan Veri Dosyası ve İlgili Sütun Adları
secondary_venue_file_name = "modified_places.csv"
secondary_venue_data_config = {
    'path': os.path.join(data_base_path, secondary_venue_file_name),
    'mekan_adi_col': 'isim',
    'type_col_name_or_index': 'tur'
}

# 5. Segment Bazlı Zenginlik Ağırlıkları
segment_weights = {
    'Otel':     {'H0': 3, 'H1': 5, 'H2': 8},
    'Restoran': {'A': 7, 'B': 4, 'C': 2},
    'Bar':      {'A': 6, 'B': 3, 'C': 2}
}
TARGET_VENUE_TYPES_MAIN = ['Otel', 'Restoran', 'Bar']

# 6. Ortalama Harcama Tutarına Göre Bonus Puanlar
spending_bonus_config = {
    'Otel':     {'0-499': 0, '500-999': 1, '1000-1999': 2, '2000+': 4, 'default': 0},
    'Restoran': {'0-499': 0, '500-999': 0.5, '1000-1999': 1.5, '2000+': 3, 'default': 0},
    'Bar':      {'0-499': 0, '500-999': 0.5, '1000-1999': 1, '2000+': 2, 'default': 0}
}

# 7. İkincil Mekan Türleri İçin Sabit Puanlar
secondary_place_fixed_weights = {
    'kahve': 1.0,
    'veteriner': 4.5
}
TARGET_VENUE_TYPES_SECONDARY = ['kahve', 'veteriner']

# 8. ÇIKTI DOSYASI
output_file = "venue_analysis_results.csv"

# --- Ayarlar Sonu ---

def preprocess_text_for_matching(series):
    if series is None: 
        return pd.Series(dtype='object')
    return series.astype(str).str.lower().str.strip()

def get_venue_type_and_quality_from_segment_vectorized(segments):
    """Vectorized version of segment processing"""
    segments = pd.Series(segments).fillna('')
    
    # Initialize result arrays
    venue_types = np.full(len(segments), None, dtype=object)
    quality_keys = np.full(len(segments), None, dtype=object)
    weights = np.zeros(len(segments))
    
    # Process Hotel segments (H*)
    hotel_mask = segments.str.startswith('H', na=False)
    for segment, weight in segment_weights['Otel'].items():
        mask = hotel_mask & (segments == segment)
        venue_types[mask] = 'Otel'
        quality_keys[mask] = segment
        weights[mask] = weight
    
    # Process Restaurant segments (R*)
    restaurant_mask = segments.str.startswith('R', na=False)
    for quality, weight in segment_weights['Restoran'].items():
        mask = restaurant_mask & segments.str.endswith(quality, na=False)
        venue_types[mask] = 'Restoran'
        quality_keys[mask] = quality
        weights[mask] = weight
    
    # Process Bar segments (D*)
    bar_mask = segments.str.startswith('D', na=False)
    for quality, weight in segment_weights['Bar'].items():
        mask = bar_mask & segments.str.endswith(quality, na=False)
        venue_types[mask] = 'Bar'
        quality_keys[mask] = quality
        weights[mask] = weight
    
    return venue_types, quality_keys, weights

def get_spending_bonus_vectorized(venue_types, spending_ranges):
    """Vectorized version of spending bonus calculation"""
    spending_ranges = pd.Series(spending_ranges).fillna('').astype(str)
    venue_types = pd.Series(venue_types).fillna('')
    
    # Clean spending ranges
    spending_cleaned = (spending_ranges.str.lower()
                       .str.replace('tl', '', regex=False)
                       .str.replace('.', '', regex=False)
                       .str.replace(' ', '', regex=False)
                       .str.strip())
    
    bonuses = np.zeros(len(venue_types))
    
    for venue_type, bonus_dict in spending_bonus_config.items():
        venue_mask = venue_types == venue_type
        if not venue_mask.any():
            continue
            
        for range_key, bonus_value in bonus_dict.items():
            if range_key == 'default':
                # Apply default bonus where no other bonus was applied
                default_mask = venue_mask & (bonuses == 0) & (spending_cleaned == '')
                bonuses[default_mask] = bonus_value
            else:
                range_mask = venue_mask & (spending_cleaned == range_key)
                bonuses[range_mask] = bonus_value
    
    return bonuses

def load_venue_data():
    """Mekan verilerini yükle ve lookup dictionary'leri oluştur"""
    print("Mekan verileri yükleniyor...")
    
    main_venue_lookup = {}
    secondary_venue_lookup = {}
    
    # Ana mekan verisi
    if os.path.exists(main_venue_data_config['path']):
        try:
            if main_venue_data_config['path'].endswith('.csv'):
                main_venues_df = pd.read_csv(main_venue_data_config['path'], dtype=str)
            elif main_venue_data_config['path'].endswith('.xlsx'):
                main_venues_df = pd.read_excel(main_venue_data_config['path'], dtype=str)
            
            if main_venues_df is not None and not main_venues_df.empty:
                mekan_adi_col = main_venue_data_config['mekan_adi_col']
                segment_col = main_venue_data_config['segment_col']
                harcama_col = main_venue_data_config['harcama_col']
                
                if mekan_adi_col in main_venues_df.columns:
                    # Create lookup dictionary for faster access
                    for _, row in main_venues_df.iterrows():
                        mekan_adi = str(row[mekan_adi_col]).lower().strip()
                        if mekan_adi and mekan_adi != 'nan':
                            main_venue_lookup[mekan_adi] = {
                                'segment': row.get(segment_col, ''),
                                'spending': row.get(harcama_col, '')
                            }
                    print(f"Ana mekan verisi yüklendi: {len(main_venue_lookup)} kayıt")
        except Exception as e:
            print(f"Ana mekan dosyası yüklenirken hata: {e}")

    # İkincil mekan verisi
    if os.path.exists(secondary_venue_data_config['path']):
        try:
            secondary_venues_df = pd.read_csv(secondary_venue_data_config['path'], dtype=str)
            if not secondary_venues_df.empty:
                type_col_id = secondary_venue_data_config['type_col_name_or_index']
                mekan_adi_col = secondary_venue_data_config['mekan_adi_col']
                
                for _, row in secondary_venues_df.iterrows():
                    mekan_adi = str(row[mekan_adi_col]).lower().strip()
                    venue_type = None
                    
                    if isinstance(type_col_id, int):
                        if 0 <= type_col_id < len(row):
                            venue_type = str(row.iloc[type_col_id]).lower().strip()
                    elif isinstance(type_col_id, str) and type_col_id in row:
                        venue_type = str(row[type_col_id]).lower().strip()
                    
                    if mekan_adi and mekan_adi != 'nan' and venue_type in TARGET_VENUE_TYPES_SECONDARY:
                        secondary_venue_lookup[mekan_adi] = {
                            'type': venue_type,
                            'weight': secondary_place_fixed_weights.get(venue_type, 0)
                        }
                
                print(f"İkincil mekan verisi yüklendi: {len(secondary_venue_lookup)} kayıt")
        except Exception as e:
            print(f"İkincil mekan dosyası yüklenirken hata: {e}")

    return main_venue_lookup, secondary_venue_lookup

def process_activities_vectorized(activities_df, main_venue_lookup, secondary_venue_lookup):
    """Tüm aktiviteleri vectorized olarak işle ve cihaz bazında topla"""
    print("Aktiviteler işleniyor (vectorized)...")
    
    # Preprocessing
    activities_clean = activities_df[[device_id_col_in_activities, musteri_tabela_adi_aktivite_col]].copy()
    activities_clean = activities_clean.dropna()
    activities_clean['mekan_processed'] = activities_clean[musteri_tabela_adi_aktivite_col].astype(str).str.lower().str.strip()
    
    n_activities = len(activities_clean)
    print(f"İşlenecek aktivite sayısı: {n_activities}")
    
    # Initialize result arrays for individual activities
    activities_results = {
        'device_aid': activities_clean[device_id_col_in_activities].values,
        'mekan_adi': activities_clean[musteri_tabela_adi_aktivite_col].values,
        'venue_type': np.full(n_activities, None, dtype=object),
        'segment': np.full(n_activities, None, dtype=object),
        'base_weight': np.zeros(n_activities),
        'spending_bonus': np.zeros(n_activities),
        'total_weight': np.zeros(n_activities),
        'match_source': np.full(n_activities, "Eşleşme Yok", dtype=object),
        'spending_range': np.full(n_activities, None, dtype=object)
    }
    
    # Process main venues
    print("Ana mekan eşleştirmeleri işleniyor...")
    segments_list = []
    spending_list = []
    main_match_indices = []
    
    for idx, mekan_processed in enumerate(tqdm(activities_clean['mekan_processed'], desc="Ana mekan lookup")):
        if mekan_processed in main_venue_lookup:
            venue_data = main_venue_lookup[mekan_processed]
            segments_list.append(venue_data['segment'])
            spending_list.append(venue_data['spending'])
            main_match_indices.append(idx)
        else:
            segments_list.append('')
            spending_list.append('')
    
    if main_match_indices:
        # Vectorized processing for main venues
        venue_types, quality_keys, base_weights = get_venue_type_and_quality_from_segment_vectorized(segments_list)
        spending_bonuses = get_spending_bonus_vectorized(venue_types, spending_list)
        
        # Update results for matched main venues
        valid_matches = [i for i in main_match_indices if venue_types[i] is not None]
        
        for idx in valid_matches:
            activities_results['venue_type'][idx] = venue_types[idx]
            activities_results['segment'][idx] = segments_list[idx]
            activities_results['base_weight'][idx] = base_weights[idx]
            activities_results['spending_bonus'][idx] = spending_bonuses[idx]
            activities_results['total_weight'][idx] = base_weights[idx] + spending_bonuses[idx]
            activities_results['match_source'][idx] = "Ana Mekan"
            activities_results['spending_range'][idx] = spending_list[idx] if spending_list[idx] else None
    
    # Process secondary venues for unmatched activities
    print("İkincil mekan eşleştirmeleri işleniyor...")
    unmatched_mask = activities_results['total_weight'] == 0
    unmatched_indices = np.where(unmatched_mask)[0]
    
    for idx in tqdm(unmatched_indices, desc="İkincil mekan lookup"):
        mekan_processed = activities_clean.iloc[idx]['mekan_processed']
        if mekan_processed in secondary_venue_lookup:
            venue_data = secondary_venue_lookup[mekan_processed]
            activities_results['venue_type'][idx] = venue_data['type'].capitalize()
            activities_results['total_weight'][idx] = venue_data['weight']
            activities_results['match_source'][idx] = "İkincil Mekan"
            activities_results['segment'][idx] = "N/A"
    
    # Convert to DataFrame for processing
    activities_df_processed = pd.DataFrame(activities_results)
    
    # CİHAZ BAZINDA TOPLAMA İŞLEMİ
    print("Cihaz bazında puanlar toplanıyor...")
    
    # Her cihaz için puanları topla
    device_scores = {}
    venue_type_columns = ['Otel_Puani', 'Restoran_Puani', 'Bar_Puani', 'Kahve_Puani', 'Veteriner_Puani']
    
    # Her cihaz için grupla
    grouped = activities_df_processed.groupby('device_aid')
    
    device_results = []
    
    for device_id, group in tqdm(grouped, desc="Cihaz bazında toplama"):
        # Her cihaz için sıfır puanlarla başla
        scores = {
            'device_aid': device_id,
            'Otel_Puani': 0.0,
            'Restoran_Puani': 0.0,
            'Bar_Puani': 0.0,
            'Kahve_Puani': 0.0,
            'Veteriner_Puani': 0.0,
            'Toplam_Puan': 0.0,
            'Ziyaret_Edilen_Mekan_Sayisi': len(group),
            'Eslesme_Sayisi': len(group[group['total_weight'] > 0]),
            'Eslesme_Orani': len(group[group['total_weight'] > 0]) / len(group) * 100
        }
        
        # Her mekan türü için puanları topla
        for _, row in group.iterrows():
            venue_type = row['venue_type']
            total_weight = row['total_weight']
            
            if pd.notna(venue_type) and total_weight > 0:
                if venue_type == 'Otel':
                    scores['Otel_Puani'] += total_weight
                elif venue_type == 'Restoran':
                    scores['Restoran_Puani'] += total_weight
                elif venue_type == 'Bar':
                    scores['Bar_Puani'] += total_weight
                elif venue_type == 'Kahve':
                    scores['Kahve_Puani'] += total_weight
                elif venue_type == 'Veteriner':
                    scores['Veteriner_Puani'] += total_weight
        
        # Toplam puanı hesapla
        scores['Toplam_Puan'] = (scores['Otel_Puani'] + scores['Restoran_Puani'] + 
                               scores['Bar_Puani'] + scores['Kahve_Puani'] + 
                               scores['Veteriner_Puani'])
        
        device_results.append(scores)
    
    return pd.DataFrame(device_results)

def main():
    print("Venue Analysis Script Başlatılıyor (Device-Based Scoring)...")
    
    # Mekan verilerini yükle ve lookup dictionary'leri oluştur
    main_venue_lookup, secondary_venue_lookup = load_venue_data()
    
    # Aktivite dosyasını yükle
    if not os.path.exists(consolidated_activity_file_path):
        print(f"Aktivite dosyası bulunamadı: {consolidated_activity_file_path}")
        return
    
    print("Aktivite dosyası yükleniyor...")
    try:
        # Chunk-wise reading for large files
        chunk_size = 50000
        all_results = []
        
        for chunk in tqdm(pd.read_csv(consolidated_activity_file_path, chunksize=chunk_size), 
                         desc="Chunks işleniyor"):
            # Gerekli sütunları kontrol et
            required_cols = [device_id_col_in_activities, musteri_tabela_adi_aktivite_col]
            if not all(col in chunk.columns for col in required_cols):
                print(f"Eksik sütunlar bulundu")
                continue
            
            # Bu chunk'ı işle
            chunk_results = process_activities_vectorized(chunk, main_venue_lookup, secondary_venue_lookup)
            all_results.append(chunk_results)
        
        # Tüm sonuçları birleştir ve final toplama yap
        if all_results:
            # Chunk'ları birleştir
            combined_chunks = pd.concat(all_results, ignore_index=True)
            
            # Son bir kez cihaz bazında topla (chunk'lar arası duplikasyon için)
            print("Final cihaz bazında toplama yapılıyor...")
            final_results = combined_chunks.groupby('device_aid').agg({
                'Otel_Puani': 'sum',
                'Restoran_Puani': 'sum',
                'Bar_Puani': 'sum',
                'Kahve_Puani': 'sum',
                'Veteriner_Puani': 'sum',
                'Toplam_Puan': 'sum',
                'Ziyaret_Edilen_Mekan_Sayisi': 'sum',
                'Eslesme_Sayisi': 'sum',
                'Eslesme_Orani': 'mean'
            }).reset_index()
            
        else:
            print("İşlenecek veri bulunamadı.")
            return
            
    except Exception as e:
        print(f"Aktivite dosyası yüklenirken hata: {e}")
        return
    
    # Sonuçları CSV'ye kaydet
    if not final_results.empty:
        output_path = os.path.join(data_base_path, output_file)
        
        try:
            final_results.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"\nSonuçlar kaydedildi: {output_path}")
            print(f"Toplam cihaz sayısı: {len(final_results)}")
            
            # Özet istatistikler
            print("\n=== ÖZET İSTATİSTİKLER ===")
            
            # Puanlanan cihaz sayısı
            scored_devices = len(final_results[final_results['Toplam_Puan'] > 0])
            print(f"Puanlanan cihaz sayısı: {scored_devices}")
            print(f"Puanlanmayan cihaz sayısı: {len(final_results) - scored_devices}")
            print(f"Puanlama oranı: {scored_devices/len(final_results)*100:.2f}%")
            
            # Ortalama puanlar
            print(f"\nOrtalama puanlar:")
            print(f"Otel: {final_results['Otel_Puani'].mean():.2f}")
            print(f"Restoran: {final_results['Restoran_Puani'].mean():.2f}")
            print(f"Bar: {final_results['Bar_Puani'].mean():.2f}")
            print(f"Kahve: {final_results['Kahve_Puani'].mean():.2f}")
            print(f"Veteriner: {final_results['Veteriner_Puani'].mean():.2f}")
            print(f"Toplam: {final_results['Toplam_Puan'].mean():.2f}")
            
            # En yüksek puanlı cihazlar
            print(f"\nEn yüksek puanlı 5 cihaz:")
            top_devices = final_results.nlargest(5, 'Toplam_Puan')[['device_aid', 'Toplam_Puan']]
            for _, row in top_devices.iterrows():
                print(f"  {row['device_aid']}: {row['Toplam_Puan']:.2f} puan")
            
        except Exception as e:
            print(f"Sonuçlar kaydedilirken hata: {e}")
    else:
        print("İşlenecek sonuç bulunamadı.")

if __name__ == "__main__":
    main()