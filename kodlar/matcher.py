import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import glob
import os
import time

def balltree_eslestirme(hareketler_df, mekanlar_df, esik_metre=50):
    """BallTree kullanarak spatial indexing ile hızlı eşleştirme"""
    # Koordinatları kontrol et ve temizle
    mekanlar_temiz = mekanlar_df.dropna(subset=['lat', 'lng']).copy()
    hareketler_temiz = hareketler_df.dropna(subset=['latitude', 'longitude']).copy()
    
    if len(mekanlar_temiz) == 0 or len(hareketler_temiz) == 0:
        return pd.DataFrame()
    
    # Koordinatları radyana çevir
    mekan_coords = np.radians(mekanlar_temiz[['lat', 'lng']].values)
    hareket_coords = np.radians(hareketler_temiz[['latitude', 'longitude']].values)
    
    # BallTree oluştur (haversine metriği ile)
    tree = BallTree(mekan_coords, metric='haversine')
    
    # Her hareket noktası için yakın mekanları bul
    esik_radyan = esik_metre / 6371000  # metreyi radyana çevir
    indices, distances = tree.query_radius(hareket_coords, r=esik_radyan, return_distance=True)
    
    # Sonuçları topla
    sonuclar = []
    for h_idx, (mekan_indices, mesafeler) in enumerate(zip(indices, distances)):
        for m_idx, mesafe_radyan in zip(mekan_indices, mesafeler):
            mesafe_metre = mesafe_radyan * 6371000
            original_h_idx = hareketler_temiz.index[h_idx]
            original_m_idx = mekanlar_temiz.index[m_idx]
            
            sonuclar.append({
                'timestamp': hareketler_df.loc[original_h_idx, 'timestamp'],
                'device_aid': hareketler_df.loc[original_h_idx, 'device_aid'],
                'gidilen_mekan': mekanlar_df.loc[original_m_idx, 'MusteriTabelaAdi'],
                'mesafe_m': mesafe_metre
            })
    
    return pd.DataFrame(sonuclar)

def toplu_eslestirme(mekanlar_dosyasi, devices_klasoru, cikti_dosyasi, esik_metre=50):
    """Tüm device dosyalarını işleyerek toplu eşleştirme yapar"""
    start_time = time.time()
    
    # Mekanlar dosyasını yükle
    try:
        mekanlar = pd.read_csv(mekanlar_dosyasi)
        print(f"Mekanlar yüklendi: {len(mekanlar)} adet")
    except Exception as e:
        print(f"Mekanlar dosyası yüklenirken hata: {e}")
        return
    
    # Device dosyalarını bul
    device_files = glob.glob(os.path.join(devices_klasoru, "*.csv"))
    if not device_files:
        print(f"'{devices_klasoru}' klasöründe CSV dosyası bulunamadı")
        return
    
    print(f"İşlenecek dosya sayısı: {len(device_files)}")
    
    tum_sonuclar = []
    basarili_dosya = 0
    
    for i, dosya_yolu in enumerate(device_files, 1):
        try:
            hareketler = pd.read_csv(dosya_yolu)
            
            if len(hareketler) == 0:
                continue
                
            sonuc = balltree_eslestirme(hareketler, mekanlar, esik_metre)
            
            if len(sonuc) > 0:
                tum_sonuclar.append(sonuc)
                basarili_dosya += 1
            
            if i % 50 == 0:
                print(f"İşlenen: {i}/{len(device_files)} dosya")
                
        except Exception as e:
            print(f"Hata - {os.path.basename(dosya_yolu)}: {e}")
            continue
    
    # Sonuçları birleştir ve kaydet
    if tum_sonuclar:
        final_df = pd.concat(tum_sonuclar, ignore_index=True)
        final_df.to_csv(cikti_dosyasi, index=False)
        
        toplam_sure = time.time() - start_time
        print(f"\n=== SONUÇ ===")
        print(f"İşlenen dosya: {basarili_dosya}/{len(device_files)}")
        print(f"Toplam eşleşme: {len(final_df)}")
        print(f"Süre: {toplam_sure:.1f} saniye")
        print(f"Çıktı dosyası: {cikti_dosyasi}")
    else:
        print("Hiç eşleşme bulunamadı")

if __name__ == "__main__":
    # Kullanım örneği
    toplu_eslestirme(
        mekanlar_dosyasi="clean_places.csv",
        devices_klasoru="devices",
        cikti_dosyasi="eslesmeler.csv",
        esik_metre=30
    )