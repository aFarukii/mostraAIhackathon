import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import folium
import warnings
from tqdm import tqdm
import time
import multiprocessing as mp
from functools import partial
warnings.filterwarnings('ignore')

class FastSpatialSocialAnalyzer:
    def __init__(self, eslesmeler_df, mekanlar_df):
        print("🚀 Hızlı analiz başlatılıyor...")
        self.eslesmeler = eslesmeler_df.copy()
        self.mekanlar = mekanlar_df.copy()
        self.user_profiles = None
        self.enriched_data = None
        self._prepare_data_fast()
        
    def _parse_mapin_segment_vectorized(self, segments):
        """Vektörize edilmiş MapinSegment parsing - çok daha hızlı"""
        segments = segments.fillna('').astype(str).str.upper()
        
        # Venue type - vectorized
        venue_types = np.where(segments.str.startswith('D'), 'bar_pub',
                     np.where(segments.str.startswith('R'), 'restaurant', 
                     np.where(segments.str.startswith('H'), 'hotel', 'other')))
        
        # Level - vectorized  
        levels = segments.str.extract(r'(\d+)')[0].fillna('0').astype(int)
        
        # Quality - vectorized
        qualities = np.where(segments.str.contains('-A'), 'A',
                   np.where(segments.str.contains('-B'), 'B', 'C'))
        
        return venue_types, levels, qualities
        
    def _prepare_data_fast(self):
        """Hızlandırılmış veri hazırlama"""
        print("📊 Veri hazırlanıyor (hızlı mod)...")
        
        # MapinSegment parse - vectorized
        if 'MapinSegment' in self.mekanlar.columns:
            print("  • MapinSegment kodları çözümleniyor...")
            venue_types, levels, qualities = self._parse_mapin_segment_vectorized(self.mekanlar['MapinSegment'])
            self.mekanlar['venue_type'] = venue_types
            self.mekanlar['venue_level'] = levels  
            self.mekanlar['venue_quality'] = qualities
        
        # Zaman özellikleri - vectorized
        print("  • Zaman özellikleri hesaplanıyor...")
        timestamps = pd.to_datetime(self.eslesmeler['timestamp'], unit='s')
        self.eslesmeler['hour'] = timestamps.dt.hour
        self.eslesmeler['day_of_week'] = timestamps.dt.dayofweek
        self.eslesmeler['is_weekend'] = self.eslesmeler['day_of_week'].isin([5, 6])
        
        # Verileri birleştir
        print("  • Veriler birleştiriliyor...")
        self.enriched_data = self.eslesmeler.merge(
            self.mekanlar[['MusteriTabelaAdi', 'venue_type', 'venue_level', 'venue_quality', 'lat', 'lng']], 
            left_on='gidilen_mekan', 
            right_on='MusteriTabelaAdi', 
            how='left'
        )
        print("✅ Veri hazırlama tamamlandı!")
        
    def create_user_profiles_fast(self):
        """Hızlandırılmış kullanıcı profilleri - Pandas groupby kullanarak"""
        print("👥 Kullanıcı profilleri oluşturuluyor (hızlı mod)...")
        
        # Tüm hesaplamaları tek seferde groupby ile yap
        user_stats = self.enriched_data.groupby('device_aid').agg({
            'gidilen_mekan': ['count', 'nunique'],  # total_visits, unique_places
            'mesafe_m': 'mean',  # avg_distance
            'is_weekend': 'mean',  # weekend_ratio
            'hour': lambda x: (x < 12).sum(),  # morning_visits (afternoon/evening ayrı hesaplanacak)
            'venue_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown',
            'venue_level': lambda x: (x >= 3).sum(),  # high_level_venues
            'venue_quality': [lambda x: (x == 'A').sum(), lambda x: (x == 'B').sum()],
            'lat': ['min', 'max'],
            'lng': ['min', 'max']
        })
        
        # Column names düzelt
        user_stats.columns = ['total_visits', 'unique_places', 'avg_distance', 'weekend_ratio', 
                             'morning_visits', 'favorite_venue_type', 'high_level_venues',
                             'quality_A_visits', 'quality_B_visits', 'lat_min', 'lat_max', 'lng_min', 'lng_max']
        
        # Venue type counts - ayrı hesapla
        venue_counts = self.enriched_data.groupby(['device_aid', 'venue_type']).size().unstack(fill_value=0)
        venue_counts = venue_counts.reindex(columns=['bar_pub', 'restaurant', 'hotel'], fill_value=0)
        
        # Hour-based visits
        hour_stats = self.enriched_data.groupby('device_aid')['hour'].agg([
            lambda x: ((x >= 12) & (x < 18)).sum(),  # afternoon
            lambda x: (x >= 18).sum(),  # evening  
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 12  # most_active_hour
        ])
        hour_stats.columns = ['afternoon_visits', 'evening_visits', 'most_active_hour']
        
        # Tüm dataframeleri birleştir
        self.user_profiles = user_stats.join([venue_counts, hour_stats])
        self.user_profiles = self.user_profiles.reset_index()
        
        # Türetilmiş özellikler
        self.user_profiles['travel_radius'] = np.sqrt(
            (self.user_profiles['lat_max'] - self.user_profiles['lat_min'])**2 + 
            (self.user_profiles['lng_max'] - self.user_profiles['lng_min'])**2
        ) * 111000
        
        # Venue type diversity
        self.user_profiles['venue_type_diversity'] = (
            (self.user_profiles[['bar_pub', 'restaurant', 'hotel']] > 0).sum(axis=1)
        )
        
        # Zenginlik skoru - vectorized
        print("  • Zenginlik skorları hesaplanıyor...")
        wealth_scores = (
            self.user_profiles['quality_A_visits'] * 15 + 
            self.user_profiles['quality_B_visits'] * 8 + 
            self.user_profiles['high_level_venues'] * 10 + 
            self.user_profiles['hotel'] * 12 + 
            self.user_profiles['bar_pub'] * 3 + 
            self.user_profiles['restaurant'] * 5 + 
            self.user_profiles['unique_places'] * 2 + 
            np.minimum(self.user_profiles['travel_radius'] / 1000, 30) + 
            self.user_profiles['evening_visits'] * 1.5
        )
        self.user_profiles['wealth_score'] = np.minimum(wealth_scores, 100)
        
        print("✅ Kullanıcı profilleri tamamlandı!")
        return self.user_profiles
        
    def segment_users_fast(self, n_clusters=5):
        """Hızlandırılmış segmentasyon"""
        print("🎯 Kullanıcı segmentasyonu yapılıyor...")
        
        if self.user_profiles is None:
            self.create_user_profiles_fast()
        
        features = ['total_visits', 'unique_places', 'weekend_ratio', 
                   'travel_radius', 'wealth_score', 'venue_type_diversity',
                   'bar_pub', 'restaurant', 'hotel',
                   'quality_A_visits', 'high_level_venues']
        
        X = self.user_profiles[features].fillna(0)
        X_scaled = StandardScaler().fit_transform(X)
        
        segments = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X_scaled)
        self.user_profiles['segment'] = segments
        
        segment_names = {0: "Casual Diners", 1: "Bar Hoppers", 2: "Fine Diners",
                        3: "Hotel Guests", 4: "Local Explorers"}
        
        self.user_profiles['segment_name'] = self.user_profiles['segment'].map(segment_names)
        
        print("✅ Segmentasyon tamamlandı!")
        return self.user_profiles['segment_name'].value_counts()
        
    def analyze_venues_fast(self):
        """Hızlandırılmış mekan analizi"""
        print("🏢 Mekan analizi yapılıyor...")
        
        # Tek groupby ile tüm hesaplamaları yap
        venue_stats = self.enriched_data.groupby('gidilen_mekan').agg({
            'device_aid': 'nunique',
            'timestamp': 'count', 
            'mesafe_m': 'mean',
            'hour': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 12,
            'is_weekend': 'mean',
            'venue_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown',
            'venue_level': 'mean',
            'venue_quality': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'C'
        }).round(2)
        
        venue_stats.columns = ['unique_visitors', 'total_visits', 'avg_distance', 
                              'peak_hour', 'weekend_ratio', 'venue_type', 
                              'avg_level', 'quality']
        
        # Popülerlik skoru - vectorized
        quality_mult = venue_stats['quality'].map({'A': 1.5, 'B': 1.2, 'C': 1.0})
        venue_stats['popularity_score'] = (
            venue_stats['unique_visitors'] * 0.4 + 
            venue_stats['total_visits'] * 0.3 +
            (1 - venue_stats['avg_distance'] / venue_stats['avg_distance'].max()) * 0.3
        ) * 100 * quality_mult + venue_stats['avg_level'] * 0.1
        
        print("✅ Mekan analizi tamamlandı!")
        return venue_stats.sort_values('popularity_score', ascending=False)
    
    def analyze_time_patterns_fast(self):
        """Hızlandırılmış zaman analizi"""
        print("⏰ Zaman desenleri analiz ediliyor...")
        
        # Tüm groupby işlemlerini tek seferde yap
        hourly = self.enriched_data['hour'].value_counts().sort_index()
        daily = self.enriched_data['day_of_week'].value_counts().sort_index()
        weekend = self.enriched_data.groupby('is_weekend').agg({
            'device_aid': 'nunique', 'timestamp': 'count'})
        
        # Venue-time patterns - pivot_table kullan
        venue_time = pd.pivot_table(
            self.enriched_data, 
            values='timestamp', 
            index='venue_type', 
            columns='hour', 
            aggfunc='count', 
            fill_value=0
        )
        
        result = {
            'hourly_activity': hourly,
            'daily_activity': daily, 
            'weekend_comparison': weekend,
            'venue_time_patterns': venue_time
        }
        
        print("✅ Zaman analizi tamamlandı!")
        return result
    
    def find_hotspots_fast(self, radius_km=0.5):
        """Hızlandırılmış hotspot analizi"""
        print("🔥 Hotspot analizi yapılıyor...")
        
        # Koordinat bazlı groupby - tek seferde
        coords_density = self.enriched_data.groupby(['lat', 'lng', 'MusteriTabelaAdi']).agg({
            'device_aid': 'nunique',
            'timestamp': 'count',
            'venue_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown',
            'venue_quality': lambda x: (x == 'A').sum()
        }).reset_index()
        
        coords_density.columns = ['lat', 'lng', 'venue_name', 'unique_users', 'total_visits', 'dominant_type', 'quality_A_count']
        
        # DBSCAN - sadece gerekli veri ile
        coords_km = coords_density[['lat', 'lng']].values * 111
        clusters = DBSCAN(eps=radius_km, min_samples=3, n_jobs=-1).fit_predict(coords_km)
        coords_density['hotspot_cluster'] = clusters
        
        # Hotspot aggregation
        hotspots = coords_density[coords_density['hotspot_cluster'] != -1].groupby('hotspot_cluster').agg({
            'lat': 'mean',
            'lng': 'mean', 
            'venue_name': lambda x: ', '.join(x.unique()),
            'unique_users': 'sum',
            'total_visits': 'sum',
            'dominant_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'mixed',
            'quality_A_count': 'sum'
        }).sort_values('total_visits', ascending=False)
        
        print("✅ Hotspot analizi tamamlandı!")
        return hotspots
    
    def create_wealth_map_fast(self, sample_size=500):
        """Hızlandırılmış harita oluşturma - daha az nokta"""
        print("🗺️ Zenginlik haritası oluşturuluyor...")
        
        if self.user_profiles is None:
            self.create_user_profiles_fast()
            
        center_lat, center_lng = self.enriched_data['lat'].mean(), self.enriched_data['lng'].mean()
        m = folium.Map(location=[center_lat, center_lng], zoom_start=12)
        
        # Daha küçük sample
        user_locations = self.enriched_data.merge(
            self.user_profiles[['device_aid', 'wealth_score']], on='device_aid')
        
        sample_data = user_locations.sample(min(sample_size, len(user_locations)))
        
        venue_colors = {'bar_pub': 'purple', 'restaurant': 'green', 'hotel': 'red', 'other': 'blue', 'unknown': 'gray'}
        
        print(f"  • {len(sample_data)} nokta haritaya ekleniyor...")
        for _, row in sample_data.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lng']],
                radius=3 + (row['wealth_score'] / 25),
                color=venue_colors.get(row['venue_type'], 'gray'),
                fillColor=venue_colors.get(row['venue_type'], 'gray'),
                popup=f"Wealth: {row['wealth_score']:.1f}<br>Type: {row['venue_type']}<br>Quality: {row['venue_quality']}<br>Place: {row['gidilen_mekan']}"
            ).add_to(m)
        
        print("✅ Harita oluşturuldu!")
        return m
    
    def comprehensive_analysis_fast(self):
        """Hızlandırılmış kapsamlı analiz"""
        print("\n" + "="*60)
        print("🚀 HIZLI KAPSAMLI ANALİZ BAŞLIYOR")
        print("="*60)
        
        start_time = time.time()
        
        # Paralel olmayan ama optimize edilmiş işlemler
        print("📈 İlerleme: 20% - Kullanıcı Profilleri")
        self.create_user_profiles_fast()
        
        print("📈 İlerleme: 40% - Kullanıcı Segmentasyonu") 
        segments = self.segment_users_fast()
        
        print("📈 İlerleme: 60% - Mekan Analizi")
        venue_analysis = self.analyze_venues_fast()
        
        print("📈 İlerleme: 80% - Zaman & Hotspot Analizi")
        time_analysis = self.analyze_time_patterns_fast()
        hotspots = self.find_hotspots_fast()
        
        print("📈 İlerleme: 100% - Sonuçlar Hazırlanıyor")
        
        # Özet istatistikler
        summary = {
            'total_users': len(self.user_profiles),
            'total_visits': len(self.enriched_data),
            'unique_venues': self.enriched_data['gidilen_mekan'].nunique(),
            'avg_wealth_score': self.user_profiles['wealth_score'].mean(),
            'segments': segments.to_dict(),
            'top_venues': venue_analysis.head(10),
            'peak_hour': time_analysis['hourly_activity'].idxmax(),
            'peak_day': time_analysis['daily_activity'].idxmax(), 
            'venue_type_distribution': self.enriched_data['venue_type'].value_counts().to_dict(),
            'quality_distribution': self.enriched_data['venue_quality'].value_counts().to_dict(),
            'hotspot_count': len(hotspots)
        }
        
        # Segment profilleri
        segment_profiles = self.user_profiles.groupby('segment_name').agg({
            'total_visits': 'mean',
            'unique_places': 'mean',
            'wealth_score': 'mean',
            'weekend_ratio': 'mean',
            'bar_pub': 'mean',
            'restaurant': 'mean', 
            'hotel': 'mean',
            'quality_A_visits': 'mean'
        }).round(2)
        
        duration = time.time() - start_time
        print(f"\n🎉 TÜM ANALİZLER TAMAMLANDI! ({duration:.1f} saniye)")
        print("="*60)
        
        return {
            'summary': summary,
            'user_profiles': self.user_profiles,
            'venue_analysis': venue_analysis,
            'time_analysis': time_analysis,
            'hotspots': hotspots,
            'segment_profiles': segment_profiles,
            'enriched_data': self.enriched_data,
            'duration': duration
        }

def run_fast_analysis(eslesmeler_path, mekanlar_path):
    """Hızlı analiz çalıştır"""
    print("⚡ HIZLI ANALİZ MODU")
    print("="*50)
    
    start_time = time.time()
    
    # Veri yükleme
    print("📂 Veriler yükleniyor...")
    eslesmeler = pd.read_csv(eslesmeler_path)
    mekanlar = pd.read_csv(mekanlar_path)
    
    # Büyük veri setleri için sampling
    if len(eslesmeler) > 100000:
        print(f"⚠️  Büyük veri seti tespit edildi ({len(eslesmeler):,} satır)")
        sample_size = min(100000, len(eslesmeler))
        eslesmeler = eslesmeler.sample(sample_size)
        print(f"🔀 {sample_size:,} satır örneklem alındı")
    
    print(f"  ✅ Eşleşmeler: {len(eslesmeler):,} satır")
    print(f"  ✅ Mekanlar: {len(mekanlar):,} satır")
    
    # Analiz 
    analyzer = FastSpatialSocialAnalyzer(eslesmeler, mekanlar)
    results = analyzer.comprehensive_analysis_fast()
    
    # Hızlı kaydetme
    print("\n💾 Sonuçlar kaydediliyor...")
    results['user_profiles'].to_csv("kullanici_profilleri_fast.csv", index=False)
    results['venue_analysis'].to_csv("mekan_analizi_fast.csv") 
    results['segment_profiles'].to_csv("segment_profilleri_fast.csv")
    results['hotspots'].to_csv("hotspots_fast.csv")
    
    # Küçük harita
    wealth_map = analyzer.create_wealth_map_fast(sample_size=300)
    wealth_map.save("zenginlik_haritasi_fast.html")
    
    # Süre
    total_duration = time.time() - start_time
    minutes = int(total_duration // 60)
    seconds = int(total_duration % 60)
    
    # Özet rapor
    print("\n" + "="*60)
    print("⚡ HIZLI ANALİZ ÖZET RAPORU")
    print("="*60)
    print(f"⏱️  Toplam Süre: {minutes} dakika {seconds} saniye")
    print(f"👥 Toplam Kullanıcı: {results['summary']['total_users']:,}")
    print(f"📍 Toplam Ziyaret: {results['summary']['total_visits']:,}")
    print(f"🏢 Benzersiz Mekan: {results['summary']['unique_venues']:,}")
    print(f"💰 Ortalama Zenginlik Skoru: {results['summary']['avg_wealth_score']:.1f}")
    print(f"⏰ En Popüler Saat: {results['summary']['peak_hour']}:00")
    print(f"🔥 Hotspot Sayısı: {results['summary']['hotspot_count']}")
    
    print(f"\n🎯 Segment Dağılımı:")
    for segment, count in results['summary']['segments'].items():
        percentage = count/results['summary']['total_users']*100
        print(f"  • {segment}: {count:,} (%{percentage:.1f})")
    
    print("\n⚡ HIZLI ANALİZ TAMAMLANDI!")
    print("="*60)
    
    return results

if __name__ == "__main__":
    results = run_fast_analysis("eslesmeler.csv", "maindataplaces.csv")