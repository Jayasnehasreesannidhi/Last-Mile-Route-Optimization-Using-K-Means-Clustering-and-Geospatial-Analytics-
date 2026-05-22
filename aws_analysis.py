# ============================================================
# Amazon Last Mile Routing — AWS Dataset Analysis
# US Cities: Los Angeles, Chicago, Boston, Austin, Seattle
# ============================================================

# pip install pandas numpy scikit-learn matplotlib folium seaborn

# ============================================================
# STEP 1: IMPORTS
# ============================================================
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries loaded successfully!")

# ============================================================
# STEP 2: LOAD THE JSON FILES
# ============================================================
BASE_PATH = r'C:\Users\saile\Downloads\amazon-training\model_build_inputs'

print("\n📦 Loading route_data.json (this may take 30 seconds)...")
with open(os.path.join(BASE_PATH, 'route_data.json'), 'r') as f:
    route_data = json.load(f)

print("📦 Loading actual_sequences.json...")
with open(os.path.join(BASE_PATH, 'actual_sequences.json'), 'r') as f:
    actual_sequences = json.load(f)

print(f"\n✅ Data loaded!")
print(f"   Total routes: {len(route_data):,}")
print(f"   Total sequences: {len(actual_sequences):,}")

# ============================================================
# STEP 3: PARSE ROUTE DATA INTO A DATAFRAME
# ============================================================
print("\n⚙️  Parsing route data...")

rows = []
for route_id, route in route_data.items():
    city_code   = route.get('station_code', 'Unknown')
    date        = route.get('date_YYYY_MM_DD', 'Unknown')
    departure   = route.get('departure_time_utc', 'Unknown')
    executor    = route.get('executor_capacity_cm3', 0)
    score       = route.get('route_score', 'Unknown')
    stops       = route.get('stops', {})

    for stop_id, stop in stops.items():
        rows.append({
            'Route_ID':     route_id,
            'Stop_ID':      stop_id,
            'City_Code':    city_code,
            'Date':         date,
            'Departure':    departure,
            'Latitude':     stop.get('lat', None),
            'Longitude':    stop.get('lng', None),
            'Stop_Type':    stop.get('type', 'Unknown'),
            'Route_Score':  score,
            'Capacity_cm3': executor,
        })

df_raw = pd.DataFrame(rows)
print(f"   Total stops parsed: {len(df_raw):,}")

# ============================================================
# STEP 4: DATA CLEANING — FULL PIPELINE
# ============================================================
print(f"\n{'='*55}")
print("🧹 DATA CLEANING REPORT")
print(f"{'='*55}")
print(f"Step 0 — Raw stops loaded:                  {len(df_raw):,}")

# 4.1 Keep only delivery stops (remove depot/station stops)
df_clean = df_raw[df_raw['Stop_Type'] == 'Dropoff'].copy()
removed = len(df_raw) - len(df_clean)
print(f"Step 1 — After removing depot stops:        {len(df_clean):,}  (removed {removed:,})")

# 4.2 Remove duplicate stops
before = len(df_clean)
df_clean = df_clean.drop_duplicates(subset=['Route_ID', 'Stop_ID'])
removed = before - len(df_clean)
print(f"Step 2 — After removing duplicates:         {len(df_clean):,}  (removed {removed:,})")

# 4.3 Remove rows with missing coordinates
before = len(df_clean)
df_clean = df_clean.dropna(subset=['Latitude', 'Longitude'])
removed = before - len(df_clean)
print(f"Step 3 — After removing missing coords:     {len(df_clean):,}  (removed {removed:,})")

# 4.4 Remove zero/null island coordinates (0,0)
before = len(df_clean)
df_clean = df_clean[
    ~((df_clean['Latitude'].abs() < 0.001) &
      (df_clean['Longitude'].abs() < 0.001))
]
removed = before - len(df_clean)
print(f"Step 4 — After removing 0,0 coords:         {len(df_clean):,}  (removed {removed:,})")

# 4.5 Keep only Continental US coordinates
before = len(df_clean)
df_clean = df_clean[
    (df_clean['Latitude'].between(24, 50)) &
    (df_clean['Longitude'].between(-130, -60))
]
removed = before - len(df_clean)
print(f"Step 5 — After US boundary filter:          {len(df_clean):,}  (removed {removed:,})")

# 4.6 Map city codes to real city names
def map_city(code):
    if not isinstance(code, str):
        return 'Other'
    code = code.upper()
    if 'LA' in code or 'DLA' in code: return 'Los Angeles'
    if 'CH' in code or 'DCH' in code: return 'Chicago'
    if 'BO' in code or 'DBO' in code: return 'Boston'
    if 'AU' in code or 'DAU' in code: return 'Austin'
    if 'SE' in code or 'DSE' in code: return 'Seattle'
    return 'Other'

df_clean['City'] = df_clean['City_Code'].apply(map_city)

# 4.7 Remove unidentified cities
before = len(df_clean)
df_clean = df_clean[df_clean['City'] != 'Other']
removed = before - len(df_clean)
print(f"Step 6 — After removing unknown cities:     {len(df_clean):,}  (removed {removed:,})")

# 4.8 Remove routes with fewer than 3 stops
before = len(df_clean)
stop_counts   = df_clean.groupby('Route_ID')['Stop_ID'].count()
valid_routes  = stop_counts[stop_counts >= 3].index
df_clean      = df_clean[df_clean['Route_ID'].isin(valid_routes)]
removed       = before - len(df_clean)
print(f"Step 7 — After removing <3 stop routes:     {len(df_clean):,}  (removed {removed:,})")

total_removed = len(df_raw) - len(df_clean)
pct_removed   = total_removed / len(df_raw) * 100
print(f"\n✅ FINAL CLEAN DATASET:                     {len(df_clean):,} delivery stops")
print(f"   Total removed:                            {total_removed:,} rows ({pct_removed:.1f}%)")
print(f"{'='*55}")

print(f"\n📊 Stops per city:")
print(df_clean['City'].value_counts())
print(f"\n✅ Clean dataset: {len(df_clean):,} stops across {df_clean['City'].nunique()} cities")

# ============================================================
# STEP 5: PARSE ACTUAL SEQUENCES
# ============================================================
print("\n⚙️  Parsing route sequences...")

seq_rows = []
for route_id, seq_data in actual_sequences.items():
    actual = seq_data.get('actual', {})
    seq_rows.append({
        'Route_ID':  route_id,
        'Num_Stops': len(actual)
    })

df_seq = pd.DataFrame(seq_rows)
print(f"   Routes with sequences: {len(df_seq):,}")
print(f"   Avg stops per route:   {df_seq['Num_Stops'].mean():.1f}")
print(f"   Max stops per route:   {df_seq['Num_Stops'].max()}")
print(f"   Min stops per route:   {df_seq['Num_Stops'].min()}")

# ============================================================
# STEP 6: BUILD ROUTE SUMMARY
# ============================================================
route_summary = (df_clean.groupby('Route_ID')
                 .agg(Total_Stops  = ('Stop_ID',      'count'),
                      City         = ('City',         'first'),
                      City_Code    = ('City_Code',    'first'),
                      Route_Score  = ('Route_Score',  'first'))
                 .reset_index())

route_summary = route_summary.merge(df_seq, on='Route_ID', how='left')

# ============================================================
# STEP 7: FEATURE ENGINEERING
# ============================================================
print("\n⚙️  Engineering features...")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2-lat1)/2)**2 +
         np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2)
    return R * 2 * np.arcsin(np.sqrt(a))

route_distances = []
for route_id, group in df_clean.groupby('Route_ID'):
    coords = group[['Latitude', 'Longitude']].values
    total_dist = sum(
        haversine(coords[i][0], coords[i][1],
                  coords[i+1][0], coords[i+1][1])
        for i in range(len(coords)-1)
    ) if len(coords) > 1 else 0
    route_distances.append({'Route_ID': route_id, 'Total_Distance_km': round(total_dist, 2)})

df_dist = pd.DataFrame(route_distances)
route_summary = route_summary.merge(df_dist, on='Route_ID', how='left')

# Remove distance outliers (above 99th percentile)
dist_cap      = route_summary['Total_Distance_km'].quantile(0.99)
before        = len(route_summary)
route_summary = route_summary[route_summary['Total_Distance_km'] <= dist_cap]
print(f"   Removed {before - len(route_summary)} distance outlier routes")

# Distance per stop + CO2 (Amazon van EPA SmartWay: 0.21 kg/km)
route_summary['Dist_Per_Stop_km'] = (
    route_summary['Total_Distance_km'] / route_summary['Total_Stops']
).round(3)
route_summary['CO2_kg'] = (route_summary['Total_Distance_km'] * 0.21).round(2)

print("✅ Features engineered!")
print(f"\n📊 Route summary stats:")
print(route_summary[['Total_Stops', 'Total_Distance_km', 'CO2_kg']].describe().round(2))

# ============================================================
# STEP 8: K-MEANS CLUSTERING
# ============================================================
print("\n🗺️  Running K-Means clustering on delivery locations...")

sample_size   = min(50000, len(df_clean))
df_sample     = df_clean.sample(sample_size, random_state=42)
coords        = df_sample[['Latitude', 'Longitude']].copy()
scaler        = StandardScaler()
coords_scaled = scaler.fit_transform(coords)

inertia           = []
silhouette_scores = []
k_range           = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(coords_scaled)
    inertia.append(km.inertia_)
    silhouette_scores.append(silhouette_score(coords_scaled, km.labels_))

# Elbow plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(k_range, inertia, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
ax1.set_ylabel('Inertia', fontsize=12)
ax1.set_title('Elbow Method — Finding Optimal K', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_title('Silhouette Score — Cluster Quality', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('aws_elbow_curve.png', dpi=150, bbox_inches='tight')
plt.show()
print("   📊 Elbow curve saved as 'aws_elbow_curve.png'")

OPTIMAL_K         = 5
kmeans            = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
df_sample['Zone'] = kmeans.fit_predict(coords_scaled)

print(f"\n✅ K-Means complete! {OPTIMAL_K} zones identified")
print("\n📊 Stops per zone:")
print(df_sample['Zone'].value_counts().sort_index())

# ============================================================
# STEP 9: VISUALIZATIONS
# ============================================================
print("\n📈 Generating visualizations...")

# 9.1 Delivery stops by city
plt.figure(figsize=(10, 6))
city_counts = df_clean['City'].value_counts()
bars = plt.bar(city_counts.index, city_counts.values,
               color=['#3498db','#e74c3c','#2ecc71','#f39c12','#9b59b6'],
               edgecolor='black')
plt.xlabel('US City', fontsize=12)
plt.ylabel('Number of Delivery Stops', fontsize=12)
plt.title('Delivery Stops by US City', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, city_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f'{val:,}', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('aws_stops_by_city.png', dpi=150, bbox_inches='tight')
plt.show()
print("   📊 City stops chart saved as 'aws_stops_by_city.png'")

# 9.2 Stops per route distribution
plt.figure(figsize=(10, 6))
plt.hist(route_summary['Total_Stops'].dropna(), bins=30,
         color='#3498db', edgecolor='black', alpha=0.8)
plt.xlabel('Number of Stops Per Route', fontsize=12)
plt.ylabel('Number of Routes', fontsize=12)
plt.title('Distribution of Stops Per Route', fontsize=14, fontweight='bold')
mean_stops = route_summary['Total_Stops'].mean()
plt.axvline(mean_stops, color='red', linestyle='--', linewidth=2,
            label=f'Mean: {mean_stops:.1f} stops')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('aws_stops_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("   📊 Stops distribution chart saved as 'aws_stops_distribution.png'")

# 9.3 Average route distance by city
plt.figure(figsize=(10, 6))
city_dist = route_summary.groupby('City')['Total_Distance_km'].mean().sort_values()
bars = plt.bar(city_dist.index, city_dist.values,
               color=['#2ecc71' if v == city_dist.min()
                      else '#e74c3c' if v == city_dist.max()
                      else '#3498db' for v in city_dist.values],
               edgecolor='black')
plt.xlabel('US City', fontsize=12)
plt.ylabel('Average Route Distance (km)', fontsize=12)
plt.title('Average Route Distance by City', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, city_dist.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}km', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('aws_distance_by_city.png', dpi=150, bbox_inches='tight')
plt.show()
print("   📊 Distance by city chart saved as 'aws_distance_by_city.png'")

# 9.4 CO2 per city
plt.figure(figsize=(10, 6))
city_co2 = route_summary.groupby('City')['CO2_kg'].mean().sort_values()
bars = plt.bar(city_co2.index, city_co2.values,
               color=['#2ecc71' if v == city_co2.min()
                      else '#e74c3c' if v == city_co2.max()
                      else '#e67e22' for v in city_co2.values],
               edgecolor='black')
plt.xlabel('US City', fontsize=12)
plt.ylabel('Average CO₂ per Route (kg)', fontsize=12)
plt.title('Average CO₂ Emissions per Route by City', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, city_co2.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.1f}kg', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('aws_co2_by_city.png', dpi=150, bbox_inches='tight')
plt.show()
print("   📊 CO₂ by city chart saved as 'aws_co2_by_city.png'")

# 9.5 Zone cluster scatter plot
plt.figure(figsize=(14, 8))
colors = plt.cm.Set1(np.linspace(0, 1, OPTIMAL_K))
for zone in range(OPTIMAL_K):
    mask = df_sample['Zone'] == zone
    plt.scatter(
        df_sample[mask]['Longitude'],
        df_sample[mask]['Latitude'],
        c=[colors[zone]], label=f'Zone {zone+1}',
        alpha=0.4, s=5
    )
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.title('Delivery Zone Clusters — US Cities (K-Means)', fontsize=14, fontweight='bold')
plt.legend(markerscale=4, fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('aws_zone_clusters.png', dpi=150, bbox_inches='tight')
plt.show()
print("   📊 Zone cluster map saved as 'aws_zone_clusters.png'")

# ============================================================
# STEP 10: INTERACTIVE FOLIUM MAP
# ============================================================
print("\n🗺️  Building interactive US map...")

center_lat = df_sample['Latitude'].mean()
center_lon = df_sample['Longitude'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=4)

zone_colors = ['red', 'blue', 'green', 'purple', 'orange']
map_sample  = df_sample.sample(min(1000, len(df_sample)), random_state=42)

for _, row in map_sample.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=3,
        color=zone_colors[int(row['Zone']) % len(zone_colors)],
        fill=True,
        fill_opacity=0.7,
        popup=f"Zone {row['Zone']+1} | City: {row['City']} | Stop: {row['Stop_ID']}"
    ).add_to(m)

m.save('aws_delivery_map.html')
print("   🗺️  Interactive US map saved as 'aws_delivery_map.html'")
print("   👉 Open in your browser to explore!")

# ============================================================
# STEP 11: KEY FINDINGS SUMMARY
# ============================================================
print("\n" + "="*55)
print("📋 KEY FINDINGS SUMMARY — AWS US DATASET")
print("="*55)
print(f"Total delivery stops:          {len(df_clean):,}")
print(f"Total routes analyzed:         {len(route_summary):,}")
print(f"US cities covered:             {df_clean['City'].nunique()}")
print(f"Avg stops per route:           {route_summary['Total_Stops'].mean():.1f}")
print(f"Avg route distance:            {route_summary['Total_Distance_km'].mean():.2f} km")
print(f"Avg CO₂ per route:             {route_summary['CO2_kg'].mean():.2f} kg")
print(f"Total CO₂ in dataset:          {route_summary['CO2_kg'].sum():.1f} kg")

print(f"\n📊 By City:")
city_stats = route_summary.groupby('City').agg(
    Routes       = ('Route_ID',         'count'),
    Avg_Stops    = ('Total_Stops',       'mean'),
    Avg_Distance = ('Total_Distance_km', 'mean'),
    Avg_CO2      = ('CO2_kg',            'mean')
).round(2)
print(city_stats)

highest_dist_city = city_stats['Avg_Distance'].idxmax()
lowest_dist_city  = city_stats['Avg_Distance'].idxmin()
highest_co2_city  = city_stats['Avg_CO2'].idxmax()
print(f"\nLongest avg routes:            {highest_dist_city}")
print(f"Shortest avg routes:           {lowest_dist_city}")
print(f"Highest CO₂ city:              {highest_co2_city}")
print(f"Zones identified:              {OPTIMAL_K}")
print("="*55)
print("\n✅ Analysis complete! Files saved:")
print("   - aws_elbow_curve.png")
print("   - aws_stops_by_city.png")
print("   - aws_stops_distribution.png")
print("   - aws_distance_by_city.png")
print("   - aws_co2_by_city.png")
print("   - aws_zone_clusters.png")
print("   - aws_delivery_map.html  ← open in browser!")