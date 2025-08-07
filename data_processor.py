import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class NYCDataProcessor:
    def __init__(self, trip_data_path, zone_data_path):
        
        self.trip_data_path = trip_data_path
        self.zone_data_path = zone_data_path
        self.trip_data = None
        self.zone_data = None
        self.processed_data = None
        self.traffic_patterns = None
        
    def load_and_preprocess(self):
        
        print("Loading NYC taxi data...")
        
        
        self.trip_data = pd.read_parquet(self.trip_data_path)
        print(f"Loaded {len(self.trip_data)} trips")
        
        
        self.zone_data = pd.read_csv(self.zone_data_path)
        print(f"Loaded {len(self.zone_data)} zones")
        
        
        self._preprocess_trip_data()
        
        
        self._create_traffic_patterns()
        
        
        self._save_processed_data()
        
        return self.processed_data
    
    def _preprocess_trip_data(self):
        
        print("Preprocessing trip data...")
        
        
        self.trip_data['tpep_pickup_datetime'] = pd.to_datetime(self.trip_data['tpep_pickup_datetime'])
        self.trip_data['tpep_dropoff_datetime'] = pd.to_datetime(self.trip_data['tpep_dropoff_datetime'])
        
        
        self.trip_data['pickup_hour'] = self.trip_data['tpep_pickup_datetime'].dt.hour
        self.trip_data['pickup_day'] = self.trip_data['tpep_pickup_datetime'].dt.dayofweek
        self.trip_data['pickup_month'] = self.trip_data['tpep_pickup_datetime'].dt.month
        self.trip_data['trip_duration'] = (
            self.trip_data['tpep_dropoff_datetime'] - self.trip_data['tpep_pickup_datetime']
        ).dt.total_seconds() / 60  
        
        
        self.trip_data['speed_mph'] = self.trip_data['trip_distance'] / (self.trip_data['trip_duration'] / 60)
        self.trip_data['fare_per_mile'] = self.trip_data['total_amount'] / self.trip_data['trip_distance']
        
        
        self.trip_data = self.trip_data[
            (self.trip_data['trip_distance'] > 0) &
            (self.trip_data['trip_duration'] > 0) &
            (self.trip_data['trip_duration'] < 180) &  
            (self.trip_data['speed_mph'] > 0) &
            (self.trip_data['speed_mph'] < 60)  
        ]
        
        print(f"After preprocessing: {len(self.trip_data)} valid trips")
        
    def _create_traffic_patterns(self):
        
        print("Creating traffic patterns...")
        
        
        zones = sorted(self.zone_data['LocationID'].unique())
        zone_to_idx = {zone: idx for idx, zone in enumerate(zones)}
        
        
        traffic_matrix = np.zeros((len(zones), len(zones)))
        
        
        for _, trip in self.trip_data.iterrows():
            pickup_zone = trip['PULocationID']
            dropoff_zone = trip['DOLocationID']
            
            if pickup_zone in zone_to_idx and dropoff_zone in zone_to_idx:
                pickup_idx = zone_to_idx[pickup_zone]
                dropoff_idx = zone_to_idx[dropoff_zone]
                
                
                weight = trip['trip_distance'] * (trip['trip_duration'] / 60) * (trip['total_amount'] / 10)
                traffic_matrix[pickup_idx][dropoff_idx] += weight
        
        
        max_traffic = np.max(traffic_matrix)
        if max_traffic > 0:
            traffic_matrix = traffic_matrix / max_traffic
        
        
        hourly_traffic = self.trip_data.groupby('pickup_hour').agg({
            'trip_distance': 'mean',
            'trip_duration': 'mean',
            'total_amount': 'mean',
            'speed_mph': 'mean'
        }).reset_index()
        
        
        zone_traffic_df = pd.DataFrame({
            'LocationID': list(zone_to_idx.keys()),
            'TrafficVolume': np.sum(traffic_matrix, axis=1)
        })
        zone_traffic_df = zone_traffic_df.merge(self.zone_data, on='LocationID')
        borough_traffic = zone_traffic_df.groupby('Borough').agg({
            'TrafficVolume': 'sum',
            'LocationID': 'count'
        }).rename(columns={'LocationID': 'ZoneCount'})
        
        
        zone_traffic = np.sum(traffic_matrix, axis=1)
        hotspot_zones = np.argsort(zone_traffic)[-20:]  
        
        self.traffic_patterns = {
            'traffic_matrix': traffic_matrix,
            'zone_mapping': zone_to_idx,
            'hourly_traffic': hourly_traffic,
            'borough_traffic': borough_traffic,
            'hotspot_zones': hotspot_zones.tolist(),
            'zone_traffic_volumes': zone_traffic
        }
        
        print(f"Created traffic patterns with {len(zones)} zones")
        
    def _save_processed_data(self):
        
        self.processed_data = {
            'trip_data': self.trip_data,
            'zone_data': self.zone_data,
            'traffic_patterns': self.traffic_patterns
        }
        
        
        with open('processed_nyc_data.pkl', 'wb') as f:
            pickle.dump(self.processed_data, f)
        
        print("Processed data saved to processed_nyc_data.pkl")
    
    def get_traffic_analysis(self):
        
        if self.traffic_patterns is None:
            self.load_and_preprocess()
        
        analysis = {
            'total_trips': len(self.trip_data),
            'avg_trip_distance': self.trip_data['trip_distance'].mean(),
            'avg_trip_duration': self.trip_data['trip_duration'].mean(),
            'avg_speed': self.trip_data['speed_mph'].mean(),
            'avg_fare': self.trip_data['total_amount'].mean(),
            'hotspot_zones': self.traffic_patterns['hotspot_zones'],
            'hourly_traffic': self.traffic_patterns['hourly_traffic'].to_dict('records'),
            'borough_traffic': self.traffic_patterns['borough_traffic'].to_dict('records'),
            'traffic_matrix_shape': self.traffic_patterns['traffic_matrix'].shape
        }
        
        return analysis
    
    def get_zone_statistics(self):
        
        if self.traffic_patterns is None:
            self.load_and_preprocess()
        
        zone_stats = []
        for zone_id, idx in self.traffic_patterns['zone_mapping'].items():
            zone_trips = self.trip_data[
                (self.trip_data['PULocationID'] == zone_id) |
                (self.trip_data['DOLocationID'] == zone_id)
            ]
            
            zone_info = self.zone_data[self.zone_data['LocationID'] == zone_id].iloc[0]
            
            stats = {
                'LocationID': zone_id,
                'Zone': zone_info['Zone'],
                'Borough': zone_info['Borough'],
                'ServiceZone': zone_info['service_zone'],
                'TotalTrips': len(zone_trips),
                'AvgDistance': zone_trips['trip_distance'].mean(),
                'AvgDuration': zone_trips['trip_duration'].mean(),
                'AvgSpeed': zone_trips['speed_mph'].mean(),
                'TrafficVolume': self.traffic_patterns['zone_traffic_volumes'][idx]
            }
            zone_stats.append(stats)
        
        return pd.DataFrame(zone_stats)
    
    def get_optimization_scenarios(self):
        
        if self.traffic_patterns is None:
            self.load_and_preprocess()
        
        
        traffic_matrix = self.traffic_patterns['traffic_matrix']
        zone_mapping = self.traffic_patterns['zone_mapping']
        
        
        routes = []
        for i in range(len(traffic_matrix)):
            for j in range(len(traffic_matrix)):
                if i != j and traffic_matrix[i][j] > 0.1:  
                    start_zone = list(zone_mapping.keys())[i]
                    end_zone = list(zone_mapping.keys())[j]
                    
                    route = {
                        'start_zone': start_zone,
                        'end_zone': end_zone,
                        'traffic_volume': traffic_matrix[i][j],
                        'optimization_priority': traffic_matrix[i][j]
                    }
                    routes.append(route)
        
        
        routes.sort(key=lambda x: x['optimization_priority'], reverse=True)
        
        return routes[:50]  
    
    def calculate_congestion_metrics(self):
        
        if self.traffic_patterns is None:
            self.load_and_preprocess()
        
        traffic_matrix = self.traffic_patterns['traffic_matrix']
        
        
        total_congestion = np.sum(traffic_matrix)
        max_congestion = np.max(traffic_matrix)
        avg_congestion = np.mean(traffic_matrix)
        
        
        congestion_levels = {
            'low': np.sum(traffic_matrix < 0.1),
            'medium': np.sum((traffic_matrix >= 0.1) & (traffic_matrix < 0.5)),
            'high': np.sum(traffic_matrix >= 0.5)
        }
        
        return {
            'total_congestion': total_congestion,
            'max_congestion': max_congestion,
            'avg_congestion': avg_congestion,
            'congestion_levels': congestion_levels
        }
    
    def get_time_based_patterns(self):
        
        if self.traffic_patterns is None:
            self.load_and_preprocess()
        
        
        hourly_patterns = self.trip_data.groupby('pickup_hour').agg({
            'trip_distance': ['mean', 'std'],
            'trip_duration': ['mean', 'std'],
            'speed_mph': ['mean', 'std'],
            'total_amount': ['mean', 'std']
        }).round(2)
        
        
        daily_patterns = self.trip_data.groupby('pickup_day').agg({
            'trip_distance': 'mean',
            'trip_duration': 'mean',
            'speed_mph': 'mean',
            'total_amount': 'mean'
        }).round(2)
        
        
        peak_hours = self.trip_data.groupby('pickup_hour').size()
        peak_hours = peak_hours.sort_values(ascending=False)
        
        return {
            'hourly_patterns': hourly_patterns.to_dict(),
            'daily_patterns': daily_patterns.to_dict(),
            'peak_hours': peak_hours.to_dict()
        }

if __name__ == "__main__":
    
    processor = NYCDataProcessor(
        trip_data_path='yellow_tripdata_2025-06.parquet',
        zone_data_path='taxi_zone_lookup.csv'
    )
    
    
    processed_data = processor.load_and_preprocess()
    
    
    analysis = processor.get_traffic_analysis()
    print("Traffic Analysis:", analysis)
    
    
    zone_stats = processor.get_zone_statistics()
    print(f"Zone Statistics: {len(zone_stats)} zones analyzed")
    
    
    scenarios = processor.get_optimization_scenarios()
    print(f"Optimization Scenarios: {len(scenarios)} routes identified")
    
    
    congestion = processor.calculate_congestion_metrics()
    print("Congestion Metrics:", congestion) 