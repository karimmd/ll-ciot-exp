#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import logging

logger = logging.getLogger(__name__)

class MedBIoTAnalyzer:
    """
    Analyzer for MedBIoT dataset to extract network characteristics
    for dynamic network condition modeling in LL-CIoT system
    """
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.patterns = {}
        self.network_profiles = {}
    
    def analyze_traffic_patterns(self) -> Dict:
        """
        Analyze MedBIoT traffic files to extract network characteristics
        """
        if not self.dataset_path.exists():
            logger.error(f"Dataset path not found: {self.dataset_path}")
            return {}
        
        csv_files = list(self.dataset_path.glob("*.csv"))
        if not csv_files:
            logger.warning("No CSV files found in dataset directory")
            return {}
        
        logger.info(f"Analyzing {len(csv_files)} MedBIoT traffic files...")
        
        for csv_file in csv_files:
            try:
                pattern_name = self._extract_pattern_name(csv_file.stem)
                characteristics = self._analyze_single_file(csv_file)
                
                if characteristics:
                    self.patterns[pattern_name] = characteristics
                    logger.debug(f"Processed pattern: {pattern_name}")
                    
            except Exception as e:
                logger.warning(f"Failed to process {csv_file}: {e}")
        
        logger.info(f"Extracted {len(self.patterns)} traffic patterns")
        return self.patterns
    
    def _extract_pattern_name(self, filename: str) -> str:
        """Extract pattern name from filename"""
        parts = filename.split('_')
        if len(parts) >= 3:
            family = parts[0]  # bashlite, mirai, torii
            device = '_'.join(parts[2:])  # fan, light, lock, etc.
            return f"{family}_{device}"
        return filename
    
    def _analyze_single_file(self, csv_file: Path) -> Dict:
        """Analyze a single CSV file for network characteristics"""
        try:
            # Read sample of data for analysis
            df = pd.read_csv(csv_file, nrows=1000)
            
            if df.empty:
                return {}
            
            # Identify numeric columns that represent network metrics
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Focus on columns that might represent network characteristics
            network_cols = [col for col in numeric_cols if any(
                keyword in col.lower() for keyword in 
                ['weight', 'mean', 'std', 'flow', 'packet', 'byte', 'time', 'dur']
            )]
            
            if not network_cols:
                # Fallback to all numeric columns
                network_cols = numeric_cols[:5]  # Take first 5 numeric columns
            
            if not network_cols:
                return {}
            
            # Calculate statistics
            stats = {}
            for col in network_cols[:5]:  # Limit to 5 columns for efficiency
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    stats[col] = {
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'median': float(col_data.median())
                    }
            
            # Derive network characteristics from statistics
            return self._derive_network_characteristics(stats)
            
        except Exception as e:
            logger.debug(f"Error analyzing {csv_file}: {e}")
            return {}
    
    def _derive_network_characteristics(self, stats: Dict) -> Dict:
        """Derive network characteristics from traffic statistics"""
        if not stats:
            return {}
        
        # Aggregate statistics across all columns
        all_means = [s['mean'] for s in stats.values()]
        all_stds = [s['std'] for s in stats.values()]
        all_ranges = [s['max'] - s['min'] for s in stats.values()]
        
        avg_mean = np.mean(all_means) if all_means else 50.0
        avg_std = np.mean(all_stds) if all_stds else 10.0
        avg_range = np.mean(all_ranges) if all_ranges else 100.0
        
        # Map to network characteristics
        # These mappings are heuristic and represent how traffic patterns 
        # might influence network conditions in IoT deployments
        
        # Bandwidth factor: higher mean values suggest more data throughput
        bandwidth_factor = max(0.3, min(2.0, avg_mean / 100.0))
        
        # Latency factor: higher variance suggests more unstable conditions
        latency_factor = max(0.5, min(3.0, avg_std / 50.0))
        
        # Packet loss factor: high range suggests burst patterns
        packet_loss_factor = max(0.0, min(0.1, avg_range / 10000.0))
        
        # Jitter factor: combination of std and range
        jitter_factor = max(0.1, min(5.0, (avg_std + avg_range/10) / 100.0))
        
        # Network quality: inverse relationship with variance
        network_quality = 1.0 / (1.0 + avg_std / 1000.0)
        network_quality = max(0.1, min(1.0, network_quality))
        
        return {
            'bandwidth_factor': bandwidth_factor,
            'latency_factor': latency_factor,
            'packet_loss_factor': packet_loss_factor,
            'jitter_factor': jitter_factor,
            'network_quality': network_quality,
            'raw_stats': {
                'avg_mean': avg_mean,
                'avg_std': avg_std,
                'avg_range': avg_range
            }
        }
    
    def generate_network_profiles(self, base_bandwidth: float = 50.0, 
                                base_latency: float = 25.0) -> Dict:
        """Generate network profiles based on analyzed patterns"""
        if not self.patterns:
            self.analyze_traffic_patterns()
        
        profiles = {}
        
        for pattern_name, characteristics in self.patterns.items():
            # Apply factors to base network conditions
            dynamic_bandwidth = base_bandwidth * characteristics['bandwidth_factor']
            dynamic_latency = base_latency * characteristics['latency_factor']
            packet_loss = characteristics['packet_loss_factor'] * 100  # Convert to percentage
            jitter = characteristics['jitter_factor'] * 10  # Scale for milliseconds
            
            profiles[pattern_name] = {
                'bandwidth_mbps': round(dynamic_bandwidth, 2),
                'latency_ms': round(dynamic_latency, 2),
                'packet_loss_percent': round(packet_loss, 3),
                'jitter_ms': round(jitter, 2),
                'reliability_score': round(characteristics['network_quality'], 3),
                'device_family': pattern_name.split('_')[0],
                'device_type': '_'.join(pattern_name.split('_')[1:])
            }
        
        self.network_profiles = profiles
        return profiles
    
    def export_profiles(self, output_path: str) -> str:
        """Export network profiles to JSON file"""
        if not self.network_profiles:
            self.generate_network_profiles()
        
        export_data = {
            'description': 'Network profiles derived from MedBIoT IoT traffic patterns',
            'base_conditions': {
                'base_bandwidth_mbps': 50.0,
                'base_latency_ms': 25.0
            },
            'profiles': self.network_profiles,
            'total_patterns': len(self.network_profiles)
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Network profiles exported to: {output_file}")
        return str(output_file)
    
    def get_random_profile(self) -> Dict:
        """Get a random network profile for testing"""
        if not self.network_profiles:
            self.generate_network_profiles()
        
        if self.network_profiles:
            import random
            pattern_name = random.choice(list(self.network_profiles.keys()))
            return {
                'pattern_name': pattern_name,
                **self.network_profiles[pattern_name]
            }
        
        # Fallback profile
        return {
            'pattern_name': 'default',
            'bandwidth_mbps': 50.0,
            'latency_ms': 25.0,
            'packet_loss_percent': 0.1,
            'jitter_ms': 2.0,
            'reliability_score': 0.9,
            'device_family': 'unknown',
            'device_type': 'unknown'
        }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='MedBIoT Dataset Analyzer')
    parser.add_argument('--dataset-path', default='../datasets/', 
                       help='Path to MedBIoT dataset directory')
    parser.add_argument('--output', default='../data/network_profiles.json',
                       help='Output file for network profiles')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize analyzer
    analyzer = MedBIoTAnalyzer(args.dataset_path)
    
    # Analyze patterns and generate profiles
    patterns = analyzer.analyze_traffic_patterns()
    profiles = analyzer.generate_network_profiles()
    
    # Export results
    output_file = analyzer.export_profiles(args.output)
    
    print(f"Analysis complete:")
    print(f"  Patterns analyzed: {len(patterns)}")
    print(f"  Network profiles generated: {len(profiles)}")
    print(f"  Output file: {output_file}")
    
    # Show sample profile
    sample = analyzer.get_random_profile()
    print(f"  Sample profile: {sample}")

if __name__ == "__main__":
    main()