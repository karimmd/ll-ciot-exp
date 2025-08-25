#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json

class NetworkPatternExtractor:
    """
    Extracts network patterns from MedBIoT dataset for realistic testbed conditions
    """
    
    def __init__(self, dataset_dir: str = "../datasets"):
        self.dataset_dir = Path(dataset_dir)
        
    def extract_patterns_from_csv(self, csv_file: Path) -> Dict:
        """Extract network characteristics from a single CSV file"""
        try:
            # Read a sample of the CSV file
            df = pd.read_csv(csv_file, nrows=100)
            
            # Parse filename for device info
            filename = csv_file.stem
            parts = filename.split('_')
            if len(parts) >= 3:
                device_family = parts[0]
                device_type = '_'.join(parts[2:])
            else:
                device_family = "unknown"
                device_type = filename
            
            # Extract numerical columns for analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return None
            
            # Calculate network pattern metrics
            sample_cols = numeric_cols[:5]  # Use first 5 numeric columns
            if len(sample_cols) > 0:
                mean_values = df[sample_cols].mean().mean()
                std_values = df[sample_cols].std().mean()
                
                # Normalize to network characteristics
                bandwidth_factor = max(0.2, min(2.0, mean_values / 50.0))
                latency_factor = max(0.5, min(3.0, std_values / 20.0))
                packet_loss_factor = max(0.0, min(0.05, std_values / 1000.0))
                network_quality = 1.0 / (1.0 + std_values / 100.0)
                
                return {
                    'device_family': device_family,
                    'device_type': device_type,
                    'bandwidth_factor': bandwidth_factor,
                    'latency_factor': latency_factor,
                    'packet_loss_factor': packet_loss_factor,
                    'network_quality': network_quality,
                    'samples_analyzed': len(df)
                }
                
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            return None
    
    def extract_all_patterns(self) -> Dict:
        """Extract patterns from all CSV files in the dataset directory"""
        patterns = {}
        csv_files = list(self.dataset_dir.glob("*.csv"))
        
        print(f"Found {len(csv_files)} CSV files to process")
        
        for csv_file in csv_files:
            pattern = self.extract_patterns_from_csv(csv_file)
            if pattern:
                pattern_key = f"{pattern['device_family']}_{pattern['device_type']}"
                patterns[pattern_key] = pattern
                print(f"Extracted pattern for {pattern_key}")
        
        return patterns
    
    def generate_network_conditions(self, patterns: Dict) -> Dict:
        """Generate realistic network conditions from extracted patterns"""
        conditions = {}
        
        for pattern_key, pattern in patterns.items():
            base_bandwidth = 50.0  # Base bandwidth in Mbps
            dynamic_bandwidth = base_bandwidth * pattern['bandwidth_factor']
            
            conditions[pattern_key] = {
                'bandwidth_mbps': round(dynamic_bandwidth, 1),
                'latency_ms': round(25.0 * pattern['latency_factor'], 1),
                'packet_loss_percent': round(pattern['packet_loss_factor'] * 100, 3),
                'jitter_ms': round(pattern['latency_factor'] * 5.0, 1),
                'reliability_score': round(pattern['network_quality'], 3),
                'device_info': {
                    'family': pattern['device_family'],
                    'type': pattern['device_type']
                }
            }
        
        return conditions
    
    def save_patterns(self, patterns: Dict, conditions: Dict):
        """Save extracted patterns and network conditions"""
        # Save raw patterns
        patterns_file = self.dataset_dir.parent / "results" / "extracted_patterns.json"
        patterns_file.parent.mkdir(exist_ok=True)
        
        with open(patterns_file, 'w') as f:
            json.dump(patterns, f, indent=2)
        print(f"Patterns saved to {patterns_file}")
        
        # Save network conditions
        conditions_file = self.dataset_dir.parent / "configs" / "network_conditions.json"
        conditions_file.parent.mkdir(exist_ok=True)
        
        with open(conditions_file, 'w') as f:
            json.dump(conditions, f, indent=2)
        print(f"Network conditions saved to {conditions_file}")
    
    def generate_summary_report(self, patterns: Dict, conditions: Dict) -> str:
        """Generate summary report of extracted patterns"""
        report = "MedBIoT Network Pattern Extraction Report\n"
        report += "=" * 50 + "\n\n"
        
        report += f"Total patterns extracted: {len(patterns)}\n"
        report += f"Network conditions generated: {len(conditions)}\n\n"
        
        # Device family distribution
        families = {}
        for pattern in patterns.values():
            family = pattern['device_family']
            families[family] = families.get(family, 0) + 1
        
        report += "Device Family Distribution:\n"
        report += "-" * 25 + "\n"
        for family, count in families.items():
            report += f"{family:15}: {count} devices\n"
        
        # Network characteristics summary
        report += "\nNetwork Characteristics Range:\n"
        report += "-" * 30 + "\n"
        
        if conditions:
            bandwidths = [c['bandwidth_mbps'] for c in conditions.values()]
            latencies = [c['latency_ms'] for c in conditions.values()]
            
            report += f"Bandwidth: {min(bandwidths):.1f} - {max(bandwidths):.1f} Mbps\n"
            report += f"Latency: {min(latencies):.1f} - {max(latencies):.1f} ms\n"
        
        report += "\nPattern Examples:\n"
        report += "-" * 15 + "\n"
        
        for i, (key, condition) in enumerate(list(conditions.items())[:5]):
            report += f"{i+1}. {key}:\n"
            report += f"   Bandwidth: {condition['bandwidth_mbps']} Mbps\n"
            report += f"   Latency: {condition['latency_ms']} ms\n"
            report += f"   Reliability: {condition['reliability_score']}\n\n"
        
        return report

def main():
    extractor = NetworkPatternExtractor()
    
    print("Starting MedBIoT Pattern Extraction...")
    print("=" * 40)
    
    # Extract patterns
    patterns = extractor.extract_all_patterns()
    
    if not patterns:
        print("No patterns extracted. Check dataset directory.")
        return
    
    # Generate network conditions
    conditions = extractor.generate_network_conditions(patterns)
    
    # Save results
    extractor.save_patterns(patterns, conditions)
    
    # Generate and print report
    report = extractor.generate_summary_report(patterns, conditions)
    print("\n" + report)
    
    # Save report
    report_file = extractor.dataset_dir.parent / "data" / "pattern_extraction_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Report saved to {report_file}")

if __name__ == "__main__":
    main()