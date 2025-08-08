#!/usr/bin/env python3
"""
CICIDS2017 Dataset Processor for AdvTG Framework

This script processes the CICIDS2017 dataset and converts it to HTTP traffic format
for training different components of the AdvTG framework with optimized data allocation.

Data allocation strategy:
- DL Training: 10% (dl_train.json)
- LLM Training: 60% (llm_train.json) 
- RL Training: 12% (rl_train.json)
- Validation: 5% (val.json)
- Testing: 13% (test.json)

Author: AdvTG Team
"""

import os
import json
import zipfile
import csv
import random
from typing import List, Dict, Any, Optional
from pathlib import Path

class CICIDS2017Processor:
    def __init__(self, dataset_dir: str = None):
        """Initialize the CICIDS2017 processor."""
        self.dataset_dir = dataset_dir or os.path.dirname(os.path.abspath(__file__))
        self.zip_file = os.path.join(self.dataset_dir, "MachineLearningCSV.zip")
        self.extracted_dir = os.path.join(self.dataset_dir, "MachineLearningCVE")
        
    def check_and_extract_zip(self) -> bool:
        """Check if zip file exists and extract if needed."""
        print("Step 1: Checking and extracting CICIDS2017 dataset...")
        
        if not os.path.exists(self.zip_file):
            print(f"❌ Error: {self.zip_file} not found!")
            print("Please download MachineLearningCSV.zip from:")
            print("https://www.unb.ca/cic/datasets/ids-2017.html")
            return False
        
        if os.path.exists(self.extracted_dir):
            print(f"✅ Dataset already extracted at {self.extracted_dir}")
            return True
        
        try:
            print(f"📦 Extracting {self.zip_file}...")
            with zipfile.ZipFile(self.zip_file, 'r') as zip_ref:
                zip_ref.extractall(self.dataset_dir)
            print("✅ Extraction completed successfully!")
            return True
        except Exception as e:
            print(f"❌ Error extracting zip file: {e}")
            return False
    
    def get_csv_files(self) -> List[str]:
        """Get list of CICIDS2017 CSV files."""
        csv_files = [
            "Monday-WorkingHours.pcap_ISCX.csv",
            "Tuesday-WorkingHours.pcap_ISCX.csv", 
            "Wednesday-workingHours.pcap_ISCX.csv",
            "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
            "Friday-WorkingHours-Morning.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
        ]
        
        existing_files = []
        for csv_file in csv_files:
            file_path = os.path.join(self.extracted_dir, csv_file)
            if os.path.exists(file_path):
                existing_files.append(file_path)
            else:
                print(f"⚠️  Warning: {csv_file} not found")
        
        return existing_files
    
    def load_csv_file(self, file_path: str, max_samples: int = None) -> List[Dict]:
        """Load data from a CSV file. If max_samples is None, load all data."""
        print(f"📊 Processing {os.path.basename(file_path)}...")
        
        try:
            data = []
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read header
                header = f.readline().strip().split(',')
                header = [col.strip() for col in header]  # Clean column names
                
                # Read data rows
                reader = csv.reader(f)
                rows = list(reader)
                total_rows = len(rows)
                
                # Sample data if max_samples is specified
                if max_samples is not None and len(rows) > max_samples:
                    rows = random.sample(rows, max_samples)
                    print(f"  📝 Sampled {max_samples} from {total_rows} total rows")
                else:
                    print(f"  📝 Loading all {total_rows} rows")
                
                # Convert rows to dictionaries
                for row in rows:
                    if len(row) == len(header):
                        row_dict = {}
                        for i, col in enumerate(header):
                            try:
                                # Try to convert to number if possible
                                if row[i].strip():
                                    try:
                                        row_dict[col] = float(row[i]) if '.' in row[i] else int(row[i])
                                    except ValueError:
                                        row_dict[col] = row[i].strip()
                                else:
                                    row_dict[col] = 0
                            except IndexError:
                                row_dict[col] = 0
                        data.append(row_dict)
            
            print(f"  ✅ Loaded {len(data)} samples")
            return data
            
        except Exception as e:
            print(f"  ❌ Error loading {file_path}: {e}")
            return []
    
    def map_attack_label(self, label: str) -> str:
        """Map CICIDS2017 labels to binary classification."""
        if isinstance(label, (int, float)):
            label = str(label)
        
        label = label.strip().upper()
        if label == 'BENIGN':
            return 'Benign'
        else:
            return 'Malicious'
    
    def create_synthetic_http(self, row: Dict, source_name: str) -> Dict[str, Any]:
        """Convert network flow features to synthetic HTTP request format."""
        # Extract network features with safe defaults
        src_ip = str(row.get('Source IP', f"192.168.1.{random.randint(10, 254)}"))
        dst_ip = str(row.get('Destination IP', f"10.0.0.{random.randint(1, 100)}"))
        src_port = int(row.get('Source Port', random.randint(1024, 65535)))
        dst_port = int(row.get('Destination Port', random.choice([80, 443, 8080])))
        flow_duration = float(row.get('Flow Duration', 0))
        fwd_packets = int(row.get('Total Fwd Packets', 1))
        bwd_packets = int(row.get('Total Backward Packets', 1))
        packet_length_mean = float(row.get('Packet Length Mean', 0))
        label = str(row.get('Label', 'BENIGN'))
        
        # Generate synthetic HTTP request line
        methods = ['GET', 'POST', 'PUT', 'DELETE', 'HEAD']
        method = methods[hash(src_ip + str(src_port)) % len(methods)]
        
        # Create path based on attack type
        if 'Web Attack' in label:
            paths = ['/admin.php', '/login.php', '/search.php?q=<script>', '/upload.php']
        elif 'DoS' in label or 'DDoS' in label:
            paths = ['/api/data', '/heavy-computation', '/resource-intensive']
        elif 'PortScan' in label:
            paths = ['/status', '/info', '/debug', '/admin']
        elif 'Brute Force' in label:
            paths = ['/login', '/auth', '/admin/login']
        else:  # Benign
            paths = ['/index.html', '/about.html', '/contact.php', '/api/users']
        
        path = paths[hash(str(fwd_packets)) % len(paths)]
        if method == 'GET' and random.random() > 0.7:
            path += f"?id={random.randint(1, 1000)}"
        
        request_line = f"{method} {path} HTTP/1.1"
        
        # Generate synthetic headers
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        ]
        
        headers = {
            "Host": f"{dst_ip}:{dst_port}",
            "User-Agent": user_agents[hash(src_ip) % len(user_agents)],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive" if flow_duration > 5000 else "close"
        }
        
        # Add attack-specific headers
        if 'Web Attack' in label:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            if 'XSS' in label:
                headers["Referer"] = "http://malicious-site.com"
        elif method in ['POST', 'PUT']:
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(int(packet_length_mean))
        
        # Generate synthetic body
        if method in ['POST', 'PUT']:
            if 'Web Attack' in label:
                if 'SQL Injection' in label:
                    body = "username=admin' OR '1'='1&password=password"
                elif 'XSS' in label:
                    body = "comment=<script>alert('XSS')</script>&submit=true"
                else:
                    body = "data=malicious_payload&action=exploit"
            elif 'Brute Force' in label:
                body = f"username=admin&password=password{random.randint(1, 1000)}"
            else:  # Benign POST
                body = f'{{"user_id": {random.randint(1, 1000)}, "action": "update", "timestamp": {int(flow_duration)}}}'
        else:
            body = ""
        
        # Create the HTTP-like structure compatible with data_processing.py
        http_item = {
            "Request Line": request_line,
            "Request Headers": headers,
            "Request Body": body,
            "Label": self.map_attack_label(label),
            "Source": source_name,
            "Original_Features": {
                "Original_Label": label,
                "Flow_Duration": flow_duration,
                "Fwd_Packets": fwd_packets,
                "Bwd_Packets": bwd_packets,
                "Packet_Length_Mean": packet_length_mean,
                "Source_IP": src_ip,
                "Destination_IP": dst_ip,
                "Source_Port": src_port,
                "Destination_Port": dst_port
            }
        }
        
        return http_item
    
    def process_all_files(self, max_samples_per_file: int = None) -> List[Dict[str, Any]]:
        """Process all CSV files and convert to HTTP format. If max_samples_per_file is None, process all data."""
        print("\nStep 2: Converting CICIDS2017 data to HTTP format...")
        
        csv_files = self.get_csv_files()
        if not csv_files:
            print("❌ No CSV files found!")
            return []
        
        all_http_data = []
        
        for csv_file in csv_files:
            source_name = os.path.basename(csv_file).split('.')[0]
            rows = self.load_csv_file(csv_file, max_samples_per_file)
            
            if rows:
                http_data = []
                for row in rows:
                    try:
                        http_item = self.create_synthetic_http(row, source_name)
                        http_data.append(http_item)
                    except Exception as e:
                        print(f"  ⚠️  Error processing row: {e}")
                        continue
                
                all_http_data.extend(http_data)
                print(f"  ✅ Converted {len(http_data)} samples from {source_name}")
        
        return all_http_data
    
    def save_processed_data(self, http_data: List[Dict[str, Any]], 
                          output_files: Dict[str, str] = None) -> Dict[str, str]:
        """Save processed data to JSON files with new data allocation strategy."""
        print("\nStep 3: Saving processed data with optimized allocation...")
        
        if output_files is None:
            output_files = {
                'dl_train': 'dl_train.json',      # DL训练用小数据集
                'llm_train': 'llm_train.json',    # LLM微调用大数据集
                'rl_train': 'rl_train.json',     # RL训练用中等数据集
                'val': 'val.json',           # 验证集
                'test': 'test.json',             # 公共测试集
                'full': 'cicids2017_full.json'         # 完整数据集
            }
        
        if not http_data:
            print("❌ No data to save!")
            return {}
        
        # Shuffle data for random distribution
        random.shuffle(http_data)
        
        # Optimized data allocation strategy:
        # - DL Training: 10% (small dataset for deep learning models)
        # - LLM Training: 60% (large dataset for LLM fine-tuning)  
        # - RL Training: 12% (medium dataset for reinforcement learning)
        # - Validation: 5% (validation set for hyperparameter tuning)
        # - Test: 13% (test set for final evaluation)
        dl_split = 0.10      # 10% for DL training
        llm_split = 0.60     # 60% for LLM training
        rl_split = 0.12      # 12% for RL training
        val_split = 0.05     # 5% for validation
        test_split = 0.13    # 13% for testing
        
        total_size = len(http_data)
        dl_size = int(total_size * dl_split)
        llm_size = int(total_size * llm_split)
        rl_size = int(total_size * rl_split)
        val_size = int(total_size * val_split)
        test_size = total_size - dl_size - llm_size - rl_size - val_size  # Remaining for test
        
        # Split data
        idx = 0
        dl_data = http_data[idx:idx + dl_size]
        idx += dl_size
        
        llm_data = http_data[idx:idx + llm_size]
        idx += llm_size
        
        rl_data = http_data[idx:idx + rl_size]
        idx += rl_size
        
        val_data = http_data[idx:idx + val_size]
        idx += val_size
        
        test_data = http_data[idx:idx + test_size]
        
        saved_files = {}
        
        # Save DL training data (small)
        dl_path = os.path.join(self.dataset_dir, output_files['dl_train'])
        with open(dl_path, 'w', encoding='utf-8') as f:
            json.dump(dl_data, f, indent=2, ensure_ascii=False)
        saved_files['dl_train'] = dl_path
        print(f"✅ DL training data saved: {dl_path} ({len(dl_data)} samples, {dl_split*100}%)")
        
        # Save LLM training data (large)
        llm_path = os.path.join(self.dataset_dir, output_files['llm_train'])
        with open(llm_path, 'w', encoding='utf-8') as f:
            json.dump(llm_data, f, indent=2, ensure_ascii=False)
        saved_files['llm_train'] = llm_path
        print(f"✅ LLM training data saved: {llm_path} ({len(llm_data)} samples, {llm_split*100}%)")
        
        # Save RL training data (medium)
        rl_path = os.path.join(self.dataset_dir, output_files['rl_train'])
        with open(rl_path, 'w', encoding='utf-8') as f:
            json.dump(rl_data, f, indent=2, ensure_ascii=False)
        saved_files['rl_train'] = rl_path
        print(f"✅ RL training data saved: {rl_path} ({len(rl_data)} samples, {rl_split*100}%)")
        
        # Save validation data
        val_path = os.path.join(self.dataset_dir, output_files['val'])
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
        saved_files['val'] = val_path
        print(f"✅ Validation data saved: {val_path} ({len(val_data)} samples, {val_split*100}%)")
        
        # Save test data (common)
        test_path = os.path.join(self.dataset_dir, output_files['test'])
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        saved_files['test'] = test_path
        print(f"✅ Test data saved: {test_path} ({len(test_data)} samples, {test_split*100:.1f}%)")
        
        # Save full dataset
        full_path = os.path.join(self.dataset_dir, output_files['full'])
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(http_data, f, indent=2, ensure_ascii=False)
        saved_files['full'] = full_path
        print(f"✅ Full dataset saved: {full_path} ({len(http_data)} samples)")
        
        # Create legacy compatibility files
        # Keep train_data2.json for backward compatibility (pointing to DL data)
        legacy_train_path = os.path.join(self.dataset_dir, 'train_data2.json')
        with open(legacy_train_path, 'w', encoding='utf-8') as f:
            json.dump(dl_data, f, indent=2, ensure_ascii=False)
        saved_files['legacy_train'] = legacy_train_path
        print(f"✅ Legacy train file saved: {legacy_train_path} (for backward compatibility)")
        
        # Keep test2.json for backward compatibility
        legacy_test_path = os.path.join(self.dataset_dir, 'test2.json')
        with open(legacy_test_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        saved_files['legacy_test'] = legacy_test_path
        print(f"✅ Legacy test file saved: {legacy_test_path} (for backward compatibility)")
        
        return saved_files
    
    def print_data_summary(self, http_data: List[Dict[str, Any]]):
        """Print summary statistics of the processed data."""
        print("\n📊 Data Summary:")
        print("=" * 50)
        
        # Label distribution
        label_counts = {}
        source_counts = {}
        
        for item in http_data:
            label = item.get('Label', 'Unknown')
            source = item.get('Source', 'Unknown')
            
            label_counts[label] = label_counts.get(label, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1
        
        print(f"Total samples: {len(http_data)}")
        
        print("\nLabel distribution:")
        for label, count in sorted(label_counts.items()):
            percentage = (count / len(http_data)) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")
        
        print("\nSource distribution:")
        for source, count in sorted(source_counts.items()):
            percentage = (count / len(http_data)) * 100
            print(f"  {source}: {count} ({percentage:.1f}%)")
    
    def run_full_pipeline(self, max_samples_per_file: int = None):
        """Run the complete processing pipeline. If max_samples_per_file is None, process all data."""
        print("🚀 Starting CICIDS2017 processing pipeline...")
        print("=" * 60)
        
        if max_samples_per_file is None:
            print("📋 Processing ALL data (no sampling)")
        else:
            print(f"📋 Processing {max_samples_per_file} samples per file")
        
        # Step 1: Extract data
        if not self.check_and_extract_zip():
            return False
        
        # Step 2: Process files
        http_data = self.process_all_files(max_samples_per_file)
        if not http_data:
            print("❌ No data processed!")
            return False
        
        # Step 3: Save data
        saved_files = self.save_processed_data(http_data)
        
        # Step 4: Print summary
        self.print_data_summary(http_data)
        
        print("\n🎉 Processing completed successfully!")
        print("=" * 60)
        print("Generated files for optimized training purposes:")
        print(f"  • DL Training (10%): {saved_files.get('dl_train', 'N/A')}")
        print(f"  • LLM Training (60%): {saved_files.get('llm_train', 'N/A')}")
        print(f"  • RL Training (12%): {saved_files.get('rl_train', 'N/A')}")
        print(f"  • Validation Set (5%): {saved_files.get('val', 'N/A')}")
        print(f"  • Test Data (13%): {saved_files.get('test', 'N/A')}")
        print(f"  • Full Dataset: {saved_files.get('full', 'N/A')}")
        print(f"  • Legacy Files: {saved_files.get('legacy_train', 'N/A')}, {saved_files.get('legacy_test', 'N/A')}")
        
        print("\nUsage examples:")
        print("  DL Training:")
        print("    from DL.data_processing import load_data, prepare_dataset")
        print("    data = load_data('dataset/dl_train_small.json')")
        print("    dataset = prepare_dataset(data)")
        print("  LLM Fine-tuning:")
        print("    # Modify LLM-Finetune/train_llama.py to use 'dataset/llm_train_large.json'")
        
        return True

def main():
    """Main function to run the processing pipeline."""
    # Process ALL data instead of sampling
    MAX_SAMPLES_PER_FILE = None  # None means process all data
    
    processor = CICIDS2017Processor()
    success = processor.run_full_pipeline(max_samples_per_file=MAX_SAMPLES_PER_FILE)
    
    if success:
        print("\n✅ All done! Your CICIDS2017 data is ready for AdvTG training.")
        print("📊 Optimized data split summary:")
        print("  • DL models will use: dl_train.json (10% of data)")
        print("  • LLM fine-tuning will use: llm_train.json (60% of data)")
        print("  • RL training will use: rl_train.json (12% of data)")
        print("  • All models will validate on: val.json (5% of data)")
        print("  • Final evaluation will use: test.json (13% of data)")
    else:
        print("\n❌ Processing failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
