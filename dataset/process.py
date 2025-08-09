#!/usr/bin/env python3
"""
CICIDS2017 Dataset Processor for AdvTG Framework

This script processes the CICIDS2017 dataset and converts it to HTTP traffic format
for training different components of the AdvTG framework with optimized data allocation.

Data allocation strategy:
- DL Training: 15,000 samples (dl_train.json) - Balanced dataset for detection models
- LLM Training: 80,000 samples (llm_train.json) - Large dataset for LLM fine-tuning  
- RL Training: 25,000 samples (rl_train.json) - Medium dataset for adversarial generation
- Validation: 10,000 samples (val.json) - Hyperparameter tuning
- Testing: 20,000 samples (test.json) - Final evaluation
- Reserved: Remaining samples (for future experiments)

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
    
    def create_balanced_dl_dataset(self, http_data: List[Dict[str, Any]], dl_size: int) -> List[Dict[str, Any]]:
        """Create balanced dataset for DL training with 1:1 malicious:benign ratio."""
        # Separate malicious and benign samples
        malicious_samples = [item for item in http_data if item.get('Label') == 'Malicious']
        benign_samples = [item for item in http_data if item.get('Label') == 'Benign']
        
        print(f"📊 Creating balanced DL dataset:")
        print(f"   Available - Malicious: {len(malicious_samples)}, Benign: {len(benign_samples)}")
        
        # Calculate required samples for 1:1 balance
        samples_per_class = dl_size // 2
        
        # Check if we have enough samples
        if len(malicious_samples) < samples_per_class:
            print(f"⚠️  Warning: Not enough malicious samples! Need {samples_per_class}, have {len(malicious_samples)}")
            samples_per_class = len(malicious_samples)
        
        if len(benign_samples) < samples_per_class:
            print(f"⚠️  Warning: Not enough benign samples! Need {samples_per_class}, have {len(benign_samples)}")
            samples_per_class = min(samples_per_class, len(benign_samples))
        
        # Randomly sample from each class
        random.shuffle(malicious_samples)
        random.shuffle(benign_samples)
        
        selected_malicious = malicious_samples[:samples_per_class]
        selected_benign = benign_samples[:samples_per_class]
        
        # Combine and shuffle
        balanced_dataset = selected_malicious + selected_benign
        random.shuffle(balanced_dataset)
        
        print(f"✅ Created balanced DL dataset: {len(selected_malicious)} malicious + {len(selected_benign)} benign = {len(balanced_dataset)} total")
        print(f"   Ratio - Malicious: {len(selected_malicious)/len(balanced_dataset)*100:.1f}%, Benign: {len(selected_benign)/len(balanced_dataset)*100:.1f}%")
        
        return balanced_dataset

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
                'test': 'test.json'             # 公共测试集
                # 'full': 'cicids2017_full.json'         # 完整数据集
            }
        
        if not http_data:
            print("❌ No data to save!")
            return {}
        
        # Shuffle data for random distribution
        random.shuffle(http_data)
        
        # Fixed sample allocation strategy (固定样本数量分配):
        # Stage 1 - DL Detection Models: 15,000 samples (balanced malicious:benign = 1:1)
        # Stage 2 - LLM Fine-tuning: 80,000 samples (large dataset for domain adaptation)  
        # Stage 3 - RL Adversarial Generation: 25,000 samples (medium dataset for policy optimization)
        # - Validation: 10,000 samples (validation set for hyperparameter tuning)
        # - Test: 20,000 samples (sufficient for final evaluation)
        # - Reserved: Remaining samples (保留用于后续实验或其他用途)
        
        dl_size = 15000      # Stage 1: DL training samples (需要1:1平衡)
        llm_size = 80000     # Stage 2: LLM training samples (大数据集用于域适应)
        rl_size = 25000      # Stage 3: RL training samples (中等数据集用于策略优化)
        val_size = 10000     # Validation samples (更大的验证集用于更可靠的超参数调优)
        test_size = 20000    # Test samples
        
        total_size = len(http_data)
        required_samples = dl_size + llm_size + rl_size + val_size + test_size
        
        if total_size < required_samples:
            print(f"⚠️  Warning: Not enough data! Required: {required_samples}, Available: {total_size}")
            print("Adjusting allocation proportionally...")
            scale_factor = total_size / required_samples
            dl_size = int(dl_size * scale_factor)
            llm_size = int(llm_size * scale_factor)
            rl_size = int(rl_size * scale_factor)
            val_size = int(val_size * scale_factor)
            test_size = total_size - dl_size - llm_size - rl_size - val_size
        else:
            print(f"✅ Sufficient data: Required: {required_samples}, Available: {total_size}")
            print(f"📊 Reserved for future use: {total_size - required_samples} samples")
        
        # Split data with balanced DL dataset
        idx = 0
        
        # Create balanced DL dataset (1:1 ratio)
        dl_data = self.create_balanced_dl_dataset(http_data, dl_size)
        
        # Remove DL samples from the remaining pool to avoid overlap
        remaining_data = http_data.copy()
        for dl_item in dl_data:
            if dl_item in remaining_data:
                remaining_data.remove(dl_item)
        
        print(f"📊 Remaining data pool after DL allocation: {len(remaining_data)} samples")
        
        # Shuffle remaining data for other allocations
        random.shuffle(remaining_data)
        
        # Allocate remaining data
        idx = 0
        llm_data = remaining_data[idx:idx + llm_size] if len(remaining_data) >= llm_size else remaining_data[idx:]
        idx += len(llm_data)
        
        rl_data = remaining_data[idx:idx + rl_size] if len(remaining_data) >= idx + rl_size else remaining_data[idx:idx + min(rl_size, len(remaining_data) - idx)]
        idx += len(rl_data)
        
        val_data = remaining_data[idx:idx + val_size] if len(remaining_data) >= idx + val_size else remaining_data[idx:idx + min(val_size, len(remaining_data) - idx)]
        idx += len(val_data)
        
        test_data = remaining_data[idx:idx + test_size] if len(remaining_data) >= idx + test_size else remaining_data[idx:]
        
        saved_files = {}
        
        # Save DL training data (balanced)
        dl_path = os.path.join(self.dataset_dir, output_files['dl_train'])
        with open(dl_path, 'w', encoding='utf-8') as f:
            json.dump(dl_data, f, indent=2, ensure_ascii=False)
        saved_files['dl_train'] = dl_path
        
        # Print DL dataset balance
        dl_malicious = len([item for item in dl_data if item.get('Label') == 'Malicious'])
        dl_benign = len([item for item in dl_data if item.get('Label') == 'Benign'])
        print(f"✅ DL training data saved: {dl_path} ({len(dl_data)} samples)")
        print(f"   └─ Balance: {dl_malicious} malicious + {dl_benign} benign (ratio: {dl_malicious/len(dl_data)*100:.1f}%:{dl_benign/len(dl_data)*100:.1f}%)")
        
        # Save LLM training data (moderate size)
        llm_path = os.path.join(self.dataset_dir, output_files['llm_train'])
        with open(llm_path, 'w', encoding='utf-8') as f:
            json.dump(llm_data, f, indent=2, ensure_ascii=False)
        saved_files['llm_train'] = llm_path
        print(f"✅ LLM training data saved: {llm_path} ({len(llm_data)} samples)")
        
        # Save RL training data (compact)
        rl_path = os.path.join(self.dataset_dir, output_files['rl_train'])
        with open(rl_path, 'w', encoding='utf-8') as f:
            json.dump(rl_data, f, indent=2, ensure_ascii=False)
        saved_files['rl_train'] = rl_path
        print(f"✅ RL training data saved: {rl_path} ({len(rl_data)} samples)")
        
        # Save validation data
        val_path = os.path.join(self.dataset_dir, output_files['val'])
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
        saved_files['val'] = val_path
        print(f"✅ Validation data saved: {val_path} ({len(val_data)} samples)")
        
        # Save test data (common)
        test_path = os.path.join(self.dataset_dir, output_files['test'])
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        saved_files['test'] = test_path
        print(f"✅ Test data saved: {test_path} ({len(test_data)} samples)")
        
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
        print(f"  • Stage 1 - DL Training (15K): {saved_files.get('dl_train', 'N/A')}")
        print(f"  • Stage 2 - LLM Training (80K): {saved_files.get('llm_train', 'N/A')}")
        print(f"  • Stage 3 - RL Training (25K): {saved_files.get('rl_train', 'N/A')}")
        print(f"  • Validation Set (10K): {saved_files.get('val', 'N/A')}")
        print(f"  • Test Data (20K): {saved_files.get('test', 'N/A')}")
        # print(f"  • Full Dataset: {saved_files.get('full', 'N/A')}")
        # print(f"  • Legacy Files: {saved_files.get('legacy_train', 'N/A')}, {saved_files.get('legacy_test', 'N/A')}")
        
        print("\nUsage examples:")
        print("  Stage 1 - DL Training (Detection Models):")
        print("    from DL.data_processing import load_data, prepare_dataset")
        print("    data = load_data('dataset/dl_train.json')  # 15K balanced samples")
        print("    dataset = prepare_dataset(data)")
        print("  Stage 2 - LLM Fine-tuning (Domain Adaptation):")
        print("    # Use 'dataset/llm_train.json' with 80K samples")
        print("  Stage 3 - RL Training (Adversarial Generation):")
        print("    # Use 'dataset/rl_train.json' with 25K samples")
        
        return True

def main():
    """Main function to run the processing pipeline."""
    # Process ALL data instead of sampling
    MAX_SAMPLES_PER_FILE = None  # None means process all data
    
    processor = CICIDS2017Processor()
    success = processor.run_full_pipeline(max_samples_per_file=MAX_SAMPLES_PER_FILE)
    
    if success:
        print("\n✅ All done! Your CICIDS2017 data is ready for AdvTG training.")
        print("📊 Three-stage allocation summary:")
        print("  • Stage 1 - DL Detection Models: dl_train.json (15,000 samples)")
        print("    └─ Balanced dataset: ~7,500 malicious + ~7,500 benign (1:1 ratio)")
        print("  • Stage 2 - LLM Fine-tuning: llm_train.json (80,000 samples)")
        print("    └─ Large dataset for domain adaptation and traffic pattern learning")
        print("  • Stage 3 - RL Adversarial Generation: rl_train.json (25,000 samples)")
        print("    └─ Medium dataset for policy optimization and reward feedback")
        print("  • Validation: val.json (10,000 samples)")
        print("  • Testing: test.json (20,000 samples)")
        print("  • Total allocated: 150,000 samples")
        print("  • Remaining samples: Reserved for future experiments")
    else:
        print("\n❌ Processing failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
