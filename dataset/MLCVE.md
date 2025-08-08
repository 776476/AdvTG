# CICIDS2017 数据

## # AdvTG Dataset Setup Guide

This directory contains the dataset processing scripts for the AdvTG framework. Due to GitHub's file size limitations, large dataset files are not included in the repository.

## Required Dataset Files

### CICIDS2017 Dataset

You need to manually download the following files:

1. **MachineLearningCSV.zip** (224 MB)
   - Source: https://www.unb.ca/cic/datasets/ids-2017.html
   - Direct link: http://cicresearch.ca/CICDataset/CIC-IDS-2017/
   - Place in: `dataset/MachineLearningCSV.zip`

2. **GeneratedLabelledFlows.zip** (271 MB) - Optional
   - Source: Same as above
   - Contains raw PCAP flows with labels
   - Place in: `dataset/GeneratedLabelledFlows.zip`

## Quick Setup

1. Download the required files to the `dataset/` directory:
   ```bash
   cd dataset/
   wget http://cicresearch.ca/CICDataset/CIC-IDS-2017/MachineLearningCSV.zip
   ```

2. Run the processing script:
   ```bash
   python process_cicids2017.py
   ```

3. This will generate the training data files:
   - `train_data2.json` - Training data for AdvTG
   - `test2.json` - Test data for evaluation
   - `cicids2017_full.json` - Complete processed dataset

## File Structure After Setup

```
dataset/
├── README_DATASET.md              # This file
├── process_cicids2017.py          # Main processing script
├── MachineLearningCSV.zip         # Downloaded CICIDS2017 data (224MB)
├── GeneratedLabelledFlows.zip     # Downloaded PCAP flows (271MB) [Optional]
├── MachineLearningCVE/            # Extracted CSV files (auto-generated)
├── train_data2.json               # Processed training data (auto-generated)
├── test2.json                     # Processed test data (auto-generated)
└── cicids2017_full.json           # Full processed dataset (auto-generated)
```

## Data Format

The processed JSON files follow this format:
```json
[
  {
    "Request Line": "GET /api/data HTTP/1.1",
    "Request Headers": {"Host": "example.com", "User-Agent": "..."},
    "Request Body": "",
    "Label": "Malicious",
    "Source": "CICIDS2017"
  }
]
```

## Dataset Information

- **Total samples**: ~80,000 network flows
- **Classes**: Benign and Malicious traffic
- **Features**: HTTP request components extracted from network flows
- **Attacks included**: DoS, DDoS, Web attacks, Brute Force, Botnet, etc.

## Citation

If you use the CICIDS2017 dataset, please cite:

```bibtex
@inproceedings{sharafaldin2018toward,
  title={Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization},
  author={Sharafaldin, Iman and Lashkari, Arash Habibi and Ghorbani, Ali A},
  booktitle={4th International Conference on Information Systems Security and Privacy (ICISSP)},
  pages={108--116},
  year={2018}
}
```

### 文件列表：

Monday-WorkingHours.pcap_ISCX.csv - 正常流量
Tuesday-WorkingHours.pcap_ISCX.csv - 暴力破解攻击
Wednesday-workingHours.pcap_ISCX.csv - DoS/DDoS攻击
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv - Web攻击
Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv - 渗透攻击
Friday-WorkingHours-Morning.pcap_ISCX.csv - 僵尸网络攻击
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv - 端口扫描
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv - DDoS攻击
数据

## 概述

这个脚本 `process_cicids2017.py` 提供了完整的自动化流程，从解压缩 CICIDS2017 数据集到转换为 AdvTG 框架所需的格式。

## 功能特性

✅ **自动解压缩** MachineLearningCSV.zip  
✅ **批量处理** 所有 CSV 文件  
✅ **智能采样** 避免内存溢出  
✅ **格式转换** 网络流特征 → HTTP 请求格式  
✅ **数据分割** 自动生成训练/测试集  
✅ **统计报告** 详细的数据分布分析  

## 使用方法

### 1. 准备工作

确保 `MachineLearningCSV.zip` 文件在 `dataset` 目录下：
```
AdvTG/
├── dataset/
│   ├── MachineLearningCSV.zip  ← 确保这个文件存在
│   └── process_cicids2017.py   ← 运行这个脚本
```

### 2. 运行脚本

```bash
cd AdvTG/dataset
python process_cicids2017.py
```

### 3. 处理流程

脚本会自动执行以下步骤：

#### Step 1: 检查和解压缩
- 检查 `MachineLearningCSV.zip` 是否存在
- 自动解压缩到 `MachineLearningCVE/` 目录
- 验证所有 CSV 文件

#### Step 2: 数据转换
- 加载所有 8 个 CSV 文件
- 每个文件采样最多 1000 条记录（可调整）
- 将网络流特征转换为合成的 HTTP 请求格式
- 保持原始特征信息以供参考

#### Step 3: 保存结果
- `train_data2.json` - 训练数据（80%）
- `test2.json` - 测试数据（20%）
- `cicids2017_full.json` - 完整数据集

## 输出格式

转换后的数据格式与现有的 `data_processing.py` 完全兼容：

```json
{
  "Request Line": "GET /admin.php HTTP/1.1",
  "Request Headers": {
    "Host": "10.0.0.50:80",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive"
  },
  "Request Body": "username=admin' OR '1'='1&password=password",
  "Label": "Malicious",
  "Source": "Thursday-WorkingHours-Morning-WebAttacks",
  "Original_Features": {
    "Original_Label": "Web Attack - SQL Injection",
    "Flow_Duration": 120000,
    "Fwd_Packets": 15,
    "Bwd_Packets": 12,
    "Source_IP": "192.168.1.100",
    "Destination_IP": "10.0.0.50"
  }
}
```

## 配置选项

### 调整采样数量
```python
# 在 main() 函数中修改
MAX_SAMPLES_PER_FILE = 1000  # 每个文件最多处理的样本数
```

### 自定义输出文件名
```python
processor = CICIDS2017Processor()
saved_files = processor.save_processed_data(
    http_data,
    output_files={
        'train': 'my_train_data.json',
        'test': 'my_test_data.json',
        'full': 'my_full_data.json'
    }
)
```

## 数据映射说明

### 标签映射
- `BENIGN` → `Benign` (正常流量)
- 所有攻击类型 → `Malicious` (恶意流量)

### HTTP 请求生成规则

1. **请求方法**: 根据源 IP 哈希确定 (GET/POST/PUT/DELETE/HEAD)
2. **路径**: 根据攻击类型生成相应路径
3. **请求头**: 基于网络特征生成合理的 HTTP 头
4. **请求体**: 根据攻击类型生成对应的载荷内容

### 攻击类型特定处理

- **Web Attack**: 生成 SQL 注入、XSS 等攻击载荷
- **DoS/DDoS**: 生成资源密集型请求
- **Port Scan**: 生成探测类请求
- **Brute Force**: 生成暴力破解载荷
- **Benign**: 生成正常的业务请求

## 与现有代码集成

处理完成后，可直接使用现有的 `data_processing.py`：

```python
from DL.data_processing import load_data, prepare_dataset, load_tokenizer

# 加载转换后的数据
data = load_data('dataset/train_data2.json')

# 准备数据集
tokenizer = load_tokenizer('bert-base-uncased')
train_dataset, val_dataset, test_dataset = prepare_dataset(data, tokenizer)

print(f"训练集: {len(train_dataset)} 样本")
print(f"验证集: {len(val_dataset)} 样本")
print(f"测试集: {len(test_dataset)} 样本")
```

## 故障排除

### 常见问题

1. **文件未找到错误**
   ```
   ❌ Error: MachineLearningCSV.zip not found!
   ```
   **解决**: 确保从 CICIDS2017 官网下载 `MachineLearningCSV.zip` 到 `dataset` 目录

2. **内存不足**
   ```
   MemoryError: Unable to allocate memory
   ```
   **解决**: 减少 `MAX_SAMPLES_PER_FILE` 的值

3. **权限错误**
   ```
   PermissionError: Access denied
   ```
   **解决**: 以管理员身份运行或检查文件权限

### 输出示例

```
🚀 Starting CICIDS2017 processing pipeline...
============================================================
Step 1: Checking and extracting CICIDS2017 dataset...
✅ Dataset already extracted at dataset\MachineLearningCVE

Step 2: Converting CICIDS2017 data to HTTP format...
📊 Processing Monday-WorkingHours.pcap_ISCX.csv...
  ✅ Converted 1000 samples from Monday-WorkingHours
📊 Processing Tuesday-WorkingHours.pcap_ISCX.csv...
  ✅ Converted 1000 samples from Tuesday-WorkingHours
...

Step 3: Saving processed data...
✅ Training data saved: dataset\train_data2.json (6400 samples)
✅ Test data saved: dataset\test2.json (1600 samples)
✅ Full dataset saved: dataset\cicids2017_full.json (8000 samples)

📊 Data Summary:
==================================================
Total samples: 8000

Label distribution:
  Benign: 3200 (40.0%)
  Malicious: 4800 (60.0%)

🎉 Processing completed successfully!
```

## 技术细节

- **内存优化**: 逐文件处理，避免一次性加载所有数据
- **错误处理**: 完善的异常处理和错误提示
- **数据完整性**: 保留原始网络流特征以供参考
- **可扩展性**: 模块化设计，易于定制和扩展
