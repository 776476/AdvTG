# AdvTG 数据分配策略说明

## 📊 优化后的数据分配方案

```
数据分配 (100% 全量数据):
├── dl_train.json      ← DL训练 (10%)
├── llm_train.json     ← LLM训练 (60%)  
├── rl_train.json     ← RL训练 (12%)   
├── val.json      ← 验证集 (5%)    
└── test.json         ← 测试集 (13%)
```

## 🎯 各数据集的用途说明

### 1. **DL训练数据 (10%)**
- **文件**: `dl_train.json`
- **用途**: 训练深度学习检测模型（BERT、TextCNN、CNN-LSTM等）
- **特点**: 小数据集即可满足DL模型训练需求
- **格式**: 标准HTTP请求格式（Request Line + Headers + Body + Label）

### 2. **LLM训练数据 (60%)**
- **文件**: `llm_traine.json`
- **用途**: Llama-3-8B模型的LoRA微调
- **特点**: 需要大量数据进行有效的语言模型微调
- **格式**: 标准HTTP请求格式，会转换为Alpaca指令格式

### 3. **RL训练数据 (12%)**
- **文件**: `rl_train.json`
- **用途**: PPO强化学习训练，生成对抗样本
- **特点**: 中等规模数据集，用于策略学习
- **格式**: 标准HTTP请求格式，会转换为指令格式

### 4. **验证集 (5%)**
- **文件**: `val.json`
- **用途**: **仅用于LLM微调阶段的验证**
- **特点**: 
  - 在LLM微调过程中评估模型性能
  - 用于早停和超参数调优
  - **不用于DL和RL模块验证**
- **格式**: 与LLM训练数据相同，会转换为Alpaca格式

### 5. **测试集 (13%)**
- **文件**: `test.json`
- **用途**: **所有模型的最终效果评估**
- **特点**: 
  - DL模型：测试分类准确率
  - LLM模型：测试生成质量
  - RL模型：测试对抗攻击效果
- **格式**: 标准HTTP请求格式

## 🔄 数据流向图

```
原始数据 (CICIDS2017)
    ↓ 处理
全量数据 (100%)
    ↓ 分配
┌─────────────────────────────────────────────────────────┐
│                    数据分配                              │
├─────────────────┬─────────────────┬─────────────────────┤
│ DL训练 (10%)    │ LLM训练 (60%)   │ RL训练 (12%)        │
│ ↓ 训练          │ ↓ 微调          │ ↓ 强化学习           │
│ DL检测模型      │ 微调后LLM       │ RL对抗生成模型       │
└─────────────────┴─────────────────┴─────────────────────┘
                  ↓ 验证 (5%)    ↓ 测试 (13%)
              LLM微调验证      所有模型最终测试
```

## ⚠️ 重要说明

### **验证集的正确使用**
1. **验证集只用于LLM微调阶段**：
   - 在`train_llama.py`中作为`eval_dataset`
   - 格式化为Alpaca提示词格式
   - 用于监控微调过程

2. **DL和RL模块的验证**：
   - DL模块：内部自动分割训练集为train/val/test
   - RL模块：使用奖励模型反馈，无需额外验证集

### **测试集的统一使用**
- 所有模块都使用 `test.json` 进行最终评估
- 确保评估结果的公平性和可比性

## 📋 修改记录

### LLM微调脚本修改
- 使用 `llm_train.json` 作为训练数据
- 使用 `val.json` 作为验证数据
- 两个数据集都格式化为Alpaca格式
- 修正了原来使用固定100个样本做验证的问题

### 数据处理脚本修改
- 更新文件命名约定
- 优化数据分配比例
- 添加详细的处理日志

### 其他模块修改
- DL模块：使用 `dl_train.json`
- RL模块：使用 `rl_train.json`
- 所有模块：使用 `test.json` 测试

## 🚀 使用方法

1. **运行数据处理**：
```bash
cd AdvTG/dataset
python process_cicids2017.py
```

2. **训练各模块**：
```bash
# DL训练
cd ../DL
python main.py

# LLM微调
cd ../LLM-Finetune
python train_llama.py

# RL训练
cd ../RL-Adv
python train_ppo.py
```

3. **最终测试**：
所有模型都在 `test.json` 上进行效果评估。

```
文件列表：
Monday-WorkingHours.pcap_ISCX.csv - 正常流量
Tuesday-WorkingHours.pcap_ISCX.csv - 暴力破解攻击
Wednesday-workingHours.pcap_ISCX.csv - DoS/DDoS攻击
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv - Web攻击
Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv - 渗透攻击
Friday-WorkingHours-Morning.pcap_ISCX.csv - 僵尸网络攻击
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv - 端口扫描
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv - DDoS攻击
```