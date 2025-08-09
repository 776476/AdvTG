"""
LLM数据处理工具
处理HTTP数据集并格式化为Alpaca格式
"""
import json
import os
from typing import Dict, List, Any, Tuple
from datasets import Dataset


class LLMDataProcessor:
    """LLM数据处理器"""
    
    def __init__(self, config):
        """
        初始化数据处理器
        
        Args:
            config: LLMConfig配置对象
        """
        self.config = config
        self.alpaca_prompt = config.ALPACA_PROMPT
        self.malicious_instruction = config.MALICIOUS_INSTRUCTION
        self.benign_instruction = config.BENIGN_INSTRUCTION
        self.eos_token = None  # 将在tokenizer加载后设置
    
    def set_eos_token(self, eos_token: str):
        """设置EOS token"""
        self.eos_token = eos_token
    
    def json_to_string(self, data: Any, indent: int = 0) -> str:
        """
        将JSON数据转换为字符串格式
        
        Args:
            data: 要转换的数据
            indent: 缩进级别
            
        Returns:
            格式化的字符串
        """
        result = []
        indent_str = ' ' * indent
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    result.append(f'{indent_str}{key}:')
                    result.append(self.json_to_string(value, indent + 1))
                else:
                    result.append(f'{indent_str}{key}: {value}')
        elif isinstance(data, list):
            for item in data:
                result.append(self.json_to_string(item, indent))
        else:
            result.append(f'{indent_str}{data}')
        
        return '\n'.join(result)
    
    def load_json_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载JSON数据文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            加载的数据列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"📁 Loaded {len(data)} samples from {file_path}")
        return data
    
    def format_single_sample(self, item: Dict[str, Any]) -> Dict[str, str]:
        """
        格式化单个样本
        
        Args:
            item: 原始数据项
            
        Returns:
            格式化后的数据项
        """
        # 构建请求内容
        request_content = (
            item["Request Line"] + "\n" + 
            self.json_to_string(item["Request Headers"]) + "\n\n" + 
            item["Request Body"]
        )
        
        # 根据标签选择指令
        instruction = (
            self.malicious_instruction if item["Label"] == "Malicious" 
            else self.benign_instruction
        )
        
        return {
            "text": request_content,
            "instruction": instruction,
            "input": item["Request Line"],
            "output": request_content
        }
    
    def format_data_for_training(self, raw_data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        将原始数据格式化为训练格式
        
        Args:
            raw_data: 原始数据列表
            
        Returns:
            格式化后的数据字典
        """
        formatted_data = {
            "text": [],
            "instruction": [],
            "input": [],
            "output": []
        }
        
        for item in raw_data:
            formatted_item = self.format_single_sample(item)
            for key in formatted_data.keys():
                formatted_data[key].append(formatted_item[key])
        
        return formatted_data
    
    def formatting_prompts_func(self, examples: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        将数据格式化为Alpaca提示词格式
        
        Args:
            examples: 批量样本数据
            
        Returns:
            格式化后的文本列表
        """
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            # 添加EOS token，否则生成会无限进行
            text = self.alpaca_prompt.format(instruction, input_text, output)
            if self.eos_token:
                text += self.eos_token
            texts.append(text)
        
        return {"text": texts}
    
    def create_dataset(self, raw_data: List[Dict[str, Any]], shuffle: bool = True) -> Dataset:
        """
        创建Dataset对象
        
        Args:
            raw_data: 原始数据列表
            shuffle: 是否打乱数据
            
        Returns:
            Dataset对象
        """
        # 格式化数据
        formatted_data = self.format_data_for_training(raw_data)
        
        # 创建Dataset
        dataset = Dataset.from_dict(formatted_data)
        
        # 打乱数据
        if shuffle:
            dataset = dataset.shuffle(seed=self.config.DATA_SHUFFLE_SEED)
        
        # 应用Alpaca格式化
        dataset = dataset.map(self.formatting_prompts_func, batched=True)
        
        return dataset
    
    def load_and_prepare_datasets(self) -> Tuple[Dataset, Dataset]:
        """
        加载并准备训练和验证数据集
        
        Returns:
            (train_dataset, val_dataset) 元组
        """
        print("📊 Loading and preparing datasets...")
        
        # 加载原始数据
        train_data = self.load_json_data(self.config.TRAIN_DATA_PATH)
        val_data = self.load_json_data(self.config.VAL_DATA_PATH)
        
        # 创建数据集
        train_dataset = self.create_dataset(train_data, shuffle=True)
        val_dataset = self.create_dataset(val_data, shuffle=True)
        
        print(f"✅ Datasets prepared:")
        print(f"   - Training samples: {len(train_dataset)}")
        print(f"   - Validation samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def print_sample_data(self, dataset: Dataset, num_samples: int = 2):
        """
        打印样本数据用于检查
        
        Args:
            dataset: 数据集
            num_samples: 打印的样本数量
        """
        print(f"\n📋 Sample data (first {num_samples} samples):")
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            print(f"\n--- Sample {i+1} ---")
            print(f"Text length: {len(sample['text'])} chars")
            print(f"Text preview: {sample['text'][:200]}...")
            if len(sample['text']) > 200:
                print("    [truncated]")
        print()


def create_data_processor(config) -> LLMDataProcessor:
    """
    创建数据处理器实例
    
    Args:
        config: LLMConfig配置对象
        
    Returns:
        LLMDataProcessor实例
    """
    return LLMDataProcessor(config)
