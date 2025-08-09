import json
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
import re
import os
from collections import Counter

# Try to import transformers, use fallback if not available
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Transformers not available, using simple tokenizer")
    TRANSFORMERS_AVAILABLE = False

class SimpleTokenizer:
    """Simple tokenizer when transformers is not available"""
    def __init__(self):
        self.vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
        self.vocab_size = 30000
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.cls_token_id = 2
        self.sep_token_id = 3
        
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        word_freq = Counter()
        for text in texts:
            words = re.findall(r'\w+', text.lower())
            word_freq.update(words)
        
        # Add most frequent words to vocab
        for word, freq in word_freq.most_common(self.vocab_size - len(self.vocab)):
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        
        print(f"Built vocabulary with {len(self.vocab)} tokens")
        
    def encode(self, text, max_length=512, padding=True, truncation=True):
        # Simple word-based tokenization
        words = re.findall(r'\w+', text.lower())
        token_ids = [self.cls_token_id]
        
        for word in words:
            token_ids.append(self.vocab.get(word, self.unk_token_id))
            if len(token_ids) >= max_length - 1:
                break
                
        token_ids.append(self.sep_token_id)
        
        if padding and len(token_ids) < max_length:
            token_ids.extend([self.pad_token_id] * (max_length - len(token_ids)))
        elif truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            
        return {
            "input_ids": token_ids,
            "attention_mask": [1 if x != self.pad_token_id else 0 for x in token_ids]
        }
        
    def __call__(self, texts, max_length=512, padding=True, truncation=True):
        if isinstance(texts, str):
            return self.encode(texts, max_length, padding, truncation)
        else:
            results = {"input_ids": [], "attention_mask": []}
            for text in texts:
                encoded = self.encode(text, max_length, padding, truncation)
                results["input_ids"].append(encoded["input_ids"])
                results["attention_mask"].append(encoded["attention_mask"])
            return results

def json_to_string(data, indent=0):
    """Convert JSON data to a formatted string."""
    result = []
    indent_str = '  ' * indent
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                result.append(f'{indent_str}{key}:')
                result.append(json_to_string(value, indent + 1))
            else:
                result.append(f'{indent_str}{key}: {value}')
    elif isinstance(data, list):
        for item in data:
            result.append(json_to_string(item, indent))
    else:
        result.append(f'{indent_str}{data}')
    return '\n'.join(result)

def load_data(file_path):
    """Load JSON data from file."""
    try:
        # Try UTF-8 encoding first
        with open(file_path, "r", encoding='utf-8') as f:
            json_data = json.load(f)
        return json_data
    except UnicodeDecodeError:
        try:
            # Try GBK encoding
            with open(file_path, "r", encoding='gbk') as f:
                json_data = json.load(f)
            return json_data
        except UnicodeDecodeError:
            # Try latin-1 as last resort
            with open(file_path, "r", encoding='latin-1') as f:
                json_data = json.load(f)
            return json_data

def prepare_dataset(json_data, tokenizer=None, max_length=512):
    """Prepare dataset from JSON data."""
    # Format data for the dataset
    formatted_data = {
        "text": [item["Request Line"]+"\n"+json_to_string(item["Request Headers"])+"\n\n"+item["Request Body"] for item in json_data],
        "label": [1 if item["Label"]=="Malicious" else 0 for item in json_data],
        "source": [item["Source"] for item in json_data]
    }
    
    # Create dataset
    dataset = Dataset.from_dict(formatted_data)
    
    if tokenizer:
        # Build vocabulary for simple tokenizer
        if isinstance(tokenizer, SimpleTokenizer):
            tokenizer.build_vocab(formatted_data["text"])
        
        # Define tokenize function
        def tokenize_function(examples):
            if isinstance(tokenizer, SimpleTokenizer):
                # Use simple tokenizer
                results = tokenizer(examples['text'], max_length=max_length, padding=True, truncation=True)
                return results
            else:
                # Use transformers tokenizer
                return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=max_length)
            
        # Apply tokenize function to dataset
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        transformer_dataset = tokenized_dataset.remove_columns(["text"])
        
        # Split dataset
        train_test_split = transformer_dataset.train_test_split(test_size=0.2)
        train_val_split = train_test_split['train'].train_test_split(test_size=0.1)
        
        # Get train, validation and test datasets
        train_dataset = train_val_split['train']
        val_dataset = train_val_split['test']
        test_dataset = train_test_split['test']
        
        # Set format for PyTorch
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        
        return train_dataset, val_dataset, test_dataset
    
    return dataset

def get_data_loaders(train_dataset, val_dataset, test_dataset=None, batch_size=32):
    """Create data loaders for training, validation and testing."""
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    if test_dataset:
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        return train_dataloader, val_dataloader, test_dataloader
    
    return train_dataloader, val_dataloader

def load_tokenizer(model_name):
    """Load tokenizer for a specific model with offline support."""
    if not TRANSFORMERS_AVAILABLE:
        print("Using SimpleTokenizer instead of transformers")
        return SimpleTokenizer()
        
    try:
        # 优先尝试从本地加载
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        print(f"Loaded {model_name} tokenizer from local cache")
        return tokenizer
    except:
        try:
            # 本地没有则从镜像网站下载
            print(f"Downloading {model_name} tokenizer from mirror...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=False,
                use_auth_token=False
                # 不设置force_download和cache_dir，使用默认行为
            )
            print(f"Downloaded {model_name} tokenizer successfully from mirror")
            return tokenizer
        except Exception as e:
            print(f"Failed to load {model_name} tokenizer from mirror: {e}")
            print("Falling back to SimpleTokenizer")
            return SimpleTokenizer() 