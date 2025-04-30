import json
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

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
    with open(file_path, "r") as f:
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
        # Define tokenize function
        def tokenize_function(examples):
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
    """Load tokenizer for a specific model."""
    return AutoTokenizer.from_pretrained(model_name) 