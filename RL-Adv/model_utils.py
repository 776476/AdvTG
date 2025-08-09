import os
import torch
import pickle
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
import torch.nn.functional as F
from peft import PeftModel
from utils import get_label, replace_traffic_type, mkdir

def load_model_configs(feature_type, features_dict=None):
    """
    Load model configurations based on feature type.
    
    Args:
        feature_type: Type of feature model ('Image' or 'Text')
        features_dict: Dictionary mapping feature types to model paths
        
    Returns:
        list: List of model configurations
    """
    if features_dict is None:
        features_dict = {"Image": "../model/imgae_model_configs.pkl", "Text": "../model/model_configs.pkl"}
    
    model_path = features_dict[feature_type]
    with open(model_path, "rb") as f:
        model_configs = pickle.load(f)
    return model_configs

def select_feature_model_type(features_dict):
    """
    Interactively select a feature model type.
    
    Args:
        features_dict: Dictionary mapping feature types to model paths
        
    Returns:
        str: Selected feature type
        list: Model configurations for the selected type
    """
    while True:
        print(features_dict.keys())
        feature_type = input("select one feature model to attack: ")
        try:
            model_path = features_dict[feature_type]
            with open(model_path, "rb") as f:
                model_configs = pickle.load(f)
                return feature_type, model_configs
        except:
            print(f"Please input Image or Text")
            continue

def setup_models(model_name, device, load_in_4bit=True):
    """
    Set up PPO model and reference model for LoRA checkpoint.
    
    Args:
        model_name: Path to the LoRA checkpoint
        device: Torch device
        load_in_4bit: Whether to load model in 4-bit quantization
        
    Returns:
        tuple: (ppo_model, ref_model, tokenizer)
    """
    from config import base_model_name
    
    # Load PPO model with LoRA adapter (don't pass quantization_config as it's already configured)
    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,  # This is the LoRA checkpoint path
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Load reference model from base model (with quantization config for base model)
    quantization_config = BitsAndBytesConfig(load_in_4bit=load_in_4bit)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model_name,  # Use the base model for reference
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        trust_remote_code=True
    )
    
    # Load tokenizer from the LoRA checkpoint (which should have the tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return ppo_model, ref_model, tokenizer

def prepare_query_tensors(batch, tokenizer, device, query_max_length=128):
    """
    Prepare query tensors for generation.
    
    Args:
        batch: Batch of data
        tokenizer: Tokenizer
        device: Torch device
        query_max_length: Maximum length of query
        
    Returns:
        tuple: (query_tensors, origin_label, requirement_label, new_query_str)
    """
    # Create query strings with instruction and input
    query_str = [(k+" \n"+o)[:query_max_length] for k, o in zip(batch['instruction'], ["GET", "GET", "GET", "GET"])]
    
    # Get original labels
    origin_label = get_label(query_str)
    
    # Generate requirement labels (inverted from original)
    requirement_label = [1 if x == 0 else 0 for x in origin_label]
    
    # Replace traffic types based on requirement labels
    new_query_str = replace_traffic_type(query_str, requirement_label)
    
    # Convert to tensors
    query_tensors = [torch.tensor(tokenizer(query)["input_ids"]).to(device) for query in new_query_str]
    
    return query_tensors, origin_label, requirement_label, new_query_str

def evaluate_responses(batch, feature_type, model_configs, padded_tensor_batch, device, requirement_label):
    """
    Evaluate model responses and calculate rewards.
    
    Args:
        batch: Batch of data
        feature_type: Type of feature model
        model_configs: Model configurations
        padded_tensor_batch: Tensor batch for evaluation
        device: Torch device
        requirement_label: Target labels
        
    Returns:
        torch.Tensor: Reward values
    """
    results = []
    
    for model_config in model_configs:
        if model_config["type"] != "custom" or model_config["name"] == "deeplog":
            continue
            
        model_path = model_config["path"]
        model = model_config["class"]
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        
        result = model(padded_tensor_batch)
        results.append(result)
    
    # Average results from all models    
    results = torch.stack(results, dim=0)
    results = torch.mean(results, dim=0)
    
    # Get rewards for required labels
    rewards = results[torch.arange(len(requirement_label)), requirement_label]
    
    # Get predictions
    prediction = torch.argmax(results, dim=1)
    
    return rewards.detach().cpu(), prediction 