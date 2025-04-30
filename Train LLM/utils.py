import os
import random
import numpy as np
import torch

def mkdir(directory):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Directory path to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_random_binary_list(length):
    """
    Generate a random list of 0s and 1s.
    
    Args:
        length: Length of the list
        
    Returns:
        list: Random binary list
    """
    return [random.randint(0, 1) for _ in range(length)]

def get_label(query_list):
    """
    Extract labels from query strings.
    
    Args:
        query_list: List of query strings
        
    Returns:
        list: List of labels (0 for benign, 1 for malicious)
    """
    labels = []
    for query in query_list:
        if "malicious" in query.lower():
            labels.append(1)  # Malicious
        else:
            labels.append(0)  # Benign
    return labels

def replace_traffic_type(query_strings, target_labels):
    """
    Replace traffic type in query strings based on target labels.
    
    Args:
        query_strings: List of query strings
        target_labels: List of target labels (0 for benign, 1 for malicious)
        
    Returns:
        list: List of modified query strings
    """
    result = []
    for i, query in enumerate(query_strings):
        if target_labels[i] == 1:  # Target is malicious
            # Replace benign with malicious
            new_query = query.replace("benign", "malicious")
        else:  # Target is benign
            # Replace malicious with benign
            new_query = query.replace("malicious", "benign")
        result.append(new_query)
    return result

def calculate_coefficient(origin_labels, requirement_labels, predictions):
    """
    Calculate coefficient based on original labels, requirement labels, and predictions.
    
    Args:
        origin_labels: Original labels
        requirement_labels: Required target labels
        predictions: Model predictions
        
    Returns:
        list: List of coefficients
    """
    coefficients = []
    for i in range(len(origin_labels)):
        if requirement_labels[i] == predictions[i]:
            # Prediction matches requirement - good
            coefficients.append(1.0)
        else:
            # Prediction doesn't match requirement - bad
            coefficients.append(0.5)
    return coefficients

def save_results(all_data, feature_type, epoch, base_dir="../dataset/PPO_data"):
    """
    Save results to a JSON file.
    
    Args:
        all_data: Data to save
        feature_type: Type of feature model
        epoch: Current epoch
        base_dir: Base directory for saving
    """
    import json
    
    output_dir = f'{base_dir}/{feature_type}/new333'
    mkdir(output_dir)
    
    output_file = os.path.join(output_dir, f'{str(epoch)}.json')
    
    # Save data to JSON file
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=4)

def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 