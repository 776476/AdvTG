import os

# Set Hugging Face mirror BEFORE importing transformers
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_URL"] = "https://hf-mirror.com"

import torch
from transformers import TrainingArguments

from models import CNNLSTMClassifier, TextCNNClassifier, DNNClassifier, DeepLog
from data_processing import load_data, prepare_dataset, load_tokenizer, SimpleTokenizer
from training import train_transformer_model, train_custom_model

def set_environment():
    """Set environment variables for GPU usage."""
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    
    # Force use of mirror for transformers library (additional method)
    try:
        from transformers import file_utils
        file_utils.HUGGINGFACE_CO_URL_HOME = "https://hf-mirror.com"
        print("Set transformers to use mirror: https://hf-mirror.com")
    except:
        try:
            import transformers.utils.hub as hub_utils
            hub_utils.HUGGINGFACE_CO_URL_HOME = "https://hf-mirror.com"
            print("Set transformers hub to use mirror")
        except:
            print("Could not set transformers mirror, using environment variables only")


def main():
    # Set environment variables
    set_environment()
    
    # Configuration
    FORCE_SIMPLE_MODE = False  # Set to True to skip BERT and only use custom models
    
    # Define constants
    DATA_PATH = "../dataset/dl_train_small.json"  # Use small dataset for DL training
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    json_data = load_data(DATA_PATH)
    
    # Create models directory
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../models/custom_models', exist_ok=True)
    
    # Train transformer model (BERT)
    print("\n====== Training BERT Model ======")
    
    if FORCE_SIMPLE_MODE:
        print("Force simple mode enabled - using SimpleTokenizer")
        tokenizer = SimpleTokenizer()
        print(f"Tokenizer type: {type(tokenizer)}")
        print(f"Is SimpleTokenizer: {isinstance(tokenizer, SimpleTokenizer)}")
    else:
        # Download from mirror
        transformer_model_name = "bert-base-uncased"
        print(f"Model endpoint: {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")
        
        tokenizer = load_tokenizer(transformer_model_name)
        print(f"Tokenizer type: {type(tokenizer)}")
        print(f"Is SimpleTokenizer: {isinstance(tokenizer, SimpleTokenizer)}")
        
    # Load and prepare data
    train_dataset, val_dataset, test_dataset = prepare_dataset(json_data, tokenizer, MAX_LENGTH)
    
    # Check if using simple tokenizer and adjust training accordingly
    if hasattr(tokenizer, 'vocab_size'):
        vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer.vocab)
    else:
        vocab_size = 30522  # BERT default vocab size
        
    # Define training arguments for transformer
    transformer_training_args = TrainingArguments(
        output_dir="./models/bert_model",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        eval_strategy="epoch",  # Changed from evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
        
    # Only train transformer model if we have real transformers tokenizer
    all_model_configs = []  # 收集所有模型配置
    
    if not isinstance(tokenizer, SimpleTokenizer):
        try:
            bert_save_path, bert_config = train_transformer_model(
                transformer_model_name,
                transformer_model_name,  # model_path parameter 
                train_dataset, 
                val_dataset, 
                transformer_training_args
            )
            all_model_configs.append(bert_config)
            print("BERT model training completed!")
        except Exception as e:
            print(f"BERT training failed: {e}")
            print("Continuing with custom models only...")
    else:
        print("Skipping BERT training - using simple tokenizer mode")
        print("Training custom models only...")
        
    # Train custom models
    embed_size = 128
    num_classes = 2
    
    # Define custom models
    models = {
        "textcnn": TextCNNClassifier(vocab_size, embed_size, num_classes, MAX_LENGTH),
        "cnn_lstm": CNNLSTMClassifier(vocab_size, embed_size, num_classes, MAX_LENGTH),
        "dnn": DNNClassifier(vocab_size, embed_size, num_classes, MAX_LENGTH),
        "deeplog": DeepLog(vocab_size, embed_size, num_classes, MAX_LENGTH)
    }
    
    custom_training_args = TrainingArguments(
        output_dir="../models/custom_models",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
    )
    
    print("\n====== Training Custom Models ======")
    
    for model_name, model in models.items():
        print(f"\nTraining model: {model_name}")
        try:
            save_path, model_config = train_custom_model(model, model_name, train_dataset, val_dataset, custom_training_args)
            all_model_configs.append(model_config)
            print(f"Completed training model: {model_name}")
        except Exception as e:
            print(f"Failed to train {model_name}: {e}")
    
    # Save model configurations for RL stage
    import pickle
    config_save_path = "../models/model_configs.pkl"
    os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
    
    with open(config_save_path, 'wb') as f:
        pickle.dump(all_model_configs, f)
    
    # Create image model configurations (placeholder)
    from create_image_configs import create_image_model_configs
    create_image_model_configs()
    
    print(f"\n✅ Saved text model configurations to: {config_save_path}")
    print(f"📊 Total text models configured: {len(all_model_configs)}")
    print("✅ Created image model configurations")
    
    print("\nAll models training completed!")

if __name__ == "__main__":
    main() 