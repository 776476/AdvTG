import os
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
    
    # Set Hugging Face mirror for Chinese users - multiple methods
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "./models/huggingface_cache"


def main():
    # Set environment variables
    set_environment()
    
    # Configuration
    FORCE_SIMPLE_MODE = False  # Set to True to skip BERT and only use custom models
    
    # Define constants
    DATA_PATH = "../dataset/train_data2.json"
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
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./models/custom_models', exist_ok=True)
    
    # Train transformer model (BERT)
    print("\n====== Training BERT Model ======")
    
    if FORCE_SIMPLE_MODE:
        print("Force simple mode enabled - using SimpleTokenizer")
        tokenizer = SimpleTokenizer()
        print(f"Tokenizer type: {type(tokenizer)}")
        print(f"Is SimpleTokenizer: {isinstance(tokenizer, SimpleTokenizer)}")
    else:
        # Use Chinese mirror or local cache
        transformer_model_name = "bert-base-uncased"
        print(f"Downloading model from: {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")
        
        # Try to load tokenizer with offline mode fallback
        print("Attempting to load BERT tokenizer...")
        tokenizer = load_tokenizer(transformer_model_name)
        print("Successfully loaded tokenizer")
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
    if not isinstance(tokenizer, SimpleTokenizer):
        try:
            transformer_trainer = train_transformer_model(
                transformer_model_name, 
                train_dataset, 
                val_dataset, 
                transformer_training_args
            )
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
        output_dir="./models/custom_models",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
    )
    
    print("\n====== Training Custom Models ======")
    for model_name, model in models.items():
        print(f"\nTraining model: {model_name}")
        try:
            train_custom_model(model, model_name, train_dataset, val_dataset, custom_training_args)
            print(f"Completed training model: {model_name}")
        except Exception as e:
            print(f"Failed to train {model_name}: {e}")
    
    print("\nAll models training completed!")

if __name__ == "__main__":
    main() 