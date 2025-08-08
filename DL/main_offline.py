import os
import torch
from transformers import TrainingArguments

from models import CNNLSTMClassifier, TextCNNClassifier, DNNClassifier, DeepLog
from data_processing import load_data, prepare_dataset, SimpleTokenizer
from training import train_custom_model

def set_environment():
    """Set environment variables for GPU usage."""
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"

def main():
    # Set environment variables
    set_environment()
    
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
    
    # Use SimpleTokenizer (offline mode)
    print("\n====== Using Offline Mode - SimpleTokenizer ======")
    tokenizer = SimpleTokenizer()
    print(f"Tokenizer type: {type(tokenizer)}")
    
    # Load and prepare data
    train_dataset, val_dataset, test_dataset = prepare_dataset(json_data, tokenizer, MAX_LENGTH)
    
    # Get vocabulary size
    vocab_size = len(tokenizer.vocab)
    embed_size = 128
    num_classes = 2
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Define custom models
    models = {
        "textcnn": TextCNNClassifier(vocab_size, embed_size, num_classes, MAX_LENGTH),
        "cnn_lstm": CNNLSTMClassifier(vocab_size, embed_size, num_classes, MAX_LENGTH),
        "dnn": DNNClassifier(vocab_size, embed_size, num_classes, MAX_LENGTH),
        "deeplog": DeepLog(vocab_size, embed_size, num_classes, MAX_LENGTH)
    }
    
    # Define training arguments for custom models
    custom_training_args = TrainingArguments(
        output_dir="./models/custom_models",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
    )
    
    print("\n====== Training Custom Models ======")
    for model_name, model in models.items():
        print(f"\nTraining model: {model_name}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        try:
            train_custom_model(model, model_name, train_dataset, val_dataset, custom_training_args)
            print(f"✅ Completed training model: {model_name}")
        except Exception as e:
            print(f"❌ Failed to train {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🎉 All models training completed!")

if __name__ == "__main__":
    main()
