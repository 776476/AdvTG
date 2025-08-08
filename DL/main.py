import os
import torch
from transformers import TrainingArguments

from models import CNNLSTMClassifier, TextCNNClassifier, DNNClassifier, DeepLog
from data_processing import load_data, prepare_dataset, load_tokenizer
from training import train_transformer_model, train_custom_model

def set_environment():
    """Set environment variables for GPU usage."""
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Uncomment if needed
    # os.environ["WORLD_SIZE"] = "2"
    # os.environ['HTTP_PROXY'] = '127.0.0.1:7890'
    # os.environ['HTTPS_PROXY'] = '127.0.0.1:7890'
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
    
    # Train transformer model (BERT)
    print("\n====== Training BERT Model ======")
    transformer_model_name = "bert-base-uncased"
    tokenizer = load_tokenizer(transformer_model_name)
    
    # Prepare dataset for transformer model
    train_dataset, val_dataset, test_dataset = prepare_dataset(json_data, tokenizer, MAX_LENGTH)
    
    # Define training arguments for transformer model
    training_args = TrainingArguments(
        output_dir=f"./models/bert",
        evaluation_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Train transformer model
    train_transformer_model("bert", transformer_model_name, train_dataset, val_dataset, training_args)
    
    # Train custom models
    print("\n====== Training Custom Models ======")
    
    # Vocabulary size (from tokenizer)
    vocab_size = len(tokenizer.vocab)
    embed_size = 128
    num_classes = 2
    
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
    )
    
    # Train each custom model
    for model_name, model in models.items():
        print(f"\nTraining model: {model_name}")
        train_custom_model(model, model_name, train_dataset, val_dataset, custom_training_args)
        print(f"Completed training model: {model_name}")
    
    print("\nTraining completed for all models!")

if __name__ == "__main__":
    main() 