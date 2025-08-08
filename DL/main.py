import os

# Set Hugging Face mirror BEFORE importing transformers
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_URL"] = "https://hf-mirror.com"

import torch
from transformers import TrainingArguments, TrainerCallback

from models import CNNLSTMClassifier, TextCNNClassifier, DNNClassifier, DeepLog
from data_processing import load_data, prepare_dataset, load_tokenizer, SimpleTokenizer
from training import train_transformer_model, train_custom_model

# 创建SwanLab回调函数用于实时记录DL训练过程
class DLSwanLabCallback(TrainerCallback):
    def __init__(self, use_swanlab=False, model_name=""):
        self.use_swanlab = use_swanlab
        self.model_name = model_name
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """在每次日志记录时调用"""
        if self.use_swanlab and logs and 'swanlab' in globals():
            try:
                # 记录损失和学习率
                log_dict = {}
                if 'loss' in logs:
                    log_dict[f'{self.model_name}_train_loss'] = logs['loss']
                if 'learning_rate' in logs:
                    log_dict[f'{self.model_name}_learning_rate'] = logs['learning_rate']
                if 'epoch' in logs:
                    log_dict[f'{self.model_name}_epoch'] = logs['epoch']
                if 'eval_loss' in logs:
                    log_dict[f'{self.model_name}_eval_loss'] = logs['eval_loss']
                if 'eval_accuracy' in logs:
                    log_dict[f'{self.model_name}_eval_accuracy'] = logs['eval_accuracy']
                if 'eval_f1' in logs:
                    log_dict[f'{self.model_name}_eval_f1'] = logs['eval_f1']
                
                # 添加step信息
                log_dict['step'] = state.global_step
                log_dict['model'] = self.model_name
                
                if log_dict:
                    swanlab.log(log_dict)
                    print(f"📊 {self.model_name} Step {state.global_step}: Loss: {logs.get('loss', 'N/A')}")
                    
            except Exception as e:
                print(f"⚠️  SwanLab logging error for {self.model_name}: {e}")
    
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """在评估时调用"""
        if self.use_swanlab and logs and 'swanlab' in globals():
            try:
                eval_dict = {}
                for k, v in logs.items():
                    if k.startswith('eval_'):
                        eval_dict[f'{self.model_name}_{k}'] = v
                
                if eval_dict:
                    swanlab.log(eval_dict)
                    print(f"📊 {self.model_name} Evaluation: {eval_dict}")
            except Exception as e:
                print(f"⚠️  SwanLab eval logging error for {self.model_name}: {e}")

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
    
    # Disable wandb completely and enable swanlab
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_SILENT"] = "true"
    
    # Initialize SwanLab for experiment tracking
    try:
        import swanlab
        swanlab.init(
            project="AdvTG-DL-Training",
            description="Deep Learning stage - BERT and Custom Models Training",
            config={
                "batch_size": 16,
                "learning_rate": 2e-5,
                "num_epochs": 3,
                "max_length": 512
            }
        )
        print("✅ SwanLab initialized successfully!")
        use_swanlab = True
    except ImportError:
        print("⚠️  SwanLab not installed, continuing without experiment tracking")
        use_swanlab = False
    except Exception as e:
        print(f"⚠️  SwanLab initialization failed: {e}, continuing without experiment tracking")
        use_swanlab = False
    
    # Configuration
    FORCE_SIMPLE_MODE = False  # Set to True to skip BERT and only use custom models
    
    # Define constants
    DATA_PATH = "../dataset/dl_train.json"  # Use small dataset for DL training
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
        report_to="none",  # Disable all automatic logging including wandb
        logging_steps=50,  # Log every 50 steps for manual tracking
    )
        
    # Only train transformer model if we have real transformers tokenizer
    all_model_configs = []  # 收集所有模型配置
    
    if not isinstance(tokenizer, SimpleTokenizer):
        try:
            # 创建BERT专用的SwanLab回调
            bert_callback = DLSwanLabCallback(use_swanlab=use_swanlab, model_name="BERT") if use_swanlab else None
            
            bert_save_path, bert_config = train_transformer_model(
                transformer_model_name,
                transformer_model_name,  # model_path parameter 
                train_dataset, 
                val_dataset, 
                transformer_training_args,
                swanlab_callback=bert_callback  # 传递回调函数
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
        report_to="none",  # Disable all automatic logging including wandb
        logging_steps=50,  # Log every 50 steps for manual tracking
    )
    
    print("\n====== Training Custom Models ======")
    
    for model_name, model in models.items():
        print(f"\nTraining model: {model_name}")
        try:
            save_path, model_config = train_custom_model(model, model_name, train_dataset, val_dataset, custom_training_args)
            all_model_configs.append(model_config)
            print(f"Completed training model: {model_name}")
            
            # 手动记录自定义模型的结果到SwanLab
            if use_swanlab and 'swanlab' in locals():
                try:
                    swanlab.log({
                        f'{model_name}_final_accuracy': model_config.get('accuracy', 0),
                        f'{model_name}_final_precision': model_config.get('precision', 0),
                        f'{model_name}_final_recall': model_config.get('recall', 0),
                        f'{model_name}_final_f1': model_config.get('f1', 0),
                        'model_type': 'custom',
                        'model_name': model_name
                    })
                    print(f"📊 {model_name} results logged to SwanLab")
                except Exception as e:
                    print(f"⚠️  SwanLab logging failed for {model_name}: {e}")
            
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
    
    # Log results to SwanLab if available
    if use_swanlab and 'swanlab' in locals():
        try:
            # Log model performance metrics
            for i, config in enumerate(all_model_configs):
                model_name = config.get('model_name', f'model_{i}')
                if 'accuracy' in config:
                    swanlab.log({f"{model_name}/accuracy": config['accuracy']})
                if 'f1' in config:
                    swanlab.log({f"{model_name}/f1_score": config['f1']})
                if 'precision' in config:
                    swanlab.log({f"{model_name}/precision": config['precision']})
                if 'recall' in config:
                    swanlab.log({f"{model_name}/recall": config['recall']})
            
            # Log summary metrics
            swanlab.log({
                "total_models_trained": len(all_model_configs),
                "training_completed": 1
            })
            
            swanlab.finish()
            print("📊 Results logged to SwanLab successfully!")
        except Exception as e:
            print(f"⚠️  SwanLab logging failed: {e}")
    
    print(f"\n✅ Saved text model configurations to: {config_save_path}")
    print(f"📊 Total text models configured: {len(all_model_configs)}")
    print("✅ Created image model configurations")
    
    print("\nAll models training completed!")

if __name__ == "__main__":
    main() 