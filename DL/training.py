import os
import gc
import torch
import torch.nn as nn
import numpy as np

# 强制设置Hugging Face镜像 - 必须在导入transformers之前
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "0"  # 允许在线但使用镜像

from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding
)

from metrics import transformer_metrics, custom_metrics

# Set default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define default model save path
MODEL_PATH = "../models/"

class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    def __init__(self, patience=3, min_delta=0.01, mode='min'):
        """
        Args:
            patience (int): Number of epochs to wait after min_delta improvement
            min_delta (float): Minimum change to qualify as improvement
            mode (str): 'min' for loss, 'max' for accuracy/f1
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else -float('inf')
        self.counter = 0
        self.best_epoch = 0
        
    def __call__(self, score, epoch):
        """
        Check if training should stop
        Returns: (should_stop, is_improvement)
        """
        is_improvement = False
        
        if self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
                self.best_epoch = epoch
                is_improvement = True
            else:
                self.counter += 1
        else:  # mode == 'max'
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
                self.best_epoch = epoch
                is_improvement = True
            else:
                self.counter += 1
        
        should_stop = self.counter >= self.patience
        return should_stop, is_improvement

def train_transformer_model(model_name, model_path, train_dataset, eval_dataset, training_args, swanlab_callback=None, early_stopping_callback=None):
    """Train a transformer model."""
    print(f"🔧 Starting {model_name} model training...")
    
    try:
        # Load model and tokenizer - 优先本地，无则从镜像下载
        print(f"📥 Loading model from {model_path}...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=False,
            use_auth_token=False
            # 移除强制下载参数，让系统自然选择本地或远程
        )
        
        print(f"📥 Loading tokenizer from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=False,
            use_auth_token=False
            # 移除强制下载参数，让系统自然选择本地或远程
        )
        
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        print(f"✅ Model and tokenizer loaded successfully")

        # 准备回调函数列表
        callbacks = []
        if swanlab_callback:
            callbacks.append(swanlab_callback)
        if early_stopping_callback:
            callbacks.append(early_stopping_callback)

        # Define trainer
        print(f"🏗️ Initializing Trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=transformer_metrics,
            callbacks=callbacks
        )

        # Train model
        print(f"🚀 Starting training...")
        trainer.train()
        print(f"✅ Training completed successfully")
        
        # Evaluate model to get metrics
        print(f"📊 Evaluating model...")
        eval_results = trainer.evaluate()
        
        # Save model
        print(f"💾 Saving model...")
        save_path = f"{MODEL_PATH}/{model_name}"
        os.makedirs(save_path, exist_ok=True)
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"✅ Model saved to {save_path}")
        
        # Create model configuration for RL stage
        model_config = {
            "model_name": model_name,
            "model_path": save_path,
            "model_type": "transformer",
            "tokenizer_path": save_path,
            "num_labels": model.num_labels,
            "eval_loss": eval_results.get("eval_loss", 0),
            "eval_accuracy": eval_results.get("eval_accuracy", 0),
            "eval_f1": eval_results.get("eval_f1", 0),
            "eval_precision": eval_results.get("eval_precision", 0),
            "eval_recall": eval_results.get("eval_recall", 0)
        }

        # Clean up
        del model
        del tokenizer
        del trainer
        torch.cuda.empty_cache()
        gc.collect()
        
        return save_path, model_config
        
    except Exception as e:
        print(f"❌ {model_name} training failed: {e}")
        # 如果是网络错误，抛出异常让上层处理
        if "Network is unreachable" in str(e) or "HTTPSConnectionPool" in str(e) or "MaxRetryError" in str(e):
            print("🌐 Network connection error detected - skipping transformer model")
            raise
        else:
            print("🔧 Other error occurred during training")
            raise
    save_path = f"{MODEL_PATH}/{model_name}"
    os.makedirs(save_path, exist_ok=True)
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Create model configuration for RL stage
    model_config = {
        "model_name": model_name,
        "model_path": save_path,
        "model_type": "transformer",
        "tokenizer_path": save_path,
        "num_labels": model.num_labels,
        "eval_loss": eval_results.get("eval_loss", 0),
        "eval_accuracy": eval_results.get("eval_accuracy", 0),
        "eval_f1": eval_results.get("eval_f1", 0),
        "eval_precision": eval_results.get("eval_precision", 0),
        "eval_recall": eval_results.get("eval_recall", 0)
    }

    # Clean up
    del model
    del tokenizer
    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    
    return save_path, model_config

def train_custom_model(model, model_name, train_dataset, eval_dataset, training_args, swanlab_run=None):
    """Train a custom model with optional SwanLab logging and early stopping."""
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=training_args.per_device_train_batch_size, 
        shuffle=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=training_args.per_device_eval_batch_size
    )
    
    # Move model to device
    model.to(device)
    
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Initialize early stopping - 监控验证损失，3个epoch没有改善就停止
    early_stopping = EarlyStopping(patience=3, min_delta=0.01, mode='min')
    best_model_state = None
    
    print(f"🛑 Early stopping enabled for {model_name}: patience=3, min_delta=0.01")

    # Training loop
    for epoch in range(training_args.num_train_epochs):
        model.train()
        epoch_train_loss = 0
        num_batches = 0
        
        for batch in train_dataloader:
            inputs = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            num_batches += 1

        # Calculate average training loss for this epoch
        avg_train_loss = epoch_train_loss / num_batches

        # Evaluation
        model.eval()
        all_preds = []
        all_labels = []
        eval_loss = 0
        num_eval_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                inputs = batch['input_ids'].to(device)
                labels = batch['label'].to(device)
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                eval_loss += loss.item()
                num_eval_batches += 1

                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate average evaluation loss
        avg_eval_loss = eval_loss / num_eval_batches if num_eval_batches > 0 else 0

        # Compute metrics
        accuracy, precision, recall, f1 = custom_metrics(np.array(all_preds), np.array(all_labels))
        
        # Early stopping check
        should_stop, is_improvement = early_stopping(avg_eval_loss, epoch + 1)
        
        # Save best model state if improvement
        if is_improvement:
            best_model_state = model.state_dict().copy()
            print(f"🎯 {model_name} Epoch {epoch + 1}: New best model saved (eval_loss: {avg_eval_loss:.4f})")
        
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        if should_stop:
            print(f"🛑 Early stopping triggered for {model_name} at epoch {epoch + 1}")
            print(f"   Best epoch: {early_stopping.best_epoch}, Best eval_loss: {early_stopping.best_score:.4f}")
            break

        # Log to SwanLab if available
        if swanlab_run:
            try:
                log_dict = {
                    # 使用命名空间分组 - 每个模型有自己的分组
                    f'CustomModels/{model_name}/train_loss': avg_train_loss,
                    f'CustomModels/{model_name}/eval_loss': avg_eval_loss,
                    f'CustomModels/{model_name}/accuracy': accuracy,
                    f'CustomModels/{model_name}/precision': precision,
                    f'CustomModels/{model_name}/recall': recall,
                    f'CustomModels/{model_name}/f1': f1,
                    f'CustomModels/{model_name}/early_stop_counter': early_stopping.counter,
                    f'CustomModels/{model_name}/best_eval_loss': early_stopping.best_score,
                    # 全局指标
                    'step': epoch + 1,
                    'epoch': epoch + 1,
                    # 汇总指标 - 所有自定义模型的比较
                    f'ModelComparison/CustomModels/{model_name}_train_loss': avg_train_loss,
                    f'ModelComparison/CustomModels/{model_name}_accuracy': accuracy,
                    f'ModelComparison/CustomModels/{model_name}_f1': f1
                }
                swanlab_run.log(log_dict)
                print(f"📊 Epoch {epoch + 1} metrics logged to SwanLab for {model_name}")
            except Exception as e:
                print(f"⚠️  SwanLab logging failed for {model_name} epoch {epoch + 1}: {e}")
    
    # Restore best model if early stopping was triggered
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ {model_name} restored to best model state (epoch {early_stopping.best_epoch})")

    # Save model
    save_path = f"{MODEL_PATH}/custom_models/{model_name}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path + ".bin")
    
    # Return model configuration for RL stage (using final metrics)
    model_config = {
        "model_name": model_name,
        "model_path": save_path + ".bin",
        "vocab_size": model.vocab_size if hasattr(model, 'vocab_size') else None,
        "embed_size": model.embed_size if hasattr(model, 'embed_size') else None,
        "num_classes": model.num_classes if hasattr(model, 'num_classes') else None,
        "max_length": model.max_length if hasattr(model, 'max_length') else None,
        "model_type": "custom",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "best_epoch": early_stopping.best_epoch,
        "best_eval_loss": early_stopping.best_score,
        "early_stopped": should_stop
    }

    # Clean up
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return save_path, model_config 