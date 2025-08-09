import os
import gc
import torch
import torch.nn as nn
import numpy as np
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
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 准备回调函数列表
    callbacks = []
    if swanlab_callback:
        callbacks.append(swanlab_callback)
    if early_stopping_callback:
        callbacks.append(early_stopping_callback)

    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=transformer_metrics,
        callbacks=callbacks  # 添加回调函数
    )

    # Train model
    trainer.train()
    
    # Evaluate model to get metrics
    eval_results = trainer.evaluate()
    
    # Save model
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
                    f'{model_name}_epoch': epoch + 1,
                    f'{model_name}_train_loss': avg_train_loss,
                    f'{model_name}_eval_loss': avg_eval_loss,
                    f'{model_name}_accuracy': accuracy,
                    f'{model_name}_precision': precision,
                    f'{model_name}_recall': recall,
                    f'{model_name}_f1': f1,
                    f'{model_name}_early_stop_counter': early_stopping.counter,
                    f'{model_name}_best_eval_loss': early_stopping.best_score
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