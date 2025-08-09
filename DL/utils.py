import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_directories(dirs):
    """Create directories if they don't exist."""
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

def plot_training_history(history, save_path=None):
    """Plot training and validation metrics."""
    plt.figure(figsize=(12, 8))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_accuracy'], label='Train')
    plt.plot(history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot F1 score
    plt.subplot(2, 2, 3)
    plt.plot(history['train_f1'], label='Train')
    plt.plot(history['val_f1'], label='Validation')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    
    # Plot precision and recall
    plt.subplot(2, 2, 4)
    plt.plot(history['train_precision'], label='Train Precision')
    plt.plot(history['train_recall'], label='Train Recall')
    plt.plot(history['val_precision'], label='Val Precision')
    plt.plot(history['val_recall'], label='Val Recall')
    plt.title('Precision and Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes=None, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class DLSwanLabCallback:
    """SwanLab回调函数 - 用于记录训练过程中的指标"""
    
    def __init__(self, use_swanlab=False, model_name="model"):
        self.use_swanlab = use_swanlab
        self.model_name = model_name
        
    def on_init_end(self, args, state, control, **kwargs):
        """初始化结束时的回调 - 新版本transformers需要"""
        if self.use_swanlab:
            print(f"📊 SwanLab callback initialized for {self.model_name}")
        
    def on_log(self, args, state, control, model=None, tokenizer=None, **kwargs):
        """训练过程中的日志回调"""
        if not self.use_swanlab:
            return
        
        try:
            import swanlab
            logs = kwargs.get('logs', {})
            
            # 记录训练指标
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    swanlab.log({f"{self.model_name}_{key}": value})
                    
        except Exception as e:
            print(f"⚠️  SwanLab logging failed: {e}")
    
    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始时的回调"""
        if self.use_swanlab:
            print(f"📊 Starting SwanLab tracking for {self.model_name}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时的回调"""
        if self.use_swanlab:
            print(f"📊 Completed SwanLab tracking for {self.model_name}") 