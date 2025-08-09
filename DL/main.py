import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# 修复分布式训练初始化问题 - 直接在代码中处理
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0" 
os.environ["LOCAL_RANK"] = "0"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"

# 导入全局多GPU配置
import sys
sys.path.append('..')
from multi_gpu_config import initialize_multi_gpu_for_stage, get_training_arguments_for_stage

# Set Hugging Face mirror BEFORE importing transformers
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_URL"] = "https://hf-mirror.com"

# Disable tokenizers parallelism to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 避免分布式训练自动初始化的环境变量
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0" 
os.environ["LOCAL_RANK"] = "0"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"

# 禁用wandb
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import torch
from transformers import TrainingArguments, TrainerCallback, EarlyStoppingCallback

# 手动初始化分布式环境以避免自动检测问题
if not torch.distributed.is_initialized():
    try:
        # 初始化单进程分布式环境
        torch.distributed.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://',
            world_size=1,
            rank=0
        )
        print("✅ 手动初始化分布式环境成功")
    except Exception as e:
        print(f"⚠️ 分布式初始化跳过: {e}")
        # 这是正常的，继续使用DataParallel

from models import CNNLSTMClassifier, TextCNNClassifier, DNNClassifier, DeepLog
from data_processing import load_data, prepare_dataset, load_tokenizer, SimpleTokenizer
from training import train_transformer_model, train_custom_model

# vLLM风格的DL并行训练配置 - 现在使用全局多GPU配置
ENABLE_VLLM_STYLE_PARALLEL = True   # 启用vLLM风格并行优化
ENABLE_TENSOR_PARALLEL = True       # 启用张量并行（多GPU）
ENABLE_DATA_PARALLEL = True         # 启用数据并行处理
MAX_PARALLEL_WORKERS = min(6, mp.cpu_count())  # 增加并行工作进程

# 创建SwanLab回调函数用于实时记录DL训练过程
class DLSwanLabCallback(TrainerCallback):
    def __init__(self, use_swanlab=False, model_name=""):
        self.use_swanlab = use_swanlab
        self.model_name = model_name
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """在每次日志记录时调用"""
        if self.use_swanlab and logs:
            try:
                import swanlab
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
                    log_dict[f'{self.model_name}_accuracy'] = logs['eval_accuracy']
                if 'eval_precision' in logs:
                    log_dict[f'{self.model_name}_precision'] = logs['eval_precision']
                if 'eval_recall' in logs:
                    log_dict[f'{self.model_name}_recall'] = logs['eval_recall']
                if 'eval_f1' in logs:
                    log_dict[f'{self.model_name}_f1'] = logs['eval_f1']
                
                # 添加step信息
                log_dict['step'] = state.global_step
                # 不添加字符串类型的model字段，因为SwanLab期望数值类型
                
                if log_dict:
                    swanlab.log(log_dict)
                    print(f"📊 {self.model_name} Step {state.global_step}: Loss: {logs.get('loss', 'N/A')}")
                    
            except Exception as e:
                print(f"⚠️  SwanLab logging error for {self.model_name}: {e}")
    
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """在评估时调用"""
        if self.use_swanlab and logs:
            try:
                import swanlab
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
    # 启用所有8张GPU用于DL训练
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    # 多GPU通信优化
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"
    
    print("🚀 启用8张RTX 4090 GPU进行DL训练!")
    
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

def get_optimal_dl_config():
    """获取DL阶段最优并行配置"""
    try:
        import torch
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        gpu_count = 0
    
    cpu_count = mp.cpu_count()
    
    # vLLM风格的动态配置 - 优化8GPU配置
    if gpu_count >= 8:
        tensor_parallel_size = 8  # 使用所有8张GPU
        optimal_batch_size = 16 * gpu_count  # 8GPU: 128 batch size
        worker_multiplier = 4
    elif gpu_count >= 4:
        tensor_parallel_size = 4
        optimal_batch_size = 24 * gpu_count  # 4GPU: 96 batch size  
        worker_multiplier = 3
    elif gpu_count >= 2:
        tensor_parallel_size = 2
        optimal_batch_size = 32 * gpu_count  # 2GPU: 64 batch size
        worker_multiplier = 2
    else:
        tensor_parallel_size = 1
        optimal_batch_size = 16
        worker_multiplier = 1
    
    optimal_workers = min(MAX_PARALLEL_WORKERS * 2, cpu_count // 2) * worker_multiplier  # 增加worker数量
    
    print(f"🚀 vLLM风格8GPU优化配置:")
    print(f"   - 检测GPU数量: {gpu_count}")
    print(f"   - 张量并行大小: {tensor_parallel_size}")
    print(f"   - 优化batch size: {optimal_batch_size}")
    print(f"   - 优化workers: {optimal_workers}")
    
    return {
        "gpu_count": gpu_count,
        "tensor_parallel_size": tensor_parallel_size,
        "optimal_batch_size": optimal_batch_size,
        "optimal_workers": optimal_workers,
        "enable_mixed_precision": gpu_count > 0,  # GPU可用时启用混合精度
        "enable_gradient_checkpointing": True,    # 内存优化
        "enable_8gpu_optimization": gpu_count >= 8,  # 8GPU特殊优化标记
    }

def main():
    """主训练函数 - 使用全局多GPU配置和完整训练流程"""
    # Set environment variables
    set_environment()
    
    # 初始化DL阶段的多GPU配置
    dl_gpu_config = initialize_multi_gpu_for_stage("DL")
    
    # Disable wandb completely and enable swanlab
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_SILENT"] = "true"
    
    print("🎯 AdvTG-DL 多GPU训练")
    print("=" * 60)
    print(f"🔧 DL阶段GPU配置:")
    print(f"   - GPU数量: {dl_gpu_config['gpu_count']}")
    print(f"   - 每设备batch size: {dl_gpu_config['per_device_batch_size']}")
    print(f"   - 梯度累积步数: {dl_gpu_config['gradient_accumulation_steps']}")
    print(f"   - 总有效batch size: {dl_gpu_config['effective_batch_size']}")
    print(f"   - 数据加载workers: {dl_gpu_config['dataloader_num_workers']}")
    print(f"   - 混合精度: {dl_gpu_config['enable_mixed_precision']}")
    
    # Initialize SwanLab for experiment tracking  
    try:
        import swanlab
        import time
        # 创建包含时间戳的自定义实验名称
        experiment_name = f"AdvTG-DL-{time.strftime('%Y%m%d-%H%M%S')}"
        
        swanlab.init(
            project="AdvTG-DL-Training",
            name=experiment_name,
            description="Deep Learning stage - BERT and Custom Models Training with multi-GPU optimization",
            config={
                # 移除字符串类型字段，SwanLab config中只保留数值类型
                "batch_size": dl_gpu_config['per_device_batch_size'],
                "learning_rate": 2e-5,
                "num_epochs": 6,  # 更新为6个epoch
                "max_length": 512,
                "gpu_count": dl_gpu_config['gpu_count'],
                "effective_batch_size": dl_gpu_config['effective_batch_size'],
                "gradient_accumulation_steps": dl_gpu_config['gradient_accumulation_steps'],
                "mixed_precision": 1 if dl_gpu_config['enable_mixed_precision'] else 0,  # 转换为数值
                "dataloader_workers": dl_gpu_config['dataloader_num_workers'],
                "multi_gpu_optimization": 1  # 用数值表示优化状态
            }
        )
        print("✅ SwanLab initialized successfully!")
        print(f"📊 Project: AdvTG-DL-Training")
        print(f"📊 Experiment: {experiment_name}")
        print(f"📊 学习率为: 2e-5")
        print(f"📊 批次大小为: {dl_gpu_config['per_device_batch_size']}")
        print(f"📊 训练轮数为: 6")
        
        # 记录训练开始状态
        swanlab.log({
            "training_started": 1,
            "initialization_timestamp": time.time(),
            "gpu_initialization_complete": 1
        })
        
        use_swanlab = True
    except ImportError:
        print("⚠️  SwanLab not installed, continuing without experiment tracking")
        use_swanlab = False
    except Exception as e:
        print(f"⚠️  SwanLab initialization failed: {e}, continuing without experiment tracking")
        use_swanlab = False
    
    # Configuration
    FORCE_SIMPLE_MODE = False  # Set to True to skip BERT and only use custom models
    
    # Define constants (使用vLLM优化的参数)
    DATA_PATH = "../dataset/dl_train.json"  # Use small dataset for DL training
    MAX_LENGTH = 512
    BATCH_SIZE = dl_gpu_config['per_device_batch_size']  # 使用多GPU优化的batch size
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 6  # 增加到6个epoch，观察收敛情况
    
    # Set device with vLLM-style optimization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎮 使用设备: {device}")
    
    # vLLM风格的CUDA优化
    if torch.cuda.is_available() and ENABLE_VLLM_STYLE_PARALLEL:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        if dl_gpu_config['gpu_count'] > 1:
            torch.cuda.set_device(0)  # 设置主GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(dl_gpu_config['gpu_count'])))
            print(f"🚀 启用多GPU并行优化")
    
    # Load data
    print("📁 Loading data...")
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
        
    # Define training arguments for transformer with 多GPU优化
    transformer_training_args_base = {
        "output_dir": "../models/bert",
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_train_epochs": NUM_EPOCHS,
        "eval_strategy": "epoch",  # Changed from evaluation_strategy
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",  # 使用验证损失作为最佳模型指标
        "greater_is_better": False,  # 损失越小越好
        "report_to": "none",  # Disable all automatic logging including wandb
        "logging_steps": 50,  # Log every 50 steps for manual tracking
    }
    
    # 合并全局多GPU配置
    transformer_training_args_kwargs = get_training_arguments_for_stage("DL", transformer_training_args_base)
    
    # 过滤掉无效的参数，只保留TrainingArguments支持的参数
    valid_training_args = {}
    from inspect import signature
    training_args_params = signature(TrainingArguments).parameters
    
    for key, value in transformer_training_args_kwargs.items():
        if key in training_args_params:
            valid_training_args[key] = value
        else:
            print(f"⚠️ 跳过无效参数: {key}={value}")
    
    # 添加必要的多GPU配置
    valid_training_args.update({
        "ddp_find_unused_parameters": False,
        "dataloader_drop_last": True,
        "local_rank": -1,  # 强制禁用分布式
    })
    
    transformer_training_args = TrainingArguments(**valid_training_args)
        
        # Only train transformer model if we have real transformers tokenizer
    all_model_configs = []  # 收集所有模型配置
    
    if not isinstance(tokenizer, SimpleTokenizer):
        try:
            # 创建BERT专用的SwanLab回调和早停回调
            bert_callback = DLSwanLabCallback(use_swanlab=use_swanlab, model_name="BERT") if use_swanlab else None
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.01
            )
            
            print("🛑 BERT Early stopping enabled: patience=3, threshold=0.01")
            
            bert_save_path, bert_config = train_transformer_model(
                transformer_model_name,
                transformer_model_name,  # model_path parameter 
                train_dataset, 
                val_dataset, 
                transformer_training_args,
                swanlab_callback=bert_callback,  # 传递回调函数
                early_stopping_callback=early_stopping_callback  # 传递早停回调
            )
            all_model_configs.append(bert_config)
            print("✅ BERT model training completed!")
            
            # 记录BERT模型的最终结果到SwanLab
            if use_swanlab:
                try:
                    swanlab.log({
                        'BERT_final_accuracy': bert_config.get('eval_accuracy', 0),
                        'BERT_final_precision': bert_config.get('eval_precision', 0),
                        'BERT_final_recall': bert_config.get('eval_recall', 0),
                        'BERT_final_f1': bert_config.get('eval_f1', 0),
                        'BERT_final_loss': bert_config.get('eval_loss', 0),
                        'transformer_model_completed': 1  # 用数值表示完成状态
                    })
                    print(f"📊 BERT final results logged to SwanLab")
                except Exception as e:
                    print(f"⚠️  SwanLab logging failed for BERT: {e}")
                    
        except Exception as e:
            print(f"❌ BERT training failed: {e}")
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
    
    # 自定义模型训练参数 - 使用全局多GPU配置
    custom_training_args_base = {
        "output_dir": "../models/custom_models",
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_train_epochs": NUM_EPOCHS,
        "report_to": "none",  # Disable all automatic logging including wandb
        "logging_steps": 50,  # Log every 50 steps for manual tracking
    }
    
    # 合并全局多GPU配置
    custom_training_args_kwargs = get_training_arguments_for_stage("DL", custom_training_args_base)
    
    # 过滤掉无效的参数，只保留TrainingArguments支持的参数
    valid_custom_args = {}
    from inspect import signature
    training_args_params = signature(TrainingArguments).parameters
    
    for key, value in custom_training_args_kwargs.items():
        if key in training_args_params:
            valid_custom_args[key] = value
        else:
            print(f"⚠️ 跳过无效参数: {key}={value}")
    
    # 添加必要的多GPU配置
    valid_custom_args.update({
        "ddp_find_unused_parameters": False,
        "dataloader_drop_last": True,
        "local_rank": -1,  # 强制禁用分布式
    })
    
    custom_training_args = TrainingArguments(**valid_custom_args)
    
    print("\n====== Training Custom Models ======")
    
    for model_name, model in models.items():
        print(f"\n🔄 Training model: {model_name}")
        try:
            # 传递SwanLab实例用于自定义模型训练实时记录
            save_path, model_config = train_custom_model(
                model, 
                model_name, 
                train_dataset, 
                val_dataset, 
                custom_training_args,
                swanlab_run=swanlab if use_swanlab else None  # 传递SwanLab实例
            )
            all_model_configs.append(model_config)
            print(f"✅ Completed training model: {model_name}")
            
            # 记录自定义模型的最终结果到SwanLab
            if use_swanlab:
                try:
                    swanlab.log({
                        f'{model_name}_final_accuracy': model_config.get('accuracy', 0),
                        f'{model_name}_final_precision': model_config.get('precision', 0),
                        f'{model_name}_final_recall': model_config.get('recall', 0),
                        f'{model_name}_final_f1': model_config.get('f1', 0),
                        f'{model_name}_final_loss': model_config.get('loss', 0),
                        f'custom_model_{model_name}_completed': 1  # 用数值表示完成状态
                    })
                    print(f"📊 {model_name} final results logged to SwanLab")
                except Exception as e:
                    print(f"⚠️  SwanLab logging failed for {model_name}: {e}")
            
        except Exception as e:
            print(f"❌ Failed to train {model_name}: {e}")
    
    # Save model configurations for RL stage
    import pickle
    config_save_path = "../models/model_configs.pkl"
    os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
    
    with open(config_save_path, 'wb') as f:
        pickle.dump(all_model_configs, f)
    
    # Create image model configurations (placeholder)
    try:
        from create_image_configs import create_image_model_configs
        create_image_model_configs()
        print("✅ Created image model configurations")
    except Exception as e:
        print(f"⚠️  Failed to create image model configs: {e}")
    
    # Log results to SwanLab if available
    if use_swanlab:
        try:
            # 记录完整的训练汇总信息
            summary_metrics = {
                "training_completed": 1,
                "total_models_trained": len(all_model_configs),
                "gpu_count": dl_gpu_config['gpu_count'],
                "final_learning_rate": LEARNING_RATE,
                "final_num_epochs": NUM_EPOCHS,
                "final_batch_size": BATCH_SIZE,
                "final_effective_batch_size": dl_gpu_config['effective_batch_size'],
                "gradient_accumulation_steps": dl_gpu_config['gradient_accumulation_steps'],
                "mixed_precision_enabled": 1 if dl_gpu_config['enable_mixed_precision'] else 0
            }
            
            # 记录每个模型的最终指标
            for config in all_model_configs:
                model_name = config.get('model_name', 'unknown')
                if 'accuracy' in config:
                    summary_metrics[f"{model_name}_final_accuracy"] = config['accuracy']
                if 'f1' in config:
                    summary_metrics[f"{model_name}_final_f1"] = config['f1']
                if 'precision' in config:
                    summary_metrics[f"{model_name}_final_precision"] = config['precision']
                if 'recall' in config:
                    summary_metrics[f"{model_name}_final_recall"] = config['recall']
                if 'loss' in config:
                    summary_metrics[f"{model_name}_final_loss"] = config['loss']
            
            # 计算并记录最佳性能指标
            best_accuracy = max([config.get('accuracy', 0) for config in all_model_configs], default=0)
            best_f1 = max([config.get('f1', 0) for config in all_model_configs], default=0)
            
            if best_accuracy > 0:
                summary_metrics["best_accuracy_across_all_models"] = best_accuracy
            if best_f1 > 0:
                summary_metrics["best_f1_across_all_models"] = best_f1
            
            # 记录硬件利用率信息
            summary_metrics.update({
                "hardware_gpu_memory_per_device": 25.3,  # RTX 4090 25.3GB each
                "hardware_total_gpu_memory": 25.3 * dl_gpu_config['gpu_count'],
                "optimization_vllm_style": 1,
                "optimization_tensor_parallel": 1 if dl_gpu_config['gpu_count'] > 1 else 0
            })
            
            swanlab.log(summary_metrics)
            swanlab.finish()
            print("📊 Comprehensive results logged to SwanLab successfully!")
            print(f"📊 Best accuracy: {best_accuracy:.4f}")
            print(f"📊 Best F1 score: {best_f1:.4f}")
        except Exception as e:
            print(f"⚠️  SwanLab logging failed: {e}")
    
    print(f"\n✅ Saved text model configurations to: {config_save_path}")
    print(f"📊 Total text models configured: {len(all_model_configs)}")
    
    print("\n🎉 All models training completed with multi-GPU optimization!")

if __name__ == "__main__":
    main()
