"""
DL阶段配置文件
统一管理深度学习训练的所有配置参数
"""
import os
import multiprocessing as mp
import torch

class DLConfig:
    """DL阶段配置类"""
    
    def __init__(self, dl_gpu_config=None):
        self.dl_gpu_config = dl_gpu_config or {}
        
        # 基础配置
        self.DATA_PATH = "../dataset/dl_train.json"
        self.MAX_LENGTH = 512
        self.LEARNING_RATE = 2e-5
        self.NUM_EPOCHS = 6
        self.FORCE_SIMPLE_MODE = False
        
        # GPU相关配置
        self.BATCH_SIZE = self.dl_gpu_config.get('per_device_batch_size', 8)
        self.GPU_COUNT = self.dl_gpu_config.get('gpu_count', 1)
        
        # vLLM风格并行配置
        self.ENABLE_VLLM_STYLE_PARALLEL = True
        self.ENABLE_TENSOR_PARALLEL = True
        self.ENABLE_DATA_PARALLEL = True
        self.MAX_PARALLEL_WORKERS = min(6, mp.cpu_count())
        
        # 模型配置
        self.EMBED_SIZE = 128
        self.NUM_CLASSES = 2
        self.VOCAB_SIZE = 30522  # BERT default
        
        # 路径配置
        self.MODEL_SAVE_DIR = '../models'
        self.CUSTOM_MODEL_DIR = '../models/custom_models'
        self.CONFIG_SAVE_PATH = "../models/model_configs.pkl"
        
        # 训练器配置
        self.BERT_MODEL_NAME = "bert-base-uncased"
        self.EARLY_STOPPING_PATIENCE = 3
        self.EARLY_STOPPING_THRESHOLD = 0.01
        self.LOGGING_STEPS = 50
        
        # SwanLab配置
        self.SWANLAB_PROJECT = "AdvTG-DL-Training"
        self.SWANLAB_DESCRIPTION = "Deep Learning stage - BERT and Custom Models Training with multi-GPU optimization"
    
    def get_transformer_training_args(self):
        """获取Transformer模型训练参数"""
        return {
            "output_dir": "./models/bert",
            "per_device_train_batch_size": self.BATCH_SIZE,
            "per_device_eval_batch_size": self.BATCH_SIZE,
            "learning_rate": self.LEARNING_RATE,
            "num_train_epochs": self.NUM_EPOCHS,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "report_to": "none",
            "logging_steps": self.LOGGING_STEPS,
            # GPU相关配置
            "dataloader_num_workers": self.dl_gpu_config.get('dataloader_num_workers', 4),
            "dataloader_pin_memory": self.dl_gpu_config.get('dataloader_pin_memory', True),
            "fp16": self.dl_gpu_config.get('enable_mixed_precision', False),
            "gradient_accumulation_steps": self.dl_gpu_config.get('gradient_accumulation_steps', 1),
        }
    
    def get_custom_training_args(self):
        """获取自定义模型训练参数"""
        return {
            "output_dir": self.CUSTOM_MODEL_DIR,
            "per_device_train_batch_size": self.BATCH_SIZE,
            "per_device_eval_batch_size": self.BATCH_SIZE,
            "learning_rate": self.LEARNING_RATE,
            "num_train_epochs": self.NUM_EPOCHS,
            "report_to": "none",
            "logging_steps": self.LOGGING_STEPS,
            # GPU相关配置
            "dataloader_num_workers": self.dl_gpu_config.get('dataloader_num_workers', 4),
            "dataloader_pin_memory": self.dl_gpu_config.get('dataloader_pin_memory', True),
            "fp16": self.dl_gpu_config.get('enable_mixed_precision', False),
            "gradient_accumulation_steps": self.dl_gpu_config.get('gradient_accumulation_steps', 1),
        }
    
    def get_swanlab_config(self):
        """获取SwanLab配置"""
        return {
            "batch_size": self.BATCH_SIZE,
            "learning_rate": self.LEARNING_RATE,
            "num_train_epochs": self.NUM_EPOCHS,
            "warmup_steps": 500,
            "gpu_count": self.GPU_COUNT,
            "effective_batch_size": self.dl_gpu_config.get('effective_batch_size', self.BATCH_SIZE),
            "mixed_precision": 1 if self.dl_gpu_config.get('enable_mixed_precision', False) else 0,
            "parallel_workers": self.dl_gpu_config.get('dataloader_num_workers', 4),
            "gradient_accumulation": self.dl_gpu_config.get('gradient_accumulation_steps', 1),
            "stage": "DL"
        }
    
    def get_model_definitions(self):
        """获取自定义模型定义"""
        from models import CNNLSTMClassifier, TextCNNClassifier, DNNClassifier, DeepLog
        
        return {
            "textcnn": TextCNNClassifier(self.VOCAB_SIZE, self.EMBED_SIZE, self.NUM_CLASSES, self.MAX_LENGTH),
            "cnn_lstm": CNNLSTMClassifier(self.VOCAB_SIZE, self.EMBED_SIZE, self.NUM_CLASSES, self.MAX_LENGTH),
            "dnn": DNNClassifier(self.VOCAB_SIZE, self.EMBED_SIZE, self.NUM_CLASSES, self.MAX_LENGTH),
            "deeplog": DeepLog(self.VOCAB_SIZE, self.EMBED_SIZE, self.NUM_CLASSES, self.MAX_LENGTH)
        }
    
    def setup_environment(self):
        """设置环境变量和GPU配置"""
        # CUDA配置
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"
        
        # Wandb禁用
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_SILENT"] = "true"
        
        # Tokenizers配置
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Hugging Face镜像
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        os.environ["HUGGINGFACE_HUB_ENDPOINT"] = "https://hf-mirror.com"
        os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"
        os.environ["HUGGINGFACE_HUB_URL"] = "https://hf-mirror.com"
        
        print("🚀 启用8张RTX 4090 GPU进行DL训练!")
        
        # 设置transformers镜像
        self._setup_transformers_mirror()
    
    def _setup_transformers_mirror(self):
        """设置transformers库镜像"""
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
    
    def setup_gpu_optimization(self):
        """设置GPU优化"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🎮 使用设备: {device}")
        
        if torch.cuda.is_available() and self.ENABLE_VLLM_STYLE_PARALLEL:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            if self.GPU_COUNT > 1:
                torch.cuda.set_device(0)
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(self.GPU_COUNT)))
                print(f"🚀 启用多GPU并行优化")
        
        return device
    
    def create_directories(self):
        """创建必要的目录"""
        os.makedirs(self.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(self.CUSTOM_MODEL_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(self.CONFIG_SAVE_PATH), exist_ok=True)
    
    def print_config_summary(self):
        """打印配置摘要"""
        print("🎯 AdvTG-DL 多GPU训练")
        print("=" * 60)
        print(f"🔧 DL阶段配置摘要:")
        print(f"   - GPU数量: {self.GPU_COUNT}")
        print(f"   - 每设备batch size: {self.BATCH_SIZE}")
        print(f"   - 学习率: {self.LEARNING_RATE}")
        print(f"   - 训练轮数: {self.NUM_EPOCHS}")
        print(f"   - 最大序列长度: {self.MAX_LENGTH}")
        print(f"   - 梯度累积步数: {self.dl_gpu_config.get('gradient_accumulation_steps', 1)}")
        print(f"   - 总有效batch size: {self.dl_gpu_config.get('effective_batch_size', self.BATCH_SIZE)}")
        print(f"   - 数据加载workers: {self.dl_gpu_config.get('dataloader_num_workers', 4)}")
        print(f"   - 混合精度: {self.dl_gpu_config.get('enable_mixed_precision', False)}")
        print(f"   - vLLM风格优化: {self.ENABLE_VLLM_STYLE_PARALLEL}")


def get_optimal_dl_config():
    """获取DL阶段最优并行配置 - 独立函数版本"""
    try:
        import torch
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        gpu_count = 0
    
    cpu_count = mp.cpu_count()
    MAX_PARALLEL_WORKERS = min(6, mp.cpu_count())
    
    # vLLM风格的动态配置 - 优化8GPU配置
    if gpu_count >= 8:
        tensor_parallel_size = 8
        optimal_batch_size = 16 * gpu_count
        worker_multiplier = 4
    elif gpu_count >= 4:
        tensor_parallel_size = 4
        optimal_batch_size = 24 * gpu_count
        worker_multiplier = 3
    elif gpu_count >= 2:
        tensor_parallel_size = 2
        optimal_batch_size = 32 * gpu_count
        worker_multiplier = 2
    else:
        tensor_parallel_size = 1
        optimal_batch_size = 16
        worker_multiplier = 1
    
    optimal_workers = min(MAX_PARALLEL_WORKERS * 2, cpu_count // 2) * worker_multiplier
    
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
        "enable_mixed_precision": gpu_count > 0,
        "enable_gradient_checkpointing": True,
        "enable_8gpu_optimization": gpu_count >= 8,
    }
