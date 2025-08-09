"""
LLM-Finetune阶段配置管理
统一管理Llama-3-8B微调的所有参数和设置
"""
import os
import time
import torch
from typing import Dict, Any, Optional


class LLMConfig:
    """LLM微调配置管理类"""
    
    def __init__(self, llm_gpu_config: Dict[str, Any]):
        """
        初始化LLM配置
        
        Args:
            llm_gpu_config: 从全局多GPU配置获取的LLM阶段配置
        """
        self.llm_gpu_config = llm_gpu_config
        self.gpu_count = llm_gpu_config.get('gpu_count', 1)
        self._setup_base_config()
        self._setup_model_config()
        self._setup_training_config()
        self._setup_data_config()
        self._setup_environment_config()
    
    def _setup_base_config(self):
        """基础配置设置"""
        # 项目配置
        self.PROJECT_NAME = "AdvTG-LLM-Finetune"
        self.EXPERIMENT_PREFIX = "AdvTG-LLM-Llama3"
        self.DESCRIPTION = "LLM Fine-tuning stage - Llama-3-8B with LoRA"
        
        # 输出目录
        self.OUTPUT_DIR = "outputs"
        self.DATASET_DIR = "../dataset"
        
        # 环境配置
        self.SEED = 3407
        self.USE_CUDA_LAUNCH_BLOCKING = True
    
    def _setup_model_config(self):
        """模型配置设置"""
        # 模型基础配置
        self.MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"
        self.MAX_SEQ_LENGTH = 2048
        self.DTYPE = None  # None for auto detection
        self.LOAD_IN_4BIT = True
        
        # LoRA配置
        self.LORA_R = 16
        self.LORA_ALPHA = 16
        self.LORA_DROPOUT = 0
        self.LORA_BIAS = "none"
        self.USE_RSLORA = False
        self.LOFTQ_CONFIG = None
        
        # LoRA目标模块
        self.TARGET_MODULES = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        
        # 训练优化
        self.USE_GRADIENT_CHECKPOINTING = "unsloth"
        self.RANDOM_STATE = 3407
    
    def _setup_training_config(self):
        """训练配置设置"""
        # 基础训练参数
        self.LEARNING_RATE = 2e-4
        self.WARMUP_STEPS = 5
        self.MAX_STEPS = 500
        self.WEIGHT_DECAY = 0.01
        self.LR_SCHEDULER_TYPE = "linear"
        
        # 精度设置 - 与Unsloth兼容
        self.FP16 = False  # 禁用fp16
        self.BF16 = True   # 使用bf16，与Unsloth模型匹配
        
        # 优化器设置
        self.OPTIM = "adamw_8bit"
        self.LOGGING_STEPS = 1
        
        # 多GPU配置
        self.PER_DEVICE_BATCH_SIZE = self.llm_gpu_config.get('per_device_batch_size', 4)
        self.GRADIENT_ACCUMULATION_STEPS = self.llm_gpu_config.get('gradient_accumulation_steps', 8)
        self.EFFECTIVE_BATCH_SIZE = self.llm_gpu_config.get('effective_batch_size', 32)
        
        # 数据处理
        self.DATASET_NUM_PROC = min(8, self.gpu_count * 2)
        self.DATALOADER_NUM_WORKERS = 0  # 避免多进程冲突
        self.PACKING = False
        
        # 分布式训练设置 - 让Unsloth处理
        self.LOCAL_RANK = -1
        self.DDP_BACKEND = None
    
    def _setup_data_config(self):
        """数据配置设置"""
        self.TRAIN_DATA_PATH = os.path.join(self.DATASET_DIR, "llm_train.json")
        self.VAL_DATA_PATH = os.path.join(self.DATASET_DIR, "val.json")
        self.DATA_SHUFFLE_SEED = 42
        
        # Alpaca提示词模板
        self.ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
        
        # 指令模板
        self.MALICIOUS_INSTRUCTION = "Follow these tips to generate malicious http traffic"
        self.BENIGN_INSTRUCTION = "Follow these tips to generate benign http traffic"
    
    def _setup_environment_config(self):
        """环境变量配置"""
        self.ENVIRONMENT_VARS = {
            # Hugging Face镜像设置
            "HF_ENDPOINT": "https://hf-mirror.com",
            "HUGGINGFACE_HUB_ENDPOINT": "https://hf-mirror.com",
            "HF_HUB_ENDPOINT": "https://hf-mirror.com",
            "HUGGINGFACE_HUB_URL": "https://hf-mirror.com",
            
            # 禁用tokenizers并行处理
            "TOKENIZERS_PARALLELISM": "false",
            
            # 禁用wandb，启用swanlab
            "WANDB_DISABLED": "true",
            "WANDB_MODE": "disabled",
            "WANDB_SILENT": "true",
            
            # CUDA设置
            "CUDA_LAUNCH_BLOCKING": "1" if self.USE_CUDA_LAUNCH_BLOCKING else "0",
            
            # NCCL设置 - 多GPU训练
            "NCCL_P2P_DISABLE": "1",
            "NCCL_IB_DISABLE": "1",
            "NCCL_DEBUG": "INFO",
            "NCCL_SOCKET_IFNAME": "lo"
        }
    
    def setup_environment(self):
        """设置环境变量"""
        for key, value in self.ENVIRONMENT_VARS.items():
            os.environ[key] = value
        
        print("🌍 Environment variables configured for LLM training")
        print(f"   - Hugging Face endpoint: {os.environ.get('HF_ENDPOINT')}")
        print(f"   - CUDA launch blocking: {os.environ.get('CUDA_LAUNCH_BLOCKING')}")
        print(f"   - Multi-GPU NCCL configured: ✅")
    
    def get_experiment_name(self) -> str:
        """生成带时间戳的实验名称"""
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        return f"{self.EXPERIMENT_PREFIX}-{timestamp}"
    
    def get_swanlab_config(self) -> Dict[str, Any]:
        """获取SwanLab配置"""
        return {
            # 数值类型配置 - SwanLab兼容
            "model_version": 3.8,  # Llama-3-8B版本
            "max_seq_length": self.MAX_SEQ_LENGTH,
            "learning_rate": self.LEARNING_RATE,
            "lora_r": self.LORA_R,
            "lora_alpha": self.LORA_ALPHA,
            "target_modules_count": len(self.TARGET_MODULES),
            
            # 多GPU配置信息
            "gpu_count": self.gpu_count,
            "per_device_batch_size": self.PER_DEVICE_BATCH_SIZE,
            "gradient_accumulation_steps": self.GRADIENT_ACCUMULATION_STEPS,
            "total_effective_batch_size": self.EFFECTIVE_BATCH_SIZE,
            "multi_gpu_training": 1 if self.gpu_count > 1 else 0,
            
            # 训练配置
            "max_steps": self.MAX_STEPS,
            "warmup_steps": self.WARMUP_STEPS,
            "weight_decay": self.WEIGHT_DECAY,
            "bf16_enabled": 1 if self.BF16 else 0,
            "use_4bit_quantization": 1 if self.LOAD_IN_4BIT else 0
        }
    
    def get_training_arguments(self) -> Dict[str, Any]:
        """获取训练参数配置"""
        return {
            "per_device_train_batch_size": self.PER_DEVICE_BATCH_SIZE,
            "gradient_accumulation_steps": self.GRADIENT_ACCUMULATION_STEPS,
            "warmup_steps": self.WARMUP_STEPS,
            "max_steps": self.MAX_STEPS,
            "learning_rate": self.LEARNING_RATE,
            "fp16": self.FP16,
            "bf16": self.BF16,
            "logging_steps": self.LOGGING_STEPS,
            "optim": self.OPTIM,
            "weight_decay": self.WEIGHT_DECAY,
            "lr_scheduler_type": self.LR_SCHEDULER_TYPE,
            "seed": self.SEED,
            "output_dir": self.OUTPUT_DIR,
            "local_rank": self.LOCAL_RANK,
            "ddp_backend": self.DDP_BACKEND,
            "dataloader_num_workers": self.DATALOADER_NUM_WORKERS
        }
    
    def get_lora_config(self) -> Dict[str, Any]:
        """获取LoRA配置"""
        return {
            "r": self.LORA_R,
            "target_modules": self.TARGET_MODULES,
            "lora_alpha": self.LORA_ALPHA,
            "lora_dropout": self.LORA_DROPOUT,
            "bias": self.LORA_BIAS,
            "use_gradient_checkpointing": self.USE_GRADIENT_CHECKPOINTING,
            "random_state": self.RANDOM_STATE,
            "use_rslora": self.USE_RSLORA,
            "loftq_config": self.LOFTQ_CONFIG
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return {
            "model_name": self.MODEL_NAME,
            "max_seq_length": self.MAX_SEQ_LENGTH,
            "dtype": self.DTYPE,
            "load_in_4bit": self.LOAD_IN_4BIT
        }
    
    def create_directories(self):
        """创建必要的目录"""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        print(f"📁 Output directory created: {self.OUTPUT_DIR}")
    
    def print_config_summary(self):
        """打印配置摘要"""
        print("\n" + "="*60)
        print("🔧 LLM-Finetune Configuration Summary")
        print("="*60)
        print(f"📊 Project: {self.PROJECT_NAME}")
        print(f"🤖 Model: {self.MODEL_NAME}")
        print(f"📏 Max sequence length: {self.MAX_SEQ_LENGTH}")
        print(f"🎯 LoRA rank: {self.LORA_R}")
        print(f"⚡ Learning rate: {self.LEARNING_RATE}")
        print(f"📈 Max steps: {self.MAX_STEPS}")
        print(f"💾 Use 4-bit: {self.LOAD_IN_4BIT}")
        print(f"🔢 BF16 precision: {self.BF16}")
        print(f"🚀 GPU count: {self.gpu_count}")
        print(f"📦 Per device batch size: {self.PER_DEVICE_BATCH_SIZE}")
        print(f"🔄 Gradient accumulation: {self.GRADIENT_ACCUMULATION_STEPS}")
        print(f"📊 Effective batch size: {self.EFFECTIVE_BATCH_SIZE}")
        print("="*60)
    
    def setup_device_and_gpu_info(self) -> torch.device:
        """设置设备并打印GPU信息"""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        gpu_count = torch.cuda.device_count()
        
        print(f"🎮 Primary device: {device}")
        print(f"🚀 Total GPU count: {gpu_count}")
        
        if gpu_count > 1:
            print(f"🔥 Multi-GPU training enabled with {gpu_count} GPUs")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        return device
