#!/usr/bin/env python3
"""
全局多GPU配置模块 - 为AdvTG三个阶段提供统一的多GPU优化
支持DL阶段、LLM阶段和RL阶段的自适应GPU配置
"""
import os
import torch
import multiprocessing as mp
from typing import Dict, Tuple, Optional

class AdvTGMultiGPUConfig:
    """AdvTG多GPU配置管理类"""
    
    def __init__(self):
        self.gpu_count = 0
        self.is_initialized = False
        self.stage_configs = {}
        
    def initialize_global_gpu_environment(self) -> bool:
        """初始化全局GPU环境"""
        if self.is_initialized:
            return True
        
        # 清除可能的GPU限制环境变量，以便检测所有GPU
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            print(f"🔍 清除现有CUDA_VISIBLE_DEVICES限制: {os.environ['CUDA_VISIBLE_DEVICES']}")
            del os.environ['CUDA_VISIBLE_DEVICES']
            
        # 检测GPU可用性
        self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        if self.gpu_count == 0:
            print("⚠️  未检测到GPU，将使用CPU模式")
            self.is_initialized = True
            return False
            
        print(f"🚀 AdvTG全局GPU配置初始化")
        print(f"检测到 {self.gpu_count} 张GPU:")
        
        # 显示所有GPU信息
        for i in range(self.gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            memory_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"   GPU {i}: {gpu_name} ({memory_gb:.1f} GB)")
        
        # 设置全局GPU环境变量
        if self.gpu_count > 1:
            # 设置所有GPU可见
            gpu_list = ','.join(str(i) for i in range(self.gpu_count))
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
            
            # 多GPU通信配置（NCCL优化）
            os.environ['NCCL_P2P_DISABLE'] = '1'
            os.environ['NCCL_IB_DISABLE'] = '1'
            os.environ['NCCL_DEBUG'] = 'INFO'
            os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
            
            # PyTorch多GPU优化
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
            print(f"✅ 多GPU环境配置完成")
        else:
            print(f"📱 单GPU模式")
            
        # CUDA基础优化
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        self.is_initialized = True
        return True
    
    def get_stage_config(self, stage: str) -> Dict:
        """获取特定阶段的GPU配置
        
        Args:
            stage: 'DL', 'LLM', 'RL'
        """
        if not self.is_initialized:
            self.initialize_global_gpu_environment()
            
        if stage in self.stage_configs:
            return self.stage_configs[stage]
            
        # 根据不同阶段计算最优配置
        config = self._calculate_stage_config(stage)
        self.stage_configs[stage] = config
        
        print(f"\n🔧 {stage}阶段多GPU配置:")
        print(f"   - GPU数量: {config['gpu_count']}")
        print(f"   - 每设备batch size: {config['per_device_batch_size']}")
        print(f"   - 梯度累积步数: {config['gradient_accumulation_steps']}")
        print(f"   - 总有效batch size: {config['effective_batch_size']}")
        print(f"   - 数据加载worker数: {config['dataloader_num_workers']}")
        print(f"   - 混合精度: {config['enable_mixed_precision']}")
        
        return config
    
    def _calculate_stage_config(self, stage: str) -> Dict:
        """根据训练阶段计算最优GPU配置"""
        cpu_count = mp.cpu_count()
        
        if self.gpu_count == 0:
            return self._get_cpu_config(stage)
            
        if stage == "DL":
            # DL阶段：BERT + 自定义模型，内存需求中等
            if self.gpu_count >= 4:
                per_device_batch_size = 16
                gradient_accumulation = 2
                num_workers = min(8, cpu_count // 2)
            elif self.gpu_count >= 2:
                per_device_batch_size = 24
                gradient_accumulation = 4
                num_workers = min(6, cpu_count // 2)
            else:
                per_device_batch_size = 32
                gradient_accumulation = 8
                num_workers = min(4, cpu_count // 2)
                
        elif stage == "LLM":
            # LLM阶段：Llama-3-8B，内存需求最高
            if self.gpu_count >= 8:
                per_device_batch_size = 2
                gradient_accumulation = 8
                num_workers = min(16, cpu_count)
            elif self.gpu_count >= 4:
                per_device_batch_size = 4
                gradient_accumulation = 16
                num_workers = min(12, cpu_count // 2)
            elif self.gpu_count >= 2:
                per_device_batch_size = 6
                gradient_accumulation = 24
                num_workers = min(8, cpu_count // 2)
            else:
                per_device_batch_size = 8
                gradient_accumulation = 64
                num_workers = min(4, cpu_count // 2)
                
        elif stage == "RL":
            # RL阶段：PPO训练，batch size较小但需要快速响应
            if self.gpu_count >= 4:
                per_device_batch_size = 4
                gradient_accumulation = 2
                num_workers = min(6, cpu_count // 4)
            elif self.gpu_count >= 2:
                per_device_batch_size = 6
                gradient_accumulation = 4
                num_workers = min(4, cpu_count // 4)
            else:
                per_device_batch_size = 8
                gradient_accumulation = 8
                num_workers = min(2, cpu_count // 4)
        else:
            # 默认配置
            per_device_batch_size = 8
            gradient_accumulation = 8
            num_workers = min(4, cpu_count // 4)
            
        effective_batch_size = per_device_batch_size * gradient_accumulation * self.gpu_count
        
        return {
            "gpu_count": self.gpu_count,
            "per_device_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": gradient_accumulation,
            "effective_batch_size": effective_batch_size,
            "dataloader_num_workers": num_workers,
            "enable_mixed_precision": True,
            "enable_gradient_checkpointing": stage in ["LLM", "RL"],  # LLM和RL启用梯度检查点
            "dataloader_pin_memory": True,
            "ddp_find_unused_parameters": False,
            "remove_unused_columns": False,
            "stage": stage
        }
    
    def _get_cpu_config(self, stage: str) -> Dict:
        """CPU模式配置"""
        cpu_count = mp.cpu_count()
        
        return {
            "gpu_count": 0,
            "per_device_batch_size": 8 if stage != "LLM" else 4,
            "gradient_accumulation_steps": 16 if stage == "LLM" else 8,
            "effective_batch_size": 128 if stage != "LLM" else 64,
            "dataloader_num_workers": min(4, cpu_count // 2),
            "enable_mixed_precision": False,
            "enable_gradient_checkpointing": True,
            "dataloader_pin_memory": False,
            "ddp_find_unused_parameters": False,
            "remove_unused_columns": False,
            "stage": stage
        }
    
    def get_training_arguments_kwargs(self, stage: str, additional_kwargs: Optional[Dict] = None) -> Dict:
        """获取TrainingArguments的关键字参数"""
        config = self.get_stage_config(stage)
        
        kwargs = {
            "per_device_train_batch_size": config["per_device_batch_size"],
            "gradient_accumulation_steps": config["gradient_accumulation_steps"],
            "dataloader_num_workers": config["dataloader_num_workers"],
            "dataloader_pin_memory": config["dataloader_pin_memory"],
            "ddp_find_unused_parameters": config["ddp_find_unused_parameters"],
            "remove_unused_columns": config["remove_unused_columns"],
        }
        
        # 混合精度配置
        if config["enable_mixed_precision"] and torch.cuda.is_available():
            kwargs["fp16"] = True
            
        # 梯度检查点
        if config["enable_gradient_checkpointing"]:
            kwargs["gradient_checkpointing"] = True
            
        # 多GPU特定配置
        if config["gpu_count"] > 1:
            kwargs["ddp_backend"] = "nccl"
            
        # 合并额外参数
        if additional_kwargs:
            kwargs.update(additional_kwargs)
            
        return kwargs
    
    def print_config_summary(self):
        """打印配置摘要"""
        print("\n" + "="*60)
        print("🚀 AdvTG多GPU配置摘要")
        print("="*60)
        print(f"总GPU数量: {self.gpu_count}")
        
        if self.stage_configs:
            for stage, config in self.stage_configs.items():
                print(f"\n{stage}阶段配置:")
                print(f"  - 每设备batch size: {config['per_device_batch_size']}")
                print(f"  - 总有效batch size: {config['effective_batch_size']}")
                print(f"  - 梯度累积: {config['gradient_accumulation_steps']}")
                print(f"  - 数据加载workers: {config['dataloader_num_workers']}")
        print("="*60)

# 全局多GPU配置实例
global_gpu_config = AdvTGMultiGPUConfig()

def get_multi_gpu_config(stage: str) -> Dict:
    """获取指定阶段的多GPU配置
    
    Args:
        stage: 'DL', 'LLM', 'RL'
        
    Returns:
        Dict: 包含GPU配置的字典
    """
    return global_gpu_config.get_stage_config(stage)

def initialize_multi_gpu_for_stage(stage: str) -> Dict:
    """为特定阶段初始化多GPU配置
    
    Args:
        stage: 'DL', 'LLM', 'RL'
        
    Returns:
        Dict: GPU配置字典
    """
    print(f"\n🎯 初始化 {stage} 阶段多GPU配置...")
    config = global_gpu_config.get_stage_config(stage)
    
    # 设置特定于阶段的环境变量
    if stage == "LLM":
        # LLM阶段特殊配置
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_SILENT"] = "true"
    elif stage == "RL":
        # RL阶段特殊配置
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
    return config

def get_training_arguments_for_stage(stage: str, base_args: Optional[Dict] = None) -> Dict:
    """获取特定阶段的TrainingArguments参数"""
    return global_gpu_config.get_training_arguments_kwargs(stage, base_args)

def print_multi_gpu_summary():
    """打印多GPU配置摘要"""
    global_gpu_config.print_config_summary()

if __name__ == "__main__":
    # 测试多GPU配置
    print("🧪 测试AdvTG多GPU配置...")
    
    # 初始化全局环境
    global_gpu_config.initialize_global_gpu_environment()
    
    # 测试三个阶段的配置
    for stage in ["DL", "LLM", "RL"]:
        config = get_multi_gpu_config(stage)
        print(f"\n{stage}阶段测试完成")
    
    # 打印配置摘要
    print_multi_gpu_summary()
