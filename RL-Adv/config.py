import os
import torch
from trl import PPOConfig

# Environment variables
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 注释掉此行以使用所有GPU
# os.environ['HTTP_PROXY'] = '127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = '127.0.0.1:7890'
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model paths
# Model paths
model_name_or_path = "../models/lamma_outputs/checkpoint-300"  # 第二阶段微调后的checkpoint

# Features dictionary
features_dict = {"Image": "../models/image_model_configs.pkl", "Text": "../models/model_configs.pkl"}

# 检查模型路径是否存在
if not os.path.exists(model_name_or_path):
    raise FileNotFoundError(f"Fine-tuned LLM not found at {model_name_or_path}. Please run stage 2 first.")

# 检查检测模型配置文件是否存在
for feature_type, config_path in features_dict.items():
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Detection model configs not found at {config_path}. Please run stage 1 first.")

# PPO Configuration
def create_ppo_config(rl_gpu_config=None):
    """
    创建PPO配置，支持多GPU优化
    
    Args:
        rl_gpu_config: RL阶段的GPU配置字典，如果为None则使用默认配置
    
    Returns:
        PPOConfig对象
    """
    # 默认配置
    default_batch_size = 4
    default_gradient_accumulation = 4
    
    # 如果提供了GPU配置，使用其中的参数
    if rl_gpu_config:
        # 尝试从多种可能的键名中获取batch size
        batch_size = rl_gpu_config.get('per_device_batch_size', 
                     rl_gpu_config.get('optimal_batch_size', 
                     rl_gpu_config.get('batch_size', default_batch_size)))
        
        # 尝试从多种可能的键名中获取梯度累积步数
        gradient_accumulation = rl_gpu_config.get('gradient_accumulation_steps',
                               rl_gpu_config.get('optimal_gradient_accumulation',
                               default_gradient_accumulation))
    else:
        batch_size = default_batch_size
        gradient_accumulation = default_gradient_accumulation
    
    return PPOConfig(
        is_peft_model=True,
        model_name=model_name_or_path,
        learning_rate=1.41e-5,
        batch_size=batch_size,
        mini_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation,
        use_score_scaling=True,  # scaling
        use_score_norm=True,  # normalization
        score_clip=1.0,
        # log_with="wandb"  # 先注释掉，避免依赖问题
    )

# Generation configurations
sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 4  # This should match config.forward_batch_size
}

# Generation parameters
output_min_length = 128
output_max_length = 256
max_length = 512
query_max_length = 128

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": None  # Will be set after tokenizer is loaded
}

# Logging configuration
columns_to_log = ['text', 'instruction', 'input', 'output', 'response'] 