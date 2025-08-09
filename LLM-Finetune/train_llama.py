import os
import torch
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 导入全局多GPU配置
import sys
sys.path.append('..')
from multi_gpu_config import AdvTGMultiGPUConfig

# 设置Hugging Face镜像 - 这行需要在导入unsloth之前或者导入之后
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_URL"] = "https://hf-mirror.com"

# Disable tokenizers parallelism to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Disable wandb and enable swanlab
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"

# 主程序入口
if __name__ == "__main__":
    # 开始训练流程
    print("🚀 Starting LLM fine-tuning with multi-GPU support...")

    # 先初始化全局多GPU配置，用于SwanLab配置
    global_gpu_config = AdvTGMultiGPUConfig()
    llm_gpu_config = global_gpu_config.get_stage_config("LLM")

    # Initialize SwanLab for LLM fine-tuning tracking
    try:
        import swanlab
        import time
        # 创建包含时间戳的自定义实验名称
        experiment_name = f"AdvTG-LLM-Llama3-{time.strftime('%Y%m%d-%H%M%S')}"
        
        swanlab.init(
            project="AdvTG-LLM-Finetune",
            name=experiment_name,  # 自定义实验名称
            description="LLM Fine-tuning stage - Llama-3-8B with LoRA",
            config={
                # 移除字符串类型字段，SwanLab config中只保留数值类型
                "model_version": 3.8,  # 用数值表示llama-3-8b版本
                "max_seq_length": 2048,
                "learning_rate": 2e-4,
                "lora_r": 16,
                "lora_alpha": 16,
                "target_modules_count": 7,  # 目标模块数量，用数值代替列表
                # 多GPU配置信息
                "gpu_count": llm_gpu_config['gpu_count'],
                "per_device_batch_size": llm_gpu_config['per_device_batch_size'],
                "gradient_accumulation_steps": llm_gpu_config['gradient_accumulation_steps'],
                "total_effective_batch_size": llm_gpu_config['effective_batch_size'],
                "multi_gpu_training": 1 if llm_gpu_config['gpu_count'] > 1 else 0
            }
        )
        print("✅ SwanLab initialized for multi-GPU LLM fine-tuning!")
        print(f"📊 实验名称: {experiment_name}")
        use_swanlab = True
    except ImportError:
        print("⚠️  SwanLab not installed, continuing without experiment tracking")
        use_swanlab = False
    except Exception as e:
        print(f"⚠️  SwanLab initialization failed: {e}")
        use_swanlab = False

    # CUDA环境基础设置
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # 设置代理环境变量 (如果代理不可用则注释掉)
    # os.environ['HTTP_PROXY'] = '127.0.0.1:7890'
    # os.environ['HTTPS_PROXY'] = '127.0.0.1:7890'

    major_version, minor_version = torch.cuda.get_device_capability()

    from unsloth import FastLanguageModel
    import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

import os
import requests

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit", # Instruct version of Gemma 7b
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit", # Instruct version of Gemma 2b
    "unsloth/llama-3-8b-bnb-4bit", # [NEW] 15 Trillion token Llama-3
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)



model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)



alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset
# dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
# dataset = dataset.map(formatting_prompts_func, batched = True,)





#加载自己的http数据集
import json

# 加载训练数据和验证数据
with open("../dataset/llm_train.json","r") as f:  # Use large dataset for LLM fine-tuning
    train_data = json.load(f)

with open("../dataset/val.json","r") as f:  # Use dedicated validation set
    val_data = json.load(f)

print(f"📊 Loaded {len(train_data)} training samples and {len(val_data)} validation samples")
from datasets import Dataset

def json_to_string(data, indent=0):
    result = []
    indent_str = ' ' * indent
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                result.append(f'{indent_str}{key}:')
                result.append(json_to_string(value, indent + 1))
            else:
                result.append(f'{indent_str}{key}: {value}')
    elif isinstance(data, list):
        for item in data:
            result.append(json_to_string(item, indent))
    else:
        result.append(f'{indent_str}{data}')
    return '\n'.join(result)

# 格式化训练数据
train_formatted_data = {
    "text": [item["Request Line"]+"\n"+json_to_string(item["Request Headers"])+"\n\n"+item["Request Body"] for item in train_data],
    "instruction": ["Follow these tips to generate malicious http traffic" if item["Label"]=="Malicious" else "Follow these tips to generate benign http traffic" for item in train_data],
    "input": [item["Request Line"] for item in train_data],
    "output": [item["Request Line"]+"\n"+json_to_string(item["Request Headers"])+"\n\n"+item["Request Body"] for item in train_data]
}
train_dataset = Dataset.from_dict(train_formatted_data)

# 格式化验证数据（相同的Alpaca格式）
val_formatted_data = {
    "text": [item["Request Line"]+"\n"+json_to_string(item["Request Headers"])+"\n\n"+item["Request Body"] for item in val_data],
    "instruction": ["Follow these tips to generate malicious http traffic" if item["Label"]=="Malicious" else "Follow these tips to generate benign http traffic" for item in val_data],
    "input": [item["Request Line"] for item in val_data],
    "output": [item["Request Line"]+"\n"+json_to_string(item["Request Headers"])+"\n\n"+item["Request Body"] for item in val_data]
}
val_dataset = Dataset.from_dict(val_formatted_data)

# 对训练和验证数据集进行洗牌
train_dataset = train_dataset.shuffle(seed=42)
val_dataset = val_dataset.shuffle(seed=42)

# 将数据集格式化为Alpaca提示词格式
train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
val_dataset = val_dataset.map(formatting_prompts_func, batched=True)

from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from unsloth import is_bfloat16_supported

# 创建SwanLab回调函数用于实时记录训练过程
class SwanLabCallback(TrainerCallback):
    def __init__(self, use_swanlab=False):
        self.use_swanlab = use_swanlab
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """在每次日志记录时调用"""
        if self.use_swanlab and logs and 'swanlab' in globals():
            try:
                # 记录损失和学习率
                log_dict = {}
                if 'loss' in logs:
                    log_dict['train_loss'] = logs['loss']
                if 'learning_rate' in logs:
                    log_dict['learning_rate'] = logs['learning_rate']
                if 'epoch' in logs:
                    log_dict['epoch'] = logs['epoch']
                if 'eval_loss' in logs:
                    log_dict['eval_loss'] = logs['eval_loss']
                if 'grad_norm' in logs:
                    log_dict['grad_norm'] = logs['grad_norm']
                
                # 添加step信息
                log_dict['step'] = state.global_step
                
                if log_dict:
                    swanlab.log(log_dict)
                    print(f"📊 Step {state.global_step}: Logged to SwanLab - Loss: {logs.get('loss', 'N/A')}")
                    
            except Exception as e:
                print(f"⚠️  SwanLab logging error: {e}")
    
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """在评估时调用"""
        if self.use_swanlab and logs and 'swanlab' in globals():
            try:
                eval_dict = {f"eval_{k}": v for k, v in logs.items() if k.startswith('eval_')}
                if eval_dict:
                    swanlab.log(eval_dict)
                    print(f"📊 Evaluation logged to SwanLab: {eval_dict}")
            except Exception as e:
                print(f"⚠️  SwanLab eval logging error: {e}")

# 初始化回调函数
swanlab_callback = SwanLabCallback(use_swanlab=use_swanlab)

# 设置环境变量以禁用 NCCL 中的 P2P 和 IB，适配多GPU训练
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
# 多GPU训练的NCCL设置
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_SOCKET_IFNAME'] = 'lo'  # 本地回环接口

# 检查GPU设备并设置主设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🎮 主设备设置为: {device}")
print(f"🚀 总GPU数量: {torch.cuda.device_count()}")

# 不要手动移动模型到设备，让DataParallel/DistributedDataParallel处理
# model.to(device)  # 注释掉，让训练器自动处理

# 动态计算最优batch size和gradient accumulation - 8GPU优化
gpu_count = torch.cuda.device_count()
optimal_per_device_batch_size = max(4, 8 // max(1, gpu_count // 4))  # 根据GPU数量调整
optimal_gradient_accumulation = max(8, 64 // gpu_count)  # 保持总batch size稳定

print(f"📊 多GPU训练优化配置:")
print(f"   - GPU数量: {gpu_count}")
print(f"   - 每设备batch size: {optimal_per_device_batch_size}")
print(f"   - 梯度累积步数: {optimal_gradient_accumulation}")
print(f"   - 总有效batch size: {optimal_per_device_batch_size * optimal_gradient_accumulation * gpu_count}")

# 构建LLM训练参数
training_args_base = {
    "per_device_train_batch_size": optimal_per_device_batch_size,
    "gradient_accumulation_steps": optimal_gradient_accumulation,
    "warmup_steps": 5,
    "max_steps": 500,
    "learning_rate": 2e-4,
    "fp16": False,  # 禁用 fp16，因为模型使用 bfloat16
    "bf16": True,   # 使用 bf16，与模型精度匹配
    "logging_steps": 1,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 3407,
    "output_dir": "outputs",
    # 禁用分布式训练相关参数，让Unsloth处理多GPU
    "local_rank": -1,
    "ddp_backend": None,
    "dataloader_num_workers": 0,  # 避免多进程冲突
}

# 创建TrainingArguments，不使用全局多GPU配置的分布式参数
training_args = TrainingArguments(**training_args_base)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,  # Use proper validation dataset
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = min(8, gpu_count * 2),  # 增加数据处理进程数
    packing = False, # Can make training 5x faster for short sequences.
    callbacks=[swanlab_callback] if use_swanlab else [],  # 添加SwanLab回调
    args = training_args,
)

print("Starting training...")

# 开始训练
trainer_stats = trainer.train()

# Log training results to SwanLab
if use_swanlab and 'swanlab' in locals():
    try:
        # Log training statistics
        if hasattr(trainer_stats, 'training_loss'):
            swanlab.log({"training_loss": trainer_stats.training_loss})
        if hasattr(trainer_stats, 'train_runtime'):
            swanlab.log({"train_runtime": trainer_stats.train_runtime})
        if hasattr(trainer_stats, 'train_samples_per_second'):
            swanlab.log({"train_samples_per_second": trainer_stats.train_samples_per_second})
        
        # Log model info (移除字符串类型字段，SwanLab期望数值类型)
        swanlab.log({
            "llama_model_version": 3.8,  # 用数值表示模型版本
            "lora_method_used": 1,  # 用数值表示LoRA方法
            "training_completed": 1
        })
        
        swanlab.finish()
        print("📊 LLM training results logged to SwanLab successfully!")
    except Exception as e:
        print(f"⚠️  SwanLab logging failed: {e}")

print("✅ LLM fine-tuning completed!")