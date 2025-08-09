import os

# Set environment variables early to avoid conflicts
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizers parallelism to avoid fork warnings

import torch
import json
import multiprocessing as mp
from tqdm import tqdm
from trl import PPOTrainer
from trl.core import LengthSampler
import torch.nn.functional as F

# Import from local modules
from environment import set_environment, check_dependencies
from config import *
from data_utils import load_http_dataset, create_dataloader, text2image
from model_utils import select_feature_model_type, setup_models, prepare_query_tensors, evaluate_responses
from utils import save_results, set_seed, mkdir

# vLLM风格的RL并行训练配置
ENABLE_VLLM_STYLE_RL_PARALLEL = True   # 启用vLLM风格RL并行优化
ENABLE_RL_TENSOR_PARALLEL = True       # 启用RL张量并行（多GPU）
ENABLE_RL_DATA_PARALLEL = True         # 启用RL数据并行处理
MAX_RL_PARALLEL_WORKERS = min(4, mp.cpu_count())  # RL并行工作进程

def get_optimal_rl_config():
    """获取RL阶段最优并行配置"""
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    cpu_count = mp.cpu_count()
    
    # vLLM风格的RL动态配置
    if gpu_count >= 2:
        tensor_parallel_size = min(2, gpu_count)  # RL训练通常不需要很大的tensor parallel
        optimal_batch_size = 8 * gpu_count  # RL batch size较小
        worker_multiplier = 2
    else:
        tensor_parallel_size = 1
        optimal_batch_size = 4
        worker_multiplier = 1
    
    optimal_workers = min(MAX_RL_PARALLEL_WORKERS, cpu_count // 4) * worker_multiplier
    
    return {
        "gpu_count": gpu_count,
        "tensor_parallel_size": tensor_parallel_size,
        "optimal_batch_size": optimal_batch_size,
        "optimal_workers": optimal_workers,
        "enable_mixed_precision": gpu_count > 0,  # GPU可用时启用混合精度
        "enable_gradient_checkpointing": True,    # RL训练内存优化
        "optimal_gradient_accumulation": 4 if gpu_count > 1 else 2,
    }

def main():
    """
    Main function to run PPO training.
    """
    print("🚀 Starting RL-Adv PPO Training...")
    print("=" * 60)
    
    # Initialize SwanLab for RL training tracking
    try:
        import swanlab
        import time
        # 创建包含时间戳的自定义实验名称
        experiment_name = f"AdvTG-RL-PPO-{time.strftime('%Y%m%d-%H%M%S')}"
        swanlab.init(
            project="AdvTG-RL-Training",
            name=experiment_name,  # 自定义实验名称
            description="RL Adversarial Training stage - PPO with reward feedback",
            config={
                # 移除字符串类型字段，SwanLab config中只保留数值类型
                "algorithm_ppo": 1,  # 用数值表示PPO算法
                "learning_rate": 1.41e-5,
                "batch_size": 4,
                "mini_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "output_min_length": 128,
                "output_max_length": 256
            }
        )
        print("✅ SwanLab initialized for RL training!")
        use_swanlab = True
    except ImportError:
        print("⚠️  SwanLab not installed, continuing without experiment tracking")
        use_swanlab = False
    except Exception as e:
        print(f"⚠️  SwanLab initialization failed: {e}")
        use_swanlab = False
    
    # Check dependencies first
    if not check_dependencies():
        print("❌ Please install missing dependencies and try again.")
        return False
    
    # Set environment variables and get device
    device = set_environment()
    
    # 获取vLLM风格的RL优化配置
    vllm_rl_config = get_optimal_rl_config()
    
    print(f"🔧 vLLM风格RL训练配置:")
    print(f"   - vLLM风格优化: {ENABLE_VLLM_STYLE_RL_PARALLEL}")
    print(f"   - GPU数量: {vllm_rl_config['gpu_count']}")
    print(f"   - 张量并行大小: {vllm_rl_config['tensor_parallel_size']}")
    print(f"   - 优化batch size: {vllm_rl_config['optimal_batch_size']}")
    print(f"   - 优化workers: {vllm_rl_config['optimal_workers']}")
    print(f"   - 混合精度: {vllm_rl_config['enable_mixed_precision']}")
    print(f"   - 梯度累积步数: {vllm_rl_config['optimal_gradient_accumulation']}")
    
    # vLLM风格的CUDA优化
    if torch.cuda.is_available() and ENABLE_VLLM_STYLE_RL_PARALLEL:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        if vllm_rl_config['gpu_count'] > 1:
            torch.cuda.set_device(0)  # 设置主GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(vllm_rl_config['gpu_count'])))
            print(f"🚀 启用vLLM风格RL多GPU优化")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    print("\n📊 Loading configuration and data...")
    
    # Create PPO configuration with vLLM optimization
    config = create_ppo_config(vllm_rl_config if ENABLE_VLLM_STYLE_RL_PARALLEL else None)
    
    # Load HTTP dataset
    dataset = load_http_dataset()
    
    # Create dataloader
    dataloader = create_dataloader(dataset)
    
    print("\n🔧 Setting up models...")
    
    # Setup models
    ppo_model, ref_model, tokenizer = setup_models(config.model_name, device)
    
    # Import configuration from config.py
    from config import (
        generation_kwargs, output_min_length, output_max_length, 
        max_length, query_max_length, columns_to_log, features_dict
    )
    
    # Update generation_kwargs with tokenizer pad token ID
    generation_kwargs["pad_token_id"] = tokenizer.eos_token_id
    
    # Configuration parameters (using config from config.py)
    feature_type, model_configs = select_feature_model_type(features_dict)
    
    output_length_sampler = LengthSampler(output_min_length, output_max_length)
    
    # Initialize PPO trainer
    ppo_trainer = PPOTrainer(config, ppo_model, ref_model, tokenizer, dataset=dataset)
    
    # Set save path
    save_path = os.path.join("../model/ppo_model/", feature_type)
    mkdir(save_path)
    
    # Load test tokenizer for text feature evaluation
    test_tokenizer = None
    if feature_type == "Text":
        from transformers import AutoTokenizer
        test_tokenizer = AutoTokenizer.from_pretrained("../models/bert/")
    
    # Training loop
    all_data = []
    reward_history = []  # 记录奖励历史
    loss_history = []    # 记录损失历史
    
    for epoch, batch in tqdm(enumerate(dataloader)):
        # Prepare query tensors
        query_tensors, origin_label, requirement_label, new_query_str = prepare_query_tensors(
            batch, tokenizer, device, query_max_length
        )
        
        # Set generation length
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        
        # Generate responses
        response_tensors_batch = ppo_trainer.generate(query_tensors, **generation_kwargs)
        response_tensors = [r.squeeze()[:max_length] for r in response_tensors_batch]
        
        # Decode responses
        batch['response'] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        
        # Prepare tensors for evaluation based on feature type
        if feature_type == "Text":
            texts = [r.split("\n", 1)[1][:max_length] for r in batch['response']]
            test_tensor = [torch.tensor(test_tokenizer(query)["input_ids"]).to(device) for query in texts]
            padded_tensors = [F.pad(t, (0, max_length - t.size(0)), 'constant', 0) 
                              if t.size(0) < max_length else t[:max_length] for t in test_tensor]
            padded_tensor_batch = torch.stack(padded_tensors)
        else:
            images = text2image(batch['response'])
            padded_tensor_batch = torch.stack(images).to(device)
        
        # Evaluate responses
        rewards, prediction = evaluate_responses(
            batch, feature_type, model_configs, padded_tensor_batch, device, requirement_label
        )
        
        # Convert rewards to list
        rewards_list = list(rewards.cpu().detach())
        
        # Update model with PPO
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards_list)
        ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=columns_to_log)
        
        # 记录奖励和损失历史
        avg_reward = torch.mean(rewards).item()
        reward_history.append(avg_reward)
        
        # Log to SwanLab every few steps with enhanced metrics
        if use_swanlab and 'swanlab' in locals():
            try:
                max_reward = torch.max(rewards).item()
                min_reward = torch.min(rewards).item()
                std_reward = torch.std(rewards).item()
                
                # 计算移动平均奖励
                window_size = min(10, len(reward_history))
                moving_avg_reward = sum(reward_history[-window_size:]) / window_size
                
                log_dict = {
                    "epoch": epoch,
                    "average_reward": avg_reward,
                    "max_reward": max_reward,
                    "min_reward": min_reward,
                    "std_reward": std_reward,
                    "moving_avg_reward": moving_avg_reward,
                    "feature_type_code": 1 if feature_type == "request" else 2,  # 用数值代替字符串
                    "batch_size": len(batch['instruction'])
                }
                
                # 记录PPO统计信息
                if 'ppo/policy/loss' in stats:
                    log_dict['policy_loss'] = stats['ppo/policy/loss']
                if 'ppo/val/loss' in stats:
                    log_dict['value_loss'] = stats['ppo/val/loss']
                if 'ppo/policy/entropy' in stats:
                    log_dict['policy_entropy'] = stats['ppo/policy/entropy']
                if 'ppo/policy/approx_kl' in stats:
                    log_dict['approx_kl'] = stats['ppo/policy/approx_kl']
                
                swanlab.log(log_dict)
                
                # 每10个epoch打印详细信息
                if epoch % 10 == 0:
                    print(f"📊 Epoch {epoch}: Avg Reward: {avg_reward:.4f}, "
                          f"Moving Avg: {moving_avg_reward:.4f}, "
                          f"Std: {std_reward:.4f}")
                    
            except Exception as e:
                print(f"⚠️  SwanLab logging failed at epoch {epoch}: {e}")
        
        # Clean up memory
        torch.cuda.empty_cache()
        
        # Collect data for later saving
        for i in range(len(batch['instruction'])):
            json_obj = {
                "Request Line": batch["input"][i],
                "Label": origin_label[i],
                "Origin Output": batch["output"][i],
                "Request Body": "",
                "Request Headers": batch["response"][i],
            }
            all_data.append(json_obj)
        
        # Save model and data periodically
        if epoch % 50 == 0:
            epoch_save_path = os.path.join(save_path, str(epoch))
            mkdir(epoch_save_path)
            
            # Save model and tokenizer
            ppo_model.save_pretrained(epoch_save_path, push_to_hub=False)
            tokenizer.save_pretrained(epoch_save_path, push_to_hub=False)
            
            # Save collected data
            save_results(all_data, feature_type, epoch)
            
            # Reset data collection
            all_data = []
    
    print("\n🎉 RL-Adv PPO training completed successfully!")
    print("=" * 60)
    print(f"📁 Models saved to: {save_path}")
    print("✅ All training stages completed.")
    
    # Finalize SwanLab logging
    if use_swanlab and 'swanlab' in locals():
        try:
            swanlab.log({
                "training_completed": 1,
                "total_epochs": epoch + 1,
                "feature_type_code": 1 if feature_type == "request" else 2,  # 用数值代替字符串
                "model_saved": 1  # 用数值表示模型保存状态
            })
            swanlab.finish()
            print("📊 RL training results logged to SwanLab successfully!")
        except Exception as e:
            print(f"⚠️  SwanLab finalization failed: {e}")
    
    return True

if __name__ == "__main__":
    import sys
    success = main()
    if success:
        print("\n✅ RL-Adv training finished successfully!")
        sys.exit(0)
    else:
        print("\n❌ RL-Adv training failed. Please check the error messages above.")
        sys.exit(1) 