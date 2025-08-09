import os

# Set environment variables early to avoid conflicts
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizers parallelism to avoid fork warnings

# 导入全局多GPU配置
import sys
sys.path.append('..')
from multi_gpu_config import AdvTGMultiGPUConfig

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

# 全局多GPU训练配置
ENABLE_MULTI_GPU_RL_PARALLEL = True   # 启用多GPU并行优化
MAX_RL_PARALLEL_WORKERS = min(4, mp.cpu_count())  # RL并行工作进程

def main():
    """
    Main function to run PPO training.
    """
    print("🚀 Starting RL-Adv PPO Training...")
    print("=" * 60)
    
    # 初始化全局多GPU配置
    global_gpu_config = AdvTGMultiGPUConfig()
    rl_gpu_config = global_gpu_config.get_stage_config("RL")
    
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
                "batch_size": rl_gpu_config['per_device_batch_size'],
                "mini_batch_size": 1,
                "gradient_accumulation_steps": rl_gpu_config['gradient_accumulation_steps'],
                "output_min_length": 128,
                "output_max_length": 256,
                # RL多GPU配置信息
                "gpu_count": rl_gpu_config['gpu_count'],
                "effective_batch_size": rl_gpu_config['effective_batch_size'],
                "multi_gpu_training": 1 if rl_gpu_config['gpu_count'] > 1 else 0
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
    
    print(f"🔧 RL阶段GPU配置:")
    print(f"   - GPU数量: {rl_gpu_config['gpu_count']}")
    print(f"   - 每设备batch size: {rl_gpu_config['per_device_batch_size']}")
    print(f"   - 梯度累积步数: {rl_gpu_config['gradient_accumulation_steps']}")
    print(f"   - 总有效batch size: {rl_gpu_config['effective_batch_size']}")
    print(f"   - 数据加载workers: {rl_gpu_config['dataloader_num_workers']}")
    print(f"   - 混合精度: {rl_gpu_config['enable_mixed_precision']}")
    
    # GPU优化设置
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        if rl_gpu_config['gpu_count'] > 1:
            torch.cuda.set_device(0)  # 设置主GPU
            print(f"🚀 启用RL多GPU优化")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    print("\n📊 Loading configuration and data...")
    
    # Create PPO configuration with 多GPU优化
    config = create_ppo_config(rl_gpu_config)
    
    # Load HTTP dataset
    dataset = load_http_dataset()
    
    # Create dataloader
    dataloader = create_dataloader(dataset)
    
    print("\n🔧 Setting up models...")
    
    # Setup models (using model_name_or_path directly since config.model_name is not available)
    ppo_model, ref_model, tokenizer = setup_models(model_name_or_path, device)
    
    # Update generation_kwargs with tokenizer pad token ID
    generation_kwargs["pad_token_id"] = tokenizer.eos_token_id
    
    # Configuration parameters (load detection models first)
    feature_type, model_configs = select_feature_model_type(features_dict)
    
    # Load detection model as reward model
    print(f"🔧 Loading detection model as reward model for feature type: {feature_type}")
    
    # Create a simple reward model wrapper using the detection models
    class DetectionRewardModel(torch.nn.Module):
        def __init__(self, model_configs, device):
            super().__init__()
            self.model_configs = model_configs
            self.device = device
            
        def forward(self, input_ids):
            # This is just a placeholder - the actual reward computation 
            # happens in evaluate_responses function
            return torch.zeros(input_ids.shape[0], 1, device=self.device)
    
    reward_model = DetectionRewardModel(model_configs, device)
    
    # Initialize PPO trainer (TRL 0.15.2 version - try without value_model first)
    # Some versions of TRL may have issues with AutoModelForCausalLMWithValueHead as value_model
    try:
        ppo_trainer = PPOTrainer(
            args=config,                    # PPOConfig
            processing_class=tokenizer,     # tokenizer  
            model=ppo_model,               # policy model
            ref_model=ref_model,           # reference model
            reward_model=reward_model,     # Detection model as reward model
            train_dataset=dataset          # dataset
            # value_model is optional in some cases
        )
    except Exception as e:
        print(f"PPOTrainer initialization failed: {e}")
        print("Trying alternative initialization...")
        # Try with a simple wrapper for value model
        class ValueModelWrapper:
            def __init__(self, model):
                self.base_model_prefix = getattr(model.pretrained_model, 'base_model_prefix', 'model')
                self.pretrained_model = model.pretrained_model
                
            def __getattr__(self, name):
                return getattr(self.pretrained_model, name)
        
        wrapped_value_model = ValueModelWrapper(ppo_model)
        ppo_trainer = PPOTrainer(
            args=config,
            processing_class=tokenizer,
            model=ppo_model,
            ref_model=ref_model,
            reward_model=reward_model,
            train_dataset=dataset,
            value_model=wrapped_value_model
        )
    
    output_length_sampler = LengthSampler(output_min_length, output_max_length)
    
    # Set save path
    save_path = os.path.join("../model/ppo_model/", feature_type)
    mkdir(save_path)
    
    # Load test tokenizer for text feature evaluation
    test_tokenizer = None
    if feature_type == "Text":
        from transformers import AutoTokenizer
        try:
            # Try to load from local trained model path first (bert_model)
            test_tokenizer = AutoTokenizer.from_pretrained("../models/bert_model/")
        except:
            try:
                # Try alternative path (bert)
                test_tokenizer = AutoTokenizer.from_pretrained("../models/bert/")
            except:
                # If both local paths fail, use standard BERT model
                print("⚠️  Local BERT model not found, using bert-base-uncased from HuggingFace")
                test_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
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