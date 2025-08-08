import os
import torch
import json
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

def main():
    """
    Main function to run PPO training.
    """
    print("🚀 Starting RL-Adv PPO Training...")
    print("=" * 60)
    
    # Initialize SwanLab for RL training tracking
    try:
        import swanlab
        swanlab.init(
            project="AdvTG-RL-Training",
            description="RL Adversarial Training stage - PPO with reward feedback",
            config={
                "algorithm": "PPO",
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
    
    # Set random seed for reproducibility
    set_seed(42)
    
    print("\n📊 Loading configuration and data...")
    
    # Create PPO configuration
    config = create_ppo_config()
    
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
        
        # Log to SwanLab every few steps
        if use_swanlab and 'swanlab' in locals() and epoch % 10 == 0:
            try:
                avg_reward = torch.mean(rewards).item()
                max_reward = torch.max(rewards).item()
                min_reward = torch.min(rewards).item()
                
                swanlab.log({
                    "epoch": epoch,
                    "average_reward": avg_reward,
                    "max_reward": max_reward,
                    "min_reward": min_reward,
                    "policy_loss": stats.get('ppo/policy/loss', 0),
                    "value_loss": stats.get('ppo/val/loss', 0),
                    "feature_type": feature_type
                })
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
                "feature_type_used": feature_type,
                "final_save_path": save_path
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