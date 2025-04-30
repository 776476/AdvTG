import os
import torch
import json
from tqdm import tqdm
from trl import PPOTrainer
from trl.core import LengthSampler
import torch.nn.functional as F

# Import from local modules
from config import *
from data_utils import load_http_dataset, create_dataloader, text2image
from model_utils import select_feature_model_type, setup_models, prepare_query_tensors, evaluate_responses
from utils import save_results, set_seed, mkdir

def main():
    """
    Main function to run PPO training.
    """
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create PPO configuration
    config = create_ppo_config()
    
    # Load HTTP dataset
    dataset = load_http_dataset()
    
    # Create dataloader
    dataloader = create_dataloader(dataset)
    
    # Set up models
    ppo_model, ref_model, tokenizer = setup_models(config.model_name, device)
    
    # Update generation_kwargs with tokenizer pad token ID
    generation_kwargs["pad_token_id"] = tokenizer.eos_token_id
    
    # Initialize PPO trainer
    ppo_trainer = PPOTrainer(config, ppo_model, ref_model, tokenizer, dataset=dataset)
    
    # Select feature model type
    feature_type, model_configs = select_feature_model_type(features_dict)
    
    # Create output length sampler
    output_length_sampler = LengthSampler(output_min_length, output_max_length)
    
    # Set save path
    save_path = os.path.join("../model/ppo_model/", feature_type)
    mkdir(save_path)
    
    # Load test tokenizer for text feature evaluation
    test_tokenizer = None
    if feature_type == "Text":
        from transformers import AutoTokenizer
        test_tokenizer = AutoTokenizer.from_pretrained("../model/bert/")
    
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

if __name__ == "__main__":
    main() 