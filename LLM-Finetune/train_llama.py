"""
LLM-Finetune阶段主训练脚本 - 重构后的简化版本
使用配置文件统一管理所有参数，支持Llama-3-8B + LoRA微调
"""
import os
import sys
import torch

# 禁用wandb - 必须在导入transformers之前设置
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

# 导入全局多GPU配置
sys.path.append('..')
from multi_gpu_config import AdvTGMultiGPUConfig

# 导入本地模块
from config import LLMConfig
from data_utils import create_data_processor
from swanlab_utils import create_swanlab_manager, create_swanlab_callback


class LLMTrainer:
    """LLM微调训练器类"""
    
    def __init__(self):
        self.config = None
        self.global_gpu_config = None
        self.llm_gpu_config = None
        self.swanlab_manager = None
        self.data_processor = None
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
    
    def initialize(self):
        """初始化训练器"""
        print("🚀 Starting LLM fine-tuning with multi-GPU support...")
        
        # 初始化全局多GPU配置
        self.global_gpu_config = AdvTGMultiGPUConfig()
        self.llm_gpu_config = self.global_gpu_config.get_stage_config("LLM")
        
        # 创建LLM配置
        self.config = LLMConfig(self.llm_gpu_config)
        
        # 设置环境和目录
        self.config.setup_environment()
        self.config.create_directories()
        self.config.print_config_summary()
        
        # 设置设备信息
        device = self.config.setup_device_and_gpu_info()
        
        # 初始化SwanLab管理器
        self.swanlab_manager = create_swanlab_manager(self.config)
        
        # 初始化数据处理器
        self.data_processor = create_data_processor(self.config)
        
        return device
    
    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        print("\n🤖 Loading Llama-3-8B model and tokenizer...")
        
        # 导入Unsloth
        from unsloth import FastLanguageModel
        
        # 获取模型配置
        model_config = self.config.get_model_config()
        
        # 加载预训练模型
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_config["model_name"],
            max_seq_length=model_config["max_seq_length"],
            dtype=model_config["dtype"],
            load_in_4bit=model_config["load_in_4bit"]
        )
        
        print(f"✅ Model loaded: {model_config['model_name']}")
        print(f"📏 Max sequence length: {model_config['max_seq_length']}")
        print(f"💾 4-bit quantization: {model_config['load_in_4bit']}")
        
        # 设置EOS token用于数据处理
        self.data_processor.set_eos_token(self.tokenizer.eos_token)
    
    def setup_lora_model(self):
        """设置LoRA模型"""
        print("\n🔧 Setting up LoRA configuration...")
        
        # 获取LoRA配置
        lora_config = self.config.get_lora_config()
        
        # 应用LoRA配置
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            **lora_config
        )
        
        print(f"✅ LoRA applied with rank: {lora_config['r']}")
        print(f"🎯 Target modules: {len(lora_config['target_modules'])} modules")
        print(f"⚡ LoRA alpha: {lora_config['lora_alpha']}")
    
    def prepare_datasets(self):
        """准备训练和验证数据集"""
        print("\n📊 Preparing datasets...")
        
        # 加载和准备数据集
        self.train_dataset, self.val_dataset = self.data_processor.load_and_prepare_datasets()
        
        # 打印样本数据
        self.data_processor.print_sample_data(self.train_dataset, num_samples=2)
        
        return self.train_dataset, self.val_dataset
    
    def create_trainer(self):
        """创建训练器"""
        print("\n🏋️ Creating SFT trainer...")
        
        from trl import SFTTrainer
        from transformers import TrainingArguments
        
        # 获取训练参数
        training_args_config = self.config.get_training_arguments()
        training_args = TrainingArguments(**training_args_config)
        
        # 创建SwanLab回调
        swanlab_callback = create_swanlab_callback(self.swanlab_manager.use_swanlab)
        
        # 创建训练器
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            dataset_text_field="text",
            max_seq_length=self.config.MAX_SEQ_LENGTH,
            dataset_num_proc=self.config.DATASET_NUM_PROC,
            packing=self.config.PACKING,
            callbacks=[swanlab_callback] if self.swanlab_manager.use_swanlab else [],
            args=training_args
        )
        
        print("✅ SFT trainer created successfully!")
        print(f"📦 Dataset processing workers: {self.config.DATASET_NUM_PROC}")
        print(f"📊 Batch size per device: {self.config.PER_DEVICE_BATCH_SIZE}")
        print(f"🔄 Gradient accumulation steps: {self.config.GRADIENT_ACCUMULATION_STEPS}")
        print(f"📈 Total effective batch size: {self.config.EFFECTIVE_BATCH_SIZE}")
        
        return trainer
    
    def train_model(self, trainer):
        """执行模型训练"""
        print("\n🚀 Starting model training...")
        print(f"📈 Max training steps: {self.config.MAX_STEPS}")
        print(f"🔥 Learning rate: {self.config.LEARNING_RATE}")
        print(f"🎯 Target precision: {'BF16' if self.config.BF16 else 'FP32'}")
        
        # 开始训练
        trainer_stats = trainer.train()
        
        print("\n✅ Training completed!")
        
        # 记录训练结果到SwanLab
        self.swanlab_manager.log_training_completion(trainer_stats)
        
        return trainer_stats
    
    def save_model(self, trainer):
        """保存训练后的模型"""
        print("\n💾 Saving trained model...")
        
        try:
            # 保存模型和tokenizer
            trainer.save_model(self.config.OUTPUT_DIR)
            self.tokenizer.save_pretrained(self.config.OUTPUT_DIR)
            
            print(f"✅ Model saved to: {self.config.OUTPUT_DIR}")
            
        except Exception as e:
            print(f"⚠️  Model saving failed: {e}")
    
    def finalize_training(self):
        """完成训练并清理资源"""
        print("\n🏁 Finalizing training...")
        
        # 完成SwanLab实验
        self.swanlab_manager.finish()
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU cache cleared")
        
        print("🎉 LLM fine-tuning pipeline completed successfully!")
    
    def run_training_pipeline(self):
        """运行完整的训练流程"""
        try:
            # 1. 初始化
            device = self.initialize()
            
            # 2. 加载模型和分词器
            self.load_model_and_tokenizer()
            
            # 3. 设置LoRA
            self.setup_lora_model()
            
            # 4. 准备数据集
            self.prepare_datasets()
            
            # 5. 创建训练器
            trainer = self.create_trainer()
            
            # 6. 训练模型
            trainer_stats = self.train_model(trainer)
            
            # 7. 保存模型
            self.save_model(trainer)
            
            # 8. 完成训练
            self.finalize_training()
            
            return trainer_stats
            
        except Exception as e:
            print(f"❌ Training pipeline failed: {e}")
            # 确保SwanLab正确关闭
            if self.swanlab_manager:
                self.swanlab_manager.finish()
            raise


def main():
    """主函数"""
    # 设置多进程启动方法（Windows兼容）
    import multiprocessing as mp
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # 已经设置过了
    
    # 创建并运行训练器
    trainer = LLMTrainer()
    trainer_stats = trainer.run_training_pipeline()
    
    return trainer_stats


if __name__ == "__main__":
    # 导入FastLanguageModel需要在main保护下
    from unsloth import FastLanguageModel
    
    # 运行主程序
    main()
