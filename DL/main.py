"""
DL阶段主训练脚本 - 重构后的简化版本
使用配置文件统一管理所有参数
"""
import os
import sys
import time
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

# 禁用wandb - 必须在导入transformers之前设置
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

# 设置镜像网站 - 优先本地，无则从镜像下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "0"  # 允许在线下载

# 导入全局多GPU配置
sys.path.append('..')
from multi_gpu_config import AdvTGMultiGPUConfig

# 导入DL配置和组件
from config import DLConfig
from models import CNNLSTMClassifier, TextCNNClassifier, DNNClassifier, DeepLog
from data_processing import load_data, prepare_dataset, load_tokenizer, SimpleTokenizer
from training import train_transformer_model, train_custom_model
from utils import DLSwanLabCallback

# 导入必要的库
import torch
from transformers import TrainingArguments, EarlyStoppingCallback


class DLTrainer:
    """DL阶段训练器类"""
    
    def __init__(self):
        self.config = None
        self.dl_gpu_config = None
        self.global_gpu_config = None
        self.use_swanlab = False
        self.swanlab_run = None
        self.all_model_configs = []
    
    def initialize(self):
        """初始化训练器"""
        # 初始化全局多GPU配置
        self.global_gpu_config = AdvTGMultiGPUConfig()
        self.dl_gpu_config = self.global_gpu_config.get_stage_config("DL")
        
        # 创建DL配置
        self.config = DLConfig(self.dl_gpu_config)
        
        # 设置环境
        self.config.setup_environment()
        self.config.create_directories()
        self.config.print_config_summary()
        
        # 设置GPU优化
        device = self.config.setup_gpu_optimization()
        return device
    
    def setup_swanlab(self):
        """设置SwanLab实验跟踪"""
        try:
            import swanlab
            experiment_name = f"AdvTG-DL-vLLM-{time.strftime('%Y%m%d-%H%M%S')}"
            
            self.swanlab_run = swanlab.init(
                project=self.config.SWANLAB_PROJECT,
                name=experiment_name,
                description=self.config.SWANLAB_DESCRIPTION,
                config=self.config.get_swanlab_config()
            )
            
            print("✅ SwanLab initialized successfully!")
            print(f"📊 Project: {self.config.SWANLAB_PROJECT}")
            print(f"📊 Experiment: {experiment_name}")
            self.use_swanlab = True
            
        except ImportError:
            print("⚠️  SwanLab not installed, continuing without experiment tracking")
            self.use_swanlab = False
        except Exception as e:
            print(f"⚠️  SwanLab initialization failed: {e}")
            self.use_swanlab = False
    
    def load_and_prepare_data(self):
        """加载和准备数据"""
        print("📁 Loading data...")
        json_data = load_data(self.config.DATA_PATH)
        
        # 根据配置选择tokenizer
        if self.config.FORCE_SIMPLE_MODE:
            print("Force simple mode enabled - using SimpleTokenizer")
            tokenizer = SimpleTokenizer()
        else:
            print(f"Model endpoint: {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")
            tokenizer = load_tokenizer(self.config.BERT_MODEL_NAME)
        
        print(f"Tokenizer type: {type(tokenizer)}")
        print(f"Is SimpleTokenizer: {isinstance(tokenizer, SimpleTokenizer)}")
        
        # 准备数据集
        train_dataset, val_dataset, test_dataset = prepare_dataset(
            json_data, tokenizer, self.config.MAX_LENGTH
        )
        
        return tokenizer, train_dataset, val_dataset, test_dataset
    
    def train_bert_model(self, tokenizer, train_dataset, val_dataset):
        """训练BERT模型"""
        if isinstance(tokenizer, SimpleTokenizer):
            print("Skipping BERT training - using simple tokenizer mode")
            return
        
        print("\n====== Training BERT Model ======")
        
        try:
            # 创建训练参数
            transformer_training_args = TrainingArguments(**self.config.get_transformer_training_args())
            
            # 创建回调函数
            bert_callback = DLSwanLabCallback(use_swanlab=self.use_swanlab, model_name="BERT")
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=self.config.EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=self.config.EARLY_STOPPING_THRESHOLD
            )
            
            print(f"🛑 BERT Early stopping enabled: patience={self.config.EARLY_STOPPING_PATIENCE}, threshold={self.config.EARLY_STOPPING_THRESHOLD}")
            
            # 训练模型
            bert_save_path, bert_config = train_transformer_model(
                self.config.BERT_MODEL_NAME,
                self.config.BERT_MODEL_NAME,
                train_dataset,
                val_dataset,
                transformer_training_args,
                swanlab_callback=bert_callback,
                early_stopping_callback=early_stopping_callback
            )
            
            self.all_model_configs.append(bert_config)
            print("✅ BERT model training completed!")
            
            # 记录结果到SwanLab
            self._log_bert_results(bert_config)
            
        except Exception as e:
            print(f"❌ BERT training failed: {e}")
            print("Continuing with custom models only...")
    
    def train_custom_models(self, train_dataset, val_dataset):
        """训练自定义模型"""
        print("\n====== Training Custom Models ======")
        
        # 获取模型定义
        models = self.config.get_model_definitions()
        
        # 创建训练参数
        custom_training_args = TrainingArguments(**self.config.get_custom_training_args())
        
        for model_name, model in models.items():
            print(f"\n🔄 Training model: {model_name}")
            try:
                # 训练模型
                save_path, model_config = train_custom_model(
                    model,
                    model_name,
                    train_dataset,
                    val_dataset,
                    custom_training_args,
                    swanlab_run=self.swanlab_run
                )
                
                self.all_model_configs.append(model_config)
                print(f"✅ Completed training model: {model_name}")
                
                # 记录结果到SwanLab
                self._log_custom_model_results(model_name, model_config)
                
            except Exception as e:
                print(f"❌ Failed to train {model_name}: {e}")
    
    def _log_bert_results(self, bert_config):
        """记录BERT结果到SwanLab"""
        if not self.use_swanlab:
            return
        
        try:
            import swanlab
            swanlab.log({
                'BERT_final_accuracy': bert_config.get('eval_accuracy', 0),
                'BERT_final_precision': bert_config.get('eval_precision', 0),
                'BERT_final_recall': bert_config.get('eval_recall', 0),
                'BERT_final_f1': bert_config.get('eval_f1', 0),
                'BERT_final_loss': bert_config.get('eval_loss', 0),
                'transformer_model_completed': 1
            })
            print(f"📊 BERT final results logged to SwanLab")
        except Exception as e:
            print(f"⚠️  SwanLab logging failed for BERT: {e}")
    
    def _log_custom_model_results(self, model_name, model_config):
        """记录自定义模型结果到SwanLab"""
        if not self.use_swanlab:
            return
        
        try:
            import swanlab
            swanlab.log({
                f'{model_name}_final_accuracy': model_config.get('accuracy', 0),
                f'{model_name}_final_precision': model_config.get('precision', 0),
                f'{model_name}_final_recall': model_config.get('recall', 0),
                f'{model_name}_final_f1': model_config.get('f1', 0),
                'custom_model_completed': 1
            })
            print(f"📊 {model_name} final results logged to SwanLab")
        except Exception as e:
            print(f"⚠️  SwanLab logging failed for {model_name}: {e}")
    
    def save_configurations(self):
        """保存模型配置"""
        with open(self.config.CONFIG_SAVE_PATH, 'wb') as f:
            pickle.dump(self.all_model_configs, f)
        
        print(f"\n✅ Saved text model configurations to: {self.config.CONFIG_SAVE_PATH}")
        print(f"📊 Total text models configured: {len(self.all_model_configs)}")
    
    def create_image_configs(self):
        """创建图像模型配置"""
        try:
            from create_image_configs import create_image_model_configs
            create_image_model_configs()
            print("✅ Created image model configurations")
        except Exception as e:
            print(f"⚠️  Failed to create image model configs: {e}")
    
    def finalize_training(self):
        """完成训练并记录最终结果"""
        if not self.use_swanlab:
            return
        
        try:
            import swanlab
            
            # 记录所有模型性能指标
            for i, config in enumerate(self.all_model_configs):
                model_name = config.get('model_name', f'model_{i}')
                for metric in ['accuracy', 'f1', 'precision', 'recall']:
                    if metric in config:
                        swanlab.log({f"{model_name}/{metric}": config[metric]})
            
            # 记录训练摘要
            swanlab.log({
                "total_models_trained": len(self.all_model_configs),
                "training_completed": 1,
                "multi_gpu_optimization_enabled": 1 if self.config.GPU_COUNT > 1 else 0,
                "final_gpu_count": self.config.GPU_COUNT,
                "final_batch_size": self.config.BATCH_SIZE,
                "final_workers": self.dl_gpu_config.get('dataloader_num_workers', 4),
                "final_effective_batch_size": self.dl_gpu_config.get('effective_batch_size', self.config.BATCH_SIZE)
            })
            
            swanlab.finish()
            print("📊 All results logged to SwanLab successfully!")
            
        except Exception as e:
            print(f"⚠️  SwanLab final logging failed: {e}")
    
    def run_training(self):
        """运行完整的训练流程"""
        print("🚀 Starting AdvTG-DL training pipeline...")
        
        # 1. 初始化
        device = self.initialize()
        
        # 2. 设置实验跟踪
        self.setup_swanlab()
        
        # 3. 加载和准备数据
        tokenizer, train_dataset, val_dataset, test_dataset = self.load_and_prepare_data()
        
        # 4. 训练BERT模型
        self.train_bert_model(tokenizer, train_dataset, val_dataset)
        
        # 5. 训练自定义模型
        self.train_custom_models(train_dataset, val_dataset)
        
        # 6. 保存配置
        self.save_configurations()
        
        # 7. 创建图像模型配置
        self.create_image_configs()
        
        # 8. 完成训练
        self.finalize_training()
        
        print("\n🎉 All models training completed with multi-GPU optimization!")


def main():
    """主函数"""
    trainer = DLTrainer()
    trainer.run_training()


if __name__ == "__main__":
    main()
