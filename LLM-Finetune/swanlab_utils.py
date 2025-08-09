"""
SwanLab实验跟踪工具
用于LLM训练过程的实验记录和监控
"""
from typing import Optional, Dict, Any
from transformers import TrainerCallback


class SwanLabManager:
    """SwanLab实验管理器"""
    
    def __init__(self, config):
        """
        初始化SwanLab管理器
        
        Args:
            config: LLMConfig配置对象
        """
        self.config = config
        self.use_swanlab = False
        self.swanlab_run = None
        self._initialize_swanlab()
    
    def _initialize_swanlab(self):
        """初始化SwanLab"""
        try:
            import swanlab
            
            # 生成实验名称
            experiment_name = self.config.get_experiment_name()
            
            # 初始化SwanLab
            self.swanlab_run = swanlab.init(
                project=self.config.PROJECT_NAME,
                name=experiment_name,
                description=self.config.DESCRIPTION,
                config=self.config.get_swanlab_config()
            )
            
            self.use_swanlab = True
            print("✅ SwanLab initialized for multi-GPU LLM fine-tuning!")
            print(f"📊 Project: {self.config.PROJECT_NAME}")
            print(f"📊 Experiment: {experiment_name}")
            
        except ImportError:
            print("⚠️  SwanLab not installed, continuing without experiment tracking")
            self.use_swanlab = False
        except Exception as e:
            print(f"⚠️  SwanLab initialization failed: {e}")
            self.use_swanlab = False
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """
        记录指标到SwanLab
        
        Args:
            metrics: 要记录的指标字典
        """
        if not self.use_swanlab:
            return
        
        try:
            import swanlab
            swanlab.log(metrics)
        except Exception as e:
            print(f"⚠️  SwanLab logging failed: {e}")
    
    def log_training_completion(self, trainer_stats):
        """
        记录训练完成状态
        
        Args:
            trainer_stats: 训练器统计信息
        """
        if not self.use_swanlab:
            return
        
        try:
            import swanlab
            
            # 记录训练统计信息
            log_dict = {}
            
            if hasattr(trainer_stats, 'training_loss'):
                log_dict['training_loss'] = trainer_stats.training_loss
            if hasattr(trainer_stats, 'train_runtime'):
                log_dict['train_runtime'] = trainer_stats.train_runtime
            if hasattr(trainer_stats, 'train_samples_per_second'):
                log_dict['train_samples_per_second'] = trainer_stats.train_samples_per_second
            
            # 记录模型信息
            log_dict.update({
                "llama_model_version": 3.8,  # 用数值表示模型版本
                "lora_method_used": 1,  # 用数值表示LoRA方法
                "training_completed": 1
            })
            
            if log_dict:
                swanlab.log(log_dict)
                print("📊 LLM training results logged to SwanLab successfully!")
                
        except Exception as e:
            print(f"⚠️  SwanLab training completion logging failed: {e}")
    
    def finish(self):
        """完成SwanLab实验"""
        if self.use_swanlab and self.swanlab_run:
            try:
                import swanlab
                swanlab.finish()
                print("📊 SwanLab experiment completed!")
            except Exception as e:
                print(f"⚠️  SwanLab finish failed: {e}")


class SwanLabCallback(TrainerCallback):
    """SwanLab训练回调函数"""
    
    def __init__(self, use_swanlab: bool = False):
        """
        初始化回调函数
        
        Args:
            use_swanlab: 是否使用SwanLab
        """
        self.use_swanlab = use_swanlab
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        训练过程中的日志回调
        
        Args:
            args: 训练参数
            state: 训练状态
            control: 训练控制
            logs: 日志数据
            **kwargs: 其他参数
        """
        if not self.use_swanlab or not logs:
            return
        
        # 检查SwanLab是否可用
        try:
            import swanlab
        except ImportError:
            return
        
        try:
            # 构建日志字典
            log_dict = {}
            
            # 记录训练指标
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
            
            # 添加步数信息
            log_dict['step'] = state.global_step
            
            # 记录到SwanLab
            if log_dict:
                swanlab.log(log_dict)
                print(f"📊 Step {state.global_step}: Logged to SwanLab - Loss: {logs.get('loss', 'N/A')}")
                
        except Exception as e:
            print(f"⚠️  SwanLab logging error: {e}")
    
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """
        评估时的回调
        
        Args:
            args: 训练参数
            state: 训练状态
            control: 训练控制
            logs: 日志数据
            **kwargs: 其他参数
        """
        if not self.use_swanlab or not logs:
            return
        
        try:
            import swanlab
            
            # 提取评估指标
            eval_dict = {f"eval_{k}": v for k, v in logs.items() if k.startswith('eval_')}
            
            if eval_dict:
                swanlab.log(eval_dict)
                print(f"📊 Evaluation logged to SwanLab: {eval_dict}")
                
        except Exception as e:
            print(f"⚠️  SwanLab eval logging error: {e}")


def create_swanlab_manager(config) -> SwanLabManager:
    """
    创建SwanLab管理器
    
    Args:
        config: LLMConfig配置对象
        
    Returns:
        SwanLabManager实例
    """
    return SwanLabManager(config)


def create_swanlab_callback(use_swanlab: bool) -> SwanLabCallback:
    """
    创建SwanLab回调函数
    
    Args:
        use_swanlab: 是否使用SwanLab
        
    Returns:
        SwanLabCallback实例
    """
    return SwanLabCallback(use_swanlab)
