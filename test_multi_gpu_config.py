#!/usr/bin/env python3
"""
AdvTG全局多GPU配置测试脚本
测试三个阶段的GPU配置是否正常工作
"""
import sys
sys.path.append('.')

def test_multi_gpu_config():
    """测试全局多GPU配置"""
    print("🧪 测试AdvTG全局多GPU配置...")
    print("=" * 60)
    
    try:
        from multi_gpu_config import (
            initialize_multi_gpu_for_stage, 
            get_multi_gpu_config, 
            get_training_arguments_for_stage,
            print_multi_gpu_summary
        )
        
        print("✅ 成功导入多GPU配置模块")
        
        # 测试三个阶段的配置
        stages = ["DL", "LLM", "RL"]
        
        for stage in stages:
            print(f"\n🔧 测试 {stage} 阶段配置:")
            try:
                # 初始化阶段配置
                config = initialize_multi_gpu_for_stage(stage)
                
                # 获取TrainingArguments配置
                training_args_config = get_training_arguments_for_stage(stage)
                
                print(f"   ✅ {stage}阶段配置成功")
                print(f"   - GPU数量: {config['gpu_count']}")
                print(f"   - 每设备batch size: {config['per_device_batch_size']}")
                print(f"   - 总有效batch size: {config['effective_batch_size']}")
                print(f"   - TrainingArguments参数数量: {len(training_args_config)}")
                
            except Exception as e:
                print(f"   ❌ {stage}阶段配置失败: {e}")
                return False
        
        # 打印配置摘要
        print_multi_gpu_summary()
        
        print("\n" + "=" * 60)
        print("✅ 全局多GPU配置测试通过!")
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_gpu_detection():
    """测试GPU检测功能"""
    print("\n🔍 GPU检测测试:")
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ 检测到 {gpu_count} 张GPU:")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                memory_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"   GPU {i}: {gpu_name} ({memory_gb:.1f} GB)")
            
            return gpu_count >= 1
        else:
            print("⚠️  未检测到CUDA GPU")
            return False
            
    except ImportError:
        print("⚠️  PyTorch未安装，无法检测GPU")
        return False
    except Exception as e:
        print(f"❌ GPU检测失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 AdvTG多GPU配置完整测试")
    print("=" * 80)
    
    # 测试GPU检测
    gpu_available = test_gpu_detection()
    
    # 测试多GPU配置
    config_success = test_multi_gpu_config()
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 测试结果摘要:")
    print(f"   GPU可用性: {'✅ 通过' if gpu_available else '⚠️  未检测到GPU'}")
    print(f"   多GPU配置: {'✅ 通过' if config_success else '❌ 失败'}")
    
    if config_success:
        print("\n🎉 AdvTG多GPU配置已准备就绪!")
        print("   - DL阶段: BERT + 自定义模型训练")
        print("   - LLM阶段: Llama-3-8B + LoRA微调")
        print("   - RL阶段: PPO强化学习训练")
        return True
    else:
        print("\n❌ 多GPU配置测试失败，请检查配置文件")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
