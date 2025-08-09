"""
Environment configuration for AdvTG RL-Adv module.

This module provides unified environment setup functions similar to the DL stage,
ensuring consistent configuration across all stages of the AdvTG framework.
"""

import os
import torch


def set_huggingface_mirror():
    """Set Hugging Face mirror for faster downloads in certain regions."""
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HUGGINGFACE_HUB_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HUGGINGFACE_HUB_URL"] = "https://hf-mirror.com"
    
    try:
        from transformers import file_utils
        file_utils.HUGGINGFACE_CO_URL_HOME = "https://hf-mirror.com"
        print("✅ Set transformers to use mirror: https://hf-mirror.com")
    except:
        try:
            import transformers.utils.hub as hub_utils
            hub_utils.HUGGINGFACE_CO_URL_HOME = "https://hf-mirror.com"
            print("✅ Set transformers hub to use mirror")
        except:
            print("⚠️  Could not set transformers mirror, using environment variables only")


def set_cuda_environment():
    """Set CUDA environment variables for optimal GPU performance."""
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 注释掉此行以使用所有GPU
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    
    # Optional: Set memory fraction (uncomment if needed)
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def get_device_info():
    """Get device information and set appropriate device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name()}")
        print(f"🔥 CUDA Version: {torch.version.cuda}")
        print(f"🔥 PyTorch Version: {torch.__version__}")
        
        # Memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🔥 GPU Memory: {total_memory:.1f} GB")
    else:
        print("⚠️  CUDA not available, using CPU")
    
    return device


def set_environment():
    """
    Set complete environment for RL-Adv training.
    
    This function combines all environment setup steps and provides
    a unified interface similar to the DL stage.
    
    Returns:
        torch.device: The device to use for training
    """
    print("🚀 Setting up RL-Adv environment...")
    
    # Set Hugging Face mirror (optional)
    set_huggingface_mirror()
    
    # Set CUDA environment
    set_cuda_environment()
    
    # Get device info
    device = get_device_info()
    
    print("✅ Environment setup completed!")
    return device


def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'torch',
        'transformers', 
        'trl',
        'datasets',
        'numpy',
        'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing!")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("🎉 All dependencies are installed!")
        return True


if __name__ == "__main__":
    # Test the environment setup
    print("Testing RL-Adv environment setup...")
    check_dependencies()
    device = set_environment()
    print(f"Setup complete. Device: {device}")
