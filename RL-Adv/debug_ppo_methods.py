#!/usr/bin/env python3
"""
Debug script to check PPOTrainer.step method signature
"""

try:
    from trl import PPOTrainer
    import inspect
    
    # Get the signature of PPOTrainer.step
    try:
        step_sig = inspect.signature(PPOTrainer.step)
        print(f"PPOTrainer.step signature: {step_sig}")
    except:
        print("Could not get step signature")
        
    # Try to get method names
    methods = [method for method in dir(PPOTrainer) if not method.startswith('_')]
    print(f"PPOTrainer public methods: {methods}")
    
    # Check if log_stats exists
    if hasattr(PPOTrainer, 'log_stats'):
        try:
            log_stats_sig = inspect.signature(PPOTrainer.log_stats)
            print(f"PPOTrainer.log_stats signature: {log_stats_sig}")
        except:
            print("Could not get log_stats signature")
    else:
        print("PPOTrainer has no log_stats method")
        
except Exception as e:
    print(f"Error: {e}")
