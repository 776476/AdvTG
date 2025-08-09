#!/usr/bin/env python3
"""
Debug script to check TRL library version and PPOTrainer signature
"""

try:
    import trl
    print(f"TRL version: {trl.__version__}")
    
    from trl import PPOTrainer
    import inspect
    
    # Get the signature of PPOTrainer.__init__
    sig = inspect.signature(PPOTrainer.__init__)
    print(f"PPOTrainer.__init__ signature: {sig}")
    
    # Get parameter names
    params = list(sig.parameters.keys())
    print(f"Parameter names: {params}")
    
    # Try to get the source code
    try:
        source = inspect.getsource(PPOTrainer.__init__)
        print(f"First 10 lines of source:")
        for i, line in enumerate(source.split('\n')[:10]):
            print(f"{i+1:2d}: {line}")
    except Exception as e:
        print(f"Could not get source: {e}")
        
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Error: {e}")
