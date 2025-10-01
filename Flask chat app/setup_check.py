#!/usr/bin/env python3
"""
Pre-flight check for Gemma 3n Therapeutic Chatbot
"""

import sys
import os

def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor} (need 3.8+)")
        return False

def check_packages():
    """Check required packages"""
    print("\nChecking required packages...")
    required = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'peft': 'PEFT',
        'bitsandbytes': 'BitsAndBytes',
        'flask': 'Flask',
        'accelerate': 'Accelerate'
    }
    
    all_ok = True
    for package, name in required.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"  ✓ {name}: {version}")
        except ImportError:
            print(f"  ✗ {name}: NOT INSTALLED")
            all_ok = False
    
    return all_ok

def check_cuda():
    """Check CUDA availability"""
    print("\nChecking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.version.cuda}")
            print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  ✓ GPU Memory: {memory_gb:.2f} GB")
            return True
        else:
            print("  ✗ CUDA not available")
            return False
    except:
        print("  ✗ Cannot check CUDA")
        return False

def check_model_files():
    """Check if model files exist"""
    print("\nChecking model files...")
    model_path = "/home/sanj-ai/Documents/SlateMate/Gemma_4b_Finetuning/gemma-3n"
    
    if not os.path.exists(model_path):
        print(f"  ✗ Model directory not found: {model_path}")
        return False
    
    print(f"  ✓ Model directory exists: {model_path}")
    
    required_files = [
        'adapter_model.safetensors',
        'adapter_config.json',
        'tokenizer.json',
        'config.json'
    ]
    
    all_ok = True
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024**2)
            print(f"  ✓ {file}: {size_mb:.2f} MB")
        else:
            print(f"  ✗ {file}: NOT FOUND")
            all_ok = False
    
    return all_ok

def main():
    print("="*70)
    print("Gemma 3n Therapeutic Chatbot - Pre-flight Check")
    print("="*70)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_packages),
        ("CUDA & GPU", check_cuda),
        ("Model Files", check_model_files)
    ]
    
    results = []
    for name, check_func in checks:
        result = check_func()
        results.append((name, result))
    
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {name}")
        if not result:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n✓ All checks passed! Ready to run the chatbot.")
        print("\nRun: python app.py")
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        print("\nInstall missing packages with:")
        print("  pip install torch transformers peft bitsandbytes flask accelerate")
    
    print("="*70)

if __name__ == "__main__":
    main()
