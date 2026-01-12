#!/usr/bin/env python3
"""
RangeLDM Demo Script - Quick Start Guide
This script provides a simple interface to run RangeLDM inference
"""

import os
import sys
import argparse
from pathlib import Path

def print_banner():
    print("=" * 60)
    print("  RangeLDM - Fast Realistic LiDAR Point Cloud Generation")
    print("=" * 60)
    print()

def check_environment():
    """Check if required packages are installed"""
    print("Checking environment...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    
    try:
        import diffusers
        print(f"✓ Diffusers {diffusers.__version__}")
    except ImportError:
        print("✗ Diffusers not installed")
        return False
    
    try:
        import accelerate
        print(f"✓ Accelerate {accelerate.__version__}")
    except ImportError:
        print("✗ Accelerate not installed")
        return False
    
    print()
    return True

def show_menu():
    """Display the main menu"""
    print("\nAvailable Options:")
    print("-" * 60)
    print("1. Generate LiDAR point clouds (inference)")
    print("2. Train VAE model")
    print("3. Train LDM unconditional")
    print("4. Train LDM conditional (upsample)")
    print("5. Show available configs")
    print("6. Check environment")
    print("q. Quit")
    print("-" * 60)

def list_configs():
    """List available configuration files"""
    ldm_configs = Path("ldm/configs")
    vae_configs = Path("vae/configs")
    
    print("\n📁 LDM Configs:")
    if ldm_configs.exists():
        for cfg in sorted(ldm_configs.glob("*.yaml")):
            print(f"  - {cfg}")
    else:
        print("  No LDM configs found")
    
    print("\n📁 VAE Configs:")
    if vae_configs.exists():
        for cfg in sorted(vae_configs.glob("*.yaml")):
            print(f"  - {cfg}")
    else:
        print("  No VAE configs found")
    print()

def run_inference():
    """Run LDM inference"""
    print("\n" + "=" * 60)
    print("  LiDAR Generation (Inference)")
    print("=" * 60)
    
    # Check if inference script exists
    inference_script = Path("ldm/inference.py")
    if not inference_script.exists():
        print("❌ Error: ldm/inference.py not found")
        return
    
    # List available configs
    print("\nAvailable config files:")
    ldm_configs = Path("ldm/configs")
    if ldm_configs.exists():
        configs = sorted(ldm_configs.glob("*.yaml"))
        for i, cfg in enumerate(configs, 1):
            print(f"  {i}. {cfg.name}")
        
        if configs:
            try:
                choice = input("\nSelect config number (or enter path): ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(configs):
                    config_path = str(configs[int(choice) - 1])
                else:
                    config_path = choice
            except (ValueError, IndexError):
                config_path = "ldm/configs/RangeLDM.yaml"
        else:
            config_path = input("Enter config path: ").strip()
    else:
        config_path = input("Enter config path: ").strip()
    
    num_samples = input("Number of samples to generate (default: 100): ").strip()
    num_samples = num_samples if num_samples else "100"
    
    batch_size = input("Batch size (default: 16): ").strip()
    batch_size = batch_size if batch_size else "16"
    
    print(f"\n🚀 Running inference with:")
    print(f"   Config: {config_path}")
    print(f"   Samples: {num_samples}")
    print(f"   Batch size: {batch_size}")
    print()
    
    # Run inference
    os.chdir("ldm")
    cmd = f"python inference.py --cfg {config_path} --samples {num_samples} --batch_size {batch_size}"
    print(f"Executing: {cmd}\n")
    os.system(cmd)

def run_vae_training():
    """Run VAE training"""
    print("\n" + "=" * 60)
    print("  VAE Training")
    print("=" * 60)
    
    vae_configs = Path("vae/configs")
    if vae_configs.exists():
        configs = sorted(vae_configs.glob("*.yaml"))
        print("\nAvailable configs:")
        for i, cfg in enumerate(configs, 1):
            print(f"  {i}. {cfg.name}")
        
        if configs:
            choice = input("\nSelect config number (or enter path): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(configs):
                config_path = configs[int(choice) - 1].name
            else:
                config_path = choice
        else:
            config_path = input("Enter config file name: ").strip()
    else:
        config_path = input("Enter config file name: ").strip()
    
    print(f"\n🚀 Starting VAE training with config: {config_path}")
    print("⚠️  Make sure dataset paths are configured correctly in the config file\n")
    
    os.chdir("vae")
    cmd = f"python main.py --base configs/{config_path}"
    print(f"Executing: {cmd}\n")
    os.system(cmd)

def run_ldm_training(conditional=False):
    """Run LDM training"""
    mode = "Conditional" if conditional else "Unconditional"
    print("\n" + "=" * 60)
    print(f"  LDM {mode} Training")
    print("=" * 60)
    
    if conditional:
        default_config = "configs/upsample.yaml"
        script = "train_conditional.py"
    else:
        default_config = "configs/RangeLDM.yaml"
        script = "train_unconditional.py"
    
    config_path = input(f"Config path (default: {default_config}): ").strip()
    config_path = config_path if config_path else default_config
    
    print(f"\n🚀 Starting LDM {mode} training")
    print(f"   Config: {config_path}")
    print("⚠️  Make sure VAE checkpoint path is configured in the config file\n")
    
    os.chdir("ldm")
    cmd = f"accelerate launch {script} --cfg {config_path}"
    print(f"Executing: {cmd}\n")
    os.system(cmd)

def main():
    parser = argparse.ArgumentParser(description="RangeLDM Demo Script")
    parser.add_argument("--auto-inference", action="store_true", 
                       help="Run inference automatically with default settings")
    args = parser.parse_args()
    
    print_banner()
    
    if args.auto_inference:
        print("Running inference with default settings...")
        run_inference()
        return
    
    if not check_environment():
        print("\n❌ Environment check failed. Please install required packages.")
        return
    
    while True:
        show_menu()
        choice = input("\nEnter your choice: ").strip().lower()
        
        if choice == '1':
            run_inference()
        elif choice == '2':
            run_vae_training()
        elif choice == '3':
            run_ldm_training(conditional=False)
        elif choice == '4':
            run_ldm_training(conditional=True)
        elif choice == '5':
            list_configs()
        elif choice == '6':
            check_environment()
        elif choice in ['q', 'quit', 'exit']:
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
