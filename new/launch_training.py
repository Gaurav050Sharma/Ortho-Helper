#!/usr/bin/env python3
"""
Training Launcher - Quick Start Script
====================================

Quick launcher for the comprehensive training pipeline.
Provides easy options to start training with different scopes.
"""

import os
import sys
import json
from datetime import datetime

def show_training_overview():
    """Show overview of what will be trained"""
    print("🏥 Medical X-Ray AI Comprehensive Training Overview")
    print("=" * 60)
    
    datasets = [
        "📊 Pneumonia (CHEST) - ~5,856 images",
        "❤️ Cardiomegaly (CHEST) - ~4,438 images", 
        "🦴 Osteoporosis (KNEE) - ~1,945 images",
        "🦵 Osteoarthritis (KNEE) - ~9,788 images",
        "🦾 Limb Abnormalities (ARM) - ~3,661 images"
    ]
    
    architectures = [
        "🧠 DenseNet121 (Recommended)",
        "⚡ EfficientNetB0 (Recommended)",
        "🏗️ ResNet50 (Recommended)", 
        "📐 VGG16 (Heavy)",
        "🔧 Custom CNN (Baseline)"
    ]
    
    configurations = [
        "⚡ Quick Test (3 epochs, 100 images/class)",
        "📊 Standard (8 epochs, 500 images/class)",
        "🔥 Intensive (15 epochs, 1000 images/class)"
    ]
    
    print("📋 DATASETS TO TRAIN:")
    for dataset in datasets:
        print(f"   {dataset}")
    
    print(f"\n🏗️ ARCHITECTURES TO TEST:")
    for arch in architectures:
        print(f"   {arch}")
    
    print(f"\n⚙️ CONFIGURATIONS:")
    for config in configurations:
        print(f"   {config}")
    
    total_combinations = 5 * 5 * 3
    print(f"\n🎯 TOTAL COMBINATIONS: {total_combinations}")
    print(f"⏱️ ESTIMATED TIME: ~{total_combinations * 10} minutes (varies by config)")
    print(f"📁 OUTPUT DIRECTORY: new/")
    print(f"💾 CRASH RECOVERY: Enabled (training_progress.json)")

def check_environment():
    """Check if environment is ready"""
    print("\n🔍 Environment Check:")
    
    # Check Python
    print(f"   🐍 Python: {sys.version.split()[0]}")
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        print(f"   🤖 TensorFlow: {tf.__version__}")
    except ImportError:
        print("   ❌ TensorFlow not found!")
        return False
    
    # Check datasets
    datasets_exist = True
    dataset_paths = [
        "Dataset/CHEST/Pneumonia_Organized",
        "Dataset/CHEST/cardiomelgy",
        "Dataset/KNEE/Osteoporosis/Combined_Osteoporosis_Dataset",
        "Dataset/KNEE/Osteoarthritis/Combined_Osteoarthritis_Dataset", 
        "Dataset/ARM/MURA_Organized/limbs"
    ]
    
    for path in dataset_paths:
        if os.path.exists(path):
            print(f"   ✅ {path}")
        else:
            print(f"   ❌ {path} - NOT FOUND")
            datasets_exist = False
    
    # Check new directory
    if not os.path.exists("new"):
        os.makedirs("new")
        print("   📁 Created 'new' directory")
    else:
        print("   📁 'new' directory exists")
    
    return datasets_exist

def show_progress():
    """Show current training progress"""
    progress_file = "new/training_progress.json"
    
    if not os.path.exists(progress_file):
        print("📊 No training progress found - Starting fresh!")
        return
    
    try:
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        
        total_combinations = 5 * 5 * 3  # 5 datasets × 5 architectures × 3 configs
        completed = len([k for k, v in progress.items() if v['status'] == 'completed'])
        failed = len([k for k, v in progress.items() if v['status'] == 'failed'])
        
        print(f"📊 TRAINING PROGRESS:")
        print(f"   ✅ Completed: {completed}/{total_combinations}")
        print(f"   ❌ Failed: {failed}")
        print(f"   ⏳ Remaining: {total_combinations - completed - failed}")
        
        if completed > 0:
            print(f"\n🏆 RECENT COMPLETIONS:")
            recent_completed = [(k, v) for k, v in progress.items() 
                              if v['status'] == 'completed'][-3:]
            
            for combo_key, combo_data in recent_completed:
                dataset = combo_data.get('dataset', 'unknown')
                arch = combo_data.get('architecture', 'unknown') 
                config = combo_data.get('configuration', 'unknown')
                accuracy = combo_data.get('results', {}).get('test_accuracy', 0)
                print(f"   ✅ {dataset}_{arch}_{config}: {accuracy*100:.1f}% accuracy")
        
    except Exception as e:
        print(f"⚠️ Error reading progress: {e}")

def run_training():
    """Launch the comprehensive training"""
    print("\n🚀 LAUNCHING COMPREHENSIVE TRAINING PIPELINE...")
    print("⚠️ This will train ALL combinations - Press Ctrl+C to cancel")
    
    try:
        # Import and run
        from comprehensive_training_pipeline import ComprehensiveTrainingPipeline
        
        pipeline = ComprehensiveTrainingPipeline()
        pipeline.run_comprehensive_training()
        
    except KeyboardInterrupt:
        print("\n⛔ Training cancelled by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main launcher function"""
    print("🏥 Medical X-Ray AI Training Launcher")
    print("🚀 Comprehensive Multi-Architecture Training Pipeline")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Show overview
    show_training_overview()
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed! Please fix issues before training.")
        return
    
    # Show current progress
    show_progress()
    
    # Ask user what to do
    print("\n🎯 TRAINING OPTIONS:")
    print("   1. 🚀 Start/Resume Comprehensive Training (ALL combinations)")
    print("   2. 📊 View Progress Only")
    print("   3. 🔧 Environment Check Only") 
    print("   4. ❌ Exit")
    
    try:
        choice = input("\n👉 Enter your choice (1-4): ").strip()
        
        if choice == "1":
            # Confirm before starting
            print(f"\n⚠️ You are about to start training {5*5*3} model combinations!")
            print("⏱️ This could take several hours depending on your hardware.")
            print("💾 Progress will be saved continuously for crash recovery.")
            confirm = input("👉 Continue? (y/N): ").strip().lower()
            
            if confirm in ['y', 'yes']:
                run_training()
            else:
                print("👋 Training cancelled.")
        
        elif choice == "2":
            show_progress()
            print("📊 Progress view complete.")
        
        elif choice == "3":
            print("🔍 Environment check complete.")
        
        elif choice == "4":
            print("👋 Goodbye!")
        
        else:
            print("❌ Invalid choice!")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()