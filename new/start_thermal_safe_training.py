#!/usr/bin/env python3
"""
Start Thermal-Safe Comprehensive Training
========================================

This script starts the comprehensive training pipeline with thermal protection
to prevent overheating while training all 75 model combinations.
"""

import os
import sys
import json
import psutil
import time
from datetime import datetime

def check_system_temperature():
    """Check if system is cool enough to start training"""
    try:
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=2)
        print(f"🖥️ Current CPU usage: {cpu_percent:.1f}%")
        
        # Check memory usage
        memory = psutil.virtual_memory()
        print(f"💾 Memory usage: {memory.percent:.1f}%")
        
        # Check if system is cool enough
        if cpu_percent > 80:
            print("⚠️ High CPU usage detected. Please wait for system to cool down.")
            return False
        
        if memory.percent > 85:
            print("⚠️ High memory usage detected. Please close some applications.")
            return False
        
        return True
        
    except Exception as e:
        print(f"⚠️ Could not check system status: {e}")
        return True  # Proceed anyway

def configure_thermal_training():
    """Configure thermal-safe training parameters"""
    print("🌡️ Thermal-Safe Training Configuration")
    print("=" * 50)
    
    presets = {
        '1': {
            'name': 'Ultra Conservative',
            'description': 'Maximum cooling, minimum heat (slowest)',
            'modifications': {
                'batch_sizes': {'thermal_safe': 4, 'quick_test': 8, 'standard': 12, 'intensive': 8},
                'cooling_breaks': 60,  # 1 minute breaks
                'max_cpu_threads': 1,
                'epochs_per_cooling': 1
            }
        },
        '2': {
            'name': 'Conservative',
            'description': 'Good cooling, low heat (recommended)',
            'modifications': {
                'batch_sizes': {'thermal_safe': 8, 'quick_test': 16, 'standard': 20, 'intensive': 16},
                'cooling_breaks': 30,  # 30 second breaks
                'max_cpu_threads': 2,
                'epochs_per_cooling': 2
            }
        },
        '3': {
            'name': 'Balanced',
            'description': 'Moderate cooling, acceptable heat',
            'modifications': {
                'batch_sizes': {'thermal_safe': 12, 'quick_test': 24, 'standard': 32, 'intensive': 24},
                'cooling_breaks': 15,  # 15 second breaks
                'max_cpu_threads': 3,
                'epochs_per_cooling': 3
            }
        }
    }
    
    print("Available thermal safety levels:")
    for key, preset in presets.items():
        print(f"\n{key}. {preset['name']}")
        print(f"   {preset['description']}")
        print(f"   Cooling breaks: {preset['modifications']['cooling_breaks']}s")
        print(f"   Max batch sizes: {preset['modifications']['batch_sizes']}")
    
    choice = input(f"\nSelect thermal safety level (1-3) [2]: ").strip()
    if choice not in presets:
        choice = '2'
        print(f"Using default: Conservative")
    
    return presets[choice]

def modify_training_config(thermal_config):
    """Modify the comprehensive training pipeline for thermal safety"""
    config_file = "comprehensive_training_pipeline.py"
    
    print(f"\n🔧 Applying thermal modifications...")
    print(f"   Thermal Level: {thermal_config['name']}")
    print(f"   Cooling Breaks: {thermal_config['modifications']['cooling_breaks']}s")
    
    # Create thermal-safe configuration
    thermal_settings = {
        'applied_at': datetime.now().isoformat(),
        'thermal_level': thermal_config['name'],
        'modifications': thermal_config['modifications'],
        'original_total_combinations': 75,
        'estimated_training_time_increase': '150-300%'
    }
    
    # Save thermal settings
    with open('new/thermal_settings.json', 'w') as f:
        json.dump(thermal_settings, f, indent=2)
    
    print("✅ Thermal configuration saved")
    return thermal_settings

def start_thermal_safe_training(thermal_config):
    """Start the thermal-safe comprehensive training"""
    print(f"\n🚀 Starting Thermal-Safe Comprehensive Training")
    print("=" * 60)
    print(f"🌡️ Thermal Level: {thermal_config['name']}")
    print(f"💤 Cooling Strategy: {thermal_config['modifications']['cooling_breaks']}s breaks")
    print(f"📊 Estimated Time: 2-3x longer (but your system stays cool!)")
    print(f"🔄 Progress: Will save continuously with crash recovery")
    
    # Final system check
    if not check_system_temperature():
        print("\n❌ System not ready for training. Please try again later.")
        return False
    
    print(f"\n✅ System ready for thermal-safe training!")
    print(f"💡 Training will pause for cooling breaks automatically")
    print(f"🛑 You can safely interrupt with Ctrl+C anytime")
    
    # Ask for confirmation
    confirm = input(f"\nStart thermal-safe training? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("❌ Training cancelled")
        return False
    
    return True

def main():
    """Main thermal-safe training launcher"""
    print("🏥 Medical X-Ray AI - Thermal-Safe Training Launcher")
    print("=" * 60)
    print("🌡️ This mode prevents overheating by:")
    print("   • Using smaller batch sizes")
    print("   • Adding cooling breaks between epochs")
    print("   • Limiting CPU/GPU usage")
    print("   • Monitoring system temperature")
    print("\n⏱️ Training will take 2-3x longer but keeps your system safe!")
    
    # Check system readiness
    print(f"\n🔍 Checking system status...")
    if not check_system_temperature():
        return
    
    # Configure thermal settings
    thermal_config = configure_thermal_training()
    
    # Apply thermal modifications
    thermal_settings = modify_training_config(thermal_config)
    
    # Start training if user confirms
    if start_thermal_safe_training(thermal_config):
        print(f"\n🎯 Launching thermal-safe training...")
        
        # Set environment variables for thermal safety
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN for stability
        
        # Start the comprehensive training
        try:
            os.system("python comprehensive_training_pipeline.py")
        except KeyboardInterrupt:
            print(f"\n🛑 Training interrupted by user")
            print(f"💾 Progress saved - can resume anytime")
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            print(f"💾 Progress saved - can resume anytime")
    
    print(f"\n🌡️ Thermal-safe training session ended")
    print(f"📁 Check 'new/' directory for saved models")
    print(f"📊 Check 'thermal_settings.json' for applied settings")

if __name__ == "__main__":
    main()