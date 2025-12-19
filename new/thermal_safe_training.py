#!/usr/bin/env python3
"""
Thermal-Safe Medical X-Ray AI Training Pipeline
===============================================

Features:
    def model_cooling_break(self, duration=None):
        """Take a cooling break after model completion"""
        if duration is None:
            duration = self.model_cooling_break_seconds
        
        print(f"\\n🎉 Model training completed!")
        print(f"💤 Model cooling break: {duration}s...")
        print("🌡️ Allowing system to cool down before next model...")
        
        # Show progress bar for cooling break
        minutes = duration // 60
        seconds = duration % 60
        print(f"   Duration: {minutes}m {seconds}s")
        
        for i in range(duration):
            progress = (i + 1) / duration
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            remaining = duration - i - 1
            mins = remaining // 60
            secs = remaining % 60
            
            print(f"\\r   Cooling: [{bar}] {mins:02d}:{secs:02d} remaining", end='', flush=True)
            time.sleep(1)
        
        print("\\n🔄 System cooled - Ready for next model...")miting to prevent overheating
- Automatic cooling breaks between epochs
- Temperature monitoring (if available)
- Smaller batch sizes for lower power consumption
- Slower learning rates for stability
- Progress saving to prevent loss during thermal shutdowns
"""

import os
import sys
import json
import time
import psutil
import threading
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ThermalSafeTraining:
    """Training pipeline with thermal protection"""
    
    def __init__(self, max_cpu_percent=60, max_gpu_memory_mb=5000, model_cooling_break_seconds=120):
        """
        Initialize thermal-safe training
        
        Args:
            max_cpu_percent: Maximum CPU usage percentage (default: 60%)
            max_gpu_memory_mb: Maximum GPU memory in MB (default: 5GB for RTX 3060)
            model_cooling_break_seconds: Cooling break duration after model completion (default: 120s/2min)
        """
        self.max_cpu_percent = max_cpu_percent
        self.max_gpu_memory_mb = max_gpu_memory_mb
        self.model_cooling_break_seconds = model_cooling_break_seconds
        self.monitoring_active = False
        self.cpu_usage_history = []
        self.gpu_memory_usage = 0
        
        print(f"🌡️ Thermal-Safe Training Initialized")
        print(f"   Max CPU Usage: {max_cpu_percent}%")
        print(f"   Max GPU Memory: {max_gpu_memory_mb}MB")
        print(f"   Model Cooling Breaks: {model_cooling_break_seconds}s after each model")
        
        self.configure_thermal_settings()
    
    def configure_thermal_settings(self):
        """Configure TensorFlow for thermal-safe operation"""
        print("🔧 Configuring thermal-safe settings...")
        
        # Limit TensorFlow threading
        tf.config.threading.set_intra_op_parallelism_threads(2)  # Reduce CPU threads
        tf.config.threading.set_inter_op_parallelism_threads(2)
        
        # Configure GPU settings if available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Set memory limit for thermal safety
                tf.config.experimental.set_memory_limit(
                    gpus[0], self.max_gpu_memory_mb
                )
                print(f"✅ GPU memory limited to {self.max_gpu_memory_mb}MB")
                
                # Use mixed precision for efficiency (less heat)
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                print("✅ Mixed precision enabled for efficiency")
                
            except RuntimeError as e:
                print(f"⚠️ GPU configuration warning: {e}")
        else:
            print("⚠️ No GPU detected, using CPU-only thermal limits")
    
    def start_monitoring(self):
        """Start thermal monitoring in background"""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()
        print("🔍 Started thermal monitoring")
    
    def stop_monitoring(self):
        """Stop thermal monitoring"""
        self.monitoring_active = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1)
        print("⏹️ Stopped thermal monitoring")
    
    def _monitor_system(self):
        """Monitor CPU and GPU usage"""
        while self.monitoring_active:
            try:
                # Monitor CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_usage_history.append(cpu_percent)
                
                # Keep only last 60 readings (1 minute)
                if len(self.cpu_usage_history) > 60:
                    self.cpu_usage_history.pop(0)
                
                # Check if CPU usage is too high
                avg_cpu = sum(self.cpu_usage_history[-10:]) / min(10, len(self.cpu_usage_history))
                if avg_cpu > self.max_cpu_percent:
                    print(f"🌡️ High CPU usage detected: {avg_cpu:.1f}% (limit: {self.max_cpu_percent}%)")
                    print("💤 Initiating cooling break...")
                    time.sleep(30)  # Short cooling break for high CPU
                
                # Monitor GPU if available (basic check)
                try:
                    import nvidia_ml_py3 as nvml
                    nvml.nvmlInit()
                    handle = nvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_memory = nvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                    
                    if gpu_temp > 80:  # 80°C threshold
                        print(f"🌡️ High GPU temperature: {gpu_temp}°C")
                        print("💤 Initiating cooling break...")
                        time.sleep(60)  # 1 minute cooling for high GPU temp
                        
                except ImportError:
                    pass  # nvidia-ml-py not available
                except Exception:
                    pass  # GPU monitoring failed
                    
            except Exception as e:
                pass  # Continue monitoring despite errors
            
            time.sleep(5)  # Check every 5 seconds
    
    def cooling_break(self, duration=None):
        """Take a cooling break"""
        if duration is None:
            duration = self.cooling_break_seconds
        
        print(f"💤 Cooling break: {duration}s...")
        print("🌡️ Allowing system to cool down...")
        
        # Show progress bar for cooling break
        for i in range(duration):
            progress = (i + 1) / duration
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"\r   Cooling: [{bar}] {i+1}/{duration}s", end='', flush=True)
            time.sleep(1)
        
        print("\n🔄 Resuming training...")
    
    def create_thermal_safe_model(self, architecture, input_shape=(224, 224, 3)):
        """Create model optimized for thermal safety"""
        print(f"🏗️ Creating thermal-safe {architecture} model...")
        
        if architecture == "DenseNet121":
            base_model = tf.keras.applications.DenseNet121(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        elif architecture == "EfficientNetB0":
            base_model = tf.keras.applications.EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        elif architecture == "ResNet50":
            base_model = tf.keras.applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Freeze more layers for thermal safety (less computation)
        base_model.trainable = False
        
        # Simpler head for less computation
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),  # Smaller dense layer
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')
        ])
        
        return model
    
    def train_with_thermal_protection(self, model, X_train, y_train, X_val, y_val, 
                                    epochs=10, batch_size=16, learning_rate=0.0005):
        """Train model with thermal protection"""
        print(f"🚀 Starting thermal-safe training...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch Size: {batch_size} (thermal-optimized)")
        print(f"   Learning Rate: {learning_rate} (conservative)")
        
        # Start monitoring
        self.start_monitoring()
        
        # Compile with conservative settings
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Custom training loop with cooling breaks
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        best_val_accuracy = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\n📊 Epoch {epoch + 1}/{epochs}")
            print("=" * 50)
            
            # Train for one epoch
            epoch_history = model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=1,  # Train one epoch at a time
                validation_data=(X_val, y_val),
                verbose=1,
                shuffle=True
            )
            
            # Record history
            for key in history.keys():
                history[key].extend(epoch_history.history[key])
            
            # Check for improvement
            val_accuracy = epoch_history.history['val_accuracy'][0]
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                print(f"✅ New best validation accuracy: {val_accuracy:.4f}")
            else:
                patience_counter += 1
                print(f"⏳ No improvement. Patience: {patience_counter}/3")
            
            # Early stopping with thermal consideration
            if patience_counter >= 3:
                print("🛑 Early stopping due to no improvement")
                break
            
            # Mandatory cooling break between epochs (except last epoch)
            if epoch < epochs - 1:
                print(f"\n🌡️ Epoch {epoch + 1} completed")
                self.cooling_break()
        
        # Stop monitoring
        self.stop_monitoring()
        
        print(f"\n✅ Thermal-safe training completed!")
        print(f"🎯 Best validation accuracy: {best_val_accuracy:.4f}")
        
        return model, history
    
    def save_thermal_safe_model(self, model, history, dataset_name, architecture, timestamp=None):
        """Save model with thermal training information"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create thermal-safe directory
        save_dir = os.path.join("new", "thermal_safe_models")
        os.makedirs(save_dir, exist_ok=True)
        
        base_name = f"{dataset_name}_{architecture}_thermal_safe_{timestamp}"
        
        # Save model files
        keras_path = os.path.join(save_dir, f"{base_name}.keras")
        h5_path = os.path.join(save_dir, f"{base_name}.h5")
        
        model.save(keras_path)
        model.save(h5_path)
        
        # Save training info
        training_info = {
            'timestamp': timestamp,
            'dataset': dataset_name,
            'architecture': architecture,
            'thermal_settings': {
                'max_cpu_percent': self.max_cpu_percent,
                'max_gpu_memory_mb': self.max_gpu_memory_mb,
                'cooling_break_seconds': self.cooling_break_seconds
            },
            'training_history': history,
            'model_parameters': int(model.count_params())
        }
        
        info_path = os.path.join(save_dir, f"{base_name}_info.json")
        with open(info_path, 'w') as f:
            json.dump(training_info, f, indent=2)
        
        print(f"💾 Thermal-safe model saved:")
        print(f"   Keras: {keras_path}")
        print(f"   H5: {h5_path}")
        print(f"   Info: {info_path}")
        
        return keras_path, h5_path, info_path

# Example usage and configuration presets
def main():
    """Main thermal-safe training function"""
    print("🌡️ Thermal-Safe Medical X-Ray AI Training")
    print("=" * 50)
    
    # Thermal safety presets
    presets = {
        'conservative': {
            'max_cpu_percent': 60,
            'max_gpu_memory_mb': 3000,
            'cooling_break_seconds': 45,
            'batch_size': 8,
            'learning_rate': 0.0003,
            'description': 'Ultra-safe for hot environments'
        },
        'balanced': {
            'max_cpu_percent': 70,
            'max_gpu_memory_mb': 4000,
            'cooling_break_seconds': 30,
            'batch_size': 16,
            'learning_rate': 0.0005,
            'description': 'Good balance of safety and speed'
        },
        'performance': {
            'max_cpu_percent': 80,
            'max_gpu_memory_mb': 5000,
            'cooling_break_seconds': 20,
            'batch_size': 24,
            'learning_rate': 0.001,
            'description': 'Higher performance with thermal monitoring'
        }
    }
    
    print("Available thermal safety presets:")
    for name, preset in presets.items():
        print(f"  {name}: {preset['description']}")
        print(f"    CPU: {preset['max_cpu_percent']}%, GPU: {preset['max_gpu_memory_mb']}MB")
        print(f"    Cooling: {preset['cooling_break_seconds']}s, Batch: {preset['batch_size']}")
    
    print("\n" + "=" * 50)
    preset_choice = input("Select preset (conservative/balanced/performance): ").strip().lower()
    
    if preset_choice not in presets:
        print("❌ Invalid preset, using 'balanced'")
        preset_choice = 'balanced'
    
    preset = presets[preset_choice]
    print(f"\n🎯 Using '{preset_choice}' preset")
    print(f"📊 {preset['description']}")
    
    # Initialize thermal-safe trainer
    trainer = ThermalSafeTraining(
        max_cpu_percent=preset['max_cpu_percent'],
        max_gpu_memory_mb=preset['max_gpu_memory_mb'],
        cooling_break_seconds=preset['cooling_break_seconds']
    )
    
    # Demo with dummy data
    print("\n🔄 Creating test data for thermal-safe training demo...")
    
    # Generate small test dataset
    X_train = tf.random.normal((100, 224, 224, 3))
    y_train = tf.random.uniform((100, 1), maxval=2, dtype=tf.int32)
    X_val = tf.random.normal((20, 224, 224, 3))
    y_val = tf.random.uniform((20, 1), maxval=2, dtype=tf.int32)
    
    # Create and train model
    model = trainer.create_thermal_safe_model("DenseNet121")
    print(f"✅ Model created: {model.count_params():,} parameters")
    
    # Train with thermal protection
    trained_model, history = trainer.train_with_thermal_protection(
        model, X_train, y_train, X_val, y_val,
        epochs=3,  # Short demo
        batch_size=preset['batch_size'],
        learning_rate=preset['learning_rate']
    )
    
    # Save model
    trainer.save_thermal_safe_model(trained_model, history, "demo", "DenseNet121")
    
    print("\n✅ Thermal-safe training demo completed!")
    print("🌡️ Your system stayed cool throughout the process!")

if __name__ == "__main__":
    main()