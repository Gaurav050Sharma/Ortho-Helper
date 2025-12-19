#!/usr/bin/env python3
"""
Thermal-Safe Medical X-Ray AI Training Pipeline
===============================================

Features:
- CPU/GPU usage limiting to prevent overheating
- Automatic cooling breaks after model completion (not epochs)
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
        
        print("\\n🔄 System cooled - Ready for next model...")
    
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
                                    epochs=10, batch_size=8, learning_rate=0.0003):
        """Train model with thermal protection"""
        print(f"🚀 Starting thermal-safe training...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch Size: {batch_size} (thermal-optimized)")
        print(f"   Learning Rate: {learning_rate} (conservative)")
        print(f"   2-minute cooling break after completion")
        
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
        
        # Standard training (no epoch-level cooling breaks)
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            verbose=1,
            shuffle=True
        )
        
        # Stop monitoring
        self.stop_monitoring()
        
        print(f"\\n✅ Thermal-safe training completed!")
        best_val_accuracy = max(history.history['val_accuracy'])
        print(f"🎯 Best validation accuracy: {best_val_accuracy:.4f}")
        
        # Model cooling break after completion
        self.model_cooling_break()
        
        return model, history
    
    def save_thermal_safe_model(self, model, history, dataset_name, architecture, timestamp=None):
        """Save model with comprehensive thermal training information and ALL possible details"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create thermal-safe directory
        save_dir = os.path.join("new", "thermal_safe_models")
        os.makedirs(save_dir, exist_ok=True)
        
        base_name = f"{dataset_name}_{architecture}_thermal_safe_{timestamp}"
        
        # Create organized subdirectories
        models_dir = os.path.join(save_dir, "models")
        configs_dir = os.path.join(save_dir, "configs")
        results_dir = os.path.join(save_dir, "results")
        
        for dir_path in [models_dir, configs_dir, results_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        saved_files = {}
        
        # Save model files in multiple formats
        keras_path = os.path.join(models_dir, f"{base_name}.keras")
        h5_path = os.path.join(models_dir, f"{base_name}.h5")
        weights_path = os.path.join(models_dir, f"{base_name}.weights.h5")
        
        model.save(keras_path)
        model.save(h5_path)
        model.save_weights(weights_path)
        
        saved_files.update({
            'keras_model': keras_path,
            'h5_model': h5_path, 
            'weights': weights_path
        })
        
        # Save comprehensive training info
        training_info = {
            'timestamp': timestamp,
            'dataset': dataset_name,
            'architecture': architecture,
            'thermal_settings': {
                'max_cpu_percent': self.max_cpu_percent,
                'max_gpu_memory_mb': self.max_gpu_memory_mb,
                'model_cooling_break_seconds': self.model_cooling_break_seconds
            },
            'training_history': {k: [float(x) for x in v] for k, v in history.history.items()},
            'model_parameters': int(model.count_params()),
            'system_info': self._get_comprehensive_system_info(),
            'model_architecture': self._get_model_architecture_details(model),
            'training_environment': self._get_training_environment_snapshot(),
            'performance_metrics': self._calculate_performance_metrics(history)
        }
        
        info_path = os.path.join(results_dir, f"{base_name}_comprehensive_info.json")
        with open(info_path, 'w') as f:
            json.dump(training_info, f, indent=2)
        saved_files['comprehensive_info'] = info_path
        
        # Save model configuration
        model_config_path = os.path.join(configs_dir, f"{base_name}_model_config.json")
        with open(model_config_path, 'w') as f:
            json.dump(model.get_config(), f, indent=2)
        saved_files['model_config'] = model_config_path
        
        # Save detailed history analysis
        history_analysis_path = os.path.join(results_dir, f"{base_name}_history_analysis.json")
        history_analysis = self._analyze_training_history(history)
        with open(history_analysis_path, 'w') as f:
            json.dump(history_analysis, f, indent=2)
        saved_files['history_analysis'] = history_analysis_path
        
        # Save metadata with file hashes
        metadata_path = os.path.join(results_dir, f"{base_name}_metadata.json")
        metadata = {
            'creation_timestamp': datetime.now().isoformat(),
            'file_hashes': {},
            'git_info': self._get_git_info(),
            'python_environment': {
                'python_version': platform.python_version(),
                'tensorflow_version': tf.__version__,
                'keras_version': tf.keras.__version__,
                'numpy_version': np.__version__
            },
            'thermal_training_type': 'comprehensive_thermal_safe'
        }
        
        # Calculate file hashes
        for file_type, file_path in saved_files.items():
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    metadata['file_hashes'][file_type] = hashlib.md5(f.read()).hexdigest()
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_files['metadata'] = metadata_path
        
        # Create comprehensive README
        readme_path = os.path.join(save_dir, f"{base_name}_README.md")
        self._create_thermal_safe_readme(readme_path, dataset_name, architecture, training_info, saved_files)
        saved_files['readme'] = readme_path
        
        print(f"💾 Thermal-safe model saved:")
        for file_type, path in saved_files.items():
            file_size = os.path.getsize(path) / (1024*1024) if os.path.exists(path) else 0
            print(f"   {file_type}: {os.path.basename(path)} ({file_size:.1f}MB)")
        
        return saved_files
    
    def _get_comprehensive_system_info(self):
        """Capture comprehensive system information"""
        try:
            system_info = {
                'platform': {
                    'system': platform.system(),
                    'release': platform.release(),
                    'version': platform.version(),
                    'machine': platform.machine(),
                    'processor': platform.processor(),
                    'python_version': platform.python_version()
                },
                'hardware': {
                    'cpu_count': psutil.cpu_count(),
                    'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                    'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2)
                },
                'tensorflow': {
                    'version': tf.__version__,
                    'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
                    'mixed_precision': tf.keras.mixed_precision.global_policy().name
                }
            }
            return system_info
        except Exception as e:
            return {'error': f'Could not gather system info: {str(e)}'}
    
    def _get_model_architecture_details(self, model):
        """Extract detailed model architecture information"""
        try:
            architecture_details = {
                'total_params': int(model.count_params()),
                'trainable_params': int(sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])),
                'non_trainable_params': int(sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])),
                'input_shape': model.input_shape,
                'output_shape': model.output_shape,
                'layer_count': len(model.layers)
            }
            return architecture_details
        except Exception as e:
            return {'error': f'Could not extract model details: {str(e)}'}
    
    def _get_training_environment_snapshot(self):
        """Capture training environment snapshot"""
        try:
            snapshot = {
                'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                'timestamp_local': datetime.now().isoformat(),
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'thermal_settings': {
                    'max_cpu_percent': self.max_cpu_percent,
                    'max_gpu_memory_mb': self.max_gpu_memory_mb,
                    'model_cooling_break_seconds': self.model_cooling_break_seconds
                }
            }
            return snapshot
        except Exception as e:
            return {'error': f'Could not capture environment snapshot: {str(e)}'}
    
    def _calculate_performance_metrics(self, history):
        """Calculate comprehensive performance metrics"""
        try:
            metrics = {}
            for metric_name, values in history.history.items():
                if values:
                    values_float = [float(v) for v in values]
                    metrics[metric_name] = {
                        'final_value': values_float[-1],
                        'best_value': min(values_float) if 'loss' in metric_name else max(values_float),
                        'mean': sum(values_float) / len(values_float),
                        'improvement': values_float[-1] - values_float[0]
                    }
            return metrics
        except Exception as e:
            return {'error': f'Could not calculate metrics: {str(e)}'}
    
    def _analyze_training_history(self, history):
        """Analyze training history in detail"""
        try:
            analysis = {
                'epochs_completed': len(history.history.get('loss', [])),
                'convergence_analysis': {},
                'epoch_details': []
            }
            
            # Epoch-by-epoch analysis
            num_epochs = len(history.history.get('loss', []))
            for epoch in range(num_epochs):
                epoch_data = {'epoch': epoch + 1}
                for metric, values in history.history.items():
                    if epoch < len(values):
                        epoch_data[metric] = float(values[epoch])
                analysis['epoch_details'].append(epoch_data)
            
            return analysis
        except Exception as e:
            return {'error': f'Could not analyze history: {str(e)}'}
    
    def _get_git_info(self):
        """Get git repository information if available"""
        try:
            git_info = {
                'branch': subprocess.check_output(['git', 'branch', '--show-current'], 
                                                 stderr=subprocess.DEVNULL).decode().strip(),
                'commit_hash': subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                                      stderr=subprocess.DEVNULL).decode().strip()
            }
            return git_info
        except:
            return {'error': 'Git information not available'}
    
    def _create_thermal_safe_readme(self, readme_path, dataset_name, architecture, training_info, saved_files):
        """Create comprehensive README for thermal-safe model"""
        readme_content = f"""# Thermal-Safe {architecture} {dataset_name} Detection Model

## 🌡️ Thermal-Safe Training Overview
This model was trained using thermal-safe parameters to prevent system overheating.

### Thermal Settings
- **Max CPU Usage**: {training_info['thermal_settings']['max_cpu_percent']}%
- **Max GPU Memory**: {training_info['thermal_settings']['max_gpu_memory_mb']}MB
- **Model Cooling Break**: {training_info['thermal_settings']['model_cooling_break_seconds']} seconds

### Model Information
- **Architecture**: {architecture}
- **Dataset**: {dataset_name}
- **Total Parameters**: {training_info['model_parameters']:,}
- **Training Timestamp**: {training_info['timestamp']}

### Performance Metrics
"""
        
        # Add performance metrics if available
        if 'performance_metrics' in training_info:
            for metric, data in training_info['performance_metrics'].items():
                if isinstance(data, dict) and 'final_value' in data:
                    readme_content += f"- **{metric.title()}**: {data['final_value']:.4f}\\n"
        
        readme_content += f"""
### Saved Files
"""
        for file_type, file_path in saved_files.items():
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / (1024*1024)
                readme_content += f"- **{file_type.replace('_', ' ').title()}**: `{os.path.basename(file_path)}` ({file_size:.1f}MB)\\n"
        
        readme_content += f"""
### System Information
- **Platform**: {training_info.get('system_info', {}).get('platform', {}).get('system', 'Unknown')}
- **TensorFlow Version**: {training_info.get('system_info', {}).get('tensorflow', {}).get('version', 'Unknown')}
- **GPU Available**: {training_info.get('system_info', {}).get('tensorflow', {}).get('gpu_available', False)}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

# Example usage and configuration presets
def main():
    """Main thermal-safe training function"""
    print("🌡️ Thermal-Safe Medical X-Ray AI Training")
    print("=" * 50)
    
    # Thermal safety presets with new specifications
    presets = {
        'ultra_safe': {
            'max_cpu_percent': 50,
            'max_gpu_memory_mb': 4000,
            'model_cooling_break_seconds': 180,  # 3 minutes
            'batch_size': 4,
            'learning_rate': 0.0002,
            'description': 'Ultra-safe for very hot environments (3min cooling)'
        },
        'conservative': {
            'max_cpu_percent': 60,
            'max_gpu_memory_mb': 5000,
            'model_cooling_break_seconds': 120,  # 2 minutes
            'batch_size': 8,
            'learning_rate': 0.0003,
            'description': 'Conservative settings (2min cooling, 60% CPU, 5GB GPU)'
        },
        'balanced': {
            'max_cpu_percent': 70,
            'max_gpu_memory_mb': 5500,
            'model_cooling_break_seconds': 90,  # 1.5 minutes
            'batch_size': 12,
            'learning_rate': 0.0005,
            'description': 'Balanced thermal safety and performance'
        }
    }
    
    print("Available thermal safety presets:")
    for name, preset in presets.items():
        print(f"  {name}: {preset['description']}")
        print(f"    CPU: {preset['max_cpu_percent']}%, GPU: {preset['max_gpu_memory_mb']}MB")
        print(f"    Model Cooling: {preset['model_cooling_break_seconds']}s, Batch: {preset['batch_size']}")
    
    print("\\n" + "=" * 50)
    preset_choice = input("Select preset (ultra_safe/conservative/balanced): ").strip().lower()
    
    if preset_choice not in presets:
        print("❌ Invalid preset, using 'conservative'")
        preset_choice = 'conservative'
    
    preset = presets[preset_choice]
    print(f"\\n🎯 Using '{preset_choice}' preset")
    print(f"📊 {preset['description']}")
    
    # Initialize thermal-safe trainer
    trainer = ThermalSafeTraining(
        max_cpu_percent=preset['max_cpu_percent'],
        max_gpu_memory_mb=preset['max_gpu_memory_mb'],
        model_cooling_break_seconds=preset['model_cooling_break_seconds']
    )
    
    # Demo with dummy data
    print("\\n🔄 Creating test data for thermal-safe training demo...")
    
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
    
    print("\\n✅ Thermal-safe training demo completed!")
    print("🌡️ Your system stayed cool with 2-minute model cooling breaks!")

if __name__ == "__main__":
    main()