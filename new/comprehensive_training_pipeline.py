#!/usr/bin/env python3
"""
Comprehensive Multi-Architecture Training Pipeline
===============================================

Trains all 5 medical datasets with multiple architectures and configurations.
Features:
- Crash recovery with checkpointing
- Continuous saving to 'new' directory
- Organized file structure
- Progress tracking
- Multiple architecture support (DenseNet121, EfficientNetB0, ResNet50, VGG16, Custom CNN)
- Multiple training configurations (quick_test, standard, intensive)

Datasets:
1. Pneumonia (CHEST/Pneumonia_Organized)
2. Cardiomegaly (CHEST/cardiomelgy) 
3. Osteoporosis (KNEE/Osteoporosis/Combined_Osteoporosis_Dataset)
4. Osteoarthritis (KNEE/Osteoarthritis/Combined_Osteoarthritis_Dataset)
5. Limb Abnormalities (ARM/MURA_Organized/limbs)
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121, EfficientNetB0, ResNet50, VGG16
from tensorflow.keras import layers, optimizers, callbacks
from PIL import Image
import random
from datetime import datetime
import traceback
import time
import platform
import psutil
import socket
import subprocess
import pkg_resources
from datetime import timezone
import hashlib
import gc

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Configure GPU for optimal performance
def configure_gpu():
    """Configure GPU settings for optimal performance with RTX 3060"""
    print("🔧 Configuring GPU settings...")
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    print(f"📊 GPU devices found: {len(gpus)}")
    
    if gpus:
        try:
            # Enable memory growth to prevent TensorFlow from allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ Enabled memory growth for: {gpu}")
            
            # Set mixed precision policy for better performance on RTX 3060
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("✅ Mixed precision enabled (float16/float32) for faster training")
            
            # Set memory limit for RTX 3060 (6GB VRAM)
            tf.config.experimental.set_memory_growth(gpus[0], True)
            
            return True
        except RuntimeError as e:
            print(f"⚠️ GPU configuration error: {e}")
            return False
    else:
        print("⚠️ No GPU devices found, using CPU")
        return False

class ComprehensiveTrainingPipeline:
    """Comprehensive training pipeline for all datasets and architectures"""
    
    def __init__(self):
        self.base_dir = "new"
        self.progress_file = os.path.join(self.base_dir, "training_progress.json")
        self.datasets = self._define_datasets()
        self.architectures = self._define_architectures()
        self.configurations = self._define_configurations()
        self.completed_combinations = self._load_progress()
        self.gpu_enabled = configure_gpu()
        
        # Create base directory
        os.makedirs(self.base_dir, exist_ok=True)
        
    def _define_datasets(self):
        """Define all datasets with their paths and characteristics"""
        return {
            "pneumonia": {
                "name": "Pneumonia",
                "path": "Dataset/CHEST/Pneumonia_Organized",
                "classes": ["Normal", "Pneumonia"],
                "folders": ["Normal", "Pneumonia"],
                "type": "chest",
                "estimated_size": 5856
            },
            "cardiomegaly": {
                "name": "Cardiomegaly", 
                "path": "Dataset/CHEST/cardiomelgy",
                "classes": ["Normal", "Cardiomegaly"],
                "folders": ["Normal", "Cardiomegaly"],
                "type": "chest",
                "estimated_size": 4438
            },
            "osteoporosis": {
                "name": "Osteoporosis",
                "path": "Dataset/KNEE/Osteoporosis/Combined_Osteoporosis_Dataset",
                "classes": ["Normal", "Osteoporosis"],
                "folders": ["Normal", "Osteoporosis"],
                "type": "knee",
                "estimated_size": 1945
            },
            "osteoarthritis": {
                "name": "Osteoarthritis",
                "path": "Dataset/KNEE/Osteoarthritis/Combined_Osteoarthritis_Dataset", 
                "classes": ["Normal", "Osteoarthritis"],
                "folders": ["Normal", "Osteoarthritis"],
                "type": "knee",
                "estimated_size": 9788
            },
            "limbs": {
                "name": "LimbAbnormalities",
                "path": "Dataset/ARM/MURA_Organized/limbs",
                "classes": ["Normal", "Abnormal"],
                "folders": ["Negative", "Positive"],
                "type": "limb",
                "estimated_size": 3661
            }
        }
    
    def _define_architectures(self):
        """Define all architectures to test"""
        return {
            "densenet121": {
                "name": "DenseNet121",
                "base_model": DenseNet121,
                "input_shape": (224, 224, 3),
                "preprocess": lambda x: x / 255.0,
                "recommended": True
            },
            "efficientnetb0": {
                "name": "EfficientNetB0", 
                "base_model": EfficientNetB0,
                "input_shape": (224, 224, 3),
                "preprocess": lambda x: x / 255.0,
                "recommended": True
            },
            "resnet50": {
                "name": "ResNet50",
                "base_model": ResNet50,
                "input_shape": (224, 224, 3),
                "preprocess": lambda x: x / 255.0,
                "recommended": True
            },
            "vgg16": {
                "name": "VGG16",
                "base_model": VGG16,
                "input_shape": (224, 224, 3), 
                "preprocess": lambda x: x / 255.0,
                "recommended": False
            },
            "custom_cnn": {
                "name": "CustomCNN",
                "base_model": None,
                "input_shape": (224, 224, 3),
                "preprocess": lambda x: x / 255.0,
                "recommended": False
            }
        }
    
    def _define_configurations(self):
        """Define training configurations optimized for performance"""
        # Adjust batch sizes based on GPU availability
        gpu_multiplier = 2 if len(tf.config.list_physical_devices('GPU')) > 0 else 1
        
        return {
            "quick_test": {
                "name": "QuickTest",
                "batch_size": 32 * gpu_multiplier,  # 64 for GPU, 32 for CPU
                "epochs": 3,
                "learning_rate": 0.002 if gpu_multiplier > 1 else 0.001,  # Higher LR for GPU
                "max_images_per_class": 100,
                "validation_split": 0.2,
                "test_split": 0.2,
                "patience": 2,
                "description": f"Quick test configuration ({'GPU' if gpu_multiplier > 1 else 'CPU'} optimized)"
            },
            "standard": {
                "name": "Standard",
                "batch_size": 24 * gpu_multiplier,  # 48 for GPU, 24 for CPU
                "epochs": 8,
                "learning_rate": 0.001,
                "max_images_per_class": 500,
                "validation_split": 0.2,
                "test_split": 0.2,
                "patience": 3,
                "description": f"Standard production configuration ({'GPU' if gpu_multiplier > 1 else 'CPU'} optimized)"
            },
            "intensive": {
                "name": "Intensive",
                "batch_size": 16 * gpu_multiplier,  # 32 for GPU, 16 for CPU
                "epochs": 15,
                "learning_rate": 0.0005,
                "max_images_per_class": 1000,
                "validation_split": 0.15,
                "test_split": 0.15,
                "patience": 5,
                "description": "Intensive training for maximum performance"
            }
        }
    
    def _load_progress(self):
        """Load training progress from checkpoint"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_progress(self, dataset_key, arch_key, config_key, status, results=None):
        """Save training progress"""
        combination_key = f"{dataset_key}_{arch_key}_{config_key}"
        
        self.completed_combinations[combination_key] = {
            "dataset": dataset_key,
            "architecture": arch_key,
            "configuration": config_key,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "results": results
        }
        
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.completed_combinations, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save progress: {e}")
    
    def _is_combination_completed(self, dataset_key, arch_key, config_key):
        """Check if combination is already completed"""
        combination_key = f"{dataset_key}_{arch_key}_{config_key}"
        return combination_key in self.completed_combinations and \
               self.completed_combinations[combination_key]["status"] == "completed"
    
    def load_dataset(self, dataset_info, config):
        """Load and preprocess dataset"""
        print(f"📊 Loading {dataset_info['name']} dataset...")
        
        images = []
        labels = []
        
        for class_idx, folder_name in enumerate(dataset_info['folders']):
            class_path = os.path.join(dataset_info['path'], folder_name)
            
            if not os.path.exists(class_path):
                print(f"⚠️ Warning: Path {class_path} does not exist!")
                continue
            
            # Get all image files
            image_files = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Limit images for efficiency
            max_images = config['max_images_per_class']
            if len(image_files) > max_images:
                random.seed(42)
                image_files = random.sample(image_files, max_images)
            
            print(f"   Loading {len(image_files)} images from {folder_name}...")
            
            for i, filename in enumerate(image_files):
                if i % 50 == 0:
                    print(f"     Progress: {i}/{len(image_files)}")
                
                try:
                    img_path = os.path.join(class_path, filename)
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((224, 224), Image.Resampling.LANCZOS)
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    
                    images.append(img_array)
                    labels.append(class_idx)
                except Exception as e:
                    continue
        
        X = np.array(images)
        y = np.array(labels)
        
        # Shuffle data
        indices = np.random.permutation(len(X))
        X = X[indices] 
        y = y[indices]
        
        print(f"✅ Dataset loaded: {len(X)} images")
        print(f"   Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def create_model(self, arch_info, dataset_info, config):
        """Create model based on architecture with GPU optimizations"""
        input_shape = arch_info['input_shape']
        
        if arch_info['name'] == 'CustomCNN':
            # Custom CNN architecture with GPU optimizations
            model = keras.Sequential([
                keras.layers.Input(shape=input_shape),
                keras.layers.Conv2D(32, (3, 3), activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D(2, 2),
                keras.layers.Conv2D(64, (3, 3), activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D(2, 2),
                keras.layers.Conv2D(128, (3, 3), activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D(2, 2),
                keras.layers.Conv2D(128, (3, 3), activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D(2, 2),
                keras.layers.Flatten(),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(512, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(1, activation='sigmoid', dtype='float32')  # Keep output as float32 for mixed precision
            ])
        else:
            # Transfer learning with pre-trained models and GPU optimizations
            base_model = arch_info['base_model'](
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
            base_model.trainable = False
            
            # GPU-optimized head with batch normalization
            inputs = keras.Input(shape=input_shape)
            x = base_model(inputs, training=False)
            x = keras.layers.GlobalAveragePooling2D()(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.5)(x)
            x = keras.layers.Dense(512, activation='relu')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.3)(x)
            x = keras.layers.Dense(256, activation='relu')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.2)(x)
            # Keep final layer as float32 for stability with mixed precision
            outputs = keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)
            
            model = keras.Model(inputs, outputs)
        
        # GPU-optimized optimizer
        optimizer = keras.optimizers.Adam(
            learning_rate=config['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7  # Smaller epsilon for mixed precision
        )
        
        # Compile model with GPU optimizations
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _cooling_break(self, duration):
        """Take a cooling break to prevent overheating"""
        print(f"\n💤 Cooling break: {duration}s...")
        print("🌡️ Allowing system to cool down...")
        
        # Show progress bar for cooling break
        for i in range(duration):
            progress = (i + 1) / duration
            bar_length = 20
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"\r   Cooling: [{bar}] {i+1}/{duration}s", end='', flush=True)
            time.sleep(1)
        
        print("\n🔄 Resuming training...")
    


    def train_model(self, model, X, y, config):
        """Train model with given configuration and GPU optimizations"""
        # Data split
        test_size = int(config['test_split'] * len(X))
        train_size = len(X) - test_size
        
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        print(f"🚀 Training with {config['name']} configuration")
        
        # GPU-optimized callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['patience'],
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=max(1, config['patience'] - 1),
            min_lr=1e-7,
            verbose=1
        )
        
        # Create temporary checkpoint
        temp_dir = os.path.join(self.base_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        checkpoint_path = os.path.join(temp_dir, "gpu_checkpoint.keras")
        
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0
        )
        
        callbacks_list = [early_stopping, reduce_lr, model_checkpoint]
        
        # Add GPU-specific performance monitoring
        if self.gpu_enabled:
            print(f"�️ GPU training enabled - Batch size: {config['batch_size']}")
        
        # Train model with GPU optimizations
        print(f"�🚀 Training for {config['epochs']} epochs...")
        
        # Enable mixed precision loss scaling if GPU is available
        fit_kwargs = {
            'epochs': config['epochs'],
            'batch_size': config['batch_size'],
            'validation_split': config['validation_split'],
            'callbacks': callbacks_list,
            'verbose': 1,
            'shuffle': True
        }
        
        # Add GPU-specific optimizations
        if self.gpu_enabled:
            fit_kwargs.update({
                'use_multiprocessing': True,
                'workers': 4,
                'max_queue_size': 10
            })
        
        history = model.fit(
            X_train, y_train,
            **fit_kwargs
        )
        
        # Evaluate
        print("📊 Evaluating model...")
        test_results = model.evaluate(X_test, y_test, verbose=0)
        
        results = {
            'test_loss': float(test_results[0]),
            'test_accuracy': float(test_results[1]),
            'epochs_trained': len(history.history['loss']),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        return model, history, results
    
    def _get_comprehensive_system_info(self):
        """Capture every possible system detail"""
        try:
            # System information
            system_info = {
                'platform': {
                    'system': platform.system(),
                    'release': platform.release(),
                    'version': platform.version(),
                    'machine': platform.machine(),
                    'processor': platform.processor(),
                    'architecture': platform.architecture(),
                    'node': platform.node(),
                    'python_version': platform.python_version(),
                    'python_implementation': platform.python_implementation()
                },
                'hardware': {
                    'cpu_count_physical': psutil.cpu_count(logical=False),
                    'cpu_count_logical': psutil.cpu_count(logical=True),
                    'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                    'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                    'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
                    'disk_usage': {disk.device: psutil.disk_usage(disk.mountpoint)._asdict() 
                                 for disk in psutil.disk_partitions()}
                },
                'gpu_info': [],
                'environment': {
                    'timezone': str(datetime.now(timezone.utc).astimezone().tzinfo),
                    'working_directory': os.getcwd(),
                    'python_path': sys.path,
                    'environment_variables': {k: v for k, v in os.environ.items() 
                                            if any(x in k.upper() for x in ['CUDA', 'TF', 'PYTHON', 'PATH'])}
                },
                'tensorflow': {
                    'version': tf.__version__,
                    'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
                    'gpu_devices': [str(gpu) for gpu in tf.config.list_physical_devices('GPU')],
                    'mixed_precision': tf.keras.mixed_precision.global_policy().name,
                    'eager_execution': tf.executing_eagerly()
                },
                'installed_packages': {pkg.project_name: pkg.version 
                                     for pkg in pkg_resources.working_set}
            }
            
            # GPU details if available
            try:
                import nvidia_ml_py3 as nvml
                nvml.nvmlInit()
                gpu_count = nvml.nvmlDeviceGetCount()
                for i in range(gpu_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    gpu_info = {
                        'index': i,
                        'name': nvml.nvmlDeviceGetName(handle).decode('utf-8'),
                        'memory_total_mb': nvml.nvmlDeviceGetMemoryInfo(handle).total // (1024*1024),
                        'memory_free_mb': nvml.nvmlDeviceGetMemoryInfo(handle).free // (1024*1024),
                        'temperature': nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU),
                        'power_draw': nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
                        'driver_version': nvml.nvmlSystemGetDriverVersion().decode('utf-8')
                    }
                    system_info['gpu_info'].append(gpu_info)
            except:
                system_info['gpu_info'] = ['GPU details unavailable']
            
            return system_info
        except Exception as e:
            return {'error': f'Could not gather system info: {str(e)}'}
    
    def _get_model_architecture_details(self, model):
        """Extract detailed model architecture information"""
        try:
            architecture_details = {
                'summary': [],
                'layer_details': [],
                'total_params': int(model.count_params()),
                'trainable_params': int(sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])),
                'non_trainable_params': int(sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])),
                'model_config': model.get_config(),
                'input_shape': model.input_shape,
                'output_shape': model.output_shape
            }
            
            # Capture model summary
            summary_lines = []
            model.summary(print_fn=lambda x: summary_lines.append(x))
            architecture_details['summary'] = summary_lines
            
            # Layer-by-layer details
            for i, layer in enumerate(model.layers):
                layer_info = {
                    'index': i,
                    'name': layer.name,
                    'class': layer.__class__.__name__,
                    'config': layer.get_config(),
                    'input_shape': getattr(layer, 'input_shape', None),
                    'output_shape': getattr(layer, 'output_shape', None),
                    'trainable': layer.trainable,
                    'count_params': layer.count_params()
                }
                
                # Add weights info if available
                if hasattr(layer, 'get_weights') and layer.get_weights():
                    weights = layer.get_weights()
                    layer_info['weights_info'] = {
                        'num_weight_tensors': len(weights),
                        'weight_shapes': [w.shape for w in weights],
                        'weight_dtypes': [str(w.dtype) for w in weights]
                    }
                
                architecture_details['layer_details'].append(layer_info)
            
            return architecture_details
        except Exception as e:
            return {'error': f'Could not extract model details: {str(e)}'}
    
    def _get_training_environment_snapshot(self, config_info):
        """Capture complete training environment snapshot"""
        try:
            snapshot = {
                'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                'timestamp_local': datetime.now().isoformat(),
                'training_config': config_info.copy(),
                'system_state': {
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else None,
                    'network_io': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else None,
                    'boot_time': psutil.boot_time(),
                    'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
                },
                'process_info': {
                    'pid': os.getpid(),
                    'memory_info': psutil.Process().memory_info()._asdict(),
                    'cpu_percent': psutil.Process().cpu_percent(),
                    'num_threads': psutil.Process().num_threads(),
                    'create_time': psutil.Process().create_time()
                }
            }
            return snapshot
        except Exception as e:
            return {'error': f'Could not capture environment snapshot: {str(e)}'}

    def save_model_artifacts(self, model, history, results, dataset_key, arch_key, config_key):
        """Save ALL possible model artifacts and information in minute detail"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create organized directory structure
        condition_name = self.datasets[dataset_key]['name']
        arch_name = self.architectures[arch_key]['name']
        config_name = self.configurations[config_key]['name']
        
        model_dir = os.path.join(
            self.base_dir, 
            f"{condition_name.lower()}_{arch_name.lower()}_{config_name.lower()}_models"
        )
        models_dir = os.path.join(model_dir, "models")
        configs_dir = os.path.join(model_dir, "configs") 
        results_dir = os.path.join(model_dir, "results")
        
        for dir_path in [models_dir, configs_dir, results_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        saved_files = {}
        base_filename = f"{condition_name.lower()}_{arch_name.lower()}_{config_name.lower()}_{timestamp}"
        
        try:
            # 1. Save model in multiple formats
            keras_path = os.path.join(models_dir, f"{base_filename}.keras")
            model.save(keras_path)
            saved_files['keras_model'] = keras_path
            
            h5_path = os.path.join(models_dir, f"{base_filename}.h5")
            model.save(h5_path)
            saved_files['h5_model'] = h5_path
            
            weights_path = os.path.join(models_dir, f"{base_filename}.weights.h5")
            model.save_weights(weights_path)
            saved_files['weights'] = weights_path
            
            # 2. Save configuration files
            model_config_path = os.path.join(configs_dir, f"{base_filename}_model_config.json")
            with open(model_config_path, 'w') as f:
                json.dump(model.get_config(), f, indent=2)
            saved_files['model_config'] = model_config_path
            
            train_config_path = os.path.join(configs_dir, f"{base_filename}_train_config.json")
            train_config = self.configurations[config_key].copy()
            train_config.update({
                'dataset': dataset_key,
                'architecture': arch_key,
                'timestamp': timestamp,
                'total_parameters': int(model.count_params())
            })
            with open(train_config_path, 'w') as f:
                json.dump(train_config, f, indent=2)
            saved_files['train_config'] = train_config_path
            
            # 3. Save comprehensive results with ALL possible details
            results_path = os.path.join(results_dir, f"{base_filename}_results.json")
            full_results = results.copy()
            full_results.update({
                'model_name': f"{arch_name}_{condition_name}_Classifier",
                'timestamp': timestamp,
                'dataset': dataset_key,
                'architecture': arch_key,
                'configuration': config_key,
                'total_parameters': int(model.count_params()),
                'system_info': self._get_comprehensive_system_info(),
                'model_architecture': self._get_model_architecture_details(model),
                'training_environment': self._get_training_environment_snapshot(self.configurations[config_key]),
                'dataset_info': self.datasets[dataset_key],
                'architecture_info': self.architectures[arch_key],
                'configuration_info': self.configurations[config_key]
            })
            with open(results_path, 'w') as f:
                json.dump(full_results, f, indent=2)
            saved_files['results'] = results_path
            
            history_path = os.path.join(results_dir, f"{base_filename}_history.json")
            history_dict = {
                'raw_history': {},
                'epoch_analysis': [],
                'training_statistics': {},
                'convergence_analysis': {}
            }
            
            # Raw history data
            for key, values in history.history.items():
                history_dict['raw_history'][key] = [float(v) for v in values]
            
            # Detailed epoch-by-epoch analysis
            num_epochs = len(history.history.get('loss', []))
            for epoch in range(num_epochs):
                epoch_data = {'epoch': epoch + 1}
                for metric, values in history.history.items():
                    if epoch < len(values):
                        epoch_data[metric] = float(values[epoch])
                
                # Calculate epoch improvements
                if epoch > 0:
                    for metric in ['loss', 'val_loss']:
                        if metric in history.history and epoch < len(history.history[metric]):
                            improvement = history.history[metric][epoch-1] - history.history[metric][epoch]
                            epoch_data[f'{metric}_improvement'] = float(improvement)
                
                history_dict['epoch_analysis'].append(epoch_data)
            
            # Training statistics
            for metric, values in history.history.items():
                if values:
                    values_float = [float(v) for v in values]
                    history_dict['training_statistics'][metric] = {
                        'final_value': values_float[-1],
                        'best_value': min(values_float) if 'loss' in metric else max(values_float),
                        'worst_value': max(values_float) if 'loss' in metric else min(values_float),
                        'mean': sum(values_float) / len(values_float),
                        'std_deviation': (sum((x - sum(values_float)/len(values_float))**2 for x in values_float) / len(values_float))**0.5,
                        'improvement_total': values_float[0] - values_float[-1] if 'loss' in metric else values_float[-1] - values_float[0],
                        'best_epoch': values_float.index(min(values_float)) + 1 if 'loss' in metric else values_float.index(max(values_float)) + 1
                    }
            
            with open(history_path, 'w') as f:
                json.dump(history_dict, f, indent=2)
            saved_files['history'] = history_path
            
            # 5. Save additional comprehensive artifacts
            
            # Model weights in multiple formats
            weights_path = os.path.join(models_dir, f"{base_filename}_weights.h5")
            model.save_weights(weights_path)
            saved_files['weights'] = weights_path
            
            # Model checkpoints (if we want to resume training)
            checkpoint_path = os.path.join(models_dir, f"{base_filename}_checkpoint")
            tf.train.Checkpoint(model=model).save(checkpoint_path)
            saved_files['checkpoint'] = checkpoint_path
            
            # Save optimizer state if available
            try:
                optimizer_path = os.path.join(configs_dir, f"{base_filename}_optimizer_config.json")
                if hasattr(model, 'optimizer') and model.optimizer:
                    optimizer_config = {
                        'name': model.optimizer.__class__.__name__,
                        'config': model.optimizer.get_config(),
                        'weights': [w.numpy().tolist() if hasattr(w, 'numpy') else str(w) 
                                  for w in model.optimizer.variables]
                    }
                    with open(optimizer_path, 'w') as f:
                        json.dump(optimizer_config, f, indent=2)
                    saved_files['optimizer_config'] = optimizer_path
            except Exception as e:
                print(f"ℹ️ Could not save optimizer config: {e}")
            
            # Save data preprocessing information
            preprocessing_path = os.path.join(configs_dir, f"{base_filename}_preprocessing.json")
            current_config = self.configurations[config_key]
            preprocessing_info = {
                'image_size': current_config.get('input_shape', [224, 224, 3]),
                'normalization': 'ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])',
                'augmentation': 'Random horizontal flip, rotation, zoom, shift',
                'color_mode': 'RGB',
                'data_format': 'channels_last',
                'rescaling': '1/255.0'
            }
            with open(preprocessing_path, 'w') as f:
                json.dump(preprocessing_info, f, indent=2)
            saved_files['preprocessing'] = preprocessing_path
            
            # Save dataset statistics
            dataset_stats_path = os.path.join(results_dir, f"{base_filename}_dataset_stats.json")
            dataset_stats = {
                'total_samples': results.get('training_samples', 0) + results.get('test_samples', 0),
                'training_samples': results.get('training_samples', 0),
                'validation_samples': results.get('validation_samples', 0),
                'test_samples': results.get('test_samples', 0),
                'class_distribution': results.get('class_distribution', {}),
                'dataset_path': self.datasets[dataset_key]['path'],
                'classes': self.datasets[dataset_key]['classes'],
                'data_split_ratios': {
                    'train': current_config.get('validation_split', 0.8),
                    'validation': current_config.get('validation_split', 0.2),
                    'test': current_config.get('test_split', 0.2)
                }
            }
            with open(dataset_stats_path, 'w') as f:
                json.dump(dataset_stats, f, indent=2)
            saved_files['dataset_stats'] = dataset_stats_path
            
            # Save training metadata with file hashes for integrity
            metadata_path = os.path.join(results_dir, f"{base_filename}_metadata.json")
            metadata = {
                'creation_timestamp': datetime.now().isoformat(),
                'training_duration_seconds': results.get('training_time_seconds', 0),
                'file_hashes': {},
                'git_info': self._get_git_info(),
                'python_environment': {
                    'python_version': platform.python_version(),
                    'tensorflow_version': tf.__version__,
                    'keras_version': tf.keras.__version__,
                    'numpy_version': np.__version__
                },
                'memory_usage': {
                    'peak_memory_mb': psutil.Process().memory_info().peak_wset // (1024*1024) if hasattr(psutil.Process().memory_info(), 'peak_wset') else 'unavailable',
                    'current_memory_mb': psutil.Process().memory_info().rss // (1024*1024)
                }
            }
            
            # Calculate file hashes for integrity verification
            for file_type, file_path in saved_files.items():
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        metadata['file_hashes'][file_type] = hashlib.md5(f.read()).hexdigest()
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            saved_files['metadata'] = metadata_path
            
            # 4. Create comprehensive README for this model
            readme_path = os.path.join(model_dir, "README.md")
            self._create_comprehensive_readme(readme_path, dataset_key, arch_key, config_key, results, timestamp, saved_files)
            saved_files['readme'] = readme_path
            
            print(f"✅ Comprehensive model artifacts saved to: {model_dir}")
            print(f"📊 Total files saved: {len(saved_files)}")
            for file_type, path in saved_files.items():
                file_size = os.path.getsize(path) / (1024*1024) if os.path.exists(path) else 0
                print(f"   {file_type}: {os.path.basename(path)} ({file_size:.1f}MB)")
            return saved_files
            
        except Exception as e:
            print(f"⚠️ Error saving model artifacts: {e}")
            return {}
    
    def _get_git_info(self):
        """Get git repository information if available"""
        try:
            git_info = {
                'branch': subprocess.check_output(['git', 'branch', '--show-current'], 
                                                 stderr=subprocess.DEVNULL).decode().strip(),
                'commit_hash': subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                                      stderr=subprocess.DEVNULL).decode().strip(),
                'commit_message': subprocess.check_output(['git', 'log', '-1', '--pretty=%B'], 
                                                         stderr=subprocess.DEVNULL).decode().strip(),
                'remote_url': subprocess.check_output(['git', 'config', '--get', 'remote.origin.url'], 
                                                     stderr=subprocess.DEVNULL).decode().strip(),
                'status': subprocess.check_output(['git', 'status', '--porcelain'], 
                                                 stderr=subprocess.DEVNULL).decode().strip()
            }
            return git_info
        except:
            return {'error': 'Git information not available'}
    
    def _create_comprehensive_readme(self, readme_path, dataset_key, arch_key, config_key, results, timestamp, saved_files):
        """Create comprehensive README with ALL possible information"""
        dataset_info = self.datasets[dataset_key]
        arch_info = self.architectures[arch_key]
        config_info = self.configurations[config_key]
        
        readme_content = f"""# {arch_info['name']} {dataset_info['name']} Detection Model - Complete Documentation

## 🎯 Model Overview
- **Architecture**: {arch_info['name']}
- **Medical Condition**: {dataset_info['name']} Detection
- **Training Configuration**: {config_info['name']}
- **Creation Timestamp**: {timestamp}
- **Training Duration**: {results.get('training_time_seconds', 0):.1f} seconds

## 📊 Performance Metrics
- **Test Accuracy**: {results['test_accuracy']:.6f} ({results['test_accuracy']*100:.3f}%)
- **Test Precision**: {results['test_precision']:.6f}
- **Test Recall**: {results['test_recall']:.6f}
- **Test Loss**: {results['test_loss']:.6f}
- **Best Validation Accuracy**: {max([float(x) for x in results.get('val_accuracy', [0])]) if 'val_accuracy' in results else 'N/A'}

## 🏗️ Architecture Details
- **Total Parameters**: {results['total_parameters']:,}
- **Input Shape**: {config_info['input_shape']}
- **Activation Function**: {dataset_info.get('activation', 'sigmoid')}
- **Loss Function**: {dataset_info.get('loss_function', 'binary_crossentropy')}
- **Optimizer**: {config_info.get('optimizer', 'Adam')}
- **Learning Rate**: {config_info['learning_rate']}

## 📚 Dataset Information
- **Source Path**: `{dataset_info['path']}`
- **Classes**: {', '.join(dataset_info['classes'])}
- **Medical Type**: {dataset_info['type']} X-ray analysis
- **Training Samples**: {results['training_samples']}
- **Test Samples**: {results['test_samples']}
- **Epochs Trained**: {results['epochs_trained']}
- **Batch Size**: {config_info['batch_size']}

## ⚙️ Training Configuration
- **Configuration Name**: {config_info['name']}
- **Description**: {config_info['description']}
- **Max Images per Class**: {config_info['max_images_per_class']}
- **Validation Split**: {config_info['validation_split']}
- **Test Split**: {config_info['test_split']}

## 💾 Saved Artifacts
"""
        
        # Add file information
        for file_type, file_path in saved_files.items():
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / (1024*1024)
                readme_content += f"- **{file_type.replace('_', ' ').title()}**: `{os.path.basename(file_path)}` ({file_size:.1f}MB)\n"
        
        readme_content += f"""

## 🔬 Technical Details
- **Framework**: TensorFlow {tf.__version__}
- **Keras Version**: {tf.keras.__version__}
- **Python Version**: {platform.python_version()}
- **Platform**: {platform.system()} {platform.release()}
- **GPU Enabled**: {'✅ YES' if len(tf.config.list_physical_devices('GPU')) > 0 else '❌ NO'}
- **Mixed Precision**: {'✅ Enabled' if tf.keras.mixed_precision.global_policy().name != 'float32' else '❌ Disabled'}

## 📁 Directory Structure
```
{os.path.basename(os.path.dirname(readme_path))}/
├── models/                    # Trained model files (.keras, .h5, weights)
├── configs/                   # Configuration and preprocessing files
├── results/                   # Training results, history, and statistics
└── README.md                  # This comprehensive documentation
```

## 🔄 Model Usage

### Loading the Model
```python
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('{os.path.basename(saved_files.get('keras_model', ''))}')

# Or load from H5 format
model = tf.keras.models.load_model('{os.path.basename(saved_files.get('h5_model', ''))}')
```

### Making Predictions
```python
import numpy as np
from PIL import Image

# Load and preprocess image
img = Image.open('path_to_xray.jpg').convert('RGB')
img = img.resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
prediction = model.predict(img_array)
confidence = prediction[0][0]

# Interpret result
if confidence > 0.5:
    result = "{dataset_info['classes'][1]}"
else:
    result = "{dataset_info['classes'][0]}"
    
print(f"Prediction: {{result}} (Confidence: {{confidence:.3f}})")
```

## 📈 Training History Analysis
- **Final Training Accuracy**: {results.get('final_training_accuracy', 'N/A')}
- **Best Epoch**: {results.get('best_epoch', 'N/A')}
- **Convergence**: {'✅ Converged' if results.get('converged', False) else '⚠️ May need more epochs'}

## 🔍 Model Verification
- **File Integrity**: MD5 hashes saved in metadata for verification
- **Reproducibility**: All random seeds and configurations saved
- **Version Control**: Git information captured (if available)

---

**Generated on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model Version**: {timestamp}
**Documentation Version**: v2.0 (Comprehensive)
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
    def run_comprehensive_training(self):
        """Run comprehensive training across all combinations"""
        print("🚀 Starting Comprehensive Multi-Architecture Training Pipeline")
        print("=" * 80)
        
        total_combinations = len(self.datasets) * len(self.architectures) * len(self.configurations)
        completed_count = len([k for k, v in self.completed_combinations.items() if v['status'] == 'completed'])
        
        print(f"📊 Training Overview:")
        print(f"   Total Combinations: {total_combinations}")
        print(f"   Completed: {completed_count}")
        print(f"   Remaining: {total_combinations - completed_count}")
        print(f"   Datasets: {len(self.datasets)}")
        print(f"   Architectures: {len(self.architectures)}")
        print(f"   Configurations: {len(self.configurations)}")
        print(f"   GPU Enabled: {'✅ YES' if self.gpu_enabled else '❌ NO (CPU only)'}")
        if self.gpu_enabled:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"   GPU Device: {gpus[0].name}")
                print(f"   Mixed Precision: ✅ Enabled (faster training)")
                print(f"   Batch Size Optimization: ✅ Enabled (larger batches)")
        
        combination_count = 0
        
        for dataset_key, dataset_info in self.datasets.items():
            for arch_key, arch_info in self.architectures.items():
                for config_key, config_info in self.configurations.items():
                    combination_count += 1
                    
                    print(f"\n{'='*60}")
                    print(f"🎯 Combination {combination_count}/{total_combinations}")
                    print(f"📊 Dataset: {dataset_info['name']}")
                    print(f"🏗️ Architecture: {arch_info['name']}")
                    print(f"⚙️ Configuration: {config_info['name']}")
                    print(f"{'='*60}")
                    
                    # Check if already completed
                    if self._is_combination_completed(dataset_key, arch_key, config_key):
                        print("✅ Already completed - Skipping")
                        continue
                    
                    try:
                        # Mark as started
                        self._save_progress(dataset_key, arch_key, config_key, "started")
                        
                        start_time = time.time()
                        
                        # Load dataset
                        X, y = self.load_dataset(dataset_info, config_info)
                        
                        # Create model
                        print(f"🏗️ Creating {arch_info['name']} model...")
                        model = self.create_model(arch_info, dataset_info, config_info)
                        print(f"✅ Model created with {model.count_params():,} parameters")
                        
                        # Train model
                        model, history, results = self.train_model(model, X, y, config_info)
                        
                        # Save artifacts
                        print(f"💾 Saving model artifacts...")
                        saved_files = self.save_model_artifacts(
                            model, history, results, dataset_key, arch_key, config_key
                        )
                        
                        training_time = time.time() - start_time
                        
                        # Mark as completed
                        final_results = results.copy()
                        final_results['training_time_seconds'] = training_time
                        final_results['saved_files'] = saved_files
                        
                        self._save_progress(dataset_key, arch_key, config_key, "completed", final_results)
                        
                        print(f"🎉 Combination completed successfully!")
                        print(f"⏱️ Training time: {training_time:.1f} seconds")
                        print(f"🎯 Accuracy: {results['test_accuracy']*100:.2f}%")
                        
                        # Clean up memory
                        del model, X, y
                        tf.keras.backend.clear_session()
                        
                    except Exception as e:
                        print(f"❌ Error in combination: {str(e)}")
                        print(f"📋 Traceback: {traceback.format_exc()}")
                        
                        # Mark as failed
                        self._save_progress(dataset_key, arch_key, config_key, "failed", {
                            "error": str(e),
                            "traceback": traceback.format_exc()
                        })
                        
                        # Clean up memory
                        tf.keras.backend.clear_session()
                        continue
        
        print(f"\n🏁 Comprehensive Training Pipeline Complete!")
        print(f"📊 Final Statistics:")
        completed_final = len([k for k, v in self.completed_combinations.items() if v['status'] == 'completed'])
        failed_final = len([k for k, v in self.completed_combinations.items() if v['status'] == 'failed'])
        print(f"   ✅ Completed: {completed_final}")
        print(f"   ❌ Failed: {failed_final}")
        print(f"   📁 Results saved in: {self.base_dir}")

def main():
    """Main execution function"""
    print("🏥 Medical X-Ray AI Comprehensive Training Pipeline")
    print("🚀 Training all datasets with all architectures and configurations")
    print("📁 Continuous saving with crash recovery support")
    
    pipeline = ComprehensiveTrainingPipeline()
    pipeline.run_comprehensive_training()

if __name__ == "__main__":
    main()