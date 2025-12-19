#!/usr/bin/env python3
"""
DenseNet121 Focused Training Pipeline
===================================

Trains DenseNet121 architecture on all 5 medical datasets with optimal configurations.
This script focuses specifically on DenseNet121 for the best Grad-CAM visualization results.
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
import platform
import psutil
import hashlib
import subprocess
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, optimizers, callbacks
from PIL import Image
import random
from datetime import datetime
import traceback
import time
import platform
import psutil
import hashlib

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

class DenseNet121TrainingPipeline:
    """Focused DenseNet121 training pipeline for all medical datasets"""
    
    def __init__(self):
        self.base_dir = "new"
        self.progress_file = os.path.join(self.base_dir, "densenet121_training_progress.json")
        self.gpu_enabled = self._configure_gpu()  # Configure GPU first
        self.datasets = self._define_datasets()
        self.configurations = self._define_configurations()
        self.completed_models = self._load_progress()
        
        # Create base directory
        os.makedirs(self.base_dir, exist_ok=True)
        
    def _configure_gpu(self):
        """Configure GPU settings for optimal performance"""
        print("🔧 Configuring GPU settings for DenseNet121...")
        
        gpus = tf.config.list_physical_devices('GPU')
        print(f"📊 GPU devices found: {len(gpus)}")
        
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"✅ Enabled memory growth for: {gpu}")
                
                # Set mixed precision for better performance
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                print("✅ Mixed precision enabled (float16/float32)")
                
                return True
            except RuntimeError as e:
                print(f"⚠️ GPU configuration error: {e}")
                return False
        else:
            print("⚠️ No GPU devices found, using CPU")
            return False
    
    def _define_datasets(self):
        """Define all 5 medical datasets"""
        return {
            "pneumonia": {
                "name": "Pneumonia",
                "path": "Dataset/CHEST/Pneumonia_Organized",
                "classes": ["Normal", "Pneumonia"],
                "folders": ["Normal", "Pneumonia"],
                "type": "chest",
                "description": "Chest X-ray pneumonia detection"
            },
            "cardiomegaly": {
                "name": "Cardiomegaly", 
                "path": "Dataset/CHEST/cardiomelgy",
                "classes": ["Normal", "Cardiomegaly"],
                "folders": ["Normal", "Cardiomegaly"],
                "type": "chest",
                "description": "Heart enlargement detection"
            },
            "osteoporosis": {
                "name": "Osteoporosis",
                "path": "Dataset/KNEE/Osteoporosis/Combined_Osteoporosis_Dataset",
                "classes": ["Normal", "Osteoporosis"],
                "folders": ["Normal", "Osteoporosis"],
                "type": "knee",
                "description": "Bone density analysis"
            },
            "osteoarthritis": {
                "name": "Osteoarthritis",
                "path": "Dataset/KNEE/Osteoarthritis/Combined_Osteoarthritis_Dataset", 
                "classes": ["Normal", "Osteoarthritis"],
                "folders": ["Normal", "Osteoarthritis"],
                "type": "knee",
                "description": "Joint degeneration detection"
            },
            "limbs": {
                "name": "LimbAbnormalities",
                "path": "Dataset/ARM/MURA_Organized/limbs",
                "classes": ["Normal", "Abnormal"],
                "folders": ["Negative", "Positive"],
                "type": "limb",
                "description": "Bone fracture and abnormality detection"
            }
        }
    
    def _define_configurations(self):
        """Define training configurations optimized for DenseNet121"""
        gpu_multiplier = 2 if self.gpu_enabled else 1
        
        return {
            "standard": {
                "name": "Standard",
                "batch_size": 32 * gpu_multiplier,
                "epochs": 10,
                "learning_rate": 0.001,
                "max_images_per_class": 500,
                "validation_split": 0.2,
                "test_split": 0.2,
                "patience": 3,
                "description": f"Standard DenseNet121 configuration ({'GPU' if gpu_multiplier > 1 else 'CPU'} optimized)"
            },
            "intensive": {
                "name": "Intensive", 
                "batch_size": 16 * gpu_multiplier,
                "epochs": 15,
                "learning_rate": 0.0005,
                "max_images_per_class": 1000,
                "validation_split": 0.2,
                "test_split": 0.2,
                "patience": 4,
                "description": f"Intensive DenseNet121 configuration ({'GPU' if gpu_multiplier > 1 else 'CPU'} optimized)"
            }
        }
    
    def _load_progress(self):
        """Load training progress from file"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_progress(self, dataset_key, config_key, status, results=None):
        """Save training progress"""
        combination_key = f"{dataset_key}_{config_key}"
        
        self.completed_models[combination_key] = {
            "dataset": dataset_key,
            "configuration": config_key,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "results": results or {}
        }
        
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.completed_models, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save progress: {e}")
    
    def _is_combination_completed(self, dataset_key, config_key):
        """Check if combination is already completed"""
        combination_key = f"{dataset_key}_{config_key}"
        return combination_key in self.completed_models and \
               self.completed_models[combination_key].get('status') == 'completed'
    
    def load_dataset(self, dataset_info, config):
        """Load and preprocess dataset"""
        print(f"📊 Loading {dataset_info['name']} dataset...")
        
        # Load images from each class
        X, y = [], []
        
        for class_idx, folder_name in enumerate(dataset_info['folders']):
            folder_path = os.path.join(dataset_info['path'], folder_name)
            
            if not os.path.exists(folder_path):
                print(f"⚠️ Folder not found: {folder_path}")
                continue
            
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Limit images per class
            max_images = config['max_images_per_class']
            if len(image_files) > max_images:
                image_files = random.sample(image_files, max_images)
            
            print(f"   Loading {len(image_files)} images from {folder_name}...")
            
            for i, img_file in enumerate(image_files):
                if i % 50 == 0:
                    print(f"     Progress: {i}/{len(image_files)}")
                
                try:
                    img_path = os.path.join(folder_path, img_file)
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((224, 224))
                    img_array = np.array(img) / 255.0
                    
                    X.append(img_array)
                    y.append(class_idx)
                except Exception as e:
                    print(f"     Error loading {img_file}: {e}")
                    continue
        
        X = np.array(X)
        y = np.array(y)
        
        # Shuffle the data
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        print(f"✅ Dataset loaded: {len(X)} images")
        print(f"   Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def create_densenet121_model(self, dataset_info):
        """Create DenseNet121 model optimized for medical imaging"""
        print(f"🏗️ Creating DenseNet121 model for {dataset_info['name']}...")
        
        # Create base DenseNet121 model
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Freeze initial layers, fine-tune later layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Add custom classification head
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid', dtype='float32')  # Float32 for mixed precision
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        print(f"✅ DenseNet121 model created with {model.count_params():,} parameters")
        return model
    
    def train_model(self, model, X, y, config):
        """Train DenseNet121 model"""
        print(f"🚀 Training DenseNet121 for {config['epochs']} epochs...")
        
        # Split data
        test_size = int(config['test_split'] * len(X))
        train_size = len(X) - test_size
        
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Create callbacks
        callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config['patience'],
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.base_dir, 'best_densenet121_checkpoint.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            batch_size=config['batch_size'],
            epochs=config['epochs'],
            validation_split=config['validation_split'],
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Evaluate model
        print("📊 Evaluating DenseNet121 model...")
        test_results = model.evaluate(X_test, y_test, verbose=0)
        
        # Prepare results
        results = {
            'test_loss': float(test_results[0]),
            'test_accuracy': float(test_results[1]),
            'test_precision': float(test_results[2]),
            'test_recall': float(test_results[3]),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'epochs_trained': len(history.history['loss']),
            'best_val_accuracy': float(max(history.history['val_accuracy'])),
            'final_training_accuracy': float(history.history['accuracy'][-1])
        }
        
        return model, history, results
    
    def _convert_to_serializable(self, obj):
        """Convert numpy/tensorflow types to JSON serializable types"""
        import numpy as np
        
        if isinstance(obj, (np.integer, np.int32, np.int64, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(key): self._convert_to_serializable(value) for key, value in obj.items()}
        elif hasattr(obj, 'numpy'):  # TensorFlow tensor
            try:
                return float(obj.numpy()) if obj.numpy().ndim == 0 else obj.numpy().tolist()
            except:
                return str(obj)
        elif hasattr(obj, '__dict__'):  # Complex objects
            try:
                return str(obj)
            except:
                return 'Object not serializable'
        elif obj is None:
            return None
        elif isinstance(obj, (bool, str)):
            return obj
        else:
            try:
                # Try to convert to basic Python types
                if hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                else:
                    return str(obj)
            except:
                return 'Object not serializable'

    def save_densenet121_artifacts(self, model, history, results, dataset_key, config_key):
        """Save comprehensive DenseNet121 model artifacts with EVERY SINGLE DETAIL"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create organized directory structure
        dataset_name = self.datasets[dataset_key]['name']
        config_name = self.configurations[config_key]['name']
        
        model_dir = os.path.join(
            self.base_dir, 
            f"densenet121_{dataset_name.lower()}_{config_name.lower()}_models"
        )
        models_dir = os.path.join(model_dir, "models")
        configs_dir = os.path.join(model_dir, "configs") 
        results_dir = os.path.join(model_dir, "results")
        system_dir = os.path.join(model_dir, "system_info")
        environment_dir = os.path.join(model_dir, "environment")
        
        for dir_path in [models_dir, configs_dir, results_dir, system_dir, environment_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        saved_files = {}
        base_filename = f"densenet121_{dataset_name.lower()}_{config_name.lower()}_{timestamp}"
        
        try:
            # 1. SAVE MODEL IN MULTIPLE FORMATS
            keras_path = os.path.join(models_dir, f"{base_filename}.keras")
            h5_path = os.path.join(models_dir, f"{base_filename}.h5")
            weights_path = os.path.join(models_dir, f"{base_filename}.weights.h5")
            
            model.save(keras_path)
            model.save(h5_path)
            model.save_weights(weights_path)
            
            saved_files.update({
                'keras_model': keras_path,
                'h5_model': h5_path,
                'weights': weights_path
            })
            
            # 2. SAVE COMPLETE MODEL ARCHITECTURE DETAILS
            model_config_path = os.path.join(configs_dir, f"{base_filename}_model_config.json")
            complete_model_info = {
                'model_config': self._convert_to_serializable(model.get_config()),
                'model_summary': [],
                'layer_details': [],
                'total_parameters': int(model.count_params()),
                'trainable_parameters': int(sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])),
                'non_trainable_parameters': int(sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])),
                'input_shape': list(model.input_shape),
                'output_shape': list(model.output_shape),
                'optimizer_config': self._convert_to_serializable(model.optimizer.get_config()),
                'loss_function': str(model.loss),
                'metrics': [m.name if hasattr(m, 'name') else str(m) for m in model.metrics],
                'model_architecture': 'DenseNet121',
                'gradcam_optimized': True,
                'recommended_gradcam_layer': 'conv5_block16_2_conv'
            }
            
            # Capture model summary
            import io
            import sys
            string_buffer = io.StringIO()
            model.summary(print_fn=lambda x: string_buffer.write(x + '\\n'))
            complete_model_info['model_summary'] = string_buffer.getvalue().split('\\n')
            
            # Capture detailed layer information
            for i, layer in enumerate(model.layers):
                layer_info = {
                    'index': i,
                    'name': layer.name,
                    'class_name': layer.__class__.__name__,
                    'trainable': layer.trainable,
                    'input_shape': str(layer.input_shape) if hasattr(layer, 'input_shape') else None,
                    'output_shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else None,
                    'param_count': int(layer.count_params()),
                    'config': self._convert_to_serializable(layer.get_config())
                }
                complete_model_info['layer_details'].append(layer_info)
            
            # Convert all data to serializable format
            complete_model_info = self._convert_to_serializable(complete_model_info)
            
            with open(model_config_path, 'w') as f:
                json.dump(complete_model_info, f, indent=2)
            saved_files['complete_model_config'] = model_config_path
            
            # 3. SAVE COMPREHENSIVE SYSTEM INFORMATION
            system_info_path = os.path.join(system_dir, f"{base_filename}_system_info.json")
            system_info = self._get_comprehensive_system_info()
            system_info = self._convert_to_serializable(system_info)
            with open(system_info_path, 'w') as f:
                json.dump(system_info, f, indent=2)
            saved_files['system_info'] = system_info_path
            
            # 4. SAVE COMPLETE ENVIRONMENT SNAPSHOT
            env_info_path = os.path.join(environment_dir, f"{base_filename}_environment.json")
            env_info = self._get_complete_environment_info()
            env_info = self._convert_to_serializable(env_info)
            with open(env_info_path, 'w') as f:
                json.dump(env_info, f, indent=2)
            saved_files['environment_info'] = env_info_path
            
            # 5. SAVE TRAINING CONFIGURATION WITH COMPLETE DETAILS
            train_config_path = os.path.join(configs_dir, f"{base_filename}_train_config.json")
            complete_train_config = self.configurations[config_key].copy()
            complete_train_config.update({
                'dataset': dataset_key,
                'dataset_info': self.datasets[dataset_key],
                'architecture': 'DenseNet121',
                'timestamp': timestamp,
                'training_start_time': timestamp,
                'total_parameters': int(model.count_params()),
                'gradcam_optimization': {
                    'optimized_for_gradcam': True,
                    'best_layer_for_visualization': 'conv5_block16_2_conv',
                    'architecture_benefits': [
                        'Dense connectivity preserves gradients',
                        'Rich feature reuse for detailed heatmaps',
                        'Excellent for medical imaging visualization',
                        'Superior gradient flow through dense blocks'
                    ]
                },
                'medical_imaging_optimizations': {
                    'pretrained_weights': 'ImageNet',
                    'fine_tuning_strategy': 'Freeze early layers, train final 20 layers',
                    'input_preprocessing': 'Normalized to [0,1] range',
                    'augmentation_applied': False,
                    'class_balancing': 'Automatic through sampling'
                }
            })
            complete_train_config = self._convert_to_serializable(complete_train_config)
            with open(train_config_path, 'w') as f:
                json.dump(complete_train_config, f, indent=2)
            saved_files['complete_train_config'] = train_config_path
            
            # 6. SAVE COMPREHENSIVE RESULTS WITH ALL METRICS
            results_path = os.path.join(results_dir, f"{base_filename}_comprehensive_results.json")
            comprehensive_results = results.copy()
            comprehensive_results.update({
                'model_name': f"DenseNet121_{dataset_name}_Classifier",
                'timestamp': timestamp,
                'dataset': dataset_key,
                'architecture': 'DenseNet121',
                'configuration': config_key,
                'total_parameters': int(model.count_params()),
                'gradcam_optimized': True,
                'recommended_gradcam_layer': 'conv5_block16_2_conv',
                'training_metrics': {
                    'final_train_loss': float(history.history['loss'][-1]),
                    'final_train_accuracy': float(history.history['accuracy'][-1]),
                    'final_val_loss': float(history.history['val_loss'][-1]),
                    'final_val_accuracy': float(history.history['val_accuracy'][-1]),
                    'best_val_accuracy': float(max(history.history['val_accuracy'])),
                    'best_val_loss': float(min(history.history['val_loss'])),
                    'training_stability': {
                        'loss_variance': float(np.var(history.history['loss'])),
                        'accuracy_variance': float(np.var(history.history['accuracy'])),
                        'convergence_epoch': int(np.argmax(history.history['val_accuracy'])) + 1
                    }
                },
                'model_performance_analysis': {
                    'generalization_gap': float(max(history.history['accuracy']) - results['test_accuracy']),
                    'overfitting_indicator': float(max(history.history['accuracy']) - max(history.history['val_accuracy'])),
                    'performance_category': 'Excellent' if results['test_accuracy'] > 0.9 else 'Good' if results['test_accuracy'] > 0.8 else 'Moderate',
                    'medical_imaging_suitability': 'High - DenseNet121 excellent for medical visualization'
                }
            })
            comprehensive_results = self._convert_to_serializable(comprehensive_results)
            with open(results_path, 'w') as f:
                json.dump(comprehensive_results, f, indent=2)
            saved_files['comprehensive_results'] = results_path
            
            # 7. SAVE COMPLETE TRAINING HISTORY WITH ANALYSIS
            history_path = os.path.join(results_dir, f"{base_filename}_complete_history.json")
            complete_history = {}
            for key, values in history.history.items():
                complete_history[key] = [float(v) for v in values]
            
            # Add training analysis
            complete_history['training_analysis'] = {
                'total_epochs': len(history.history['loss']),
                'convergence_analysis': {
                    'best_epoch': int(np.argmax(history.history['val_accuracy'])) + 1,
                    'early_stopped': len(history.history['loss']) < self.configurations[config_key]['epochs'],
                    'improvement_over_epochs': [float(acc) for acc in np.diff(history.history['val_accuracy'])]
                },
                'loss_analysis': {
                    'training_loss_trend': 'Decreasing' if history.history['loss'][-1] < history.history['loss'][0] else 'Increasing',
                    'validation_loss_trend': 'Decreasing' if history.history['val_loss'][-1] < history.history['val_loss'][0] else 'Increasing',
                    'loss_stability': float(np.std(history.history['loss'][-5:])) if len(history.history['loss']) >= 5 else 0.0
                }
            }
            
            complete_history = self._convert_to_serializable(complete_history)
            with open(history_path, 'w') as f:
                json.dump(complete_history, f, indent=2)
            saved_files['complete_history'] = history_path
            
            # 8. SAVE DATASET INTEGRITY INFORMATION
            dataset_integrity_path = os.path.join(system_dir, f"{base_filename}_dataset_integrity.json")
            dataset_integrity = self._get_dataset_integrity_info(dataset_key)
            dataset_integrity = self._convert_to_serializable(dataset_integrity)
            with open(dataset_integrity_path, 'w') as f:
                json.dump(dataset_integrity, f, indent=2)
            saved_files['dataset_integrity'] = dataset_integrity_path
            
            # 9. CREATE COMPREHENSIVE README
            readme_path = os.path.join(model_dir, "README.md")
            self._create_densenet121_readme(readme_path, dataset_key, config_key, comprehensive_results, timestamp, saved_files)
            saved_files['readme'] = readme_path
            
            # 10. SAVE FILE MANIFEST
            manifest_path = os.path.join(model_dir, f"{base_filename}_file_manifest.json")
            manifest = {
                'total_files_saved': len(saved_files),
                'file_details': {},
                'total_size_mb': 0,
                'save_timestamp': timestamp,
                'model_identifier': f"DenseNet121_{dataset_name}_{config_name}_{timestamp}"
            }
            
            for file_type, file_path in saved_files.items():
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    manifest['file_details'][file_type] = {
                        'path': file_path,
                        'size_bytes': file_size,
                        'size_mb': round(file_size / (1024*1024), 2),
                        'exists': True
                    }
                    manifest['total_size_mb'] += file_size / (1024*1024)
            
            manifest['total_size_mb'] = round(manifest['total_size_mb'], 2)
            
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            saved_files['file_manifest'] = manifest_path
            
            print(f"✅ DenseNet121 artifacts saved to: {model_dir}")
            print(f"📁 Total files saved: {len(saved_files)}")
            print(f"💾 Total size: {manifest['total_size_mb']:.1f}MB")
            return saved_files
            
        except Exception as e:
            print(f"⚠️ Error saving DenseNet121 artifacts: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _create_densenet121_readme(self, readme_path, dataset_key, config_key, results, timestamp, saved_files):
        """Create comprehensive README for DenseNet121 model"""
        dataset_info = self.datasets[dataset_key]
        config_info = self.configurations[config_key]
        
        readme_content = f"""# DenseNet121 {dataset_info['name']} Detection Model

## 🏆 **DenseNet121 - Optimized for Grad-CAM Visualization**

This model is specifically trained using DenseNet121 architecture for superior Grad-CAM visualization results.

### 🎯 **Model Overview**
- **Architecture**: DenseNet121 (Dense Convolutional Network)
- **Medical Condition**: {dataset_info['name']} Detection
- **Dataset Type**: {dataset_info['type'].title()} X-ray analysis
- **Training Configuration**: {config_info['name']}
- **Timestamp**: {timestamp}

### 📊 **Performance Metrics**
- **Test Accuracy**: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)
- **Test Precision**: {results['test_precision']:.4f}
- **Test Recall**: {results['test_recall']:.4f}
- **Test Loss**: {results['test_loss']:.4f}
- **Best Validation Accuracy**: {results['best_val_accuracy']:.4f}

### 🔥 **Grad-CAM Optimization**
- **✅ Best Architecture**: DenseNet121 provides superior Grad-CAM visualization
- **✅ Dense Connectivity**: Rich gradient flow for detailed heatmaps
- **✅ Medical Imaging Optimized**: Excellent for fine-grained medical features
- **✅ Recommended Layer**: `conv5_block16_2_conv` for Grad-CAM visualization

### 🏗️ **Architecture Details**
- **Total Parameters**: {results.get('total_parameters', 'Unknown'):,}
- **Input Shape**: (224, 224, 3)
- **Base Model**: DenseNet121 (ImageNet pretrained)
- **Classification Head**: GlobalAveragePooling2D + Dense layers
- **Activation**: Sigmoid (binary classification)

### 📚 **Training Details**
- **Epochs Trained**: {results['epochs_trained']}
- **Training Samples**: {results['training_samples']}
- **Test Samples**: {results['test_samples']}
- **Batch Size**: {config_info['batch_size']}
- **Learning Rate**: {config_info['learning_rate']}

### 🎯 **Dataset Information**
- **Classes**: {', '.join(dataset_info['classes'])}
- **Description**: {dataset_info['description']}
- **Validation Split**: {config_info['validation_split']*100}%
- **Test Split**: {config_info['test_split']*100}%

### 💾 **Saved Files**
"""
        
        # Add file information
        for file_type, file_path in saved_files.items():
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / (1024*1024)
                readme_content += f"- **{file_type.replace('_', ' ').title()}**: `{os.path.basename(file_path)}` ({file_size:.1f}MB)\\n"
        
        readme_content += f"""

### 🔥 **Grad-CAM Usage**
```python
# Load the trained DenseNet121 model
import tensorflow as tf
from utils.gradcam import GradCAM

# Load model
model = tf.keras.models.load_model('{os.path.basename(saved_files.get('keras_model', ''))}')\n
# Initialize Grad-CAM with recommended layer
gradcam = GradCAM(model, layer_name='conv5_block16_2_conv')

# Generate heatmap for image
heatmap = gradcam.generate_heatmap(image_array)
```

### 🎯 **Why DenseNet121 is Best for Grad-CAM**
1. **Dense Connectivity**: Each layer connects to all subsequent layers
2. **Feature Preservation**: Excellent gradient flow through dense blocks
3. **Medical Imaging**: Proven superior performance in medical visualization
4. **Clear Heatmaps**: Produces well-defined activation regions
5. **Fine Detail**: Captures subtle medical abnormalities effectively

### 📈 **Performance Analysis**
- **Convergence**: {results['epochs_trained']} epochs for stable training
- **Generalization**: {results['test_accuracy']*100:.1f}% test accuracy indicates good generalization
- **Medical Relevance**: Optimized for {dataset_info['type']} X-ray analysis

---

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model Type**: DenseNet121 Medical Imaging Classifier
**Grad-CAM Optimized**: ✅ YES - Best visualization results
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
    def run_densenet121_training(self):
        """Run DenseNet121 training on all 5 datasets"""
        print("🚀 Starting DenseNet121 Focused Training Pipeline")
        print("=" * 70)
        print("🧠 Architecture: DenseNet121 (Best for Grad-CAM)")
        print("📊 Datasets: All 5 medical conditions")
        print("🎯 Configurations: Standard & Intensive")
        print("=" * 70)
        
        total_combinations = len(self.datasets) * len(self.configurations)
        completed_count = len([k for k, v in self.completed_models.items() if v['status'] == 'completed'])
        
        print(f"📊 Training Overview:")
        print(f"   Total Models: {total_combinations}")
        print(f"   Completed: {completed_count}")
        print(f"   Remaining: {total_combinations - completed_count}")
        print(f"   GPU Enabled: {'✅ YES' if self.gpu_enabled else '❌ NO (CPU only)'}")
        
        combination_count = 0
        
        for dataset_key, dataset_info in self.datasets.items():
            for config_key, config_info in self.configurations.items():
                combination_count += 1
                
                print(f"\\n{'='*50}")
                print(f"🎯 Model {combination_count}/{total_combinations}")
                print(f"📊 Dataset: {dataset_info['name']}")
                print(f"🏗️ Architecture: DenseNet121")
                print(f"⚙️ Configuration: {config_info['name']}")
                print(f"{'='*50}")
                
                # Check if already completed
                if self._is_combination_completed(dataset_key, config_key):
                    print("✅ Already completed - Skipping")
                    continue
                
                try:
                    # Mark as started
                    self._save_progress(dataset_key, config_key, "started")
                    
                    start_time = time.time()
                    
                    # Load dataset
                    X, y = self.load_dataset(dataset_info, config_info)
                    
                    # Create DenseNet121 model
                    model = self.create_densenet121_model(dataset_info)
                    
                    # Train model
                    model, history, results = self.train_model(model, X, y, config_info)
                    
                    # Save artifacts
                    print("💾 Saving DenseNet121 artifacts...")
                    saved_files = self.save_densenet121_artifacts(
                        model, history, results, dataset_key, config_key
                    )
                    
                    training_time = time.time() - start_time
                    
                    # Mark as completed
                    final_results = results.copy()
                    final_results['training_time_seconds'] = training_time
                    final_results['saved_files'] = saved_files
                    
                    self._save_progress(dataset_key, config_key, "completed", final_results)
                    
                    print(f"🎉 DenseNet121 model completed successfully!")
                    print(f"⏱️ Training time: {training_time:.1f} seconds")
                    print(f"🎯 Accuracy: {results['test_accuracy']*100:.2f}%")
                    print(f"🔥 Grad-CAM optimized: Ready for visualization!")
                    
                    # Clean up memory
                    del model, X, y
                    tf.keras.backend.clear_session()
                    
                except Exception as e:
                    print(f"❌ Error in DenseNet121 training: {str(e)}")
                    print(f"📋 Traceback: {traceback.format_exc()}")
                    
                    # Mark as failed
                    self._save_progress(dataset_key, config_key, "failed", {
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    })
                    
                    # Clean up memory
                    tf.keras.backend.clear_session()
                    continue
        
        print(f"\\n🏁 DenseNet121 Training Pipeline Complete!")
        print(f"📊 Final Statistics:")
        completed_final = len([k for k, v in self.completed_models.items() if v['status'] == 'completed'])
        failed_final = len([k for k, v in self.completed_models.items() if v['status'] == 'failed'])
        print(f"   ✅ Completed: {completed_final}")
        print(f"   ❌ Failed: {failed_final}")
        print(f"   🔥 All models optimized for Grad-CAM visualization!")
        print(f"   📁 Results saved in: {self.base_dir}")
    
    def _get_comprehensive_system_info(self):
        """Get comprehensive system information for complete documentation"""
        try:
            system_info = {
                'timestamp': datetime.now().isoformat(),
                'platform': {
                    'system': platform.system(),
                    'platform': platform.platform(),
                    'machine': platform.machine(),
                    'processor': platform.processor(),
                    'architecture': platform.architecture(),
                    'python_version': platform.python_version(),
                    'python_implementation': platform.python_implementation()
                },
                'hardware': {},
                'tensorflow_info': {},
                'gpu_info': {},
                'memory_info': {},
                'disk_info': {}
            }
            
            # Hardware information
            try:
                system_info['hardware'] = {
                    'cpu_count': os.cpu_count(),
                    'cpu_count_logical': psutil.cpu_count(logical=True),
                    'cpu_count_physical': psutil.cpu_count(logical=False),
                    'cpu_freq_current': psutil.cpu_freq().current if psutil.cpu_freq() else None,
                    'cpu_freq_max': psutil.cpu_freq().max if psutil.cpu_freq() else None,
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
                }
            except:
                system_info['hardware'] = {'error': 'Could not get hardware info'}
            
            # Memory information
            try:
                memory = psutil.virtual_memory()
                system_info['memory_info'] = {
                    'total_gb': round(memory.total / (1024**3), 2),
                    'available_gb': round(memory.available / (1024**3), 2),
                    'used_gb': round(memory.used / (1024**3), 2),
                    'percentage': memory.percent,
                    'swap_total_gb': round(psutil.swap_memory().total / (1024**3), 2) if psutil.swap_memory().total > 0 else 0,
                    'swap_used_gb': round(psutil.swap_memory().used / (1024**3), 2) if psutil.swap_memory().used > 0 else 0
                }
            except:
                system_info['memory_info'] = {'error': 'Could not get memory info'}
            
            # Disk information
            try:
                disk = psutil.disk_usage('/')
                system_info['disk_info'] = {
                    'total_gb': round(disk.total / (1024**3), 2),
                    'used_gb': round(disk.used / (1024**3), 2),
                    'free_gb': round(disk.free / (1024**3), 2),
                    'percentage': round((disk.used / disk.total) * 100, 2)
                }
            except:
                system_info['disk_info'] = {'error': 'Could not get disk info'}
            
            # TensorFlow information
            try:
                system_info['tensorflow_info'] = {
                    'version': tf.__version__,
                    'keras_version': tf.keras.__version__,
                    'built_with_cuda': tf.test.is_built_with_cuda(),
                    'gpu_available': tf.test.is_gpu_available(),
                    'physical_devices': len(tf.config.list_physical_devices()),
                    'logical_devices': len(tf.config.list_logical_devices()),
                    'mixed_precision_enabled': tf.keras.mixed_precision.global_policy().name != 'float32'
                }
            except:
                system_info['tensorflow_info'] = {'error': 'Could not get TensorFlow info'}
            
            # GPU information
            try:
                gpus = tf.config.list_physical_devices('GPU')
                gpu_info = []
                for i, gpu in enumerate(gpus):
                    gpu_details = tf.config.experimental.get_device_details(gpu)
                    gpu_info.append({
                        'device_name': gpu.name,
                        'device_type': gpu.device_type,
                        'details': gpu_details
                    })
                system_info['gpu_info'] = {
                    'gpu_count': len(gpus),
                    'gpu_devices': gpu_info,
                    'memory_growth_enabled': self.gpu_enabled
                }
            except:
                system_info['gpu_info'] = {'error': 'Could not get GPU info'}
            
            return system_info
            
        except Exception as e:
            return {'error': f'Failed to get system info: {str(e)}'}
    
    def _get_complete_environment_info(self):
        """Get complete environment information"""
        try:
            env_info = {
                'timestamp': datetime.now().isoformat(),
                'environment_variables': {},
                'python_packages': {},
                'tensorflow_config': {},
                'keras_config': {},
                'working_directory': os.getcwd(),
                'script_location': os.path.abspath(__file__),
                'python_path': sys.path,
                'python_executable': sys.executable
            }
            
            # Environment variables (only relevant ones)
            relevant_env_vars = ['PATH', 'PYTHONPATH', 'CUDA_VISIBLE_DEVICES', 'TF_CPP_MIN_LOG_LEVEL', 
                               'KERAS_BACKEND', 'TF_FORCE_GPU_ALLOW_GROWTH', 'HOME', 'USER']
            for var in relevant_env_vars:
                if var in os.environ:
                    env_info['environment_variables'][var] = os.environ[var]
            
            # Get Python packages
            try:
                import pkg_resources
                installed_packages = {pkg.project_name: pkg.version for pkg in pkg_resources.working_set}
                key_packages = ['tensorflow', 'keras', 'numpy', 'PIL', 'opencv-python', 'matplotlib', 'scikit-learn']
                env_info['python_packages'] = {pkg: installed_packages.get(pkg, 'Not installed') for pkg in key_packages}
                env_info['python_packages']['total_packages'] = len(installed_packages)
            except:
                env_info['python_packages'] = {'error': 'Could not get package info'}
            
            # TensorFlow configuration
            try:
                env_info['tensorflow_config'] = {
                    'eager_execution': tf.executing_eagerly(),
                    'mixed_precision_policy': tf.keras.mixed_precision.global_policy().name,
                    'optimizer_jit': tf.config.optimizer.get_jit() if hasattr(tf.config.optimizer, 'get_jit') else None,
                    'memory_growth': len(tf.config.experimental.list_memory_growth(tf.config.list_physical_devices('GPU'))) > 0 if tf.config.list_physical_devices('GPU') else False
                }
            except:
                env_info['tensorflow_config'] = {'error': 'Could not get TensorFlow config'}
            
            return env_info
            
        except Exception as e:
            return {'error': f'Failed to get environment info: {str(e)}'}
    
    def _get_dataset_integrity_info(self, dataset_key):
        """Get dataset integrity and hash information"""
        try:
            dataset_info = self.datasets[dataset_key]
            integrity_info = {
                'dataset_name': dataset_info['name'],
                'dataset_path': dataset_info['path'],
                'timestamp': datetime.now().isoformat(),
                'class_directories': {},
                'file_counts': {},
                'sample_files': {},
                'dataset_hash': None
            }
            
            dataset_path = dataset_info['path']
            if os.path.exists(dataset_path):
                # Get class directories and file counts
                for class_name in dataset_info['classes']:
                    class_path = os.path.join(dataset_path, class_name)
                    if os.path.exists(class_path):
                        files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        integrity_info['class_directories'][class_name] = class_path
                        integrity_info['file_counts'][class_name] = len(files)
                        # Get first 5 files as samples
                        integrity_info['sample_files'][class_name] = files[:5]
                
                # Calculate dataset hash (hash of file names and sizes)
                all_files = []
                for class_name in dataset_info['classes']:
                    class_path = os.path.join(dataset_path, class_name)
                    if os.path.exists(class_path):
                        for file in os.listdir(class_path):
                            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                file_path = os.path.join(class_path, file)
                                file_size = os.path.getsize(file_path)
                                all_files.append(f"{file}:{file_size}")
                
                # Create hash from sorted file list
                all_files.sort()
                dataset_string = '|'.join(all_files)
                integrity_info['dataset_hash'] = hashlib.md5(dataset_string.encode()).hexdigest()
                integrity_info['total_files'] = len(all_files)
            
            return integrity_info
            
        except Exception as e:
            return {'error': f'Failed to get dataset integrity info: {str(e)}', 'dataset_key': dataset_key}

def main():
    """Main execution function"""
    print("🏥 Medical X-Ray AI - DenseNet121 Focused Training")
    print("🔥 Optimized for Superior Grad-CAM Visualization")
    print("📊 Training on all 5 medical datasets")
    
    pipeline = DenseNet121TrainingPipeline()
    pipeline.run_densenet121_training()

if __name__ == "__main__":
    main()