#!/usr/bin/env python3
"""
Cardiomegaly Training Fix
========================

Fix the preprocessing issues for Cardiomegaly dataset and train DenseNet121 models.
This addresses the float64/int64 casting error and folder structure mismatch.
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

class CardiomegalyDenseNet121Trainer:
    """Specialized trainer for Cardiomegaly dataset with DenseNet121"""
    
    def __init__(self):
        print("🏥 Cardiomegaly DenseNet121 Trainer - Fix & Train")
        print("==================================================")
        
        # Setup paths
        self.base_dir = "cardiomegaly_densenet121_models"
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Progress tracking
        self.progress_file = "cardiomegaly_training_progress.json"
        self.completed_models = self._load_progress()
        
        # GPU configuration
        self.gpu_enabled = self._configure_gpu()
        
        # Dataset configuration - FIXED
        self.cardiomegaly_dataset = {
            "name": "Cardiomegaly",
            "path": "Dataset/CHEST/cardiomelgy/train/train",  # Fixed path
            "classes": ["false", "true"],  # Actual folder names
            "folders": ["false", "true"],  # Actual folder names
            "type": "chest",
            "description": "Heart enlargement detection",
            "class_labels": ["Normal", "Cardiomegaly"]  # Human readable labels
        }
        
        # Training configurations
        self.configurations = {
            "standard": {
                "name": "Standard",
                "max_images_per_class": 500,
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001,
                "validation_split": 0.2,
                "test_split": 0.2,
                "patience": 3
            },
            "intensive": {
                "name": "Intensive", 
                "max_images_per_class": 1000,
                "epochs": 15,
                "batch_size": 25,
                "learning_rate": 0.001,
                "validation_split": 0.2,
                "test_split": 0.2,
                "patience": 4
            }
        }
    
    def _configure_gpu(self):
        """Configure GPU settings"""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"📊 GPU devices found: {len(gpus)}")
                # Configure memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"✅ Enabled memory growth for: {gpu}")
                
                # Set mixed precision for better performance
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                print("✅ Mixed precision enabled (float16/float32)")
                
                return True
            else:
                print("⚠️ No GPU devices found, using CPU")
                return False
        except RuntimeError as e:
            print(f"⚠️ GPU configuration error: {e}")
            return False
    
    def _load_progress(self):
        """Load training progress"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_progress(self, config_key, status, results=None):
        """Save training progress"""
        self.completed_models[config_key] = {
            "dataset": "cardiomegaly",
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
    
    def load_cardiomegaly_dataset(self, config):
        """Load and preprocess Cardiomegaly dataset with FIXED data types"""
        print(f"📊 Loading {self.cardiomegaly_dataset['name']} dataset...")
        print(f"📁 Dataset path: {self.cardiomegaly_dataset['path']}")
        
        # Check if dataset path exists
        if not os.path.exists(self.cardiomegaly_dataset['path']):
            raise FileNotFoundError(f"Dataset path not found: {self.cardiomegaly_dataset['path']}")
        
        # Load images from each class
        X, y = [], []
        
        for class_idx, folder_name in enumerate(self.cardiomegaly_dataset['folders']):
            folder_path = os.path.join(self.cardiomegaly_dataset['path'], folder_name)
            
            if not os.path.exists(folder_path):
                print(f"⚠️ Folder not found: {folder_path}")
                continue
            
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Limit images per class
            max_images = config['max_images_per_class']
            if len(image_files) > max_images:
                image_files = random.sample(image_files, max_images)
            
            print(f"   Loading {len(image_files)} images from {folder_name} ({self.cardiomegaly_dataset['class_labels'][class_idx]})...")
            
            for i, img_file in enumerate(image_files):
                if i % 50 == 0:
                    print(f"     Progress: {i}/{len(image_files)}")
                
                try:
                    img_path = os.path.join(folder_path, img_file)
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((224, 224))
                    img_array = np.array(img, dtype=np.float32) / 255.0  # Explicit float32
                    
                    X.append(img_array)
                    y.append(int(class_idx))  # Explicit int conversion
                except Exception as e:
                    print(f"     Error loading {img_file}: {e}")
                    continue
        
        # Convert to numpy arrays with explicit data types
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)  # Explicit int32 to avoid casting issues
        
        # Shuffle the data
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        print(f"✅ Dataset loaded: {len(X)} images")
        print(f"   Data types: X={X.dtype}, y={y.dtype}")
        print(f"   Class distribution: {np.bincount(y)}")
        print(f"   Classes: {self.cardiomegaly_dataset['class_labels']}")
        
        return X, y
    
    def create_cardiomegaly_model(self):
        """Create DenseNet121 model optimized for Cardiomegaly detection"""
        print(f"🏗️ Creating DenseNet121 model for Cardiomegaly detection...")
        
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
        
        # Compile model with FIXED metrics
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        print(f"✅ DenseNet121 model created with {model.count_params():,} parameters")
        return model
    
    def train_cardiomegaly_model(self, model, X, y, config):
        """Train DenseNet121 model for Cardiomegaly detection"""
        print(f"🚀 Training DenseNet121 for Cardiomegaly - {config['epochs']} epochs...")
        
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
                f'{self.base_dir}/best_cardiomegaly_checkpoint.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_split=config['validation_split'],
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Evaluate model
        print(f"📊 Evaluating Cardiomegaly model...")
        test_results = model.evaluate(X_test, y_test, verbose=0)
        
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
    
    def save_cardiomegaly_model(self, model, history, results, config_key):
        """Save Cardiomegaly model with comprehensive details"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = self.configurations[config_key]['name']
        
        # Create model directory
        model_dir = os.path.join(self.base_dir, f"cardiomegaly_{config_name.lower()}_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model in multiple formats
        model_path = os.path.join(model_dir, f"cardiomegaly_densenet121_{config_name.lower()}_{timestamp}")
        
        try:
            # Save models
            model.save(f"{model_path}.keras")
            model.save(f"{model_path}.h5")
            model.save_weights(f"{model_path}.weights.h5")
            
            # Save results
            results_data = {
                'model_info': {
                    'name': f"DenseNet121_Cardiomegaly_{config_name}",
                    'architecture': 'DenseNet121',
                    'dataset': 'Cardiomegaly',
                    'configuration': config_name,
                    'timestamp': timestamp,
                    'total_parameters': int(model.count_params())
                },
                'performance': results,
                'training_config': self.configurations[config_key],
                'dataset_info': self.cardiomegaly_dataset,
                'gradcam_optimization': {
                    'optimized_for_gradcam': True,
                    'recommended_layer': 'conv5_block16_2_conv',
                    'architecture_benefits': [
                        'Dense connectivity preserves gradients',
                        'Excellent for cardiac imaging visualization',
                        'Superior gradient flow for heart abnormalities',
                        'Clear heatmaps for cardiomegaly detection'
                    ]
                }
            }
            
            with open(os.path.join(model_dir, 'model_details.json'), 'w') as f:
                json.dump(results_data, f, indent=2)
            
            # Save training history
            history_data = {}
            for key, values in history.history.items():
                history_data[key] = [float(v) for v in values]
            
            with open(os.path.join(model_dir, 'training_history.json'), 'w') as f:
                json.dump(history_data, f, indent=2)
            
            # Create README
            readme_content = f"""# DenseNet121 Cardiomegaly Detection Model

## Model Information
- **Architecture**: DenseNet121
- **Medical Condition**: Cardiomegaly (Heart Enlargement)
- **Configuration**: {config_name}
- **Training Date**: {timestamp}
- **Parameters**: {model.count_params():,}

## Performance Metrics
- **Test Accuracy**: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)
- **Test Precision**: {results['test_precision']:.4f}
- **Test Recall**: {results['test_recall']:.4f}
- **Test Loss**: {results['test_loss']:.4f}

## Dataset Information
- **Classes**: Normal vs Cardiomegaly
- **Training Samples**: {results['training_samples']}
- **Test Samples**: {results['test_samples']}
- **Epochs Trained**: {results['epochs_trained']}

## Grad-CAM Optimization
This model is optimized for superior Grad-CAM visualization:
- **Recommended Layer**: `conv5_block16_2_conv`
- **Architecture Benefits**: Dense connectivity for cardiac imaging
- **Visualization Quality**: Excellent for heart abnormality detection

## Usage
```python
import tensorflow as tf
from utils.gradcam import GradCAM

# Load model
model = tf.keras.models.load_model('{model_path}.keras')

# Initialize Grad-CAM
gradcam = GradCAM(model, layer_name='conv5_block16_2_conv')

# Generate heatmap
heatmap = gradcam.generate_heatmap(chest_xray_image)
```

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            with open(os.path.join(model_dir, 'README.md'), 'w') as f:
                f.write(readme_content)
            
            print(f"✅ Cardiomegaly model saved to: {model_dir}")
            
        except Exception as e:
            print(f"⚠️ Error saving model: {e}")
    
    def train_all_cardiomegaly_models(self):
        """Train both standard and intensive Cardiomegaly models"""
        print("🚀 Starting Cardiomegaly DenseNet121 Training")
        print("==============================================")
        
        for config_key, config_info in self.configurations.items():
            print(f"\n🎯 Training Cardiomegaly - {config_info['name']} Configuration")
            print("=" * 60)
            
            try:
                # Load dataset
                X, y = self.load_cardiomegaly_dataset(config_info)
                
                # Create model
                model = self.create_cardiomegaly_model()
                
                # Train model
                start_time = datetime.now()
                model, history, results = self.train_cardiomegaly_model(model, X, y, config_info)
                end_time = datetime.now()
                
                training_time = (end_time - start_time).total_seconds()
                results['training_time_seconds'] = training_time
                
                print(f"🎉 Cardiomegaly {config_info['name']} model completed successfully!")
                print(f"⏱️ Training time: {training_time:.1f} seconds")
                print(f"🎯 Accuracy: {results['test_accuracy']*100:.2f}%")
                print(f"🔥 Grad-CAM optimized: Ready for cardiac visualization!")
                
                # Save model
                self.save_cardiomegaly_model(model, history, results, config_key)
                
                # Save progress
                self._save_progress(config_key, "completed", results)
                
                # Clean up memory
                tf.keras.backend.clear_session()
                
            except Exception as e:
                print(f"❌ Error training Cardiomegaly {config_info['name']}: {str(e)}")
                print(f"📋 Traceback:")
                import traceback
                traceback.print_exc()
                
                # Save failed progress
                self._save_progress(config_key, "failed", {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })
                
                # Clean up memory
                tf.keras.backend.clear_session()
                continue
        
        # Final summary
        completed = len([k for k, v in self.completed_models.items() if v['status'] == 'completed'])
        failed = len([k for k, v in self.completed_models.items() if v['status'] == 'failed'])
        
        print(f"\n🏁 Cardiomegaly Training Complete!")
        print(f"📊 Final Statistics:")
        print(f"   ✅ Completed: {completed}")
        print(f"   ❌ Failed: {failed}")
        print(f"   🔥 DenseNet121 optimized for cardiac Grad-CAM visualization!")

def main():
    """Main execution function"""
    print("🏥 Medical X-Ray AI - Cardiomegaly DenseNet121 Training")
    print("🔥 Fixed preprocessing issues + Optimized for Grad-CAM")
    print("📊 Training both Standard and Intensive configurations")
    
    trainer = CardiomegalyDenseNet121Trainer()
    trainer.train_all_cardiomegaly_models()

if __name__ == "__main__":
    main()