#!/usr/bin/env python3
"""
GPU-Optimized Medical X-Ray AI Training Pipeline
Optimized for RTX 3060 with 6GB VRAM
"""

import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure GPU settings
def configure_gpu():
    """Configure GPU settings for optimal performance"""
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
            
            # Set mixed precision policy for better performance
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("✅ Mixed precision enabled (float16/float32)")
            
            return True
        except RuntimeError as e:
            print(f"⚠️ GPU configuration error: {e}")
            return False
    else:
        print("⚠️ No GPU devices found, using CPU")
        return False

# Optimized data loading for GPU
def create_optimized_data_generators(train_dir, val_dir, batch_size=32, img_size=(224, 224)):
    """Create optimized data generators for GPU training"""
    
    # Increased batch size for GPU training
    gpu_batch_size = batch_size * 2 if len(tf.config.list_physical_devices('GPU')) > 0 else batch_size
    
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode='nearest',
        # GPU-optimized preprocessing
        preprocessing_function=None
    )
    
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=gpu_batch_size,
        class_mode='binary',
        shuffle=True,
        # Optimize for GPU
        interpolation='bilinear'
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=gpu_batch_size,
        class_mode='binary',
        shuffle=False,
        interpolation='bilinear'
    )
    
    return train_generator, val_generator, gpu_batch_size

# GPU-optimized model architectures
def create_gpu_optimized_model(architecture, input_shape=(224, 224, 3), num_classes=1):
    """Create GPU-optimized models with mixed precision support"""
    
    print(f"🏗️ Creating GPU-optimized {architecture} model...")
    
    if architecture == "EfficientNetB0":
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif architecture == "DenseNet121":
        base_model = tf.keras.applications.DenseNet121(
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
    elif architecture == "VGG16":
        base_model = tf.keras.applications.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Add GPU-optimized head
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='sigmoid', dtype='float32')  # Keep output as float32
    ])
    
    return model

# GPU-optimized training function
def train_gpu_optimized_model(model, train_gen, val_gen, epochs=10, learning_rate=0.001):
    """Train model with GPU optimizations"""
    
    print(f"🚀 Starting GPU-optimized training for {epochs} epochs...")
    
    # GPU-optimized optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    # Compile with mixed precision
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # GPU-optimized callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'temp/gpu_best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
        # GPU optimizations
        use_multiprocessing=True,
        workers=4,
        max_queue_size=10
    )
    
    return model, history

# GPU Performance Monitor
class GPUMonitor:
    def __init__(self):
        self.gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        
    def monitor_training(self, dataset_name, architecture):
        """Monitor GPU usage during training"""
        print(f"\n🖥️ GPU Performance Monitor")
        print(f"Dataset: {dataset_name}")
        print(f"Architecture: {architecture}")
        print(f"GPU Available: {self.gpu_available}")
        
        if self.gpu_available:
            try:
                # Get GPU memory info
                gpu_details = tf.config.experimental.get_device_details(tf.config.list_logical_devices('GPU')[0])
                print(f"GPU Details: {gpu_details}")
            except:
                print("Could not retrieve GPU details")

# Main GPU training pipeline
def main():
    """Main GPU-optimized training pipeline"""
    
    print("🏥 Medical X-Ray AI GPU-Optimized Training Pipeline")
    print("=" * 60)
    
    # Configure GPU
    gpu_enabled = configure_gpu()
    
    # Initialize GPU monitor
    monitor = GPUMonitor()
    
    # Training configurations optimized for RTX 3060
    configs = {
        'fast_gpu': {
            'epochs': 5,
            'batch_size': 64,  # Larger batch size for GPU
            'learning_rate': 0.002,
            'description': 'Fast GPU training (5 epochs, large batch)'
        },
        'balanced_gpu': {
            'epochs': 10,
            'batch_size': 48,
            'learning_rate': 0.001,
            'description': 'Balanced GPU training (10 epochs, medium batch)'
        },
        'high_quality_gpu': {
            'epochs': 20,
            'batch_size': 32,
            'learning_rate': 0.0005,
            'description': 'High quality GPU training (20 epochs, careful learning)'
        }
    }
    
    # Dataset configurations
    datasets = {
        'pneumonia': 'Dataset/CHEST/Pneumonia_Organized',
        'cardiomegaly': 'Dataset/CHEST/cardiomelgy',
        'osteoporosis': 'Dataset/KNEE/Osteoporosis',
        'osteoarthritis': 'Dataset/KNEE/Osteoarthritis'
    }
    
    # Model architectures optimized for GPU
    architectures = ['EfficientNetB0', 'DenseNet121', 'ResNet50']
    
    print(f"\n📋 Available configurations:")
    for config_name, config in configs.items():
        print(f"  {config_name}: {config['description']}")
    
    print(f"\n📋 Available datasets: {list(datasets.keys())}")
    print(f"📋 Available architectures: {architectures}")
    
    # Get user input
    print(f"\n{'='*60}")
    dataset_choice = input("Select dataset (pneumonia/cardiomegaly/osteoporosis/osteoarthritis): ").strip().lower()
    arch_choice = input("Select architecture (efficientnetb0/densenet121/resnet50): ").strip()
    config_choice = input("Select config (fast_gpu/balanced_gpu/high_quality_gpu): ").strip().lower()
    
    if dataset_choice not in datasets:
        print("❌ Invalid dataset choice")
        return
    
    if arch_choice not in [arch.lower() for arch in architectures]:
        print("❌ Invalid architecture choice")
        return
        
    if config_choice not in configs:
        print("❌ Invalid configuration choice")
        return
    
    # Get proper architecture name
    arch_name = next(arch for arch in architectures if arch.lower() == arch_choice.lower())
    
    # Start training
    config = configs[config_choice]
    dataset_path = datasets[dataset_choice]
    
    print(f"\n🎯 Starting GPU-optimized training:")
    print(f"   Dataset: {dataset_choice}")
    print(f"   Architecture: {arch_name}")
    print(f"   Configuration: {config['description']}")
    print(f"   GPU Enabled: {gpu_enabled}")
    
    # Monitor setup
    monitor.monitor_training(dataset_choice, arch_name)
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    try:
        # Create model
        model = create_gpu_optimized_model(arch_name)
        print(f"✅ Model created: {model.count_params():,} parameters")
        
        # For now, create dummy data to test GPU functionality
        print("🔄 Creating test data for GPU training validation...")
        
        # Generate dummy data to test GPU training
        batch_size = config['batch_size']
        train_data = tf.random.normal((batch_size * 4, 224, 224, 3))
        train_labels = tf.random.uniform((batch_size * 4, 1), maxval=2, dtype=tf.int32)
        val_data = tf.random.normal((batch_size, 224, 224, 3))
        val_labels = tf.random.uniform((batch_size, 1), maxval=2, dtype=tf.int32)
        
        # Convert to datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels)).batch(batch_size)
        
        # Optimize datasets for GPU
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        
        print("🚀 Starting GPU training test...")
        start_time = time.time()
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train for a few epochs to test GPU
        history = model.fit(
            train_dataset,
            epochs=2,  # Short test
            validation_data=val_dataset,
            verbose=1
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"\n✅ GPU Training Test Completed!")
        print(f"⏱️ Training time: {training_time:.2f} seconds")
        print(f"🎯 Final accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"📊 GPU optimization: {'ENABLED' if gpu_enabled else 'DISABLED'}")
        
        # Save test results
        os.makedirs('gpu_test_results', exist_ok=True)
        results = {
            'timestamp': datetime.now().isoformat(),
            'dataset': dataset_choice,
            'architecture': arch_name,
            'configuration': config_choice,
            'gpu_enabled': gpu_enabled,
            'training_time': training_time,
            'final_accuracy': float(history.history['accuracy'][-1]),
            'model_parameters': model.count_params()
        }
        
        with open('gpu_test_results/gpu_training_test.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📁 Results saved to: gpu_test_results/gpu_training_test.json")
        
    except Exception as e:
        print(f"❌ Error during GPU training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()