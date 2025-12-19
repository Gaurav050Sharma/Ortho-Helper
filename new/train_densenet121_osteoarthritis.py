#!/usr/bin/env python3
"""
DenseNet121 Osteoarthritis Classification Training Script
========================================================

Robust training script for osteoarthritis vs normal knee X-ray classification
using DenseNet121 architecture with transfer learning.

Dataset: Combined_Osteoarthritis_Dataset
- Normal: 6,942 images
- Osteoarthritis: 2,846 images
- Total: 9,788 images

Output: All model formats (.h5, .keras, weights, config, etc.) saved in /new/
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers
from PIL import Image
import json
from datetime import datetime
import random

# Suppress warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train_densenet121_osteoarthritis():
    """Train DenseNet121 for osteoarthritis classification"""
    
    print("🦴 DenseNet121 Osteoarthritis Training")
    print("=" * 50)
    
    # Configuration
    config = {
        'input_shape': (224, 224, 3),
        'batch_size': 16,
        'epochs': 8,
        'learning_rate': 0.001,
        'max_images_per_class': 500,  # Limit for reasonable training time
        'validation_split': 0.2,
        'test_split': 0.2
    }
    
    # Paths
    dataset_path = "Dataset/KNEE/Osteoarthritis/Combined_Osteoarthritis_Dataset"
    output_dir = "new"
    os.makedirs(output_dir, exist_ok=True)
    
    print("📊 Loading osteoarthritis dataset...")
    print(f"   Max images per class: {config['max_images_per_class']}")
    
    # Load data
    images = []
    labels = []
    
    # Load Normal images (limited for demo)
    normal_path = os.path.join(dataset_path, 'Normal')
    normal_files = [f for f in os.listdir(normal_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Randomly sample for balanced dataset
    random.seed(42)
    random.shuffle(normal_files)
    normal_files = normal_files[:config['max_images_per_class']]
    
    print(f"📁 Loading {len(normal_files)} normal images...")
    for i, filename in enumerate(normal_files):
        if i % 100 == 0:
            print(f"   Progress: {i}/{len(normal_files)}")
        try:
            img_path = os.path.join(normal_path, filename)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(config['input_shape'][:2], Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            images.append(img_array)
            labels.append(0)  # Normal = 0
        except Exception as e:
            continue
    
    # Load Osteoarthritis images (limited for demo)
    arthritis_path = os.path.join(dataset_path, 'Osteoarthritis')
    arthritis_files = [f for f in os.listdir(arthritis_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Randomly sample
    random.shuffle(arthritis_files)
    arthritis_files = arthritis_files[:config['max_images_per_class']]
    
    print(f"📁 Loading {len(arthritis_files)} osteoarthritis images...")
    for i, filename in enumerate(arthritis_files):
        if i % 100 == 0:
            print(f"   Progress: {i}/{len(arthritis_files)}")
        try:
            img_path = os.path.join(arthritis_path, filename)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(config['input_shape'][:2], Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            images.append(img_array)
            labels.append(1)  # Osteoarthritis = 1
        except Exception as e:
            continue
    
    # Convert to arrays and shuffle
    X = np.array(images)
    y = np.array(labels)
    
    # Shuffle the data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    print(f"\\n✅ Dataset loaded successfully!")
    print(f"   Total images: {len(X)}")
    print(f"   Normal: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
    print(f"   Osteoarthritis: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
    
    # Data split
    test_size = int(config['test_split'] * len(X))
    train_size = len(X) - test_size
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"\\n📈 Data split:")
    print(f"   Training: {len(X_train)} images")
    print(f"   Testing: {len(X_test)} images")
    
    # Create DenseNet121 model
    print("\\n🏗️ Creating DenseNet121 model for osteoarthritis...")
    
    # Base model
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=config['input_shape']
    )
    base_model.trainable = False  # Freeze initially
    
    # Custom classification head for osteoarthritis
    inputs = keras.Input(shape=config['input_shape'])
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu', name='arthritis_dense_512')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu', name='arthritis_dense_256')(x)
    x = layers.Dropout(0.2)(x)
    # Binary classification for osteoarthritis detection
    outputs = layers.Dense(1, activation='sigmoid', name='arthritis_prediction')(x)
    
    model = keras.Model(inputs, outputs, name='DenseNet121_Osteoarthritis')
    
    # Compile for binary classification
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print(f"✅ Osteoarthritis model created!")
    print(f"   Total parameters: {model.count_params():,}")
    print(f"   Architecture: DenseNet121 + Custom Arthritis Head")
    print(f"   Output: Binary classification (Normal vs Osteoarthritis)")
    
    # Training callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
    
    print("\\n🚀 Starting osteoarthritis training...")
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_split=config['validation_split'],
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate model
    print("\\n📊 Evaluating osteoarthritis model...")
    test_results = model.evaluate(X_test, y_test, verbose=0)
    test_loss = test_results[0]
    test_acc = test_results[1]
    test_precision = test_results[2] if len(test_results) > 2 else 0.0
    test_recall = test_results[3] if len(test_results) > 3 else 0.0
    
    print(f"🎯 Osteoarthritis Classification Results:")
    print(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Precision: {test_precision:.4f}")
    print(f"   Test Recall: {test_recall:.4f}")
    
    # Save all model formats
    print("\\n💾 Saving osteoarthritis model in all formats...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    saved_files = {}
    
    try:
        # 1. .keras format (recommended)
        keras_path = os.path.join(output_dir, f'densenet121_osteoarthritis_{timestamp}.keras')
        model.save(keras_path)
        saved_files['keras_model'] = keras_path
        print(f"✅ Saved .keras: {os.path.basename(keras_path)}")
    except Exception as e:
        print(f"⚠️ Error saving .keras: {e}")
    
    try:
        # 2. .h5 format
        h5_path = os.path.join(output_dir, f'densenet121_osteoarthritis_{timestamp}.h5')
        model.save(h5_path)
        saved_files['h5_model'] = h5_path
        print(f"✅ Saved .h5: {os.path.basename(h5_path)}")
    except Exception as e:
        print(f"⚠️ Error saving .h5: {e}")
    
    try:
        # 3. Weights only
        weights_path = os.path.join(output_dir, f'osteoarthritis_weights_{timestamp}.weights.h5')
        model.save_weights(weights_path)
        saved_files['weights'] = weights_path
        print(f"✅ Saved weights: {os.path.basename(weights_path)}")
    except Exception as e:
        print(f"⚠️ Error saving weights: {e}")
    
    try:
        # 4. Model configuration
        config_path = os.path.join(output_dir, f'osteoarthritis_config_{timestamp}.json')
        with open(config_path, 'w') as f:
            json.dump(model.get_config(), f, indent=2)
        saved_files['config'] = config_path
        print(f"✅ Saved config: {os.path.basename(config_path)}")
    except Exception as e:
        print(f"⚠️ Error saving config: {e}")
    
    try:
        # 5. Training history
        history_path = os.path.join(output_dir, f'osteoarthritis_history_{timestamp}.json')
        history_dict = {}
        for key, values in history.history.items():
            history_dict[key] = [float(v) for v in values]
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        saved_files['history'] = history_path
        print(f"✅ Saved history: {os.path.basename(history_path)}")
    except Exception as e:
        print(f"⚠️ Error saving history: {e}")
    
    try:
        # 6. Model summary
        summary_path = os.path.join(output_dir, f'osteoarthritis_summary_{timestamp}.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            model.summary(print_fn=lambda x: f.write(x + '\\n'))
        saved_files['summary'] = summary_path
        print(f"✅ Saved summary: {os.path.basename(summary_path)}")
    except Exception as e:
        print(f"⚠️ Error saving summary: {e}")
    
    try:
        # 7. Training configuration
        train_config_path = os.path.join(output_dir, f'osteoarthritis_train_config_{timestamp}.json')
        config_to_save = config.copy()
        config_to_save.update({
            'timestamp': timestamp,
            'dataset_size': len(X),
            'model_params': int(model.count_params()),
            'final_accuracy': float(test_acc),
            'epochs_trained': len(history.history['loss'])
        })
        with open(train_config_path, 'w') as f:
            json.dump(config_to_save, f, indent=2)
        saved_files['train_config'] = train_config_path
        print(f"✅ Saved train config: {os.path.basename(train_config_path)}")
    except Exception as e:
        print(f"⚠️ Error saving train config: {e}")
    
    try:
        # 8. Results summary
        results = {
            'model_name': 'DenseNet121_Osteoarthritis_Classifier',
            'timestamp': timestamp,
            'condition': 'Osteoarthritis',
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'total_parameters': int(model.count_params()),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'epochs_trained': len(history.history['loss']),
            'architecture': 'DenseNet121 + Custom Osteoarthritis Head',
            'input_shape': config['input_shape'],
            'classes': ['Normal', 'Osteoarthritis'],
            'activation': 'sigmoid',
            'loss_function': 'binary_crossentropy'
        }
        
        results_path = os.path.join(output_dir, f'osteoarthritis_results_{timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        saved_files['results'] = results_path
        print(f"✅ Saved results: {os.path.basename(results_path)}")
    except Exception as e:
        print(f"⚠️ Error saving results: {e}")
    
    try:
        # 9. SavedModel directory
        savedmodel_path = os.path.join(output_dir, f'osteoarthritis_savedmodel_{timestamp}')
        tf.saved_model.save(model, savedmodel_path)
        saved_files['savedmodel'] = savedmodel_path
        print(f"✅ Saved SavedModel: {os.path.basename(savedmodel_path)}")
    except Exception as e:
        print(f"⚠️ Error saving SavedModel: {e}")
    
    # Final summary
    print("\\n🎉 Osteoarthritis Training Complete!")
    print("=" * 50)
    print(f"📁 Files saved to: {output_dir}")
    print(f"🎯 Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"📊 Model Parameters: {model.count_params():,}")
    print(f"🦴 Condition: Osteoarthritis Detection")
    print(f"⏰ Timestamp: {timestamp}")
    
    print("\\n📂 Osteoarthritis Model Files:")
    for key, path in saved_files.items():
        if os.path.isdir(path):
            print(f"   📁 {key}: {os.path.basename(path)}/")
        else:
            print(f"   📄 {key}: {os.path.basename(path)}")
    
    return model, results, saved_files

if __name__ == "__main__":
    model, results, files = train_densenet121_osteoarthritis()