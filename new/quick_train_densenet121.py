#!/usr/bin/env python3
"""
Quick DenseNet121 Osteoporosis Training Demo
Fast training with subset of data for demonstration
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

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def quick_train_densenet121():
    """Quick training demo with subset of data"""
    
    print("🚀 Quick DenseNet121 Osteoporosis Training Demo")
    print("=" * 55)
    
    # Configuration
    config = {
        'input_shape': (224, 224, 3),
        'batch_size': 16,
        'epochs': 5,  # Quick demo
        'learning_rate': 0.001,
        'max_images_per_class': 100  # Limit for quick demo
    }
    
    # Dataset paths
    dataset_path = "Dataset/KNEE/Osteoporosis/Combined_Osteoporosis_Dataset"
    output_dir = "new"
    os.makedirs(output_dir, exist_ok=True)
    
    print("📊 Loading subset of dataset for quick demo...")
    
    # Load limited data for quick training
    images = []
    labels = []
    
    # Load Normal images (limited)
    normal_path = os.path.join(dataset_path, 'Normal')
    normal_files = [f for f in os.listdir(normal_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:config['max_images_per_class']]
    
    print(f"📁 Loading {len(normal_files)} normal images...")
    for filename in normal_files:
        try:
            img_path = os.path.join(normal_path, filename)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(config['input_shape'][:2], Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            images.append(img_array)
            labels.append(0)
        except:
            continue
    
    # Load Osteoporosis images (limited)
    osteo_path = os.path.join(dataset_path, 'Osteoporosis')
    osteo_files = [f for f in os.listdir(osteo_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:config['max_images_per_class']]
    
    print(f"📁 Loading {len(osteo_files)} osteoporosis images...")
    for filename in osteo_files:
        try:
            img_path = os.path.join(osteo_path, filename)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(config['input_shape'][:2], Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            images.append(img_array)
            labels.append(1)
        except:
            continue
    
    # Convert to arrays
    X = np.array(images)
    y = np.array(labels)
    
    print(f"✅ Loaded {len(X)} images total")
    print(f"   Normal: {np.sum(y == 0)}")
    print(f"   Osteoporosis: {np.sum(y == 1)}")
    
    # Simple train/test split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"📈 Data split: {len(X_train)} train, {len(X_test)} test")
    
    # Create DenseNet121 model
    print("🏗️ Creating DenseNet121 model...")
    
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=config['input_shape']
    )
    base_model.trainable = False  # Freeze base model
    
    # Add custom head
    inputs = keras.Input(shape=config['input_shape'])
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Model created with {model.count_params():,} parameters")
    
    # Quick training
    print("🚀 Starting quick training...")
    
    history = model.fit(
        X_train, y_train,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate
    print("📊 Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"🎯 Test Accuracy: {test_acc:.4f}")
    print(f"🎯 Test Loss: {test_loss:.4f}")
    
    # Save all formats
    print("💾 Saving model in all formats...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. .h5 format
    h5_path = os.path.join(output_dir, f'densenet121_osteoporosis_{timestamp}.h5')
    model.save(h5_path)
    print(f"✅ Saved .h5: {h5_path}")
    
    # 2. .keras format
    keras_path = os.path.join(output_dir, f'densenet121_osteoporosis_{timestamp}.keras')
    model.save(keras_path)
    print(f"✅ Saved .keras: {keras_path}")
    
    # 3. Weights only
    weights_path = os.path.join(output_dir, f'densenet121_weights_{timestamp}.weights.h5')
    model.save_weights(weights_path)
    print(f"✅ Saved weights: {weights_path}")
    
    # 4. Model config
    config_path = os.path.join(output_dir, f'model_config_{timestamp}.json')
    with open(config_path, 'w') as f:
        json.dump(model.get_config(), f, indent=2)
    print(f"✅ Saved config: {config_path}")
    
    # 5. SavedModel format
    savedmodel_path = os.path.join(output_dir, f'savedmodel_{timestamp}')
    model.save(savedmodel_path, save_format='tf')
    print(f"✅ Saved SavedModel: {savedmodel_path}")
    
    # 6. Training history
    history_path = os.path.join(output_dir, f'training_history_{timestamp}.json')
    with open(history_path, 'w') as f:
        # Convert numpy types to regular Python types for JSON serialization
        history_dict = {}
        for key, values in history.history.items():
            history_dict[key] = [float(v) for v in values]
        json.dump(history_dict, f, indent=2)
    print(f"✅ Saved history: {history_path}")
    
    # 7. Model summary
    summary_path = os.path.join(output_dir, f'model_summary_{timestamp}.txt')
    with open(summary_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\\n'))
    print(f"✅ Saved summary: {summary_path}")
    
    # 8. Training configuration
    config_save_path = os.path.join(output_dir, f'training_config_{timestamp}.json')
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Saved config: {config_save_path}")
    
    # Results summary
    results = {
        'model_name': 'DenseNet121_Osteoporosis',
        'timestamp': timestamp,
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'total_params': int(model.count_params()),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'epochs': config['epochs'],
        'architecture': 'DenseNet121 + Custom Head'
    }
    
    results_path = os.path.join(output_dir, f'training_results_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved results: {results_path}")
    
    print("\\n🎉 Quick Training Complete!")
    print("=" * 55)
    print(f"📁 All files saved to: {output_dir}")
    print(f"🎯 Final test accuracy: {test_acc:.4f}")
    print(f"📊 Total parameters: {model.count_params():,}")
    
    return model, results

if __name__ == "__main__":
    model, results = quick_train_densenet121()