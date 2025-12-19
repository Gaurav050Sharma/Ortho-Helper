#!/usr/bin/env python3
"""
Robust DenseNet121 Osteoporosis Training Script
Simplified version that avoids dimensional issues
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

def train_densenet121_robust():
    """Robust DenseNet121 training without dimensional issues"""
    
    print("🏥 Robust DenseNet121 Osteoporosis Training")
    print("=" * 50)
    
    # Configuration
    config = {
        'input_shape': (224, 224, 3),
        'batch_size': 8,  # Smaller batch size
        'epochs': 5,
        'learning_rate': 0.001,
        'max_images': 100  # Quick demo
    }
    
    # Dataset paths
    dataset_path = "Dataset/KNEE/Osteoporosis/Combined_Osteoporosis_Dataset"
    output_dir = "new"
    
    print("📊 Loading dataset...")
    
    # Load data (simplified)
    X_data = []
    y_data = []
    
    # Load normal images
    normal_path = os.path.join(dataset_path, 'Normal')
    normal_files = os.listdir(normal_path)[:config['max_images']]
    
    for filename in normal_files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img_path = os.path.join(normal_path, filename)
                img = Image.open(img_path).convert('RGB')
                img = img.resize((224, 224), Image.Resampling.LANCZOS)
                img_array = np.array(img, dtype=np.float32) / 255.0
                X_data.append(img_array)
                y_data.append(0)
            except:
                continue
    
    # Load osteoporosis images
    osteo_path = os.path.join(dataset_path, 'Osteoporosis')
    osteo_files = os.listdir(osteo_path)[:config['max_images']]
    
    for filename in osteo_files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img_path = os.path.join(osteo_path, filename)
                img = Image.open(img_path).convert('RGB')
                img = img.resize((224, 224), Image.Resampling.LANCZOS)
                img_array = np.array(img, dtype=np.float32) / 255.0
                X_data.append(img_array)
                y_data.append(1)
            except:
                continue
    
    X = np.array(X_data)
    y = np.array(y_data)
    
    print(f"✅ Loaded {len(X)} images")
    print(f"   Normal: {np.sum(y == 0)}")
    print(f"   Osteoporosis: {np.sum(y == 1)}")
    
    # Simple train/test split
    indices = np.random.permutation(len(X))
    split_point = int(0.8 * len(X))
    
    train_idx = indices[:split_point]
    test_idx = indices[split_point:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"📈 Split: {len(X_train)} train, {len(X_test)} test")
    
    # Create simple DenseNet121 model
    print("🏗️ Creating model...")
    
    model = keras.Sequential([
        DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        ),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Freeze base model
    model.layers[0].trainable = False
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Model ready: {model.count_params():,} parameters")
    
    # Train
    print("🚀 Training...")
    
    history = model.fit(
        X_train, y_train,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate
    print("📊 Evaluating...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"🎯 Test Accuracy: {test_acc:.4f}")
    print(f"🎯 Test Loss: {test_loss:.4f}")
    
    # Save files
    print("💾 Saving model files...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create comprehensive file set
    files_created = []
    
    # 1. Keras model (.keras)
    try:
        keras_path = os.path.join(output_dir, f'densenet121_robust_{timestamp}.keras')
        model.save(keras_path)
        files_created.append(f'✅ {os.path.basename(keras_path)}')
    except Exception as e:
        files_created.append(f'❌ keras model: {e}')
    
    # 2. H5 model (.h5)
    try:
        h5_path = os.path.join(output_dir, f'densenet121_robust_{timestamp}.h5')
        model.save(h5_path)
        files_created.append(f'✅ {os.path.basename(h5_path)}')
    except Exception as e:
        files_created.append(f'❌ h5 model: {e}')
    
    # 3. Weights (.weights.h5)
    try:
        weights_path = os.path.join(output_dir, f'weights_robust_{timestamp}.weights.h5')
        model.save_weights(weights_path)
        files_created.append(f'✅ {os.path.basename(weights_path)}')
    except Exception as e:
        files_created.append(f'❌ weights: {e}')
    
    # 4. Model config (JSON)
    try:
        config_path = os.path.join(output_dir, f'config_robust_{timestamp}.json')
        with open(config_path, 'w') as f:
            json.dump(model.get_config(), f, indent=2)
        files_created.append(f'✅ {os.path.basename(config_path)}')
    except Exception as e:
        files_created.append(f'❌ config: {e}')
    
    # 5. Training history (JSON)
    try:
        history_path = os.path.join(output_dir, f'history_robust_{timestamp}.json')
        history_data = {key: [float(v) for v in values] for key, values in history.history.items()}
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        files_created.append(f'✅ {os.path.basename(history_path)}')
    except Exception as e:
        files_created.append(f'❌ history: {e}')
    
    # 6. Model summary (TXT)
    try:
        summary_path = os.path.join(output_dir, f'summary_robust_{timestamp}.txt')
        with open(summary_path, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\\n'))
        files_created.append(f'✅ {os.path.basename(summary_path)}')
    except Exception as e:
        files_created.append(f'❌ summary: {e}')
    
    # 7. Complete results (JSON)
    try:
        results = {
            'model_name': 'DenseNet121_Osteoporosis_Robust',
            'timestamp': timestamp,
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss),
            'total_parameters': int(model.count_params()),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'epochs': config['epochs'],
            'architecture': 'DenseNet121 + Global Average Pooling + Dense Layers',
            'final_activation': 'sigmoid',
            'loss_function': 'binary_crossentropy'
        }
        
        results_path = os.path.join(output_dir, f'results_robust_{timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        files_created.append(f'✅ {os.path.basename(results_path)}')
    except Exception as e:
        files_created.append(f'❌ results: {e}')
    
    # 8. TensorFlow SavedModel
    try:
        saved_model_dir = os.path.join(output_dir, f'savedmodel_robust_{timestamp}')
        tf.saved_model.save(model, saved_model_dir)
        files_created.append(f'✅ SavedModel directory: {os.path.basename(saved_model_dir)}')
    except Exception as e:
        files_created.append(f'❌ savedmodel: {e}')
    
    # Final summary
    print("\\n🎉 Training Complete!")
    print("=" * 50)
    print(f"📁 Files saved to: {output_dir}")
    print(f"🎯 Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"📊 Parameters: {model.count_params():,}")
    print("\\n📂 Files Created:")
    for file_info in files_created:
        print(f"   {file_info}")
    
    return model, results, files_created

if __name__ == "__main__":
    model, results, files = train_densenet121_robust()