#!/usr/bin/env python3
"""
Final DenseNet121 Osteoporosis Training Script
Fixed version with all correct file formats
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

def train_densenet121_osteoporosis():
    """Complete DenseNet121 training with all file formats"""
    
    print("🏥 DenseNet121 Osteoporosis Classification Training")
    print("=" * 60)
    
    # Configuration
    config = {
        'input_shape': (224, 224, 3),
        'batch_size': 16,
        'epochs': 10,
        'learning_rate': 0.001,
        'max_images_per_class': 200  # Increased for better training
    }
    
    # Dataset paths
    dataset_path = "Dataset/KNEE/Osteoporosis/Combined_Osteoporosis_Dataset"
    output_dir = "new"
    os.makedirs(output_dir, exist_ok=True)
    
    print("📊 Loading osteoporosis dataset...")
    
    # Load data
    images = []
    labels = []
    
    # Load Normal images
    normal_path = os.path.join(dataset_path, 'Normal')
    normal_files = [f for f in os.listdir(normal_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:config['max_images_per_class']]
    
    print(f"📁 Loading {len(normal_files)} normal images...")
    for i, filename in enumerate(normal_files):
        if i % 50 == 0:
            print(f"   Progress: {i}/{len(normal_files)}")
        try:
            img_path = os.path.join(normal_path, filename)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(config['input_shape'][:2], Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            images.append(img_array)
            labels.append(0)
        except Exception as e:
            continue
    
    # Load Osteoporosis images
    osteo_path = os.path.join(dataset_path, 'Osteoporosis')
    osteo_files = [f for f in os.listdir(osteo_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:config['max_images_per_class']]
    
    print(f"📁 Loading {len(osteo_files)} osteoporosis images...")
    for i, filename in enumerate(osteo_files):
        if i % 50 == 0:
            print(f"   Progress: {i}/{len(osteo_files)}")
        try:
            img_path = os.path.join(osteo_path, filename)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(config['input_shape'][:2], Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            images.append(img_array)
            labels.append(1)
        except Exception as e:
            continue
    
    # Convert to arrays
    X = np.array(images)
    y = np.array(labels)
    
    print(f"\\n✅ Dataset loaded successfully!")
    print(f"   Total images: {len(X)}")
    print(f"   Normal: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
    print(f"   Osteoporosis: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
    
    # Data split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\\n📈 Data split:")
    print(f"   Training: {len(X_train)} images")
    print(f"   Testing: {len(X_test)} images")
    
    # Create DenseNet121 model
    print("\\n🏗️ Creating DenseNet121 model...")
    
    # Base model
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=config['input_shape']
    )
    base_model.trainable = False  # Freeze initially
    
    # Custom classification head
    inputs = keras.Input(shape=config['input_shape'])
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu', name='dense_512')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu', name='dense_256')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(2, activation='softmax', name='predictions')(x)
    
    model = keras.Model(inputs, outputs, name='DenseNet121_Osteoporosis')
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print(f"✅ Model created successfully!")
    print(f"   Total parameters: {model.count_params():,}")
    print(f"   Architecture: DenseNet121 + Custom Head")
    
    # Training
    print("\\n🚀 Starting training...")
    
    # Simple callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluation
    print("\\n📊 Evaluating model...")
    test_results = model.evaluate(X_test, y_test, verbose=0)
    test_loss = test_results[0]
    test_acc = test_results[1]
    test_precision = test_results[2] if len(test_results) > 2 else 0.0
    test_recall = test_results[3] if len(test_results) > 3 else 0.0
    
    print(f"🎯 Final Results:")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Precision: {test_precision:.4f}")
    print(f"   Test Recall: {test_recall:.4f}")
    
    # Save all model formats
    print("\\n💾 Saving model in all formats...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    saved_files = {}
    
    try:
        # 1. .h5 format (complete model)
        h5_path = os.path.join(output_dir, f'densenet121_osteoporosis_complete_{timestamp}.h5')
        model.save(h5_path)
        saved_files['h5_model'] = h5_path
        print(f"✅ Saved .h5 model: {os.path.basename(h5_path)}")
    except Exception as e:
        print(f"⚠️ Error saving .h5: {e}")
    
    try:
        # 2. .keras format (recommended)
        keras_path = os.path.join(output_dir, f'densenet121_osteoporosis_complete_{timestamp}.keras')
        model.save(keras_path)
        saved_files['keras_model'] = keras_path
        print(f"✅ Saved .keras model: {os.path.basename(keras_path)}")
    except Exception as e:
        print(f"⚠️ Error saving .keras: {e}")
    
    try:
        # 3. Weights only
        weights_path = os.path.join(output_dir, f'densenet121_weights_{timestamp}.weights.h5')
        model.save_weights(weights_path)
        saved_files['weights'] = weights_path
        print(f"✅ Saved weights: {os.path.basename(weights_path)}")
    except Exception as e:
        print(f"⚠️ Error saving weights: {e}")
    
    try:
        # 4. Model architecture (JSON)
        config_path = os.path.join(output_dir, f'model_architecture_{timestamp}.json')
        with open(config_path, 'w') as f:
            json.dump(model.get_config(), f, indent=2)
        saved_files['architecture'] = config_path
        print(f"✅ Saved architecture: {os.path.basename(config_path)}")
    except Exception as e:
        print(f"⚠️ Error saving architecture: {e}")
    
    try:
        # 5. Model summary
        summary_path = os.path.join(output_dir, f'model_summary_{timestamp}.txt')
        with open(summary_path, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\\n'))
        saved_files['summary'] = summary_path
        print(f"✅ Saved summary: {os.path.basename(summary_path)}")
    except Exception as e:
        print(f"⚠️ Error saving summary: {e}")
    
    try:
        # 6. Training history
        history_path = os.path.join(output_dir, f'training_history_{timestamp}.json')
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
        # 7. Training configuration
        config_save_path = os.path.join(output_dir, f'training_config_{timestamp}.json')
        config_to_save = config.copy()
        config_to_save['timestamp'] = timestamp
        config_to_save['dataset_size'] = len(X)
        config_to_save['model_params'] = int(model.count_params())
        with open(config_save_path, 'w') as f:
            json.dump(config_to_save, f, indent=2)
        saved_files['config'] = config_save_path
        print(f"✅ Saved config: {os.path.basename(config_save_path)}")
    except Exception as e:
        print(f"⚠️ Error saving config: {e}")
    
    try:
        # 8. Final results summary
        results = {
            'model_name': 'DenseNet121_Osteoporosis_Classifier',
            'timestamp': timestamp,
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'total_parameters': int(model.count_params()),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'epochs_trained': len(history.history['loss']),
            'architecture': 'DenseNet121 + Custom Classification Head',
            'input_shape': config['input_shape'],
            'classes': ['Normal', 'Osteoporosis']
        }
        
        results_path = os.path.join(output_dir, f'final_results_{timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        saved_files['results'] = results_path
        print(f"✅ Saved results: {os.path.basename(results_path)}")
    except Exception as e:
        print(f"⚠️ Error saving results: {e}")
    
    # Final summary
    print("\\n🎉 Training and Saving Complete!")
    print("=" * 60)
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"📊 Model Parameters: {model.count_params():,}")
    print(f"⏰ Timestamp: {timestamp}")
    print("\\n📂 Generated Files:")
    for key, path in saved_files.items():
        print(f"   • {key}: {os.path.basename(path)}")
    
    return model, results, saved_files

if __name__ == "__main__":
    model, results, files = train_densenet121_osteoporosis()