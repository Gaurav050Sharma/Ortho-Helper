#!/usr/bin/env python3
"""
DenseNet121 Osteoporosis Classification Training Script
======================================================

This script trains a DenseNet121 model on the osteoporosis dataset and saves all model artifacts.

Dataset Structure:
- Normal: 966 images
- Osteoporosis: 979 images
- Total: 1,945 images

Model Architecture: DenseNet121 with custom classification head
Training Strategy: Transfer learning with fine-tuning
Output: Multiple model formats (.h5, .keras, weights, config, etc.)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import pickle

# Deep Learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Image processing
from PIL import Image
import cv2

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

class DenseNet121OsteoporosisTrainer:
    """
    Comprehensive DenseNet121 trainer for osteoporosis classification
    """
    
    def __init__(self, dataset_path, output_dir):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.model = None
        self.history = None
        self.class_names = ['Normal', 'Osteoporosis']
        
        # Training configuration
        self.config = {
            'input_shape': (224, 224, 3),
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 0.001,
            'fine_tune_at': 100,  # Unfreeze top layers for fine-tuning
            'validation_split': 0.2,
            'test_split': 0.1,
            'early_stopping_patience': 10,
            'reduce_lr_patience': 5,
            'seed': 42
        }
        
        # Set random seeds for reproducibility
        np.random.seed(self.config['seed'])
        tf.random.set_seed(self.config['seed'])
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"🏥 DenseNet121 Osteoporosis Trainer Initialized")
        print(f"📁 Dataset: {self.dataset_path}")
        print(f"💾 Output: {self.output_dir}")
        
    def load_and_preprocess_data(self):
        """
        Load and preprocess the osteoporosis dataset
        """
        print("\\n📊 Loading and preprocessing dataset...")
        
        # Initialize data containers
        images = []
        labels = []
        
        # Load Normal images
        normal_path = os.path.join(self.dataset_path, 'Normal')
        print(f"📁 Loading Normal images from: {normal_path}")
        
        normal_files = [f for f in os.listdir(normal_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for i, filename in enumerate(normal_files):
            if i % 100 == 0:
                print(f"   Loaded {i}/{len(normal_files)} normal images...")
            
            try:
                img_path = os.path.join(normal_path, filename)
                img = Image.open(img_path).convert('RGB')
                img = img.resize(self.config['input_shape'][:2], Image.Resampling.LANCZOS)
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                images.append(img_array)
                labels.append(0)  # Normal = 0
                
            except Exception as e:
                print(f"   Error loading {filename}: {e}")
                continue
        
        print(f"✅ Loaded {len([l for l in labels if l == 0])} normal images")
        
        # Load Osteoporosis images
        osteo_path = os.path.join(self.dataset_path, 'Osteoporosis')
        print(f"📁 Loading Osteoporosis images from: {osteo_path}")
        
        osteo_files = [f for f in os.listdir(osteo_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for i, filename in enumerate(osteo_files):
            if i % 100 == 0:
                print(f"   Loaded {i}/{len(osteo_files)} osteoporosis images...")
            
            try:
                img_path = os.path.join(osteo_path, filename)
                img = Image.open(img_path).convert('RGB')
                img = img.resize(self.config['input_shape'][:2], Image.Resampling.LANCZOS)
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                images.append(img_array)
                labels.append(1)  # Osteoporosis = 1
                
            except Exception as e:
                print(f"   Error loading {filename}: {e}")
                continue
        
        print(f"✅ Loaded {len([l for l in labels if l == 1])} osteoporosis images")
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        print(f"\\n📈 Dataset Summary:")
        print(f"   Total images: {len(X)}")
        print(f"   Normal: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
        print(f"   Osteoporosis: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
        print(f"   Image shape: {X.shape[1:]}")
        
        # Split data: train/val/test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.config['test_split'], 
            stratify=y, random_state=self.config['seed']
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=self.config['validation_split']/(1-self.config['test_split']), 
            stratify=y_temp, random_state=self.config['seed']
        )
        
        print(f"\\n🔄 Data Split:")
        print(f"   Training: {len(X_train)} images")
        print(f"   Validation: {len(X_val)} images")
        print(f"   Test: {len(X_test)} images")
        
        # Save data splits info
        split_info = {
            'total_images': len(X),
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'class_distribution': {
                'normal': int(np.sum(y == 0)),
                'osteoporosis': int(np.sum(y == 1))
            }
        }
        
        with open(os.path.join(self.output_dir, 'data_split_info.json'), 'w') as f:
            json.dump(split_info, f, indent=2)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_densenet121_model(self):
        """
        Create DenseNet121 model with custom classification head
        """
        print("\\n🏗️ Creating DenseNet121 model...")
        
        # Create base DenseNet121 model
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=self.config['input_shape']
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom classification head
        inputs = keras.Input(shape=self.config['input_shape'])
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(2, activation='softmax')(x)  # Binary classification
        
        model = keras.Model(inputs, outputs)
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(f"✅ Model created successfully!")
        print(f"   Total parameters: {model.count_params():,}")
        print(f"   Trainable parameters: {sum([keras.backend.count_params(w) for w in model.trainable_weights]):,}")
        
        # Save model architecture
        model_config = model.get_config()
        with open(os.path.join(self.output_dir, 'model_architecture.json'), 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # Save model summary
        with open(os.path.join(self.output_dir, 'model_summary.txt'), 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\\n'))
        
        self.model = model
        return model
    
    def setup_callbacks(self):
        """
        Setup training callbacks
        """
        callbacks_list = [
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint
            callbacks.ModelCheckpoint(
                filepath=os.path.join(self.output_dir, 'best_model_checkpoint.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # CSV logger
            callbacks.CSVLogger(
                filename=os.path.join(self.output_dir, 'training_log.csv')
            ),
            
            # TensorBoard
            callbacks.TensorBoard(
                log_dir=os.path.join(self.output_dir, 'tensorboard_logs'),
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
        
        return callbacks_list
    
    def train_model(self, X_train, X_val, y_train, y_val):
        """
        Train the DenseNet121 model
        """
        print("\\n🚀 Starting model training...")
        
        # Calculate class weights for balanced training
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        print(f"📊 Class weights: {class_weight_dict}")
        
        # Setup callbacks
        callbacks_list = self.setup_callbacks()
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator()
        
        # Create data generators
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=self.config['batch_size'],
            shuffle=False
        )
        
        # Phase 1: Train with frozen base model
        print("\\n🔄 Phase 1: Training with frozen base model...")
        history_phase1 = self.model.fit(
            train_generator,
            epochs=20,
            validation_data=val_generator,
            class_weight=class_weight_dict,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Phase 2: Fine-tune with unfrozen layers
        print("\\n🔄 Phase 2: Fine-tuning with unfrozen layers...")
        
        # Unfreeze top layers
        self.model.layers[1].trainable = True  # base_model
        
        # Fine-tune from this layer onwards
        fine_tune_at = self.config['fine_tune_at']
        for layer in self.model.layers[1].layers[:fine_tune_at]:
            layer.trainable = False
        
        # Use lower learning rate for fine-tuning
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config['learning_rate']/10),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(f"   Unfrozen layers: {sum([layer.trainable for layer in self.model.layers[1].layers])}")
        print(f"   Trainable parameters: {sum([keras.backend.count_params(w) for w in self.model.trainable_weights]):,}")
        
        # Continue training
        history_phase2 = self.model.fit(
            train_generator,
            epochs=self.config['epochs'],
            initial_epoch=20,
            validation_data=val_generator,
            class_weight=class_weight_dict,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Combine histories
        self.history = {
            'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
            'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
            'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss'],
            'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']
        }
        
        print("✅ Training completed!")
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """
        Comprehensive model evaluation
        """
        print("\\n📊 Evaluating model performance...")
        
        # Predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Test accuracy
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\\n🎯 Test Results:")
        print(f"   Accuracy: {test_acc:.4f}")
        print(f"   Precision: {test_precision:.4f}")
        print(f"   Recall: {test_recall:.4f}")
        print(f"   Loss: {test_loss:.4f}")
        
        # Classification report
        report = classification_report(y_test, y_pred, target_names=self.class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # Save evaluation results
        evaluation_results = {
            'test_accuracy': float(test_acc),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'test_loss': float(test_loss),
            'roc_auc': float(roc_auc),
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        with open(os.path.join(self.output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        return evaluation_results
    
    def save_all_model_formats(self):
        """
        Save model in all possible formats
        """
        print("\\n💾 Saving model in multiple formats...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save complete model (.h5 format)
        h5_path = os.path.join(self.output_dir, f'densenet121_osteoporosis_complete_{timestamp}.h5')
        self.model.save(h5_path)
        print(f"✅ Saved .h5 model: {h5_path}")
        
        # 2. Save in Keras format (.keras)
        keras_path = os.path.join(self.output_dir, f'densenet121_osteoporosis_complete_{timestamp}.keras')
        self.model.save(keras_path)
        print(f"✅ Saved .keras model: {keras_path}")
        
        # 3. Save weights only
        weights_path = os.path.join(self.output_dir, f'densenet121_osteoporosis_weights_{timestamp}.h5')
        self.model.save_weights(weights_path)
        print(f"✅ Saved weights: {weights_path}")
        
        # 4. Save model configuration
        config_path = os.path.join(self.output_dir, f'densenet121_osteoporosis_config_{timestamp}.json')
        with open(config_path, 'w') as f:
            json.dump(self.model.get_config(), f, indent=2)
        print(f"✅ Saved config: {config_path}")
        
        # 5. Save as SavedModel format
        savedmodel_path = os.path.join(self.output_dir, f'densenet121_osteoporosis_savedmodel_{timestamp}')
        self.model.save(savedmodel_path, save_format='tf')
        print(f"✅ Saved SavedModel: {savedmodel_path}")
        
        # 6. Save training history
        history_path = os.path.join(self.output_dir, f'training_history_{timestamp}.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✅ Saved training history: {history_path}")
        
        # 7. Save training configuration
        config_save_path = os.path.join(self.output_dir, f'training_config_{timestamp}.json')
        with open(config_save_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"✅ Saved training config: {config_save_path}")
        
        # 8. Save model as pickle (for compatibility)
        try:
            pickle_path = os.path.join(self.output_dir, f'densenet121_osteoporosis_pickle_{timestamp}.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"✅ Saved pickle model: {pickle_path}")
        except Exception as e:
            print(f"⚠️ Could not save pickle format: {e}")
        
        print(f"\\n🎉 All model formats saved to: {self.output_dir}")
        
        return {
            'h5_model': h5_path,
            'keras_model': keras_path,
            'weights': weights_path,
            'config': config_path,
            'savedmodel': savedmodel_path,
            'history': history_path,
            'training_config': config_save_path
        }
    
    def create_visualizations(self):
        """
        Create training visualizations and save plots
        """
        print("\\n📈 Creating visualizations...")
        
        if self.history is None:
            print("⚠️ No training history available for visualization")
            return
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Training history plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DenseNet121 Osteoporosis Classification Training Results', fontsize=16)
        
        # Accuracy plot
        axes[0, 0].plot(self.history['accuracy'], label='Training Accuracy', color='blue')
        axes[0, 0].plot(self.history['val_accuracy'], label='Validation Accuracy', color='red')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss plot
        axes[0, 1].plot(self.history['loss'], label='Training Loss', color='blue')
        axes[0, 1].plot(self.history['val_loss'], label='Validation Loss', color='red')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Model architecture summary
        axes[1, 0].text(0.1, 0.8, f'Model: DenseNet121', fontsize=12, weight='bold')
        axes[1, 0].text(0.1, 0.7, f'Input Shape: {self.config["input_shape"]}', fontsize=10)
        axes[1, 0].text(0.1, 0.6, f'Total Parameters: {self.model.count_params():,}', fontsize=10)
        axes[1, 0].text(0.1, 0.5, f'Batch Size: {self.config["batch_size"]}', fontsize=10)
        axes[1, 0].text(0.1, 0.4, f'Epochs: {self.config["epochs"]}', fontsize=10)
        axes[1, 0].text(0.1, 0.3, f'Learning Rate: {self.config["learning_rate"]}', fontsize=10)
        axes[1, 0].set_title('Training Configuration')
        axes[1, 0].axis('off')
        
        # Final metrics
        final_train_acc = self.history['accuracy'][-1]
        final_val_acc = self.history['val_accuracy'][-1]
        final_train_loss = self.history['loss'][-1]
        final_val_loss = self.history['val_loss'][-1]
        
        axes[1, 1].text(0.1, 0.8, 'Final Training Metrics:', fontsize=12, weight='bold')
        axes[1, 1].text(0.1, 0.7, f'Training Accuracy: {final_train_acc:.4f}', fontsize=10)
        axes[1, 1].text(0.1, 0.6, f'Validation Accuracy: {final_val_acc:.4f}', fontsize=10)
        axes[1, 1].text(0.1, 0.5, f'Training Loss: {final_train_loss:.4f}', fontsize=10)
        axes[1, 1].text(0.1, 0.4, f'Validation Loss: {final_val_loss:.4f}', fontsize=10)
        axes[1, 1].set_title('Final Results')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'training_visualization.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Training visualization saved: {plot_path}")
        
        return plot_path

def main():
    """
    Main training function
    """
    print("🏥 DenseNet121 Osteoporosis Classification Training")
    print("=" * 60)
    
    # Configuration
    dataset_path = "Dataset/KNEE/Osteoporosis/Combined_Osteoporosis_Dataset"
    output_dir = "new"
    
    # Initialize trainer
    trainer = DenseNet121OsteoporosisTrainer(dataset_path, output_dir)
    
    # Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.load_and_preprocess_data()
    
    # Create model
    model = trainer.create_densenet121_model()
    
    # Train model
    history = trainer.train_model(X_train, X_val, y_train, y_val)
    
    # Evaluate model
    evaluation_results = trainer.evaluate_model(X_test, y_test)
    
    # Save all model formats
    saved_files = trainer.save_all_model_formats()
    
    # Create visualizations
    trainer.create_visualizations()
    
    print("\\n🎉 Training Complete!")
    print("=" * 60)
    print(f"📁 All files saved to: {output_dir}")
    print(f"🎯 Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
    print(f"🎯 Test accuracy: {evaluation_results['test_accuracy']:.4f}")
    print("\\n📂 Generated Files:")
    for key, path in saved_files.items():
        print(f"   • {key}: {path}")

if __name__ == "__main__":
    main()