# 📁 DenseNet121 Models - Organized Structure

## 🎯 Quick Overview
```
new/
├── README.md                           # 📖 Complete documentation
├── 🦴 osteoporosis_models/            # 80% accuracy model
│   ├── models/                        # 📦 All model files (.keras, .h5, weights, SavedModel)
│   ├── configs/                       # ⚙️ Architecture & training configs  
│   └── results/                       # 📊 Training history & final metrics
├── 🦵 osteoarthritis_models/          # 82% accuracy model
│   ├── models/                        # 📦 All model files (.keras, .h5, weights, SavedModel)
│   ├── configs/                       # ⚙️ Architecture & training configs
│   └── results/                       # 📊 Training history & final metrics
├── ❤️ cardiomegaly_models/            # 62% accuracy model
│   ├── models/                        # 📦 All model files (.keras, .h5, weights, SavedModel)
│   ├── configs/                       # ⚙️ Architecture & training configs
│   └── results/                       # 📊 Training history & final metrics
├── 🫁 pneumonia_models/               # 93% accuracy model 🏆
│   ├── models/                        # 📦 All model files (.keras, .h5, weights, SavedModel)
│   ├── configs/                       # ⚙️ Architecture & training configs
│   └── results/                       # 📊 Training history & final metrics
└── misc_files/                        # 📋 Additional data files
```

## 🏆 Model Performance
| Condition | Accuracy | Dataset Size | Parameters | Anatomy |
|-----------|----------|--------------|------------|---------|
| Osteoporosis | 80.00% | 1,945 images | 7.7M | Knee |
| Osteoarthritis | 82.00% | 9,788 images | 7.7M | Knee |
| Cardiomegaly | 62.00% | 4,438 images | 7.7M | Chest |
| **Pneumonia** | **93.00%** 🏆 | 5,856 images | 7.7M | Chest |

## 🚀 Quick Usage
```python
# Load any model
import tensorflow as tf
model = tf.keras.models.load_model('path/to/model.keras')

# Or load specific condition
osteoporosis_model = tf.keras.models.load_model('osteoporosis_models/models/densenet121_robust_20251005_193045.keras')
osteoarthritis_model = tf.keras.models.load_model('osteoarthritis_models/models/densenet121_osteoarthritis_20251005_194016.keras')
cardiomegaly_model = tf.keras.models.load_model('cardiomegaly_models/models/densenet121_cardiomegaly_20251005_195658.keras')
pneumonia_model = tf.keras.models.load_model('pneumonia_models/models/densenet121_pneumonia_20251005_200721.keras')  # Best performer!
```

## 📂 File Types Available
✅ `.keras` - Native Keras format (recommended)  
✅ `.h5` - Legacy HDF5 format  
✅ `.weights.h5` - Weights only  
✅ `savedmodel/` - TensorFlow production format  
✅ `.json` - Configurations and results  
✅ `.txt` - Model summaries  

---
*All models trained on October 5, 2025 using DenseNet121 architecture*