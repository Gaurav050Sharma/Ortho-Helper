# DenseNet121 Limb Abnormalities Detection Model

## Model Information
- **Architecture**: DenseNet121
- **Medical Condition**: Limb Abnormalities Detection
- **Configuration**: Standard
- **Training Date**: 20251006_185751
- **Parameters**: 7,305,281

## Performance Metrics
- **Test Accuracy**: 0.7250 (72.50%)
- **Test Precision**: 0.7647
- **Test Recall**: 0.6500
- **Training Time**: 2.9 minutes
- **Epochs Trained**: 6/10

## Dataset Information
- **Classes**: Normal vs Abnormal
- **Training Data**: Medical imaging dataset
- **Dataset Path**: Dataset/ARM/MURA_Organized/train
- **Image Size**: 224x224 pixels

## Grad-CAM Optimization
This model is optimized for superior Grad-CAM visualization:
- **Recommended Layer**: `conv5_block16_2_conv`
- **Architecture Benefits**: Dense connectivity for medical imaging
- **Visualization Quality**: Excellent for limb abnormalities detection
- **Medical Interpretation**: Clear region highlighting

## Usage
```python
import tensorflow as tf
from utils.gradcam import GradCAM

# Load model
model = tf.keras.models.load_model('models/densenet121_limbabnormalities_standard_models/models/densenet121_limbabnormalities_standard_20251006_185751.h5')

# Initialize Grad-CAM
gradcam = GradCAM(model, layer_name='conv5_block16_2_conv')

# Generate heatmap
heatmap = gradcam.generate_heatmap(medical_image)
```

## Clinical Application
- **Primary Use**: Limb Abnormalities Detection
- **Accuracy Level**: Moderate (72.50%)
- **Deployment Ready**: Research and development
- **Visualization**: Grad-CAM heatmaps for medical interpretation

Generated: 2025-10-06 19:17:18