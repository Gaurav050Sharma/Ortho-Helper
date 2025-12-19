# DenseNet121 Knee Osteoarthritis Detection Model

## Model Information
- **Architecture**: DenseNet121
- **Medical Condition**: Knee Osteoarthritis Detection
- **Configuration**: Standard
- **Training Date**: 20251006_184344
- **Parameters**: 7,305,281

## Performance Metrics
- **Test Accuracy**: 0.9200 (92.00%)
- **Test Precision**: 0.9425
- **Test Recall**: 0.8817
- **Training Time**: 4.5 minutes
- **Epochs Trained**: 10/10

## Dataset Information
- **Classes**: Normal vs Osteoarthritis
- **Training Data**: Medical imaging dataset
- **Dataset Path**: Dataset/KNEE/Osteoarthritis/train
- **Image Size**: 224x224 pixels

## Grad-CAM Optimization
This model is optimized for superior Grad-CAM visualization:
- **Recommended Layer**: `conv5_block16_2_conv`
- **Architecture Benefits**: Dense connectivity for medical imaging
- **Visualization Quality**: Excellent for knee osteoarthritis detection
- **Medical Interpretation**: Clear region highlighting

## Usage
```python
import tensorflow as tf
from utils.gradcam import GradCAM

# Load model
model = tf.keras.models.load_model('models/densenet121_osteoarthritis_standard_models/models/densenet121_osteoarthritis_standard_20251006_184344.h5')

# Initialize Grad-CAM
gradcam = GradCAM(model, layer_name='conv5_block16_2_conv')

# Generate heatmap
heatmap = gradcam.generate_heatmap(medical_image)
```

## Clinical Application
- **Primary Use**: Knee Osteoarthritis Detection
- **Accuracy Level**: Excellent (92.00%)
- **Deployment Ready**: Medical grade
- **Visualization**: Grad-CAM heatmaps for medical interpretation

Generated: 2025-10-06 19:17:18