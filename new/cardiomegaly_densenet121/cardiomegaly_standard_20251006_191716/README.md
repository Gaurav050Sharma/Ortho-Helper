# DenseNet121 Cardiomegaly Detection Model

## Model Information
- **Architecture**: DenseNet121
- **Medical Condition**: Cardiomegaly (Heart Enlargement)
- **Configuration**: Standard
- **Training Date**: 20251006_191716
- **Parameters**: 7,305,281

## Performance Metrics
- **Test Accuracy**: 0.6800 (68.00%)
- **Test Precision**: 0.7216
- **Test Recall**: 0.6542
- **Test Loss**: 0.6093

## Dataset Information
- **Classes**: Normal vs Cardiomegaly
- **Training Samples**: 800
- **Test Samples**: 200
- **Epochs Trained**: 8

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
model = tf.keras.models.load_model('cardiomegaly_densenet121_models\cardiomegaly_standard_20251006_191716\cardiomegaly_densenet121_standard_20251006_191716.keras')

# Initialize Grad-CAM
gradcam = GradCAM(model, layer_name='conv5_block16_2_conv')

# Generate heatmap
heatmap = gradcam.generate_heatmap(chest_xray_image)
```

Generated: 2025-10-06 19:17:18
