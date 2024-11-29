# Model Documentation for Plant Disease Detection

## Introduction
This document provides comprehensive details about the plant disease detection model built using YOLOv8. The model identifies various plant diseases and suggests relevant preventive measures, cures, and related products. This model is designed for agricultural applications to help farmers identify plant health issues early and take appropriate actions.

## Model Architecture
The model used for plant disease detection is **YOLOv8**, a popular deep learning model known for its accuracy and speed in object detection tasks. We have trained this model on a custom dataset containing images of different plant diseases. The architecture is designed to detect bounding boxes around affected areas and classify the type of disease.

### Key Features
- **Input Size**: 640x640 pixels.
- **Model Type**: YOLOv8 (You Only Look Once version 8).
- **Output**: Detection bounding boxes, class labels, and confidence scores.

## Data Used for Training
The model was trained on a dataset that includes:
- **Source**: Publicly available plant disease datasets and custom images collected from local farms.
- **Preprocessing Steps**:
  - Image resizing and normalization.
  - Data augmentation techniques, such as rotation, flipping, and brightness adjustments.

## Training Details
### Environment
- **Programming Language**: Python 3.8
- **Framework**: PyTorch 1.13
- **Training Time**: Approximately 6 hours on an NVIDIA GPU (e.g., RTX 3080)
- **Hyperparameters**:
  - **Learning Rate**: 0.001
  - **Batch Size**: 16
  - **Epochs**: 25


![confusion matrix](assets/confusion_matrix (1).png)
*Confusion matrix showing the performance of the model across different classes.*

![F1_curve](assets/F1_curve (1).png)
*F1 curve illustrating the trade-off between precision and recall across various thresholds.*

![labels_correlogram](assets/labels_correlogram.jpg)
*Correlogram depicting the correlation between different labels in the dataset.*

![results](assets/results.png)
*Summary of the model's overall performance, including key metrics and evaluation results.*



### Training Procedure
1. The model was initialized with pre-trained weights from the YOLOv8 repository.
2. The dataset was divided into training (80%), validation (10%), and test (10%) sets.
3. The training loop included standard augmentation techniques and an early stopping condition for optimal performance.

## Model Performance
### Evaluation Metrics
- **Precision**: 0.97
- **Recall**: 0.92
- **F1 Score**: 0.90
- **Mean Average Precision (mAP)**: 0.85

### Performance Analysis
The model performed well on unseen test data, with a high recall indicating effective detection of diseases. Some challenges included handling images with multiple overlapping diseases, which occasionally reduced accuracy.

## How to Use the Model
To use the trained model, refer to the [README.md](../README.md) file for detailed instructions on setup, model usage, and examples.

