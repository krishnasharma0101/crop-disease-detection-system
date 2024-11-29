# Plant Disease Detection Web App

## Project Overview
The **Plant Disease Detection Web App** is a machine learning-based application that can identify plant diseases from images. It uses the YOLOv8 object detection model for disease detection and provides valuable information such as disease descriptions, prevention methods, and product recommendations (e.g., pesticides and fertilizers) from sources like Wikipedia and Amazon India.

## Features
- **Image Upload**: Upload an image of a plant for disease detection.
- **Disease Detection**: Use the YOLOv8 model to detect diseases.
- **Information Retrieval**: Fetch detailed disease descriptions from Wikipedia.
- **Product Recommendations**: Search for relevant products from Amazon India for disease prevention and treatment.
- **User-Friendly Interface**: Built using Streamlit for an intuitive user experience.

## Model Training
### Data Collection
The model was trained using a dataset containing images of healthy plants and plants affected by various diseases. The dataset was sourced from public plant disease repositories and augmented to improve model robustness.

### Model Architecture
The core of the detection system is the **YOLOv8** model, which is well-suited for real-time object detection due to its balance of speed and accuracy. The model was trained using [Ultralytics YOLO](https://github.com/ultralytics/yolov8) library.

### Training Process
1. **Preprocessing**: Image data was preprocessed to standardize input dimensions.
2. **Augmentation**: Image augmentation techniques (e.g., flipping, rotation, and scaling) were applied to increase the model's generalization capabilities.
3. **Training Configuration**:
   - **Batch Size**: 16
   - **Learning Rate**: 0.001
   - **Epochs**: 50
   - **Optimizer**: Adam
4. **Evaluation**: The model was evaluated using precision, recall, and F1-score metrics on a validation set.

### Model File
The trained model is available as `last (2).pt` in the `model/` directory.

## Installation Instructions
To set up the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/krishnasharma0101/crop-disease-detection-system.git


**Navigate to the Project Directory**:
```bash
    cd plant-disease-detection
**Set Up a Virtual Environment**:
```bash
    python -m venv venv
** On Windows, use**
 ```bash
    venv\Scripts\activate`
```bash
    source venv/bin/activate  # On macOS/Linux

**Install Required Packages**:

pip install -r requirements.txt


**Usage Guide**:

Run the application using Streamlit:

streamlit run app.py
Open your web browser and go to http://localhost:8501 to access the app.

## **Demo Video**
### Watch the Project Demo
<video width="640" height="360" controls>
  <source src="assets/demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>



**License**

This project is licensed under the MIT License. You are free to use, modify, or distribute this project as needed.

**Acknowledgements**

1. Ultralytics YOLOv8 for the object detection model.
2. Wikipedia API for fetching disease information.
3. Amazon India for sourcing product recommendations.
4. Other libraries and tools that contributed to the development of this project.

**Contact**

For questions, feedback, or contributions, please reach out to your- krishna.gwl11@gmail.com or create an issue on the GitHub repository.


