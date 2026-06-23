# MobileNetV2-Based Clinical Decision Support System for Automated Pneumonia Screening from Chest X-Ray Images
## LIVE DEMO : https://pneumovision-ai.streamlit.app/
## Overview

This project presents an AI-powered Clinical Decision Support System (CDSS) for automated pneumonia screening using chest X-ray images. The system leverages Transfer Learning with MobileNetV2 to classify chest radiographs into Normal and Pneumonia categories.

The objective is to assist healthcare professionals by providing rapid and reliable screening predictions along with confidence scores, enabling faster clinical decision-making.

---

## Problem Statement

Pneumonia is one of the leading causes of respiratory illness worldwide. Accurate diagnosis often relies on expert interpretation of chest X-rays, which can be time-consuming and resource-intensive.

This project aims to develop an automated deep learning-based screening system capable of identifying pneumonia from chest X-ray images while maintaining high diagnostic reliability.

---

## Objectives

- Develop an automated pneumonia screening system.
- Compare baseline CNN performance against transfer learning models.
- Improve diagnostic performance using MobileNetV2.
- Minimize false-negative predictions.
- Build a deployment-ready clinical decision support application.
- Provide confidence-based prediction outputs.

---

## Dataset

### Dataset Used

Chest X-Ray Images (Pneumonia) Dataset

### Classes

- Normal
- Pneumonia

### Dataset Split

- Training Set
- Validation Set
- Test Set

### Dataset Characteristics

- Medical chest radiograph images
- Binary classification problem
- Class imbalance toward pneumonia samples
- Variable image dimensions before preprocessing

---

## Project Workflow

### 1. Dataset Audit

Performed dataset inspection to understand:

- Class distribution
- Number of images
- Dataset structure
- Image characteristics

Notebook:
`01_dataset_audit.ipynb`

---

### 2. Exploratory Data Analysis (EDA)

Conducted exploratory analysis to identify:

- Class imbalance
- Image dimension variations
- Sample chest X-ray visualization
- Dataset quality assessment

Notebook:
`02_eda.ipynb`

---

### 3. Data Preprocessing

Preprocessing steps included:

- Image resizing to 160×160
- Pixel normalization
- Dataset generation using ImageDataGenerator
- Batch preparation for training

Notebook:
`03_preprocessing.ipynb`

---

### 4. Baseline CNN Development

A custom CNN architecture was developed as a benchmark model.

Architecture included:

- Convolution Layers
- Max Pooling Layers
- Dense Layers
- Binary Output Layer

Notebook:
`04_baseline_cnn.ipynb`

---

### 5. Transfer Learning with MobileNetV2

Transfer learning was implemented using MobileNetV2 pretrained on ImageNet.

Advantages:

- Lightweight architecture
- Efficient feature extraction
- Faster convergence
- Improved generalization

Custom classification layers were added on top of the pretrained backbone.

Notebook:
`05_transfer_learning.ipynb`

---

### 6. Model Inference

Implemented inference pipeline for:

- Model loading
- Image preprocessing
- Prediction generation
- Confidence score calculation
- Result visualization

Notebook:
`06_model_inference.ipynb`

---

## Experiments Conducted

### Experiment 1: Baseline CNN

Purpose:

Evaluate custom CNN performance.

Results:

- Accuracy: 62.5%
- High bias toward pneumonia predictions

---

### Experiment 2: MobileNetV2 Transfer Learning

Purpose:

Improve feature extraction through pretrained weights.

Results:

- Accuracy: 79.8%
- High pneumonia recall
- Low false-negative count

---

### Experiment 3: MobileNetV2 with Data Augmentation

Purpose:

Analyze impact of augmentation techniques.

Techniques:

- Rotation
- Zoom
- Horizontal Flip
- Width Shift
- Height Shift

Results:

- Accuracy: 89.3%
- Increased false negatives

---

## Model Comparison

| Model | Accuracy (%) |
|---------|---------|
| Baseline CNN | 62.5 |
| MobileNetV2 | 79.8 |
| MobileNetV2 + Augmentation | 89.3 |

---

## Final Model Selection

The original MobileNetV2 model was selected as the deployment model.

Reasons:

- Strong diagnostic performance
- High recall
- Lower false negatives
- Better clinical reliability

Final Model:

`mobilenetv2_original.keras`

---

## Streamlit Clinical Decision Support System

A web-based application was developed using Streamlit.

### Features

- Upload chest X-ray images
- Automated pneumonia screening
- Confidence score generation
- Clinical recommendation display
- Interactive user interface

Application File:

`app.py`

---

## Project Structure

```text
Pneumonia_CDSS_Project/
│
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_baseline_cnn.ipynb
│   ├── 05_transfer_learning.ipynb
│   └── 06_model_inference.ipynb
│
├── outputs/
│   ├── model_comparison.csv
│   └── sample_prediction.png
│
├── reports/
│   └── project_report.docx
│
├── app.py
├── requirements.txt
│
├── mobilenetv2_original.keras
└── mobilenetv2_augmented.keras
```

## Technologies Used

### Programming Language

- Python

### Deep Learning

- TensorFlow
- Keras
- MobileNetV2

### Data Processing

- NumPy
- OpenCV
- PIL

### Visualization

- Matplotlib

### Deployment

- Streamlit

---

## Installation

```bash
pip install -r requirements.txt
```

## Run Application

```bash
streamlit run app.py
```

---

## Sample Output

Input:

- Chest X-ray Image

Output:

- Prediction: Normal / Pneumonia
- Confidence Score
- Clinical Recommendation

---

## Future Enhancements

- Grad-CAM Explainability
- Multi-Class Lung Disease Classification
- Real-Time Clinical Deployment
- Cloud-Based Inference
- Electronic Health Record (EHR) Integration
- Model Monitoring and Performance Tracking

---

## Author

**Disha R**

B.E. Artificial Intelligence & Data Science

Nitte Meenakshi Institute of Technology (NMIT)

Bengaluru, India
