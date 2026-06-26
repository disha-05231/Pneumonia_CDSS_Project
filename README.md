```markdown
# 🫁 PneumoVision AI
### MobileNetV2-Based Clinical Decision Support System for Automated Pneumonia Screening using Chest X-Ray Images

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-blue)

---

## 🚀 Live Demo

🔗 **https://pneumovision-ai.streamlit.app/**

---

# 📌 Overview

PneumoVision AI is an AI-powered Clinical Decision Support System (CDSS) developed for automated pneumonia screening from chest X-ray images. The system leverages **Transfer Learning with MobileNetV2** to classify radiographs into **Normal** and **Pneumonia** categories while providing confidence scores, probability analysis, clinical recommendations, and downloadable diagnostic reports.

The application demonstrates an end-to-end deep learning pipeline, including dataset exploration, preprocessing, model development, evaluation, deployment, and an interactive Streamlit dashboard for real-time inference.

This project is intended for **educational and research purposes** and showcases how deep learning can assist clinicians by providing rapid preliminary screening.

---

# 🎯 Problem Statement

Pneumonia is one of the leading causes of respiratory illness worldwide. Diagnosis typically requires interpretation of chest radiographs by trained radiologists, which may not always be immediately available.

This project aims to build an AI-assisted clinical decision support system capable of automatically screening chest X-rays for pneumonia, enabling faster preliminary assessment while supporting healthcare professionals in clinical decision-making.

---

# 🎯 Objectives

- Develop an automated pneumonia screening system.
- Compare a baseline CNN against Transfer Learning models.
- Improve diagnostic performance using MobileNetV2.
- Reduce false-negative predictions.
- Build a deployment-ready Streamlit application.
- Generate confidence-based predictions.
- Provide AI-assisted clinical recommendations.
- Visualize image preprocessing for interpretability.

---

# ⭐ Key Features

- Upload Chest X-ray Images
- AI Diagnostic Report
- Pneumonia Prediction
- Confidence Score
- Clinical Risk Assessment
- AI Probability Analysis
- Computer Vision Processing Pipeline
- Image Metadata Extraction
- AI Processing Summary
- Clinical Interpretation
- Clinical Recommendation
- Downloadable Clinical Report
- Interactive Streamlit Dashboard

---

# 📂 Dataset

## Dataset Used

**Chest X-Ray Images (Pneumonia)**

Dataset contains two categories:

- Normal
- Pneumonia

---

## Dataset Split

- Training Set
- Validation Set
- Test Set

---

## Dataset Characteristics

- Binary classification dataset
- Medical chest radiographs
- Class imbalance toward pneumonia samples
- Variable image dimensions
- JPEG image format

---

# ⚙️ Complete Project Workflow

```

Chest X-ray Image
│
▼
Image Upload
│
▼
Dataset Inspection
│
▼
Exploratory Data Analysis
│
▼
Image Preprocessing
│
├── Resize (160×160)
├── Pixel Normalization
├── RGB Formatting
├── CLAHE Visualization
├── Grayscale Visualization
└── Edge Detection Visualization
│
▼
Transfer Learning
(MobileNetV2)
│
▼
Prediction
│
▼
Confidence Score
│
▼
Probability Analysis
│
▼
Clinical Recommendation
│
▼
Download Clinical Report

```

---

# 📊 Dataset Audit

Performed an initial inspection to understand the dataset.

Activities performed:

- Dataset structure analysis
- Image count verification
- Class distribution analysis
- Folder organization
- Sample visualization
- Dataset consistency check

Notebook:

```

01_dataset_audit.ipynb

```

---

# 📈 Exploratory Data Analysis (EDA)

EDA was performed to better understand the dataset characteristics.

Analysis included:

- Class imbalance visualization
- Sample chest X-rays
- Image dimension analysis
- Dataset quality inspection
- Pixel intensity observations

Notebook:

```

02_eda.ipynb

```

---

# 🖼️ Image Preprocessing

Each uploaded chest X-ray undergoes preprocessing before inference.

Preprocessing includes:

- Image resizing to **160 × 160**
- RGB formatting
- Pixel normalization
- Batch generation using ImageDataGenerator

For visualization within the dashboard, additional computer vision techniques are displayed:

- Grayscale Conversion
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Canny Edge Detection

> **Note:**  
> CLAHE, Grayscale, and Edge Detection are shown for visualization and interpretability.  
> The MobileNetV2 model performs inference using the resized RGB image.

Notebook:

```

03_preprocessing.ipynb

```

---

# 🤖 Baseline CNN

A custom CNN model was initially developed to establish baseline performance.

Architecture included:

- Convolution Layers
- ReLU Activation
- Max Pooling
- Fully Connected Layers
- Sigmoid Output Layer

Notebook:

```

04_baseline_cnn.ipynb

```

---

# 🚀 Transfer Learning with MobileNetV2

Transfer Learning was implemented using MobileNetV2 pretrained on ImageNet.

### Why MobileNetV2?

- Lightweight architecture
- Efficient feature extraction
- Faster convergence
- Lower computational cost
- Better generalization
- Suitable for real-time deployment

Custom classification layers were added above the pretrained backbone.

Notebook:

```

05_transfer_learning.ipynb

```

---

# 🔍 Model Inference

Inference pipeline performs:

- Image loading
- Image preprocessing
- Model prediction
- Confidence calculation
- Probability estimation
- Clinical recommendation generation

Notebook:

```

06_model_inference.ipynb

```

---

# 🧪 Experiments Conducted

## Experiment 1 — Baseline CNN

Purpose:

Evaluate custom CNN performance.

Results:

- Accuracy: **62.5%**
- High bias toward pneumonia predictions

---

## Experiment 2 — MobileNetV2

Purpose:

Improve feature extraction using pretrained weights.

Results:

- Accuracy: **79.8%**
- Improved recall
- Lower false negatives

---

## Experiment 3 — MobileNetV2 + Data Augmentation

Techniques Used:

- Rotation
- Zoom
- Horizontal Flip
- Width Shift
- Height Shift

Results:

- Accuracy: **89.3%**
- Higher overall accuracy
- Increased false negatives

---

# 📊 Model Comparison

| Model | Accuracy | Observation |
|---------|-----------|------------------------------|
| Baseline CNN | 62.5% | Underfitting |
| MobileNetV2 | 79.8% | Better Feature Extraction |
| MobileNetV2 + Augmentation | 89.3% | Higher Accuracy but More False Negatives |

---

# ✅ Final Model Selection

The original MobileNetV2 model was selected for deployment.

Reasons:

- Better clinical reliability
- Higher pneumonia recall
- Lower false-negative rate
- Stable inference
- Lightweight architecture

Final model:

```

mobilenetv2_original.keras

```

---

# 🖥️ Streamlit Clinical Decision Support System

An interactive web application was developed using Streamlit.

### Dashboard Features

- Upload Chest X-ray
- AI Diagnostic Report
- Prediction
- Confidence Score
- Clinical Risk Level
- AI Probability Analysis
- Computer Vision Processing Pipeline
- Image Information
- AI Processing Summary
- Clinical Recommendation
- Clinical Interpretation
- Download Clinical Report

---

# 🧠 Computer Vision Processing

To improve transparency and interpretability, the dashboard visualizes the image preprocessing pipeline.

Displayed visualizations include:

- Model Input
- Grayscale Conversion
- CLAHE Enhanced Image
- Edge Detection

These visualizations help users understand how the uploaded X-ray is processed before inference.

---

# 📸 Dashboard Preview

## Home Screen

```

Add Screenshot Here

```

---

## Prediction Dashboard

```

Add Screenshot Here

```

---

## Clinical Report

```

Add Screenshot Here

````

---

# 📁 Project Structure

```text
PneumoVisionAI/
│
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_baseline_cnn.ipynb
│   ├── 05_transfer_learning.ipynb
│   └── 06_model_inference.ipynb
│
├── assets/
│   ├── dashboard.png
│   ├── workflow.png
│   └── prediction.png
│
├── outputs/
│   ├── model_comparison.csv
│   └── sample_prediction.png
│
├── reports/
│   └── project_report.docx
│
├── mobilenetv2_original.keras
├── mobilenetv2_augmented.keras
├── app.py
├── requirements.txt
└── README.md
````

---

# 🛠️ Technologies Used

| Category            | Technology        |
| ------------------- | ----------------- |
| Programming         | Python            |
| Deep Learning       | TensorFlow, Keras |
| Transfer Learning   | MobileNetV2       |
| Computer Vision     | OpenCV            |
| Image Processing    | PIL               |
| Numerical Computing | NumPy             |
| Visualization       | Matplotlib        |
| Deployment          | Streamlit         |

---

# ⚡ Installation

```bash
git clone https://github.com/your-username/PneumoVisionAI.git

cd PneumoVisionAI

pip install -r requirements.txt
```

---

# ▶️ Run Application

```bash
streamlit run app.py
```

---

# 📤 Sample Output

Input:

* Chest X-ray Image

Output:

* Prediction (Normal / Pneumonia)
* Confidence Score
* Clinical Risk Level
* Probability Analysis
* Clinical Recommendation
* Downloadable Clinical Report

---

# ⚠️ Limitations

* Intended for educational and research purposes.
* Does not replace professional medical diagnosis.
* Supports binary classification only.
* Performance depends on image quality.
* Requires further clinical validation before real-world deployment.

---

# 🚀 Future Enhancements

* Grad-CAM Explainability
* Lung Segmentation
* Multi-class Lung Disease Classification
* DICOM Image Support
* REST API Integration
* Electronic Health Record (EHR) Integration
* Cloud Deployment
* Model Monitoring Dashboard

---

# 👩‍💻 Author

**Disha R**

B.E. Artificial Intelligence & Data Science

Nitte Meenakshi Institute of Technology (NMIT)

Bengaluru, Karnataka, India

---

# 📜 Disclaimer

This Clinical Decision Support System (CDSS) is intended solely for educational and research purposes. Predictions generated by the model should not be considered a substitute for professional medical diagnosis or clinical judgment. Final diagnosis and treatment decisions should always be made by qualified healthcare professionals.

```
```
