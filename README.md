# 🍇 GreenVision – Grape Leaf Disease Detection & Severity Estimation

<p align="center">
  <img src="images/banner.png" width="100%" alt="GreenVision Banner">
</p>

<p align="center">
An end-to-end deep learning application that detects grape leaf diseases, estimates infection severity, and visually explains predictions using Grad-CAM.
</p>

---

## 🌿 Overview

GreenVision is an AI-powered agricultural diagnostic system built using Deep Learning and Explainable AI techniques.

The application can:

- Detect grape leaf diseases
- Estimate severity percentage
- Generate Grad-CAM visual explanations
- Provide treatment recommendations

Built with PyTorch and Streamlit using a scientific botanical interface design.

> Developed as a Computer Vision course project at Umm Al-Qura University under the supervision of Dr. Serene Noor Wali.

---

# ✨ Application Screenshots

## 🖥 Main Interface

> ضع هنا صورة الصفحة الرئيسية للتطبيق

<p align="center">
  <img src="images/home.png" width="90%">
</p>

---

## 🔍 Disease Prediction Result

> ضع هنا صورة النتيجة بعد رفع الصورة

<p align="center">
  <img src="images/result.png" width="90%">
</p>

---

## 🧠 Grad-CAM Visualization

> ضع هنا صورة الـ Heatmap

<p align="center">
  <img src="images/gradcam.png" width="90%">
</p>

---

## 📊 Probability Distribution & Severity Estimation

> ضع هنا صورة الاحتمالات ونسبة الإصابة

<p align="center">
  <img src="images/probabilities.png" width="90%">
</p>

---

# ✨ Features

- 🍇 Classifies grape leaves into 4 categories
  - Black Rot
  - Esca
  - Leaf Blight
  - Healthy

- 📈 Estimates infection severity percentage

- 🧠 Generates Grad-CAM heatmaps for explainability

- 💊 Provides disease-specific treatment recommendations

- 🎨 Botanical editorial UI using custom CSS

- ☁️ Automatic model download from Hugging Face Hub

- 🧩 Beginner-friendly structure

---

# 🛠 Technologies Used

| Category | Technologies |
|---|---|
| Language | Python |
| Deep Learning | PyTorch, Torchvision |
| Architecture | ResNet34 |
| Explainability | Grad-CAM |
| Web Framework | Streamlit |
| Image Processing | Pillow, NumPy |
| Model Hosting | Hugging Face Hub |

---

# 🧹 Data Preparation

The dataset preparation pipeline included:

- Resizing images to 224×224
- Data augmentation
- Image normalization
- Train / Validation / Test split
- Class balancing

Dataset used:
PlantVillage grape leaf subset (4 classes)

---

# 🔍 Model Training

## 1️⃣ Classification Model

- Backbone: ResNet34 pretrained on ImageNet
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Learning Rate: 1e-4

### ✅ Performance

Achieved **97.22% test accuracy**

---

## 2️⃣ Severity Regression Model

Custom regression head:

```python
Linear → ReLU → Dropout(0.3) → Linear → Sigmoid
