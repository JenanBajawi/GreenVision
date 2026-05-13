# 🍇 GreenVision – Grape Leaf Disease Detection & Severity Estimation

An end-to-end deep learning application that detects diseases on grape leaves, estimates infection severity, and visually explains its predictions using Grad-CAM. Built as a scientific diagnostic instrument with an editorial-botanical interface using **Streamlit**.

This project was developed as a Computer Vision course project at **Umm Al-Qura University · Group 19**, supervised by **Dr. Serene Noor Wali**.

---

## ✨ Features

- Classifies grape leaves into **4 categories** (Black Rot, Esca, Leaf Blight, Healthy)
- Estimates **infection severity** as a continuous percentage
- Generates **Grad-CAM heatmaps** to visualize which regions influenced the diagnosis
- Provides **disease-specific treatment recommendations**
- Editorial botanical UI built with custom CSS (Fraunces + Manrope typography)
- Automatic model download from **Hugging Face Hub** (no manual setup)
- Beginner-friendly and well-structured

---

## 🛠 Technologies Used

- **Language:** Python
- **Deep Learning:** PyTorch, Torchvision
- **Architecture:** ResNet34 (Transfer Learning)
- **Explainability:** Grad-CAM (pytorch-grad-cam)
- **Web App:** Streamlit
- **Image Processing:** Pillow, NumPy
- **Model Hosting:** Hugging Face Hub
- **Environment:** Python 3.9+ · Jupyter Notebook (training)

---

## 🧹 Data Preparation

Data preparation ensures the model trains on clean, balanced grape leaf imagery.

### Steps Performed
- Used the **PlantVillage** grape leaf subset (4 classes)
- Resized all images to **224 × 224**
- Applied **data augmentation**: random flips, rotations, color jitter
- Normalized using ImageNet mean/std statistics
- Split into **train / validation / test** (70 / 15 / 15)
- Balanced class distributions to reduce bias

---

## 🔍 Model Training

Two separate models were trained on the same backbone (ResNet34) for two different tasks.

### 1. Classification Model
- ResNet34 pretrained on ImageNet
- Final fully-connected layer replaced with `Linear(in_features, 4)`
- Loss: **CrossEntropyLoss**
- Optimizer: Adam, lr = 1e-4
- Achieved **97.22% accuracy** on the test set

### 2. Severity Regression Model
- ResNet34 backbone with a custom regression head:
  - `Linear → ReLU → Dropout(0.3) → Linear → Sigmoid`
- Loss: **MSELoss**
- Outputs a value in **[0, 1]** interpreted as percentage of affected leaf tissue

---

## 🧠 Grad-CAM Visual Explanation

Gradient-weighted Class Activation Mapping highlights the leaf regions that drove each diagnostic decision — turning the model from a black box into an interpretable instrument.

- Hooks into the last convolutional block of ResNet34
- Produces a heatmap overlay on the original leaf image
- Helps validate that the model focuses on **lesions** rather than background

---

## 📊 Application Workflow

| Step | What Happens |
|------|--------------|
| 1 | User uploads a grape leaf image |
| 2 | Image is preprocessed (resize + normalize) |
| 3 | Classification model predicts the disease class |
| 4 | Regression model estimates infection severity |
| 5 | Grad-CAM produces a visual explanation heatmap |
| 6 | App displays probability distribution + treatment protocol |

---

## 💡 Key Insights

- **Esca** is the most difficult class — its tiger-stripe pattern is sometimes confused with Black Rot
- **Healthy specimens** are detected with the highest confidence (>98%)
- Severity correlates strongly with **lesion coverage** seen in Grad-CAM
- Grad-CAM confirms the model relies on **disease patches**, not background pixels — a sign of a well-generalized model

---

## 🌿 Supported Specimens

| № | Common Name | Latin Name |
|----|------------|------------|
| 01 | Black Rot | *Guignardia bidwellii* |
| 02 | Esca | *Phaeomoniella chlamydospora* |
| 03 | Leaf Blight | *Isariopsis griseola* |
| 04 | Healthy Specimen | *Vitis vinifera* |

---

## 🎯 Project Purpose

- Apply deep learning to a real agricultural problem
- Practice transfer learning with **ResNet34**
- Combine **classification + regression + XAI** in a single pipeline
- Build a polished, scientific user interface with Streamlit
- Lay a foundation for future precision-agriculture applications

---

## 🚀 Try It

### Option 1 — Streamlit Cloud (Online)
🔗 **Live Demo:** *https://your-streamlit-app-url.streamlit.app*

### Option 2 — Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/greenvision.git
cd greenvision

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

> The trained models will be downloaded automatically from
> [`JenanB/greenvision-model`](https://huggingface.co/JenanB/greenvision-model)
> on first launch.

---

## 📦 Requirements

```
streamlit>=1.28.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
numpy>=1.24.0
huggingface_hub>=0.20.0
grad-cam>=1.4.8
opencv-python-headless>=4.8.0
```

---

## 📂 Project Structure

```
greenvision/
 ├── app.py                  # Streamlit application
 ├── requirements.txt        # Python dependencies
 ├── notebooks/
 │    ├── training.ipynb     # Classification model training
 │    └── severity.ipynb     # Regression model training
 ├── images/                 # Screenshots & sample images
 └── README.md
```
