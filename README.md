# 🧠 Brain Tumor Detection Using Deep Learning

This project detects brain tumors in MRI images via deep learning—leveraging **VGG16 (Transfer Learning)** and custom CNN layers for accurate classification into **Tumor** or **No Tumor**.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [How It Works](#how-it-works)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

---

## 📝 Overview
Brain tumors, when detected early, allow for timely treatment and improved outcomes. This project addresses this need by classifying brain MRI scans into:
- **Tumor**
- **No Tumor**

Deep learning models are used to analyze MRI images and generate predictions that can assist medical professionals.

---

## 📁 Dataset
- Source: **Brain Tumor MRI Dataset** by Masoud Nickparvar on Kaggle  
  https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset 
- Contains ~7,022 images divided into four classes:  
  - *Glioma*  
  - *Meningioma*  
  - *Pituitary tumor*  
  - *No tumor*  
- Classification tasks in this project may focus on binary classification (Tumor vs No Tumor) or multi‑class depending on your approach.

---

## 🛠️ Tech Stack
- Python  
- TensorFlow & Keras  
- VGG16 (ImageNet pre-trained, fine‑tuned)  
- NumPy, Matplotlib  
- PIL (Pillow) for image operations  
- Jupyter Notebook / Google Colab

---

## ⚙️ How It Works
1. Download and extract the Kaggle dataset  
2. Preprocess images: resize, normalize, and augment  
3. Use VGG16 (without top layers) as a feature extractor  
4. Add classification layers (dense, dropout, output)  
5. Train and validate the model  
6. Evaluate using accuracy/loss curves and a confusion matrix  
7. Enable real-time single-image prediction via a helper function

---

## 🧠 Model Architecture
- **Base layer**: Pre-trained VGG16 without the top classification layers  
- **Custom classifier** layers:
  - GlobalAveragePooling2D  
  - Dense layer with ReLU activation  
  - Dropout for regularization  
  - Final output layer (Sigmoid for binary, Softmax for multi-class)

---

## 📊 Results
- *Training Accuracy*: ~97%  
- *Validation Accuracy*: ~95% or above  
- Evaluation metrics include confusion matrix and model performance plots  



---

## ▶️ Usage

### 🔧 Clone the repository
```bash
git clone https://github.com/nbr233/brain-tumour-detection-using-deep-learning.git
cd brain-tumour-detection-using-deep-learning
