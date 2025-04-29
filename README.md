# Breast Cancer Detection using Fusion of TSBTC and MobileNet

This project focuses on the **automated classification of breast cancer histopathology images** using a combination of **TSBTC** (Two-Stage Breast Tissue Classifier) and **MobileNet** features. We utilized the **BreakHis dataset** and performed extensive data augmentation to balance classes and avoid overfitting. The final fusion model achieved an impressive **accuracy of 99.7%** using an MLP classifier.

## 🚀 Project Overview

- **Dataset:** [BreakHis Dataset](https://www.kaggle.com/datasets/ambarish/breakhis)  
- **Objective:** Classify histopathological breast cancer images into **Benign** or **Malignant**
- **Techniques Used:** MobileNet, TSBTC, MLP, Data Augmentation
- **Best Accuracy:** 99.7% with feature fusion model

## 🗂️ Repository Structure

| File | Description |
|------|-------------|
| `augment_dataset.py` | Performs image augmentation (rotation, transformation) on BreakHis dataset to balance benign and malignant classes. |
| `fusion_features_extractor.py` | Extracts and fuses features from MobileNet and TSBTC models. |
| `fusion_model_mlp.h5` | Pretrained MLP model trained on fused features (99.7% accuracy). |
| `mnetxtsbtc_pipeline.py` | Pipeline implementation combining MobileNet and TSBTC feature extractors. |
| `mobile_net_feature_extractor.py` | Extracts deep features from MobileNet for classification. |
| `mobilenet_unfreeze_6_adamw.pth` | Fine-tuned MobileNet model with last 6 layers unfrozen using AdamW optimizer. |
| `results_by_csv.py` | Converts classification results into a CSV format for analysis or reporting. |

## 📊 Methodology

1. **Data Augmentation**
   - Balanced the dataset by augmenting benign class (rotations, flips, zoom, etc.)
2. **Feature Extraction**
   - Extracted deep features using MobileNet and TSBTC
3. **Fusion**
   - Combined features from both models to create a powerful feature vector
4. **Classification**
   - Used a Multi-Layer Perceptron (MLP), SMO (Sequential Minimal Optimization ), and KNN (K-Nearest Neighbors) classifier
   - Achieved 99.7% accuracy

## 📌 Dependencies

- Python 3.11
- TensorFlow 2.11+
- Keras 2.11+
- PyTorch
- OpenCV
- NumPy
- scikit-learn
- Pandas

## 🧪 Results





<p align="center">
  <img src="https://github.com/user-attachments/assets/e1499f0e-7e4a-4acd-b4c8-cd2fd8795c93" alt="hybrid" width="600"/>
</p>


The figure above presents a comparative analysis of various machine learning classifiers applied to a hybrid feature space, which combines handcrafted features and fine-tuned MobileNetV2 features.

MLP (Multilayer Perceptron), SMO (Sequential Minimal Optimization), and KNN (K-Nearest Neighbors) demonstrated exceptional performance, each achieving a remarkable accuracy of 99.7%.

Logistic Regression and SGD (Stochastic Gradient Descent) also performed robustly, with accuracy values exceeding 99%.

These results emphasize the strength of the hybrid feature representation and its compatibility with diverse model architectures, showcasing its powerful pattern recognition capabilities. From a research standpoint, the findings underline the effectiveness of fusing deep and handcrafted features in significantly enhancing classification performance for breast cancer detection.


## 👨‍💻 Authors

- [**Abhijeet Fasate**](https://github.com/AbhijeetFasate13)  
- [**Chetan Padhen**](https://github.com/Chetax)  
- [**Akshay Patil**](https://github.com/devbyakshay)  
- [**Pallive Kadam**](https://github.com/Pallavik24)  
