# # Project Title: Heart Disease Prediction and Breast Cancer Detection

## Heart Disease Prediction Machine Learning Project

### 1. Project Overview
This project uses machine learning methods to predict heart disease and compares the performance of multiple classic machine learning models. The data comes from https://www.kaggle.com/datasets/shantanugarg274/heart-prediction-dataset-quantum/data on kaggle. It is the final assignment 1 of my master's AI and Machine Learning course.

### Dataset
- `Heart Prediction Quantum Dataset.csv`: A heart disease prediction dataset with 500 records and 7 features.

### File Description
- `AI_And_MachineLearning_FinalTask02.ipynb`: Jupyter notebook with complete data analysis and model implementation code.
- `AI_And_MachineLearning_FinalTask02.pdf`: PDF version of the project report.

### Implemented Models
- L1 Regression (Lasso)
- L2 Regression (Ridge)
- Polynomial Regression
- Support Vector Machine (SVM)
- Decision Tree
- Naive Bayes

### Results
L1 Regression performs best in terms of AUC-ROC metric (0.981), while Decision Tree performs best in terms of accuracy and precision (92.7% accuracy and 96.5% precision).

### 2. Project Overview  
This project implements convolutional neural networks (CNNs) for classifying breast cancer thermal imaging data into three categories: **normal**, **sick**, and **unknown**. It compares classic and modern CNN architectures to optimize medical image analysis accuracy and computational efficiency. Developed as the final project for the AI and Machine Learning course at Tomsk State University.

---

### Dataset  
- **Source**: [Breast Cancer Detection Using Thermography](https://www.kaggle.com/datasets/thilak02/breast-cancer-detection-using-thermography/data)  
- **Samples**: 362 thermal images (162 normal, 100 sick, 100 unknown)  
- **Split**: Stratified 70%-15%-15% partitioning for training, validation, and testing  

---

### File Description  
- `AI_And_MachineLearning_FinalTask02(second).ipynb`:  
  - Data preprocessing pipeline with stratified sampling  
  - CNN model implementations (LeNet-5, AlexNet, VGG16, ResNet50)  
  - Training/validation workflows with early stopping  
  - Performance visualization (loss curves, confusion matrices)  
- `BCD_Dataset/`: Directory for thermal images (excluded from repository)  

---

### Implemented Models  
1. **LeNet-5**: Lightweight baseline CNN for edge-device benchmarking  
2. **AlexNet**: Deep CNN with pooling layers for feature abstraction  
3. **VGG16**: 16-layer network with batch normalization  
4. **ResNet50**: Transfer learning using ImageNet-pretrained weights  

---

### Results  
- **LeNet-5**: 92.7% test accuracy, optimal for resource-constrained environments  
- **AlexNet**: 89.1% accuracy with moderate computational demands  
- **VGG16**: 96.4% accuracy but slower inference on Apple M1 hardware  
- **ResNet50**: 100% accuracy across all classes (normal/sick/unknown)  

**Key Insight**: ResNet50 with transfer learning achieved perfect classification, demonstrating the effectiveness of pretrained models in medical imaging. LeNet-5 remains viable for lightweight screening applications.  