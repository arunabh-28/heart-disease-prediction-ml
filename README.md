# Heart Disease Prediction using Machine Learning

A machine learning project that predicts the likelihood of heart disease using patient health data. It compares multiple classification algorithms and deploys the selected K-Nearest Neighbors (KNN) model through a Flask web application.

## Overview

The project uses the Heart Failure Prediction dataset containing 918 patient records. It covers the complete machine learning workflow, from data exploration and preprocessing to model evaluation and web application development.

The trained model takes clinical attributes as input and predicts whether the patient belongs to the heart disease or no heart disease class.

## Models and Results

Five classification algorithms were evaluated using accuracy and F1 score.

| Model | Accuracy | F1 Score |
|---|---:|---:|
| Logistic Regression | 87.50% | 88.78% |
| K-Nearest Neighbors (KNN) | 88.59% | 89.86% |
| Gaussian Naive Bayes | 86.96% | 87.88% |
| Decision Tree | 74.46% | 75.13% |
| Support Vector Machine (SVM) | 86.41% | 88.04% |

KNN achieved the highest accuracy and F1 score among the models tested. It was selected for the Flask application based on these evaluation results.

*These results reflect the evaluation performed in this project and are not evidence of clinical performance.*

## Features

- Exploratory data analysis and visualization
- Data cleaning and preprocessing
- Handling invalid zero values
- Encoding categorical variables
- Feature scaling using `StandardScaler`
- Training and comparing multiple classification models
- Evaluating models using accuracy and F1 score
- Saving the trained model and preprocessing artifacts
- Flask web interface for entering patient information and generating predictions

## Dataset

The dataset contains 918 records and 12 input features, along with the target column `HeartDisease`.

**Numerical features**
- Age
- RestingBP
- Cholesterol
- MaxHR
- Oldpeak

**Categorical and binary features**
- Sex
- ChestPainType
- FastingBS
- RestingECG
- ExerciseAngina
- ST_Slope

**Target:** `HeartDisease`

## Tech Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Flask
- HTML and CSS
- Jupyter Notebook

## Project Structure

```text
heart-disease-prediction-ml/
│
├── app.py
├── heart.csv
├── Heart_Disease_ML_EDA.ipynb
├── heart_disease_knn.pkl
├── heart_disease_scaler.pkl
├── heart_disease_features.pkl
├── requirements.txt
├── README.md
│
└── templates/
    └── index.html
```

The `.pkl` files contain the trained KNN model, fitted scaler, and expected feature names. They are required by the Flask application and should be kept consistent with the training pipeline.

