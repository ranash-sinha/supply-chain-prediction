\# AI-Powered Supply Chain Risk Intelligence System



A Machine Learning system for predicting shipment delivery risk and identifying high-risk orders before dispatch.



\---



\## Overview



This project develops an end-to-end Machine Learning pipeline to predict \*\*Late Delivery Risk\*\* using the \*\*DataCo Supply Chain Dataset\*\*, containing \*\*180K+ records\*\*.



The project combines exploratory data analysis, feature engineering, preprocessing, model comparison, and evaluation to build a classification system that can support \*\*data-driven supply chain and logistics decisions\*\*.



\---



\## Problem Statement



Late deliveries can negatively impact customer satisfaction, operational efficiency, and supply chain performance.



This project formulates late delivery prediction as a \*\*binary classification problem\*\*, where the model predicts whether an order is at risk of being delivered late.



\*\*Target Variable:\*\* `Late\_delivery\_risk`



\---



\## Key Features



\- Exploratory Data Analysis (EDA)

\- Data Cleaning \& Preprocessing

\- Feature Engineering

\- Multiple Machine Learning Model Comparison

\- Extra Trees Classifier

\- Model Performance Evaluation

\- Feature Importance Analysis

\- 5-Fold Cross-Validation

\- ROC \& Precision-Recall Analysis



\---



\## Tech Stack



\- \*\*Python\*\*

\- \*\*Pandas\*\*

\- \*\*NumPy\*\*

\- \*\*Scikit-learn\*\*

\- \*\*Matplotlib\*\*

\- \*\*Seaborn\*\*

\- \*\*Google Colab\*\*



\---



\## Model Performance



| Model | Accuracy | Precision | Recall | F1 Score |

|---|---:|---:|---:|---:|

| \*\*Extra Trees\*\* | \*\*90.04%\*\* | \*\*94.02%\*\* | \*\*87.39%\*\* | \*\*90.58%\*\* |

| Decision Tree | 83.23% | 85.28% | 83.91% | 84.59% |

| Random Forest | 79.86% | 89.72% | 71.46% | 79.56% |

| Logistic Regression | 73.43% | 80.84% | 67.55% | 73.60% |

| Gradient Boosting | 71.57% | 87.18% | 56.46% | 68.53% |



\### Best Model



\*\*Extra Trees Classifier\*\* achieved the best overall performance:



\- \*\*Accuracy:\*\* 90.04%

\- \*\*Precision:\*\* 94.02%

\- \*\*Recall:\*\* 87.39%

\- \*\*F1 Score:\*\* 90.58%

\---



\## 🔍 Exploratory Data Analysis



\### Correlation Heatmap



The correlation heatmap shows the relationships between numerical features in the dataset.



<p align="center">

&#x20; <img src="images/correlation-heatmap.png" width="850">

</p>



\### Pair Plot



<p align="center">

&#x20; <img src="images/pair-plot.png" width="850">

</p>



\---



\## 📈 Model Evaluation



\### ROC Curve



<p align="center">

&#x20; <img src="images/roc-curve-final.png" width="800">

</p>



\### Feature Importance



<p align="center">

&#x20; <img src="images/imp-features-bar-plot.png" width="800">

</p>



\### Confusion Matrix



<p align="center">

&#x20; <img src="images/confusion-matrix.png" width="600">

</p>



\## Project Workflow



```text

DataCo Supply Chain Dataset

&#x20;           ↓

&#x20;    Data Cleaning

&#x20;           ↓

&#x20;Exploratory Data Analysis

&#x20;           ↓

&#x20;   Feature Engineering

&#x20;           ↓

&#x20;   Train / Test Split

&#x20;           ↓

&#x20;     Preprocessing

&#x20;           ↓

&#x20;    Model Training

&#x20;           ↓

&#x20;   Model Comparison

&#x20;           ↓

&#x20;   Model Evaluation

&#x20;           ↓

&#x20;Feature Importance Analysis

&#x20;           ↓

&#x20;  Cross-Validation



