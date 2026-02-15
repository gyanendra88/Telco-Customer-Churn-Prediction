# 📞 Telco Customer Churn Prediction

## a. Problem Statement

Customer churn refers to customers leaving a company’s services over time, which directly impacts business revenue and customer retention strategies.

The objective of this project is to:

> Predict whether a telecom customer will **churn (Yes/No)** based on customer demographics, subscription details, account information, and service usage behaviour.

This is a **Binary Classification Problem** where:

- **Target Variable:** `Churn`
- **Classes:** Yes (Customer leaves), No (Customer stays)

The objective is to compare multiple machine learning models and evaluate their effectiveness using standard classification metrics.

---

## b. Dataset Description

### 📊 Dataset Overview

- **Dataset:** WA_Fn-UseC_-Telco-Customer-Churn  
- **Source:** Kaggle  
- **Instances:** 7043 customers  
- **Features:** 20+ input features (after dropping ID column)  
- **Target Variable:** Churn (Yes/No)

The dataset contains information about a fictional telecom company’s customers including demographics, account details, subscribed services, and billing information.

---

### 🧾 Feature Description

#### 1️⃣ Customer Demographics

| Feature | Description |
|---|---|
| customerID | Unique customer identifier (dropped) |
| gender | Male/Female |
| SeniorCitizen | Whether customer is senior citizen (0/1) |
| Partner | Has partner (Yes/No) |
| Dependents | Has dependents (Yes/No) |

---

#### 2️⃣ Account & Subscription Information

| Feature | Description |
|---|---|
| tenure | Months customer has stayed |
| Contract | Month-to-month / One year / Two year |
| PaperlessBilling | Uses electronic billing |
| PaymentMethod | Method of payment |
| MonthlyCharges | Monthly bill amount |
| TotalCharges | Total amount billed |

---

#### 3️⃣ Services Subscribed

| Feature | Description |
|---|---|
| PhoneService | Phone service availability |
| MultipleLines | Multiple phone lines |
| InternetService | DSL / Fiber optic / None |
| OnlineSecurity | Online security service |
| OnlineBackup | Online backup |
| DeviceProtection | Device protection |
| TechSupport | Technical support |
| StreamingTV | Streaming TV subscription |
| StreamingMovies | Streaming Movies subscription |

---

#### 🎯 Target Variable

| Feature | Description |
|---|---|
| Churn | Yes = customer left, No = customer stayed |

---

### ⚙️ Preprocessing Applied

- Removed `customerID`
- Converted `TotalCharges` safely to numeric (handled blanks)
- Train/Test split with stratification
- ColumnTransformer pipeline:
  - Numeric → StandardScaler
  - Categorical → OneHotEncoder
- Same preprocessing reused via Pipeline for all models

---

## c. Models Used

The following 6 classification models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor (kNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

---

## 📊 Comparison Table (Evaluation Metrics)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8055 | 0.8419 | 0.6572 | 0.5588 | 0.6040 | 0.4790 |
| XGBoost (Ensemble) | 0.8006 | 0.8378 | 0.6525 | 0.5321 | 0.5862 | 0.4607 |
| Random Forest (Ensemble) | 0.7793 | 0.8171 | 0.6068 | 0.4786 | 0.5351 | 0.3978 |
| Naive Bayes (Gaussian) | 0.6948 | 0.8074 | 0.4589 | 0.8369 | 0.5928 | 0.4245 |
| kNN | 0.7651 | 0.8049 | 0.5556 | 0.5749 | 0.5650 | 0.4043 |
| Decision Tree | 0.7282 | 0.6573 | 0.4884 | 0.5053 | 0.4967 | 0.3107 |

---

## 📈 Observations (Model Performance)

| ML Model | Observation |
|---|---|
| **Logistic Regression** | Logistic Regression achieved the best overall balanced performance across all evaluation metrics. It recorded the highest Accuracy (0.8055), highest AUC (0.8419), highest F1 Score (0.6040), and highest MCC (0.4790), indicating strong generalization and consistent classification quality. This suggests that the churn dataset is relatively well separated after preprocessing and one-hot encoding, allowing a linear model with regularization to perform extremely well. It serves as a strong baseline and the most stable deployment candidate. |
| **Decision Tree** | Decision Tree produced the weakest overall performance among all models with the lowest AUC (0.6573) and MCC (0.3107). Although the model is simple and highly interpretable, it likely suffered from overfitting due to greedy splits on categorical features after encoding. The lower generalization ability indicates that single trees struggle to capture stable decision boundaries compared to ensemble methods. |
| **kNN** | The k-Nearest Neighbour model showed moderate performance with balanced precision and recall, achieving Accuracy of 0.7651 and AUC of 0.8049. However, performance remained below Logistic Regression and ensemble models. This is expected because distance-based algorithms are sensitive to high-dimensional feature spaces created by one-hot encoding (curse of dimensionality), which reduces neighborhood quality and impacts prediction effectiveness. |
| **Naive Bayes (Gaussian)** | Naive Bayes achieved the highest Recall (0.8369), meaning it detected the largest number of churners among all models. However, this came at the cost of low Precision (0.4589), indicating many false positives. This behavior aligns with the strong independence assumptions made by Naive Bayes, which are rarely fully satisfied in real-world telecom data. Despite lower overall Accuracy (0.6948), the model is useful in business scenarios where missing churn customers is more costly than raising extra alerts. |
| **Random Forest (Ensemble)** | Random Forest improved significantly over the single Decision Tree due to bagging and ensemble averaging, resulting in better stability and generalization. It achieved balanced metrics (Accuracy 0.7793, AUC 0.8171). The ensemble reduces variance compared to a standalone tree, but performance remained slightly lower than Logistic Regression and XGBoost, suggesting that the dataset does not strongly require complex non-linear partitioning. |
| **XGBoost (Ensemble)** | XGBoost was one of the strongest performers with high AUC (0.8378) and strong precision (0.6525). The boosting mechanism helped sequentially reduce errors and capture complex relationships in data. However, its overall Accuracy and F1 score were marginally lower than Logistic Regression, indicating that additional model complexity did not significantly improve performance for this dataset. This shows that boosted ensembles are powerful but not always superior to well-regularized linear models. |

---

## 🔎 Key Insight

The results demonstrate that higher model complexity does not necessarily lead to better performance. While ensemble methods such as Random Forest and XGBoost improved stability and learning capacity, Logistic Regression achieved the best overall balance across metrics.

---

## 📏 Why Multiple Evaluation Metrics Were Used

Since churn prediction is a binary classification problem with class imbalance, relying only on accuracy can be misleading.

| Metric | Why Important |
|---|---|
| Accuracy | Overall correctness |
| AUC | Discrimination ability across thresholds |
| Precision | Important when false positives are costly |
| Recall | Important to detect maximum churners |
| F1 Score | Balance between precision & recall |
| MCC | Balanced metric considering TP, TN, FP, FN |

---

## 🎯 Model Selection Insight

Although XGBoost and Random Forest are ensemble models, Logistic Regression achieved the best overall performance.

Possible reasons:

- Many linearly separable one-hot encoded features  
- Regularization prevents overfitting  
- Tree models may overfit sparse categorical splits  

---

## 💼 Business Interpretation

- Naive Bayes achieved highest Recall → good for detecting churners.
- Logistic Regression provides best balance → strong deployment candidate.
- Model selection depends on business priorities, not accuracy alone.

---

## 🏗 Project Structure

```
ml_assignment_2/
├── app.py                                     # Streamlit web application
├── requirements.txt                           # Python dependencies
├── README.md                                  # Project documentation
├── .gitignore                                 # Git ignore file
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Original dataset
│   └── telco_test_data.csv                    # Test split for Streamlit app
└── model/
    ├── 2025AA05771_modeltraining.ipynb        # Model training notebook 
    ├── logistic_regression.joblib           
    ├── decision_tree.joblib 
    ├── knn.joblib
    ├── naive_bayes_gaussian.joblib
    ├── random_forest.joblib
    ├── xgboost.joblib

```

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🚀 Streamlit App Features

- **Dataset Upload (CSV):** Upload test data for prediction and evaluation.  
- **Model Selection Dropdown:** Choose from 6 trained ML models.  
- **Evaluation Metrics Display:** Shows Accuracy, AUC, Precision, Recall, F1 Score, and MCC (if Churn label is available).  
- **Confusion Matrix & Classification Report:** Visual confusion matrix with detailed classification metrics.  
- **Prediction Summary:** Displays churn prediction distribution and model insights.  
- **Download Predictions:** Export prediction results as a CSV file.  
- **Sample Test CSV Download:** Users can download a sample input file for reference.

---

## 🔗 Links

- **Dataset (Kaggle):** https://www.kaggle.com/datasets/ditisolanki/wa-fn-usec-telco-customer-churn
- **Live Streamlit App:** https://predicttelecomchurn.streamlit.app/

---
