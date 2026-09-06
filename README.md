# 🚚 AI-Powered Supply Chain Risk Intelligence System

> **Predict late-delivery risk before dispatch and turn shipment data into actionable supply-chain intelligence.**

An end-to-end Machine Learning project built on the **DataCo Supply Chain Dataset** to identify orders that are at risk of late delivery. The project covers the complete workflow from data cleaning and exploratory analysis to feature engineering, model comparison, evaluation, feature-importance analysis, cross-validation, and model export.

---

## 📌 Project at a Glance

| | |
|---|---|
| **Problem** | Late-delivery risk prediction |
| **Task** | Binary classification |
| **Target** | `Late_delivery_risk` |
| **Dataset** | DataCo Supply Chain Dataset |
| **Records used** | **180,519** |
| **Best model** | **Extra Trees Classifier** |
| **Accuracy** | **90.04%** |
| **Precision** | **94.02%** |
| **Recall** | **87.39%** |
| **F1 Score** | **90.58%** |
| **ROC-AUC** | **96.75%** |
| **Environment** | Google Colab |

### Why this matters

Late deliveries affect customer satisfaction, service-level performance, operational planning, and supply-chain efficiency. A risk-prediction system can help logistics teams **prioritize potentially problematic orders before they become delivery failures**.

---

## 🎯 Problem Statement

The objective is to predict whether an order is likely to be delivered late.

The problem is formulated as a **binary classification task**:

- `0` → On-time / not identified as late
- `1` → Late-delivery risk

**Target variable:** `Late_delivery_risk`

The dataset contains more than **180K order records**, providing a substantial base for analyzing relationships between order characteristics, shipping configuration, customer segments, product information, geography, and delivery risk.

---

## 🧠 What This Project Does

The pipeline performs:

- 🔎 Exploratory Data Analysis (EDA)
- 🧹 Data cleaning and duplicate removal
- 🛡️ Leakage-aware feature selection
- 📅 Date-based feature extraction
- 🧩 Feature engineering
- 🔄 Automated numerical/categorical preprocessing
- 🤖 Multiple model comparison
- 🌲 Extra Trees model training
- 📊 Classification evaluation
- 📈 ROC and Precision-Recall analysis
- 🔍 Feature-importance analysis
- 🔁 5-fold cross-validation
- 💾 Model and feature-importance export

---

# 🏗️ End-to-End Workflow

```text
                    DataCo Supply Chain Dataset
                                │
                                ▼
                         Data Cleaning
                                │
                                ▼
                    Exploratory Data Analysis
                                │
                                ▼
                       Feature Engineering
                                │
                                ▼
                       Train / Test Split
                           (80 / 20)
                                │
                                ▼
                         Preprocessing
                  ┌─────────────┴─────────────┐
                  │                           │
             Numerical                  Categorical
             Imputation                  Imputation
                  │                           │
            Standard Scaling              One-Hot Encoding
                  └─────────────┬─────────────┘
                                │
                                ▼
                        Model Training
                                │
                                ▼
                        Model Comparison
                                │
                                ▼
                    Select Extra Trees Model
                                │
                                ▼
                       Model Evaluation
                                │
                 ┌──────────────┼──────────────┐
                 ▼              ▼              ▼
            Confusion       ROC-AUC       Precision-Recall
             Matrix
                                │
                                ▼
                       Feature Importance
                                │
                                ▼
                         5-Fold CV Check
                                │
                                ▼
                    Export Trained Model
```

---

# 📊 Dataset & Data Preparation

The project loads the **DataCo Supply Chain Dataset** from CSV format and performs an initial inspection using:

- Dataset shape
- Data types
- Descriptive statistics
- Missing-value analysis
- Duplicate detection
- Unique-value analysis

### Duplicate handling

Duplicate rows are identified and removed before modeling.

### Columns removed

The following columns were dropped because they were not useful for the modeling objective or contained high-cardinality, identifying, geographic-coordinate, or descriptive information:

```text
Customer Email
Customer Fname
Customer Lname
Customer Password
Customer Street
Product Description
Product Image
Order Customer Id
Order Id
Order Item Id
Customer Id
Product Card Id
Category Id
Department Id
Product Category Id
Latitude
Longitude
```

### Leakage control

Several fields that can reveal information about the actual delivery outcome were explicitly removed:

```text
Days for shipping (real)
Delivery Status
shipping date (DateOrders)
```

This is important because a delivery-risk model should not depend on information that becomes available only after the shipment outcome is known.

---

# 📅 Date Feature Engineering

The original order timestamp:

```text
order date (DateOrders)
```

is converted into useful temporal features:

- `Order_Year`
- `Order_Month`
- `Order_Day`
- `Order_DayOfWeek`

The original timestamp is then removed.

This allows the model to capture potential temporal patterns without directly using the raw timestamp.

---

# 🛠️ Feature Engineering

Several domain-inspired features were created to give the model additional supply-chain and commercial signals.

| Feature | Description |
|---|---|
| `High_Value_Order` | Flags orders above the median sales value |
| `Discount_Percentage` | Discount relative to the item's actual price |
| `Sales_Per_Item` | Total sales normalized by quantity ordered |
| `Profit_Margin` | Benefit per order expressed as a percentage of sales |
| `Urgent_Shipment` | Flags shipments with a scheduled delivery window of 2 days or less |
| `Category_Avg_Sales` | Average sales value for the corresponding product category |
| `Category_Frequency` | Frequency encoding of product categories |
| `Market_Frequency` | Frequency encoding of markets |
| `Order_Size` | Quantity × product price; proxy for overall order size |
| `Price_Difference` | Difference between listed product price and charged price |

These features attempt to convert raw transactional fields into signals that are easier for a machine-learning model to use.

---

# 🔬 Exploratory Data Analysis

The project investigates delivery risk from multiple perspectives.

### Delivery-risk distribution

The target distribution contains:

- **98,977** records with `Late_delivery_risk = 1`
- **81,542** records with `Late_delivery_risk = 0`

This corresponds to approximately:

- **54.83%** late-risk class
- **45.17%** non-late-risk class

The target is therefore not perfectly balanced, making metrics such as **precision, recall, F1, ROC-AUC, and Precision-Recall analysis** useful alongside accuracy.

### EDA areas covered

- Numerical feature distributions
- Outlier inspection using boxplots
- Correlation heatmap
- Shipping Mode vs. delivery risk
- Market vs. delivery risk
- Customer Segment vs. delivery risk
- Product Category vs. delivery risk
- Top countries by order volume
- Delivery-risk rate by region
- Sales distribution
- Sales vs. delivery risk
- Scheduled shipping time vs. delivery risk
- Pairplot of selected numerical variables

For computational efficiency, the pairplot uses a **random sample of 2,000 records** rather than the full dataset.

---

## 📸 EDA Visualizations

### Correlation Heatmap

<p align="center">
  <img src="images/correlation-heatmap.png" width="850" alt="Correlation heatmap">
</p>

### Pair Plot

<p align="center">
  <img src="images/pair-plot.png" width="850" alt="Pair plot">
</p>

---

# 🤖 Machine Learning Approach

## Train / Test Split

The dataset is split into:

- **80% training data**
- **20% test data**

The split is **stratified on the target variable** so that the class distribution remains approximately consistent across the training and test sets.

---

## ⚙️ Preprocessing Pipeline

A `ColumnTransformer` is used to apply different preprocessing strategies to numerical and categorical variables.

### Numerical features

```text
Median Imputation
        ↓
Standard Scaling
```

### Categorical features

```text
Most-Frequent Imputation
        ↓
One-Hot Encoding
```

Categorical encoding uses:

```python
OneHotEncoder(handle_unknown="ignore")
```

The complete preprocessing stage is wrapped together with the classifier inside a Scikit-learn `Pipeline`.

### Why use a pipeline?

This keeps preprocessing consistent between training and inference and helps prevent **data leakage**, because preprocessing transformations are fitted using the training data rather than the test set.

---

# 🧪 Models Compared

Five classification algorithms were evaluated using the same preprocessing workflow:

1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**
4. **Extra Trees**
5. **Gradient Boosting**

The model-comparison loop evaluates:

- Accuracy
- Precision
- Recall
- F1 Score

---

# 🏆 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| 🥇 **Extra Trees** | **90.04%** | **94.02%** | **87.39%** | **90.58%** |
| Decision Tree | 83.23% | 85.28% | 83.91% | 84.59% |
| Random Forest | 79.86% | 89.72% | 71.46% | 79.56% |
| Logistic Regression | 73.43% | 80.84% | 67.55% | 73.60% |
| Gradient Boosting | 71.57% | 87.18% | 56.46% | 68.53% |

> **Extra Trees achieved the strongest overall performance in the recorded model comparison, with an F1 Score of 90.58% and Accuracy of 90.04%.**

---

# 🌲 Final Model: Extra Trees Classifier

The final selected model is:

```python
ExtraTreesClassifier(random_state=42)
```

It was selected based on its performance across the evaluated metrics.

### Final test-set metrics

| Metric | Score |
|---|---:|
| Accuracy | **90.04%** |
| Precision | **94.02%** |
| Recall | **87.39%** |
| F1 Score | **90.58%** |
| ROC-AUC | **96.75%** |

### What the metrics mean

**Accuracy — 90.04%**  
Approximately 90% of test-set predictions were classified correctly.

**Precision — 94.02%**  
Among orders predicted as being at risk, approximately 94% were actually in the positive class.

**Recall — 87.39%**  
The model identifies approximately 87% of the positive late-risk cases.

**F1 Score — 90.58%**  
Provides a balance between precision and recall and is especially useful when accuracy alone is not sufficient.

**ROC-AUC — 96.75%**  
Indicates strong ranking/discrimination ability between the two target classes on the held-out test set.

---

# 📈 Model Evaluation

The final model is evaluated using several complementary diagnostics.

## Confusion Matrix

<p align="center">
  <img src="images/confusion-matrix.png" width="600" alt="Confusion matrix">
</p>

The confusion matrix provides a direct view of correct and incorrect predictions for the two classes.

---

## ROC Curve

<p align="center">
  <img src="images/roc-curve-final.png" width="800" alt="ROC curve">
</p>

The ROC curve examines the trade-off between:

- True Positive Rate
- False Positive Rate

The final model achieved a **ROC-AUC of 0.9675**.

---

## Precision-Recall Curve

The project also evaluates the **Precision-Recall curve**, which provides another useful perspective on classification performance, particularly when the positive class is operationally important.

---

## Feature Importance

Feature importance is extracted directly from the trained Extra Trees classifier.

### Top features recorded by the model

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `Shipping Mode_Standard Class` | 0.05250 |
| 2 | `Urgent_Shipment` | 0.04677 |
| 3 | `Days for shipment (scheduled)` | 0.04464 |
| 4 | `Shipping Mode_First Class` | 0.03614 |
| 5 | `Order Status_SUSPECTED_FRAUD` | 0.01840 |
| 6 | `Shipping Mode_Second Class` | 0.01729 |
| 7 | `Order Status_CANCELED` | 0.01640 |
| 8 | `Order_Day` | 0.01383 |
| 9 | `Order_Month` | 0.01212 |
| 10 | `Profit_Margin` | 0.01040 |

The results suggest that **shipping configuration and scheduled delivery constraints are among the strongest predictive signals** in the trained model.

> Feature importance indicates predictive contribution within this trained model; it should **not** be interpreted as causal evidence that a feature directly causes late delivery.

<p align="center">
  <img src="images/imp-features-bar-plot.png" width="800" alt="Feature importance">
</p>

---

# 🔁 Cross-Validation

To check whether the reported performance depends too heavily on a single train/test split, the project includes a **5-fold cross-validation** step using **F1 score** as the evaluation metric.

```python
cross_val_score(
    best_et,
    X_train,
    y_train,
    cv=5,
    scoring="f1",
    n_jobs=-1
)
```

The notebook keeps this computation commented by default because it adds substantial runtime. It can be uncommented when a fresh validation check is required.

---

# 💾 Saved Model & Results

The project exports the trained model using `joblib`:

```text
supply_chain_risk_model.pkl
```

Feature-importance results are exported as:

```text
feature_importance.csv
```

This makes the project easier to reuse for future inference, analysis, or deployment.

---

# 📁 Suggested Repository Structure

```text
supply-chain-risk-intelligence/
│
├── 📓 supply_chain_final(2).ipynb
├── 📄 README.md
│
├── 🖼️ images/
│   ├── correlation-heatmap.png
│   ├── pair-plot.png
│   ├── roc-curve-final.png
│   ├── imp-features-bar-plot.png
│   ├── confusion-matrix.png
│   └── model-evaluation.png
│
├── 📊 feature_importance.csv
└── 🤖 supply_chain_risk_model.pkl
```

> The dataset itself is not included in the repository structure above. The notebook currently loads it from Google Drive.

---

# 🚀 Getting Started

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd supply-chain-risk-intelligence
```

## 2. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## 3. Open the notebook

Run:

```text
supply_chain_final(2).ipynb
```

The notebook was developed in **Google Colab**.

## 4. Provide the dataset

The notebook currently expects:

```text
DataCoSupplyChainDataset.csv
```

and loads it from Google Drive.

If running locally, replace the Google Drive loading cell with your local CSV path.

---

# 🧰 Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computing |
| **Scikit-learn** | Preprocessing, modeling, evaluation |
| **Matplotlib** | Visualization |
| **Seaborn** | Statistical visualization |
| **Joblib** | Model serialization |
| **Google Colab** | Development environment |

---

# 💡 Business Value

This project demonstrates how machine learning can support supply-chain operations by moving from **reactive delivery tracking** toward **proactive risk identification**.

A production version of the system could potentially support workflows such as:

```text
New Order
   ↓
Risk Prediction
   ↓
Low Risk ───────────────► Normal Processing
   │
   └── High Risk ───────► Operational Review
                              │
                              ├── Expedite shipment
                              ├── Review shipping mode
                              ├── Prioritize fulfillment
                              ├── Contact logistics partner
                              └── Proactively manage customer expectations
```

This creates a bridge between a machine-learning prediction and an operational decision.

---

# 🔍 Key Insights from the Model

Based on the recorded feature-importance analysis:

- **Shipping mode** is one of the strongest predictive signals.
- **Urgent shipments** are highly important to the model.
- **Scheduled shipping duration** is a major predictor.
- Temporal features such as **order day and month** contribute to prediction.
- Commercial features such as **profit margin, discounts, sales, and order size** also contribute.
- The model benefits from combining operational, transactional, categorical, and temporal information.

These findings can help guide further investigation into **where and why delivery risk emerges** in the supply chain.

---

# ⚠️ Limitations & Next Steps

This project is a strong modeling baseline, but there are several ways it could be made more production-ready.

### 1. Time-aware validation

The current evaluation uses a stratified 80/20 train/test split. For a real forecasting system, a **time-based split** would be more representative:

```text
Past orders → Training
Future orders → Testing
```

This would better simulate how the model behaves when predicting future shipments.

### 2. Threshold optimization

The classifier's default decision threshold is used. In a real logistics operation, the threshold could be optimized according to the business cost of:

- Missing a genuinely risky shipment
- Flagging an order that would have arrived on time

### 3. Probability calibration

Predicted probabilities could be calibrated so that a risk score has a more reliable operational interpretation.

### 4. Hyperparameter tuning

The current Extra Trees model uses:

```python
ExtraTreesClassifier(random_state=42)
```

Further tuning of parameters such as:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`

could potentially improve generalization.

### 5. Model explainability

SHAP or permutation-based analysis could complement tree feature importance and provide **order-level explanations**, such as:

> "This order was flagged as high risk primarily because of its shipping mode, scheduled delivery window, and temporal characteristics."

### 6. Production monitoring

A deployed system should monitor:

- Prediction accuracy
- Precision / recall
- Data drift
- Feature drift
- Prediction distribution
- False-positive and false-negative rates

### 7. Feature availability before dispatch

For a true **pre-dispatch** system, every feature used at inference time should be verified to be available at the exact moment the risk decision is made. This is especially important for operational fields such as order status.

---

# 🧪 Reproducibility Notes

The notebook uses `random_state=42` for:

- Train/test splitting
- Extra Trees
- Other tree-based models where configured
- Pairplot sampling

This helps make the experiment reproducible.

The full model-comparison loop and 5-fold cross-validation are intentionally left commented in the notebook because they can take significant runtime on the full dataset.

---

# 📌 Project Highlights

### Dataset
**180K+ supply-chain order records**

### Target
**Late delivery risk**

### Best Model
**Extra Trees Classifier**

### Best F1
**90.58%**

### Precision
**94.02%**

### Recall
**87.39%**

### ROC-AUC
**96.75%**

### Validation
**80/20 stratified split + optional 5-fold cross-validation**

### Output
**Reusable serialized ML pipeline + feature-importance CSV**

---

# 👨‍💻 Project Focus

This project combines:

**Machine Learning + Supply Chain Analytics + Feature Engineering + Operational Risk Intelligence**

The main goal is not simply to maximize a classification score, but to demonstrate how a structured ML pipeline can transform historical supply-chain data into **early-warning signals for logistics decision-making**.

---

## ⭐ If you found this project useful

Feel free to ⭐ the repository and use the workflow as a foundation for building more advanced supply-chain forecasting and risk-intelligence systems.

---

### 📜 License

Add the license appropriate for your repository and the dataset's usage terms before publishing.
