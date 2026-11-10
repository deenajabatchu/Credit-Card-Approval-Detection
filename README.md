
# Credit Card Approval Detection Using Machine Learning

## 1. Project Overview

Credit Card Approval Detection is a supervised machine learning classification project that predicts whether a credit card application is likely to be approved or rejected based on applicant information.

Financial institutions receive numerous credit card applications from individuals with different financial backgrounds and personal details. Evaluating these applications requires analyzing multiple attributes before making an approval decision.

This project uses historical credit approval data and a Random Forest Classification algorithm to identify patterns in applicant records and predict credit card application outcomes.

The project demonstrates a complete machine learning workflow, starting from data collection and preprocessing to model training, prediction, and evaluation.

---

## 2. Problem Statement

Banks and financial institutions need to evaluate credit card applications before approving or rejecting them.

Each applicant provides different details, including personal information, financial characteristics, and other application-related attributes.

The main problem addressed in this project is:

**How can machine learning be used to predict whether a credit card application will be approved or rejected based on historical applicant data?**

### Project Objectives

- Analyze historical credit card application data.
- Identify and handle missing and invalid values.
- Perform Exploratory Data Analysis (EDA).
- Prepare numerical and categorical features for machine learning.
- Train a classification model using Random Forest.
- Predict credit card application approval outcomes.
- Evaluate model performance using different classification metrics.
- Identify important features influencing model predictions.

---

## 3. Dataset Description

**Dataset Name:** Credit Approval Dataset (CRX)

**File Name:** `crx.csv`

**Dataset Source:** [Kaggle - Credit Card Approval Clean Data](https://www.kaggle.com/datasets/samuelcortinhas/credit-card-approval-clean-data)

The CRX dataset contains historical credit card application records with 15 input attributes and one target attribute.

### Dataset Information

- Total records: 690
- Input features: 15
- Target feature: 1
- Feature types: Numerical and categorical
- Problem type: Binary Classification

### Target Variable

The target column is `A16`.

| Target Value | Meaning | Encoded Value |
|---|---|---|
| `+` | Application Approved | 1 |
| `-` | Application Rejected | 0 |

The target values are converted into numerical labels to make them suitable for machine learning.

### Dataset Features

**Numerical Features:**

- A2
- A3
- A8
- A11
- A14
- A15

**Categorical Features:**

- A1
- A4
- A5
- A6
- A7
- A9
- A10
- A12
- A13

The original CRX dataset uses anonymized attribute names. Therefore, their exact business meanings are not assumed in this project.

---

## 4. Technologies and Libraries Used

| Technology | Purpose | Why It Is Used |
|---|---|---|
| Python | Programming Language | Implements the complete machine learning workflow. |
| Pandas | Data Handling | Loads CSV files, organizes records, and handles missing values. |
| NumPy | Numerical Computing | Supports numerical operations and data processing. |
| Matplotlib | Data Visualization | Creates charts to understand feature distributions. |
| Seaborn | Statistical Visualization | Visualizes approval distribution and missing-value patterns. |
| Scikit-learn | Machine Learning | Provides preprocessing, classification, training, and evaluation tools. |

### Python Libraries and Their Usage

**Pandas**
- Reads the credit approval dataset.
- Handles missing values.
- Performs data preparation and feature selection.

**NumPy**
- Supports numerical operations and data manipulation.

**Matplotlib**
- Displays numerical feature distributions.
- Visualizes model evaluation results.

**Seaborn**
- Creates approval distribution plots.
- Displays missing-value patterns.
- Generates confusion matrix heatmaps.
- Visualizes feature importance.

**Scikit-learn**
- Splits the dataset into training and testing sets.
- Handles missing numerical and categorical values.
- Encodes categorical features.
- Scales numerical features.
- Builds and trains the Random Forest model.
- Evaluates model performance.

---

## 5. Project Methodology

The project follows these machine learning steps:

### Step 1: Data Collection

The CRX Credit Approval dataset is collected in CSV format and loaded using Pandas.

**Why?**

Historical application records provide examples that help the model learn patterns associated with approval and rejection decisions.

### Step 2: Data Preparation

The dataset is loaded with column names ranging from A1 to A16.

The following operations are performed:

- Missing-value markers represented by `?` are identified.
- Numerical columns are converted into numeric data types.
- Records without valid target values are removed.
- Invalid target labels are filtered.
- Approval labels are converted into numerical values.

**Why?**

Raw datasets may contain missing, invalid, or inconsistent values. Data preparation ensures that the dataset is suitable for further processing.

### Step 3: Exploratory Data Analysis (EDA)

EDA is performed to understand the dataset before training the model.

The project generates:

1. Credit Card Approval Distribution
2. Numerical Feature Distributions
3. Missing Values Visualization

**Why?**

EDA helps understand the distribution of application outcomes, numerical feature patterns, and missing values in the dataset.

### Step 4: Feature Selection

The dataset is divided into:

- Independent variables (X): Applicant attributes from A1 to A15.
- Dependent variable (y): Application decision A16.

**Why?**

The model uses applicant attributes as input and learns to predict the application decision.

### Step 5: Data Preprocessing

Different preprocessing techniques are applied to numerical and categorical features.

**Numerical Features:**

1. SimpleImputer(strategy="median")
2. StandardScaler()

**Why?**

- Median imputation replaces missing numerical values with the median of each feature.
- StandardScaler transforms numerical features to a common scale.

**Categorical Features:**

1. SimpleImputer(strategy="most_frequent")
2. OneHotEncoder(handle_unknown="ignore")

**Why?**

- Most-frequent imputation fills missing categorical values using the most common category.
- OneHotEncoder converts categorical values into numerical representations.

### Step 6: Data Transformation

Scikit-learn's ColumnTransformer combines the numerical and categorical preprocessing pipelines.

**Why?**

The dataset contains different types of features. ColumnTransformer applies the correct preprocessing method to each feature group within a single workflow.

### Step 7: Train-Test Split

The dataset is divided into:

- 80% Training Data
- 20% Testing Data

The `train_test_split()` function uses:

- `test_size=0.2`
- `random_state=42`
- `stratify=y`

**Why?**

Training data helps the model learn patterns from historical applications.

Testing data evaluates the model on unseen records.

Stratification helps maintain the class distribution in both training and testing datasets.

### Step 8: Model Building

**Algorithm: Random Forest Classifier**

Random Forest is a supervised ensemble learning algorithm that combines multiple decision trees to make a final classification prediction.

In this project, the Random Forest classifier is configured with:

- Number of estimators: 100
- Random state: 42

**Why Random Forest?**

- Handles structured datasets containing different types of features after preprocessing.
- Learns complex relationships between input features and target classes.
- Combines multiple decision trees to make predictions.
- Can capture non-linear patterns.
- Provides feature importance scores.

### Step 9: Model Training and Prediction

The model is trained using the training dataset.

After training, it predicts the approval outcomes of the testing dataset.

**Why?**

This process allows the model to learn from historical application records and evaluate its ability to predict outcomes for unseen applications.

### Step 10: Model Evaluation

The following evaluation metrics are calculated:

| Metric | Purpose |
|---|---|
| Accuracy | Measures the percentage of correctly classified applications. |
| Precision | Measures how many applications predicted as approved were actually approved. |
| Recall | Measures how many actual approved applications were correctly identified. |
| F1-Score | Balances precision and recall using their harmonic mean. |
| Confusion Matrix | Shows correct and incorrect predictions for both classes. |

**Why?**

Accuracy alone may not provide a complete understanding of classification performance. Precision, recall, F1-score, and the confusion matrix help identify different types of prediction errors.

### Step 11: Feature Importance Analysis

Random Forest provides feature importance scores based on how much each transformed feature contributes to reducing impurity across the decision trees.

The project:

- Extracts transformed feature names.
- Calculates feature importance scores.
- Sorts features by importance.
- Displays the top 10 features using a bar chart.

**Why?**

Feature importance helps understand which input features were most useful to the trained model when making predictions.

It indicates predictive contribution, not causation.

---

## 6. Project Workflow

```text
             CRX Dataset
                  |
                  v
             Load CSV Data
                  |
                  v
            Data Preparation
                  |
                  v
        Handle Missing Values
                  |
                  v
       Exploratory Data Analysis
                  |
                  v
          Feature Selection
                  |
                  v
         Data Preprocessing
            /          \
           v            v
      Numerical      Categorical
       Features       Features
           |              |
           v              v
      Median Impute   Mode Impute
           |              |
           v              v
      StandardScaler  OneHotEncoder
            \          /
             \        /
                  |
                  v
           Train-Test Split
                  |
                  v
        Random Forest Classifier
                  |
                  v
             Model Training
                  |
                  v
               Prediction
                  |
                  v
            Model Evaluation
                  |
                  v
          Feature Importance
                  |
                  v
       Approval / Rejection
```

---

## 7. Output and Result Discussion

After executing the Python program, the following outputs are generated:

### 1. Dataset Information

Displays:
- First five dataset records.
- Total number of rows and columns.
- Dataset information.
- Missing values in each column.

### 2. Exploratory Data Analysis

The following visualizations are generated:

- Credit Card Approval Distribution
- Numerical Feature Distributions
- Missing Values Heatmap

These visualizations help understand the dataset and identify patterns before model training.

### 3. Model Evaluation Results

The program displays:

- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report

The actual evaluation scores are calculated when the model is executed.

### 4. Confusion Matrix

The confusion matrix represents the model's classification results.

| Actual / Predicted | Rejected (0) | Approved (1) |
|---|---|---|
| Rejected (0) | True Negative | False Positive |
| Approved (1) | False Negative | True Positive |

**Interpretation:**

- True Positive: An approved application correctly predicted as approved.
- True Negative: A rejected application correctly predicted as rejected.
- False Positive: A rejected application incorrectly predicted as approved.
- False Negative: An approved application incorrectly predicted as rejected.

This helps identify where the model makes incorrect predictions.

### 5. Feature Importance

The top 10 important transformed features are displayed in a bar chart.

This helps understand which features contributed most to the Random Forest model's predictive decisions.

### Final Prediction Meaning

| Model Output | Interpretation |
|---|---|
| 1 | Credit Card Application Approved |
| 0 | Credit Card Application Rejected |

### Expected Outcome

The expected outcome is a trained machine learning classification model that predicts whether an application belongs to the approved or rejected category based on patterns learned from historical data.

The model's actual performance must be evaluated using the generated metrics rather than assuming a fixed accuracy.

---

## 8. Project Structure

```text
Credit-Card-Approval/
│
├── data/
│   └── crx.csv
│
├── credit_card_approval.py
│
├── requirements.txt
│
├── README.md
│
├── .gitignore
│
└── LICENSE
```

---

## 9. Installation and Execution

### Prerequisites

- Python 3.x
- Visual Studio Code
- Git (optional)

### Step 1: Clone the Repository

```bash
git clone https://github.com/deenajabatchu/Credit-Card-Approval-Detection.git
```

### Step 2: Navigate to the Project Folder

```bash
cd Credit-Card-Approval-Detection
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Virtual Environment

For Windows:

```bash
.venv\Scripts\activate
```

### Step 5: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 6: Run the Project

```bash
python credit_card_approval.py
```

**Note:** Ensure that `crx.csv` is placed inside the `data` folder before executing the program.

---

## 10. Limitations and Future Improvements

### Current Limitations

- The dataset contains anonymized attributes, limiting direct interpretation of individual features.
- Historical approval decisions may contain bias, which the model could learn.
- Accuracy alone cannot determine whether a model is suitable for financial decisions.
- The current implementation uses only one classification algorithm.
- Feature importance does not explain causation or guarantee fair decision-making.
- The project is an educational prototype and is not intended for real banking decisions.

### Future Improvements

- Compare Random Forest with Logistic Regression and Decision Tree classifiers.
- Apply hyperparameter tuning to improve model performance.
- Perform additional exploratory data analysis.
- Evaluate class imbalance and use suitable techniques if required.
- Explore explainable AI techniques to understand model predictions.
- Evaluate fairness and potential bias in model predictions.
- Develop a simple web application using Flask or FastAPI.

---

## 11. Conclusion

Credit Card Approval Detection demonstrates how supervised machine learning can be applied to a binary classification problem.

The project uses historical credit application data, data preparation, missing-value handling, feature transformation, and a Random Forest classifier to predict approval or rejection outcomes.

Through this project, practical experience is gained in:

- Data preprocessing and transformation.
- Exploratory Data Analysis.
- Supervised machine learning.
- Classification model development.
- Model evaluation.
- Feature importance analysis.

The project highlights how machine learning can support the analysis of credit application data while emphasizing the importance of data quality, evaluation, explainability, and responsible use of predictive models.

---

## 12. Author

**Deenaja Batchu**

B.Tech - Computer Science and Engineering

- GitHub: [deenajabatchu](https://github.com/deenajabatchu)
- LinkedIn: [Deenaja Batchu](https://www.linkedin.com/in/deenaja-batchu/)

---

**Project Category:** Machine Learning | Supervised Learning | Classification

**Algorithm:** Random Forest Classifier

**Programming Language:** Python
