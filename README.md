# Task 1: Data Cleaning and Preprocessing

**ElevateLabs Machine Learning Internship**
**Dataset:** Titanic Passenger Survival Dataset
*

---

## Overview

This task focuses on one of the most critical stages of any machine learning pipeline — data cleaning and preprocessing. Raw datasets in the real world are rarely clean. They contain missing values, inconsistent formats, categorical text that models cannot process, and numerical features at vastly different scales. This project demonstrates a complete, structured approach to transforming the raw Titanic dataset into a clean, model-ready format.

---

## Dataset Description

The Titanic dataset contains records of 891 passengers who were aboard the RMS Titanic. The goal is to predict whether a passenger survived or not based on their attributes.

| Column | Data Type | Description |
|---|---|---|
| PassengerId | Integer | Unique identifier per passenger |
| Survived | Integer | Target variable — 0 = Did not survive, 1 = Survived |
| Pclass | Integer | Passenger ticket class (1 = First, 2 = Second, 3 = Third) |
| Name | String | Full name of the passenger |
| Sex | String | Gender of the passenger |
| Age | Float | Age in years |
| SibSp | Integer | Number of siblings or spouses aboard |
| Parch | Integer | Number of parents or children aboard |
| Ticket | String | Ticket number |
| Fare | Float | Fare paid by the passenger in British pounds |
| Cabin | String | Cabin number assigned |
| Embarked | String | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

---

## Tools and Libraries

| Library | Version | Purpose |
|---|---|---|
| Python | 3.x | Core programming language |
| pandas | Latest | Data loading, manipulation, and cleaning |
| numpy | Latest | Numerical operations |
| matplotlib | Latest | Base plotting and visualization |
| seaborn | Latest | Statistical visualizations |
| scikit-learn | Latest | Encoding, scaling, and model training |

Install all required dependencies using:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
Task-1/
|
|-- Titanic-Dataset.csv          # Original raw dataset
|-- data_preprocessing.py        # Main preprocessing script
|-- Cleaned_Titanic_Data.csv     # Output: cleaned and processed dataset
|-- requirements.txt             # Python dependencies
|-- README.md                    # Project documentation
|
|-- plots/
    |-- missing_before.png       # Missing value heatmap before cleaning
    |-- missing_after.png        # Missing value heatmap after cleaning
    |-- outliers_before.png      # Boxplot before outlier removal
    |-- outliers_after.png       # Boxplot after outlier removal
    |-- distributions.png        # Feature distribution histograms
    |-- correlation.png          # Feature correlation heatmap
```

---

## Code Architecture

The script is organized into dedicated functions, one per preprocessing step. This structure follows professional coding standards — each function has a single responsibility, is independently testable, and is documented with a docstring.

```
data_preprocessing.py
|
|-- load_data()              Step 1 — Load dataset and print summary statistics
|-- plot_missing_before()    Step 2 — Visualize missing values before cleaning
|-- handle_missing()         Step 3 — Impute missing values
|-- plot_missing_after()     Step 4 — Visualize missing values after cleaning
|-- encode_features()        Step 5 — Encode categorical columns to numerical
|-- remove_outliers()        Step 6 — Detect and remove outliers using IQR
|-- scale_features()         Step 7 — Standardize numerical feature scales
|-- plot_distributions()     Step 8 — Plot feature histograms
|-- plot_correlation()       Step 9 — Generate correlation heatmap
|-- run_baseline_model()     Bonus  — Train and evaluate a Logistic Regression model
```

---

## Preprocessing Steps

### Step 1 — Data Loading and Exploration

The dataset was loaded using `pandas.read_csv()` and inspected using `.info()`, `.isnull().sum()`, and `.describe()` to understand its structure before any modification.

**Missing values identified:**

| Column | Missing Count | Percentage |
|---|---|---|
| Age | 177 | 19.9% |
| Cabin | 687 | 77.1% |
| Embarked | 2 | 0.2% |

---

### Step 2 — Visualizing Missing Values (Before)

A heatmap was generated using Seaborn to provide a visual representation of where missing values exist in the dataset. This establishes a clear "before" reference point.

```python
sns.heatmap(original_df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title("Missing Values - Before Cleaning")
plt.savefig('plots/missing_before.png')
```

---

### Step 3 — Handling Missing Values

Three different imputation strategies were applied based on the nature of each column.

**Age — Filled with Median**

```python
df['Age'] = df['Age'].fillna(df['Age'].median())
```

The median was chosen over the mean because the Age distribution is slightly right-skewed. The median is resistant to the influence of extreme values (such as very young or very old passengers), making it a more reliable estimate of the central tendency.

**Embarked — Filled with Mode**

```python
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
```

Embarked is a categorical column, so arithmetic operations such as mean or median do not apply. The mode (most frequently occurring value, which is Southampton) was used as the best assumption for the two missing entries.

**Cabin — Replaced with Placeholder**

```python
df['Cabin'] = df['Cabin'].fillna('Unknown')
```

With over 77% of values missing, reliable imputation of Cabin was not feasible. The column was assigned a placeholder value and subsequently dropped in the encoding step, as it carries no meaningful signal at this density of missingness.

---

### Step 4 — Visualizing Missing Values (After)

The same heatmap was regenerated after imputation to confirm that all missing values had been addressed. The resulting plot shows a completely uniform surface, confirming zero missing values remain.

---

### Step 5 — Encoding Categorical Features

Machine learning models require numerical input. All text-based columns were converted accordingly.

**Sex — Label Encoding**

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
# female = 0, male = 1
```

Since Sex contains only two unique values, binary Label Encoding is appropriate and does not introduce any artificial ordinal relationship.

**Embarked — One-Hot Encoding**

```python
df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked')
# Produces: Embarked_C, Embarked_Q, Embarked_S
```

Embarked contains three unordered categories. Label Encoding would incorrectly imply a numerical ranking (e.g., C < Q < S), which does not exist. One-Hot Encoding treats each category independently as a binary flag, preserving the true nature of the variable.

**Columns Dropped**

```python
df.drop(['Name', 'Ticket', 'PassengerId', 'Cabin'], axis=1, inplace=True)
```

These columns were removed as they are either unique identifiers with no generalisable pattern or too sparse to contribute meaningful signal to a model.

---

### Step 6 — Outlier Detection and Removal

Outliers are extreme data points that can disproportionately influence model training, particularly for linear models. Box plots were generated before and after removal to document the impact.

**Method: Interquartile Range (IQR)**

```python
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR   # = -26.72
upper_bound = Q3 + 1.5 * IQR   # = +65.63

df = df[(df['Fare'] >= lower_bound) & (df['Fare'] <= upper_bound)]
```

The IQR method was selected over the Z-score method because it is non-parametric — it does not assume a normal distribution. The Fare column is heavily right-skewed due to a small number of very high first-class fares, making IQR the more robust choice.

**Result:** 116 rows with extreme Fare values were removed, reducing the dataset from 891 to 775 records.

---

### Step 7 — Feature Scaling (Standardization)

Numerical features existed on different scales. Age ranged from approximately 0 to 80, while Fare (after outlier removal) ranged from 0 to 65. Models such as Logistic Regression, SVM, and KNN are sensitive to feature magnitude — features with larger values dominate the learning process unfairly.

**StandardScaler was applied:**

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
```

StandardScaler transforms each feature using the formula:

```
z = (x - mean) / standard_deviation
```

After scaling:

| Feature | Mean | Std Dev |
|---|---|---|
| Age | ~0.0000 | ~1.0006 |
| Fare | ~0.0000 | ~1.0006 |

Both features are now on the same scale, centered at zero with unit variance.

---

### Step 8 — Feature Distributions

Histograms of all features were plotted after preprocessing to confirm the final distribution of each variable and verify that scaling and encoding were applied correctly.

```python
df.hist(figsize=(12, 8), bins=20, color='steelblue', edgecolor='white')
plt.suptitle("Feature Distributions After Preprocessing")
plt.savefig('plots/distributions.png')
```

---

### Step 9 — Correlation Heatmap

A correlation heatmap was generated to understand the linear relationships between all features and, most importantly, their relationship with the target variable `Survived`.

```python
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.savefig('plots/correlation.png')
```

**Key findings:**

| Feature | Correlation with Survived | Interpretation |
|---|---|---|
| Sex | Strong negative | Female passengers (encoded as 0) survived at a significantly higher rate |
| Pclass | Negative | Higher class number (lower class) correlates with lower survival |
| Fare | Positive | Higher fares generally indicate higher class, correlating with better survival |

---

### Bonus Step — Baseline Logistic Regression Model

Although a predictive model was not required for this task, a simple Logistic Regression was trained on the cleaned dataset to validate the preprocessing pipeline. A model that trains and produces reasonable accuracy is direct confirmation that the data is correctly structured and clean.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**Results:**

| Metric | Value |
|---|---|
| Training samples | 620 |
| Test samples | 155 |
| Overall accuracy | 75.48% |

```
Classification Report:

                 Precision   Recall   F1-Score   Support
Did Not Survive   0.77        0.86     0.81        95
Survived          0.73        0.58     0.65        60

Accuracy                               0.75       155
```

A 75% baseline accuracy with no feature engineering or hyperparameter tuning confirms the preprocessing pipeline is correct and the data is ready for further modelling.

---

## Before vs After Comparison

| Property | Before Preprocessing | After Preprocessing |
|---|---|---|
| Total rows | 891 | 775 |
| Total columns | 12 | 10 |
| Missing values | 866 | 0 |
| Text-based columns | 4 | 0 |
| Outliers in Fare | Present | Removed |
| Feature scale | Inconsistent | Standardized (mean = 0, std = 1) |
| Model-ready | No | Yes |

---

## Key Concepts Applied

**Median vs Mean Imputation**
The median is preferred over the mean when a distribution is skewed or contains outliers, as it is not influenced by extreme values.

**Label Encoding vs One-Hot Encoding**
Label Encoding is appropriate for binary categories. One-Hot Encoding is required for multi-class nominal categories to avoid introducing a false ordinal relationship.

**IQR vs Z-Score for Outlier Detection**
IQR is non-parametric and works reliably on skewed data. Z-score assumes a Gaussian distribution and may miss outliers in heavily skewed distributions.

**Standardization vs Normalization**
StandardScaler (z-score standardization) is preferred when data may contain outliers or does not follow a normal distribution. MinMaxScaler (normalization to [0, 1]) is sensitive to extreme values and better suited to data that is already normally distributed.

**Baseline Model Validation**
Training a simple model on cleaned data is a best practice to verify the preprocessing was applied correctly before investing time in complex model development.

---

## How to Run

```bash
# Step 1 — Install dependencies
pip install -r requirements.txt

# Step 2 — Run the full preprocessing pipeline
python data_preprocessing.py

# Output files generated:
#   Cleaned_Titanic_Data.csv
#   plots/missing_before.png
#   plots/missing_after.png
#   plots/outliers_before.png
#   plots/outliers_after.png
#   plots/distributions.png
#   plots/correlation.png
```

---

## Output Files

| File | Description |
|---|---|
| `Cleaned_Titanic_Data.csv` | Final preprocessed dataset, ready for ML model training |
| `plots/missing_before.png` | Heatmap of missing values in the raw dataset |
| `plots/missing_after.png` | Heatmap confirming zero missing values after cleaning |
| `plots/outliers_before.png` | Boxplot of Age and Fare before outlier removal |
| `plots/outliers_after.png` | Boxplot of Age and Fare after outlier removal |
| `plots/distributions.png` | Histograms of all features after preprocessing |
| `plots/correlation.png` | Pairwise feature correlation heatmap |

---

*ElevateLabs Machine Learning Internship — Task 1*
*Submitted: April 2026*
