import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os

# Setup
os.makedirs('plots', exist_ok=True)
sns.set_theme(style="darkgrid")


# Step 1: Load & Explore
def load_data(filepath):
    """Load the dataset and print basic info."""
    df = pd.read_csv(filepath)
    print("=" * 55)
    print("STEP 1: Dataset Overview")
    print("=" * 55)
    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nBasic Statistics:\n", df.describe())
    return df


# Step 2: Visualize Missing Values (Before)
def plot_missing_before(df):
    """Save a heatmap of missing values in the original dataset."""
    plt.figure(figsize=(12, 5))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title("Missing Values — Before Cleaning", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/missing_before.png', dpi=150)
    plt.close()
    print("\n[Saved] plots/missing_before.png")


# Step 3: Handle Missing Values
def handle_missing(df):
    """Impute missing values using appropriate strategies per column."""
    print("\n" + "=" * 55)
    print("STEP 3: Handling Missing Values")
    print("=" * 55)

    # Age → Median (robust to skew/outliers)
    median_age = df['Age'].median()
    df['Age'] = df['Age'].fillna(median_age)
    print(f"\n  Age: filled {df['Age'].isnull().sum()} nulls with median ({median_age:.1f})")

    # Embarked → Mode (most frequent port)
    mode_embarked = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(mode_embarked)
    print(f"  Embarked: filled nulls with mode ('{mode_embarked}')")

    # Cabin → 'Unknown' (too sparse to impute meaningfully)
    df['Cabin'] = df['Cabin'].fillna('Unknown')
    print(f"  Cabin: replaced nulls with 'Unknown'")

    return df


# Step 4: Visualize Missing Values (After)
def plot_missing_after(df):
    """Save a heatmap of missing values after imputation."""
    plt.figure(figsize=(12, 5))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title("Missing Values — After Cleaning", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/missing_after.png', dpi=150)
    plt.close()
    print("\n[Saved] plots/missing_after.png")


# Step 5: Encode Categorical Features
def encode_features(df):
    """Convert text categories to numerical representations."""
    print("\n" + "=" * 55)
    print("STEP 5: Encoding Categorical Features")
    print("=" * 55)

    # Sex → Binary Label Encoding
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    print(f"\n  Sex: Label Encoded -> {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Embarked → One-Hot Encoding (no ordinal relationship between ports)
    df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked')
    print("  Embarked: One-Hot Encoded -> Embarked_C, Embarked_Q, Embarked_S")

    # Drop columns with no predictive value
    cols_to_drop = ['Name', 'Ticket', 'PassengerId', 'Cabin']
    df.drop(cols_to_drop, axis=1, inplace=True)
    print(f"  Dropped: {cols_to_drop}")

    return df


# Step 6: Outlier Detection & Removal
def remove_outliers(df):
    """Detect and remove outliers in 'Fare' using the IQR method."""
    print("\n" + "=" * 55)
    print("STEP 6: Outlier Removal (IQR Method)")
    print("=" * 55)

    # Boxplot BEFORE
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df[['Age', 'Fare']], palette='Set2')
    plt.title("Boxplot — Before Outlier Removal", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/outliers_before.png', dpi=150)
    plt.close()
    print("\n[Saved] plots/outliers_before.png")

    # IQR calculation on Fare
    Q1 = df['Fare'].quantile(0.25)
    Q3 = df['Fare'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    before = df.shape[0]
    df = df[(df['Fare'] >= lower) & (df['Fare'] <= upper)]
    after = df.shape[0]
    print(f"\n  Fare IQR bounds: [{lower:.2f}, {upper:.2f}]")
    print(f"  Rows removed: {before - after}  ({before} -> {after})")

    # Boxplot AFTER
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df[['Age', 'Fare']], palette='Set2')
    plt.title("Boxplot — After Outlier Removal", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/outliers_after.png', dpi=150)
    plt.close()
    print("[Saved] plots/outliers_after.png")

    return df


# Step 7: Feature Scaling
def scale_features(df):
    """Standardize numerical features to zero mean and unit variance."""
    print("\n" + "=" * 55)
    print("STEP 7: Feature Scaling (Standardization)")
    print("=" * 55)

    scaler = StandardScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
    print(f"\n  Age  -> mean: {df['Age'].mean():.4f}, std: {df['Age'].std():.4f}")
    print(f"  Fare -> mean: {df['Fare'].mean():.4f}, std: {df['Fare'].std():.4f}")

    return df


# Step 8: Feature Distributions
def plot_distributions(df):
    """Plot histograms of all numerical features."""
    df.hist(figsize=(12, 8), bins=20, color='steelblue', edgecolor='white')
    plt.suptitle("Feature Distributions (After Preprocessing)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/distributions.png', dpi=150)
    plt.close()
    print("\n[Saved] plots/distributions.png")


# Step 9: Correlation Heatmap
def plot_correlation(df):
    """Plot a heatmap showing correlations between all features."""
    # Ensure only numeric columns are used
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        linewidths=0.5,
        square=True
    )
    plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/correlation.png', dpi=150)
    plt.close()
    print("[Saved] plots/correlation.png")


# Step 10: Bonus — Logistic Regression Model
def run_baseline_model(df):
    """Train a basic Logistic Regression to validate the cleaned data."""
    print("\n" + "=" * 55)
    print("BONUS STEP: Baseline Logistic Regression Model")
    print("=" * 55)

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"  Accuracy: {acc * 100:.2f}%")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Did Not Survive', 'Survived']))


# Main Pipeline
if __name__ == "__main__":

    # Load raw data and keep a copy for comparison
    original_df = load_data('Titanic-Dataset.csv')

    # Visualize missing values BEFORE
    plot_missing_before(original_df)

    # Run cleaning pipeline
    df = original_df.copy()
    df = handle_missing(df)

    # Visualize missing values AFTER
    plot_missing_after(df)

    df = encode_features(df)
    df = remove_outliers(df)
    df = scale_features(df)

    # Generate visual reports
    plot_distributions(df)
    plot_correlation(df)

    # Before vs After Summary
    print("\n" + "=" * 55)
    print("SUMMARY: Before vs After Preprocessing")
    print("=" * 55)
    print(f"\n  Rows:     {original_df.shape[0]:>4}  ->  {df.shape[0]}")
    print(f"  Columns:  {original_df.shape[1]:>4}  ->  {df.shape[1]}")
    print(f"  Nulls:    {original_df.isnull().sum().sum():>4}  ->  {df.isnull().sum().sum()}")

    # Save cleaned dataset
    df.to_csv('Cleaned_Titanic_Data.csv', index=False)
    print("\n  Cleaned data saved -> Cleaned_Titanic_Data.csv")

    # Bonus: run a quick model to show preprocessing worked
    run_baseline_model(df)

    print("\nDone! All steps complete. Check the 'plots/' folder for visualizations.")
