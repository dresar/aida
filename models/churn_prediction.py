import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Set plot style
sns.set(style="whitegrid")

def load_and_preprocess_data(filepath):
    print("--- Loading Data ---")
    df = pd.read_csv(filepath)
    
    # 1. Handle TotalCharges (convert to numeric, handle errors)
    # The dataset typically has empty strings " " for TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill missing values (mostly where tenure=0) with 0 or median
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Drop customerID as it's not predictive
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        
    print(f"Data Shape: {df.shape}")
    print(f"Missing Values:\n{df.isnull().sum().sum()}")
    
    return df

def build_pipeline(X_train):
    # Identify numerical and categorical columns
    # We explicitly listed: tenure, MonthlyCharges, TotalCharges as numerical
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [col for col in X_train.columns if col not in numeric_features]
    
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    # Preprocessing for numerical data: Scaling
    numeric_transformer = StandardScaler()
    
    # Preprocessing for categorical data: One-Hot Encoding
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        verbose_feature_names_out=False
    )
    
    # Define the full pipeline: Preprocess -> SMOTE -> XGBoost
    # Note: We use imblearn's Pipeline to handle SMOTE correctly during cross-validation (if used)
    # and to ensure it's only applied to training data during fit.
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            eval_metric='logloss'
        ))
    ])
    
    return pipeline

def evaluate_model(model, X_test, y_test):
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return y_pred, y_prob, roc_auc

def plot_results(y_test, y_pred, y_prob):
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_prob):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('roc_curve.png')
    print("ROC curve saved to roc_curve.png")

def analyze_feature_importance(pipeline, feature_names_in):
    print("\n--- Feature Importance ---")
    
    # Extract the model and preprocessor
    model = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Get feature names after one-hot encoding
    # This requires scikit-learn >= 1.0
    try:
        feature_names_out = preprocessor.get_feature_names_out()
    except AttributeError:
        # Fallback for older versions or if extraction fails
        print("Could not extract specific feature names automatically.")
        return

    importances = model.feature_importances_
    
    # Create a DataFrame
    feat_imp = pd.DataFrame({
        'Feature': feature_names_out,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("Top 10 Important Features:")
    print(feat_imp.head(10))
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_imp.head(15))
    plt.title('Top 15 Feature Importances (XGBoost)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Feature importance plot saved to feature_importance.png")

def main():
    # 0. Check for dataset
    import os
    dataset_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found. Creating dummy data...")
        import create_dummy_data
        create_dummy_data.create_dummy_data(dataset_path)
    
    # 1. Load Data
    df = load_and_preprocess_data(dataset_path)
    
    # 2. Split Features and Target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Encode Target (Yes/No -> 1/0) if not already numeric
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        print(f"Target encoded: {le.classes_}")
        
    # 3. Train Test Split
    # Stratify is important for imbalanced datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Training Set: {X_train.shape}")
    print(f"Testing Set: {X_test.shape}")
    
    # 4. Build and Train Pipeline (includes Preprocessing, SMOTE, XGBoost)
    pipeline = build_pipeline(X_train)
    
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # 5. Evaluate
    y_pred, y_prob, roc_auc = evaluate_model(pipeline, X_test, y_test)
    
    # 6. Visualize
    plot_results(y_test, y_pred, y_prob)
    
    # 7. Feature Importance
    analyze_feature_importance(pipeline, X.columns)

if __name__ == "__main__":
    main()
