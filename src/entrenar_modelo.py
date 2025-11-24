import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier

# Set up directories
DATA_DIR = "data"
MODELS_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    print("Preprocessing data...")
    # Convert boolean columns to int
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)

    return df

def create_pipeline(categorical_cols):
    print("Creating pipeline...")
    # Categorical preprocessing
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough' # Keep numerical/boolean columns as is
    )

    # Define the model
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    # Create the pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', xgb)])
    return pipeline

def plot_distribution(y_train, y_test):
    print("Plotting data distribution...")
    labels = ['Training Set', 'Test Set']
    sizes = [len(y_train), len(y_test)]
    colors = ['#ff9999','#66b3ff']
    explode = (0.1, 0)

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')
    plt.title("Data Split Distribution")
    plt.savefig(os.path.join(RESULTS_DIR, "distribution_split.png"))
    plt.close()
    print(f"Distribution plot saved to {RESULTS_DIR}/distribution_split.png")

def plot_confusion_matrix(y_true, y_pred, labels):
    print("Plotting confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()
    print(f"Confusion matrix plot saved to {RESULTS_DIR}/confusion_matrix.png")

def main():
    # 1. Load Data
    filepath = os.path.join(DATA_DIR, "entrenamientos_datos_credito_train_ready.csv")
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return

    df = load_data(filepath)

    # 2. Identify Features and Target
    target_col = 'obtuvo_credito'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Convert target to int if boolean
    if y.dtype == 'bool':
        y = y.astype(int)

    # Preprocess X (mostly bool to int conversion for consistency, though Passthrough handles it)
    X = preprocess_data(X)

    # Identify categorical columns for OneHotEncoder
    # We assume object type columns are categorical
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns identified: {categorical_cols}")

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Plot Distribution
    plot_distribution(y_train, y_test)

    # 5. Setup Pipeline and GridSearch
    pipeline = create_pipeline(categorical_cols)

    # Parameters for Grid Search
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__subsample': [0.8, 1.0]
    }

    print("Starting Grid Search...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters found: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_

    # 6. Evaluate
    print("Evaluating model...")
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nModel Performance Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Save Metrics to text file
    with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
        f.write("Model Performance Metrics:\n")
        f.write(f"Best Params: {grid_search.best_params_}\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred))

    # Plot Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, labels=['No Credito', 'Si Credito'])

    # 7. Save Model
    model_path = os.path.join(MODELS_DIR, "modelo_xgboost_credito.pkl")
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
