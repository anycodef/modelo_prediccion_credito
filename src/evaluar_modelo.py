import pandas as pd
import numpy as np
import joblib
import os
import random

# Set up directories
DATA_DIR = "data"
MODELS_DIR = "models"
RESULTS_DIR = "results"

def generate_synthetic_data(original_df, num_rows=100):
    print(f"Generating synthetic data with {num_rows} rows...")
    synthetic_data = {}

    for col in original_df.columns:
        if col == 'obtuvo_credito':
            continue # Skip target, we will generate a random one later for comparison

        if original_df[col].dtype == 'object':
            # Categorical: sample from unique values
            unique_values = original_df[col].unique()
            synthetic_data[col] = np.random.choice(unique_values, size=num_rows)
        elif original_df[col].dtype == 'bool':
             # Boolean: sample True/False
            synthetic_data[col] = np.random.choice([True, False], size=num_rows)
        elif np.issubdtype(original_df[col].dtype, np.number):
            # Numerical: sample from a range or distribution
            min_val = original_df[col].min()
            max_val = original_df[col].max()
            if np.issubdtype(original_df[col].dtype, np.integer):
                 synthetic_data[col] = np.random.randint(min_val, max_val + 1, size=num_rows)
            else:
                 synthetic_data[col] = np.random.uniform(min_val, max_val, size=num_rows)

    return pd.DataFrame(synthetic_data)

def main():
    # 1. Load Original Data to understand structure
    original_data_path = os.path.join(DATA_DIR, "entrenamientos_datos_credito_train_ready.csv")
    if not os.path.exists(original_data_path):
        print(f"Error: Original data not found at {original_data_path}")
        return

    original_df = pd.read_csv(original_data_path)

    # 2. Load Model
    model_path = os.path.join(MODELS_DIR, "modelo_xgboost_credito.pkl")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    # 3. Generate Synthetic Data
    synthetic_df = generate_synthetic_data(original_df, num_rows=50)

    # Save synthetic data
    synthetic_data_path = os.path.join(DATA_DIR, "datos_inventados_evaluacion.csv")
    synthetic_df.to_csv(synthetic_data_path, index=False)
    print(f"Synthetic data saved to {synthetic_data_path}")

    # 4. Preprocess Synthetic Data (Pre-pipeline steps like bool conversion)
    # The pipeline handles OneHot, but we need to ensure booleans are ints if the pipeline expects ints or pass them as is if pipeline handles it.
    # In train script, we converted bools to int before passing to pipeline. We must do the same here.
    print("Preprocessing synthetic data...")
    processed_df = synthetic_df.copy()
    bool_cols = processed_df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        processed_df[col] = processed_df[col].astype(int)

    # 5. Predict
    print("Predicting...")
    predictions = model.predict(processed_df)
    probabilities = model.predict_proba(processed_df)

    # 6. Generate "Real" Labels (Invented) for comparison
    real_labels = np.random.choice([0, 1], size=len(synthetic_df))

    # 7. Create Evaluation Table
    # Select a few important features to show in the table
    display_cols = ['nivel_educacion', 'departamento', 'edad', 'numero_parcelas']
    # Ensure these columns exist
    display_cols = [c for c in display_cols if c in synthetic_df.columns]

    eval_table = synthetic_df[display_cols].copy()
    eval_table['Real Label (Invented)'] = real_labels
    eval_table['Predicted Label'] = predictions
    # Probability of class 1 (Si Credito)
    eval_table['Prediction Probability (Si)'] = probabilities[:, 1]

    # Map labels to meaningful strings
    eval_table['Real Label (Invented)'] = eval_table['Real Label (Invented)'].map({0: 'No', 1: 'Si'})
    eval_table['Predicted Label'] = eval_table['Predicted Label'].map({0: 'No', 1: 'Si'})

    # Save table
    results_path = os.path.join(RESULTS_DIR, "evaluacion_datos_inventados.csv")
    eval_table.to_csv(results_path, index=False)
    print(f"Evaluation table saved to {results_path}")

    # Print first few rows
    print("\nEvaluation Table Preview:")
    print(eval_table.head())

if __name__ == "__main__":
    main()
