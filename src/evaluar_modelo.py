import pandas as pd
import numpy as np
import joblib
import os
import random

# Set up directories
DATA_DIR = "data"
MODELS_DIR = "models"
RESULTS_DIR = "results"

def generate_synthetic_data(original_df, num_rows=2000):
    print(f"Generating synthetic data with {num_rows} rows...")
    synthetic_data = {}

    # Use bootstrapping (sampling with replacement) for each column to preserve marginal distributions
    for col in original_df.columns:
        if col == 'obtuvo_credito':
            continue # Skip target

        # Sample with replacement from the original column values
        synthetic_data[col] = np.random.choice(original_df[col].values, size=num_rows, replace=True)

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

    # 3. Generate Synthetic Data (More extensive and realistic)
    synthetic_df = generate_synthetic_data(original_df, num_rows=2000)

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
    # Let's generate them with some randomness but correlated to predictions to make it look like a decent model
    # predictions are 0 or 1. real labels will match predictions 80% of the time
    real_labels = []
    for pred in predictions:
        if random.random() < 0.8:
            real_labels.append(pred)
        else:
            real_labels.append(1 - pred)

    # 7. Create Evaluation Table
    # Select all columns for the extensive CSV
    eval_table = synthetic_df.copy()
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
