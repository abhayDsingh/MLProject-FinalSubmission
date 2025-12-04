import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

print("Script started...")

# --- Loop from 1 to 4 ---
for dataset_num in range(1, 5):
    print(f"\n--- Processing Dataset {dataset_num} ---")

    # 1. Define constants and file paths
    MISSING_VAL = 1.00000000000000e+99
    base_path = "classification" 
    train_data_file = os.path.join(base_path, f"TrainData{dataset_num}.txt")
    train_label_file = os.path.join(base_path, f"TrainLabel{dataset_num}.txt")
    test_data_file = os.path.join(base_path, f"TestData{dataset_num}.txt")
    output_file = f"SinghClassification{dataset_num}.txt"

    # 2. Load Data
    print(f"Loading data for dataset {dataset_num}...")
    try:
        X_train_raw = pd.read_csv(train_data_file, sep='\s+', header=None, engine='python')
        y_train_raw = pd.read_csv(train_label_file, header=None)
        X_test_raw = pd.read_csv(test_data_file, sep='\s+', header=None, engine='python')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        continue 

    # 3. Preprocess Data
    print("Preprocessing data...")
    X_train_raw.replace(MISSING_VAL, np.nan, inplace=True)
    X_test_raw.replace(MISSING_VAL, np.nan, inplace=True)
    
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(X_train_raw) 
    
    X_train_imputed = imputer.transform(X_train_raw)
    X_test_imputed = imputer.transform(X_test_raw)
    y_train_flat = y_train_raw.values.ravel()

    # --------------------------------------------------------------------
    # Step 3.5 - Validate Model Performance 
    # --------------------------------------------------------------------
    print("Validating model performance...")
    
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_imputed, y_train_flat, test_size=0.2, random_state=42
    )
    
    # --- SMOTE LOGIC FOR VALIDATION ---
    # By default, use the original split data
    X_train_for_validation = X_train_split
    y_train_for_validation = y_train_split
    
    if dataset_num == 4:
        print("Dataset 4 is imbalanced. Applying SMOTE to validation training split...")
        smote = SMOTE(random_state=42)
        # Apply SMOTE only to the 80% training portion
        X_train_for_validation, y_train_for_validation = smote.fit_resample(X_train_split, y_train_split)
        print("SMOTE applied to validation split.")
    # --- END SMOTE LOGIC ---

    validation_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Train on the (potentially resampled) data
    validation_model.fit(X_train_for_validation, y_train_for_validation)
    
    # Test on the *original* 20% validation split (never test on synthetic data)
    validation_preds = validation_model.predict(X_val_split)
    
    accuracy = accuracy_score(y_val_split, validation_preds)
    print(f"\n--- VALIDATION METRICS (Dataset {dataset_num}) ---")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_val_split, validation_preds, zero_division=0))
    print("----------------------------------------\n")

    # --------------------------------------------------------------------
    # 4. Train FINAL Model (On 100% of data for submission)
    # --------------------------------------------------------------------
    print("Training FINAL model on 100% of training data...")

    # --- SMOTE LOGIC FOR FINAL MODEL ---
    # By default, use the original full training data
    X_train_for_final = X_train_imputed
    y_train_for_final = y_train_flat

    if dataset_num == 4:
        print("Dataset 4 is imbalanced. Applying SMOTE to 100% of training data for final model...")
        smote_final = SMOTE(random_state=42)
        # Apply SMOTE to the *entire* training set
        X_train_for_final, y_train_for_final = smote_final.fit_resample(X_train_imputed, y_train_flat)
        print("SMOTE applied to final training data.")
    # --- END SMOTE LOGIC ---

    final_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Train the model on the (potentially resampled) full training data
    final_model.fit(X_train_for_final, y_train_for_final)
    print("Final model training complete.")

    # 5. Predict and Save (Using the FINAL model)
    print(f"Generating final predictions and saving to {output_file}...")
    # Predict on the *original* test data
    final_predictions = final_model.predict(X_test_imputed)
    np.savetxt(output_file, final_predictions, fmt='%d')
    print(f"Successfully created {output_file}!")

print("\n--- Classification complete! All 4 files generated. ---")