import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

print("Spam detection script started...")

# 1. Define constants and file paths
base_path = "Spam Email Detection"
train_file_1 = os.path.join(base_path, "spam_train1.csv")
train_file_2 = os.path.join(base_path, "spam_train2.csv")
test_file = os.path.join(base_path, "spam_test.csv")

# !!! IMPORTANT: Change "MyLastName" to your actual last name !!!
output_file = "SinghSpam.txt"

# 2. Load and Combine Training Data
print("Loading and combining training data...")
try:
    # Added encoding='latin1' to handle special characters
    df_train1 = pd.read_csv(train_file_1, usecols=['v1', 'v2'], encoding='latin1')
    df_train1.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)
    
    df_train2 = pd.read_csv(train_file_2, usecols=['label', 'text'], encoding='latin1')
    
    df_train = pd.concat([df_train1, df_train2], ignore_index=True)
    df_train.dropna(subset=['label', 'text'], inplace=True)
    
    print(f"Total training samples loaded: {len(df_train)}")
    
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please make sure the 'Spam Email Detection' folder is in the same directory as this script")
    exit()

# 3. Prepare Data for ML
print("Preparing data for machine learning...")
X_train_text = df_train['text']
y_train = df_train['label'].map({'ham': 0, 'spam': 1})

# 4. Vectorize the Text (TF-IDF)
print("Initializing and fitting TF-IDF Vectorizer...")
# We must fit the vectorizer on ALL training text to build a complete vocabulary
vectorizer = TfidfVectorizer(
    stop_words='english',
    lowercase=True,
    max_features=5000
)
# .fit_transform() learns the vocab and converts all training text
X_train_numeric = vectorizer.fit_transform(X_train_text)
print(f"Training data transformed into shape: {X_train_numeric.shape}")


# --------------------------------------------------------------------
# NEW: Step 4.5 - Validate Model Performance
# --------------------------------------------------------------------
print("Validating model performance...")
# Now we split the *numerical* data for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_numeric, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print("Training validation model...")
validation_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
validation_model.fit(X_train_split, y_train_split)

print("Generating validation predictions...")
validation_preds = validation_model.predict(X_val_split)
accuracy = accuracy_score(y_val_split, validation_preds)
print("\n--- VALIDATION METRICS (Spam Detection) ---")
print(f"Accuracy: {accuracy * 100:.2f}%")
# 'target_names' makes the report readable (0=ham, 1=spam)
print(classification_report(y_val_split, validation_preds, target_names=['ham (0)', 'spam (1)'], zero_division=0))
print("----------------------------------------\n")

# --------------------------------------------------------------------
# 5. Train FINAL Model (On 100% of data for submission)
# --------------------------------------------------------------------
print("Training FINAL model on 100% of training data...")
final_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
# Train on the *entire* numerical training set
final_model.fit(X_train_numeric, y_train)
print("Final model training complete.")

# 6. Predict on Test Data
print("Loading and preparing test data...")
try:
    df_test = pd.read_csv(test_file, encoding='latin1')
    # Use .fillna('') to handle any potential empty messages
    X_test_text = df_test['message'].fillna('') 
except FileNotFoundError:
    print(f"Error: Could not find {test_file}")
    exit()

print("Transforming test data using the fitted vectorizer...")
# Use .transform() only - this uses the vocabulary we already learned
X_test_numeric = vectorizer.transform(X_test_text)

print("Generating final predictions...")
final_predictions = final_model.predict(X_test_numeric)

# 7. Save the Final Output
print(f"Saving predictions to {output_file}...")
np.savetxt(output_file, final_predictions, fmt='%d')

print(f"\n--- Spam detection complete! File created: {output_file} ---")