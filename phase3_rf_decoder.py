import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train_rf_decoder():
    print("Phase 3: Loading Stim Generated Quantum Dataset")

    # 1.Loading pre-generated numpy arrays
    X_path= "data/X_stim_syndromes.npy"
    y_path= "data/y_stim_labels.npy"

    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError("Dataset files not found! Please run the Stim generator script first.")

    X=np.load(X_path)   
    y=np.load(y_path)

    print(f"Dataset loaded successfully.Features Shape: {X.shape}, Labels Shape:{y.shape}")

    #2. Split into Training (80%) and Testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    # Note: Stratify=y ensures both train and test sets get same percentage of rare logical errors!

    #3. Initializing and Training the Random Forest Classifier
    print("\n🌲 Training the Random Forest Decoder (this might take a few seconds)...")
    # n_estimators=50 means we are building an ensemble of 50 independent decision trees
    # max_depth=10 prevents the trees from over-complicating and memorizing noise
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Training complete!")

    # 4. Run Inference on Unseen Testing Data
    print("\n🔮 Running decoder inference on 20,000 unseen quantum snapshots...")
    y_pred = model.predict(X_test)

    # 5. Comprehensive Performance Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n================== DECODER PERFORMANCE REPORT ==================")
    print(f"Overall Model Accuracy: {accuracy * 100:.2f}%")
    print("================================================================")
    
    print("\n📊 Confusion Matrix Breakdown:")
    print(f"True Negatives  (No error, AI said Safe):       {cm[0][0]}")
    print(f"False Positives (No error, AI said Failed):     {cm[0][1]}")
    print(f"False Negatives (Logical failure, AI MISSED IT): {cm[1][0]}")
    print(f"True Positives  (Logical failure, AI CAUGHT IT): {cm[1][1]}")
    
    print("\n📋 Detailed Classification Metrics:")
    print(classification_report(y_test, y_pred, target_names=["Safe (0)", "Logical Flip (1)"]))


if __name__ == "__main__":
    train_rf_decoder()