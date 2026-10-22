import numpy as np
import stim
import pymatching
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# 1. Load the SAME dataset your ML models used
X = np.load("data/X_stim_syndromes.npy")   # detection events, shape (100000, 24)
y = np.load("data/y_stim_labels.npy")      # logical flip labels

# 2. Recreate the SAME test split (same random_state and stratify as the ML scripts)
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Rebuild the SAME circuit that generated the data.
#    These parameters must match phase2_stim_generator.py exactly.
circuit = stim.Circuit.generated(
    "repetition_code:memory",
    distance=5,
    rounds=5,
    after_clifford_depolarization=0.03,
    before_measure_flip_probability=0.03,
)

# 4. Build the detector error model and the MWPM decoder from it
dem = circuit.detector_error_model(decompose_errors=True)
print("Detectors in circuit:", dem.num_detectors, "| Features in dataset:", X.shape[1])
assert dem.num_detectors == X.shape[1], "Circuit does not match the dataset!"

matcher = pymatching.Matching.from_detector_error_model(dem)

# 5. Decode the test syndromes
pred = matcher.decode_batch(X_test.astype(np.uint8)).flatten()
y_true = y_test.astype(np.uint8)

# 6. Evaluate
cm = confusion_matrix(y_true, pred)
errors = (pred != y_true).sum()

print("\n=========== MWPM (PyMatching) RESULTS ===========")
print("Confusion matrix:\n", cm)
print(f"Precision (flip): {precision_score(y_true, pred):.3f}")
print(f"Recall (flip):    {recall_score(y_true, pred):.3f}")
print(f"F1 (flip):        {f1_score(y_true, pred):.3f}")
print(f"Accuracy:         {(pred == y_true).mean() * 100:.2f}%")
print(f"Logical error rate (wrong predictions / shots): {errors / len(y_true):.4f}")

# 7. Compare with your CNN (from your earlier run: 897 FP + 81 FN out of 20,000)
cnn_errors = 897 + 81
print(f"\nCNN logical error rate (from your results): {cnn_errors / 20000:.4f}")