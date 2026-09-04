# 🚀 Quantum Error Correction: Deep Learning Decoders for Surface Codes

An end-to-end Machine Learning engineering pipeline designed to decode topological quantum errors. This project simulates a Distance-5 repetition code using **Stim** and evaluates ML architectures—from Random Forests to Spatiotemporal Convolutional Neural Networks (2D-CNNs)—to protect logical qubits from hardware noise.

## 🧠 The Problem: The Accuracy Paradox in QEC
Quantum error correction suffers from severe class imbalance. At a 3% physical error rate, catastrophic logical flips are rare (~10% of the time). Standard ML models optimize for overall accuracy by simply guessing that the quantum state is "safe," missing the critical errors that corrupt quantum memory.

## 🛠️ Tech Stack
* **Quantum Simulation:** Stim (High-speed Clifford simulator)
* **Machine Learning:** PyTorch, Scikit-Learn
* **Data Processing:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn

## 📈 Model Evolution & Algorithmic Strategy

1. **Random Forest (Baseline):** Processed flat 20-bit syndrome vectors. Achieved 91.5% accuracy but failed catastrophically on **Recall (19%)**, completely missing the majority of logical errors due to class imbalance.
2. **PyTorch Deep MLP:** Implemented `BCEWithLogitsLoss` with a highly weighted positive class penalty. This forced the network to prioritize rare failures, rocketing **Recall to 93%**.
3. **2D-CNN (Spatiotemporal):** Reshaped the flat detector data into a `6x4` matrix (Time/Rounds × Space/Detectors). Localized convolutional kernels tracked error propagation timelines, pushing **Recall to 96%** while recovering precision and achieving a **0.79 F1-Score**.

## 📊 Results Summary

| Model Architecture | Overall Accuracy | Logical Flip Recall | F1-Score |
|-------------------|------------------|----------------------|----------|
| Random Forest     | 91.56%           | 19.00%               | 0.32     |
| PyTorch MLP       | 93.12%           | 93.47%               | 0.73     |
| PyTorch 2D-CNN    | **95.00%**       | **96.00%**           | **0.79** |

![Model Evolution](model_evolution.png)
![Model Evolution and Confusion Matrices](triple_confusion_matrix.png)
## 🚀 How to Run Locally

1. **Clone the repo and set up the environment:**
   ```bash
   conda create -n qec_ml python=3.10
   conda activate qec_ml
   pip install stim torch scikit-learn numpy matplotlib seaborn