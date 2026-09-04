import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load and prepare data
X = np.load("data/X_stim_syndromes.npy")
y = np.load("data/y_stim_labels.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert numpy arrays to PyTorch Tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)

# 2. Define the Neural Network Architecture
class QECDecoderMLP(nn.Module):
    def __init__(self, input_dim=20):
        super(QECDecoderMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Outputs raw logits
        )
        
    def forward(self, x):
        return self.net(x)

model = QECDecoderMLP(input_dim=X.shape[1])

# 3. Calculate Class Weight to heavily penalize missing Class 1 errors
num_zeros = (y_train == 0).sum()
num_ones = (y_train == 1).sum()
pos_weight = torch.tensor([num_zeros / num_ones], dtype=torch.float32)

# BCEWithLogitsLoss combines Sigmoid + Binary Cross Entropy with class weighting
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training Loop
print("🧠 Training PyTorch Deep MLP Decoder with Weighted Loss...")
model.train()
for epoch in range(15):
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1:02d}/15 | Loss: {total_loss/len(train_loader):.4f}")

# 5. Evaluation
model.eval()
with torch.no_grad():
    raw_preds = model(X_test_t)
    probs = torch.sigmoid(raw_preds)
    preds = (probs > 0.5).float().numpy()

print("\n================== PYTORCH MLP PERFORMANCE REPORT ==================")
print(confusion_matrix(y_test, preds))
print("\nDetailed Metrics:")
print(classification_report(y_test, preds, target_names=["Safe (0)", "Logical Flip (1)"]))