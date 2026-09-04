import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load Data
X = np.load("data/X_stim_syndromes.npy")
y = np.load("data/y_stim_labels.npy")

# 2. FIXED: Reshape into (Batch, Channels, Rounds, Detectors)
# 24 features -> 6 temporal steps x 4 spatial detectors
# Using X.shape[0] explicitly prevents silent batch-size mismatches
X_2d = X.reshape(X.shape[0], 1, 6, 4)

X_train, X_test, y_train, y_test = train_test_split(
    X_2d, y, test_size=0.2, random_state=42, stratify=y
)

# Convert to Tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)

# 3. Define 2D-CNN Architecture
class QECDecoderCNN(nn.Module):
    def __init__(self):
        super(QECDecoderCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # First Conv Layer: Scans for local 3x3 error clusters
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            # Second Conv Layer: Detects cross-round temporal tracks
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            # FIXED: Updated flattened dimension from 5 to 6 rounds (32 channels * 6 * 4)
            nn.Linear(32 * 6 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)

model = QECDecoderCNN()

# 4. Weighted Loss Setup
num_zeros = (y_train == 0).sum()
num_ones = (y_train == 1).sum()
pos_weight = torch.tensor([num_zeros / num_ones], dtype=torch.float32)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training Loop
print("🖼️ Training 2D Spatiotemporal CNN Decoder...")
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

# 6. Evaluation
model.eval()
with torch.no_grad():
    raw_preds = model(X_test_t)
    probs = torch.sigmoid(raw_preds)
    preds = (probs > 0.5).float().numpy()

print("\n================== PYTORCH 2D-CNN PERFORMANCE REPORT ==================")
print(confusion_matrix(y_test, preds))
print("\nDetailed Metrics:")
print(classification_report(y_test, preds, target_names=["Safe (0)", "Logical Flip (1)"]))