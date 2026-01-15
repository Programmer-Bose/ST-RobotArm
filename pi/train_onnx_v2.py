import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.onnx

# 1. Load Data
try:
    data = pd.read_csv("ik_data_orientation.csv")
except FileNotFoundError:
    print("Error: ik_data_orientation.csv not found!")
    exit()

# INPUTS: Pixel X, Pixel Y, AND Angle
X = data[['pixel_x', 'pixel_y', 'obj_angle']].values.astype(np.float32)
# OUTPUTS: 5 Joints (Base, Shoulder, Elbow, Pitch, Roll)
y = data[['j11', 'j12', 'j14', 'j15', 'j16']].values.astype(np.float32)

# Normalize
# X (0-640), Y (0-480) -> roughly divide by 640
X_norm = X.copy()
X_norm[:, 0] = X[:, 0] / 640.0
X_norm[:, 1] = X[:, 1] / 640.0
# Angle (Assuming 0-180 range from collector) -> divide by 180
X_norm[:, 2] = X[:, 2] / 180.0

y_norm = y / 180.0

# 2. Define Model (Inputs = 3)
class IKModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),   # <--- CHANGED FROM 2 TO 3
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)    
        )

    def forward(self, x):
        return self.net(x)

model = IKModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 3. Train
inputs = torch.tensor(X_norm)
targets = torch.tensor(y_norm)

print("Training V2 (With Orientation)...")
for epoch in range(5000):
    optimizer.zero_grad()
    preds = model(inputs)
    loss = criterion(preds, targets)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# 4. Export ONNX
model.eval()
# Dummy input now has 3 values (x, y, angle)
dummy_input = torch.randn(1, 3) 

torch.onnx.export(
    model, dummy_input, "ik_model_v2.onnx", 
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
)
print("Saved 'ik_model_v2.onnx'")