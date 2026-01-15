import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.onnx

# 1. Load Data
try:
    data = pd.read_csv("ik_data.csv")
except FileNotFoundError:
    print("Error: ik_data.csv not found. Run the Data Collector first!")
    exit()

X = data[['pixel_x', 'pixel_y']].values.astype(np.float32)
y = data[['j11', 'j12', 'j14', 'j15', 'j16']].values.astype(np.float32)

# Normalize
X_norm = X / 640.0 
y_norm = y / 180.0

# 2. Define Model
class IKModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),   # Input: x, y
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)    # Output: 5 Joints
        )

    def forward(self, x):
        return self.net(x)

model = IKModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 3. Train
inputs = torch.tensor(X_norm)
targets = torch.tensor(y_norm)

print("Training...")
for epoch in range(5000):
    optimizer.zero_grad()
    preds = model(inputs)
    loss = criterion(preds, targets)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# 4. EXPORT TO ONNX
model.eval()
dummy_input = torch.randn(1, 2) # Example input (Batch size 1, 2 features)

torch.onnx.export(
    model, 
    dummy_input, 
    "ik_model.onnx", 
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names = ['input'],   # We will use this name in the App
    output_names = ['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("Success! Saved 'ik_model.onnx'")