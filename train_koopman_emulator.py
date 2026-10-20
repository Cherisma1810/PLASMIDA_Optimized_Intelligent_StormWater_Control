import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# 1. Load the generated PySWMM dataset
df = pd.read_csv('koopman_training_data.csv')

state_cols = [c for c in df.columns if c.startswith('state_')]
action_cols = [c for c in df.columns if c.startswith('action_')]
next_state_cols = [c for c in df.columns if c.startswith('next_state_')]

# Combine state and action as the input; next_state is the target
X = df[state_cols + action_cols].values
Y = df[next_state_cols].values

X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

dataset = TensorDataset(X_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. Define the Koopman-inspired Surrogate Neural Network
class KoopmanEmulator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(KoopmanEmulator, self).__init__()
        # A lightweight Multi-Layer Perceptron (MLP) for high-speed inference
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

input_dim = len(state_cols) + len(action_cols)
output_dim = len(next_state_cols)

model = KoopmanEmulator(input_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 3. Train the Emulator
print("Training Koopman Emulator Surrogate Model...")
epochs = 100

for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_X, batch_Y in dataloader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_Y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:<3}/{epochs} | Mean Squared Error Loss: {epoch_loss/len(dataloader):.6f}")

# 4. Save the trained weights for the DRQN to use
torch.save(model.state_dict(), 'koopman_emulator.pth')
print("\nTraining complete! Surrogate model saved as 'koopman_emulator.pth'.")