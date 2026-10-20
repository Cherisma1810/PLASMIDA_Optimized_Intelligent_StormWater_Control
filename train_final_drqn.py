import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from plasmida_drqn_optimizer import FastKoopmanEnv, DRQN

env = FastKoopmanEnv()

# Optimal Hyperparameters found by PLASMIDA
best_lr = 0.0114
best_hidden = 128
best_seq_len = 7

print("Training Final DRQN Agent using PLASMIDA-optimized hyperparameters...")
drqn = DRQN(obs_dim=12, hidden_size=best_hidden)
optimizer = optim.Adam(drqn.parameters(), lr=best_lr)
criterion = nn.MSELoss()

epochs = 200
steps = 100
epsilon = 1.0
epsilon_decay = 0.96
epsilon_min = 0.05
gamma = 0.90 # Discount factor for future rewards

for ep in range(epochs):
    state = env.reset()
    # Initialize LSTM hidden states
    hidden = (torch.zeros(1, 1, best_hidden), torch.zeros(1, 1, best_hidden))
    state_history = [state] * best_seq_len
    total_reward = 0

    for step in range(steps):
        state_tensor = torch.tensor(np.array(state_history[-best_seq_len:]), dtype=torch.float32).unsqueeze(0)
        q_vals, hidden = drqn(state_tensor, hidden)

        # Epsilon-Greedy Exploration
        if random.random() < epsilon:
            action = random.randint(0, 127)
        else:
            action = torch.argmax(q_vals).item()

        next_state, reward = env.step(action)
        total_reward += reward

        # Standard Q-learning Bellman Update
        next_state_history = state_history[1:] + [next_state]
        next_state_tensor = torch.tensor(np.array(next_state_history[-best_seq_len:]), dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            next_q_vals, _ = drqn(next_state_tensor, hidden)
            max_next_q = torch.max(next_q_vals)

        target_q = reward + gamma * max_next_q

        # Update only the Q-value for the action taken
        target_q_tensor = q_vals.clone().detach()
        target_q_tensor[0][action] = target_q

        optimizer.zero_grad()
        loss = criterion(q_vals, target_q_tensor)
        loss.backward()
        optimizer.step()

        # Detach hidden state to prevent backpropagation graph errors
        hidden = (hidden[0].detach(), hidden[1].detach())
        state_history.append(next_state)

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    if (ep + 1) % 20 == 0:
        print(f"Episode {ep+1:<3}/{epochs} | Total Reward (Flooding Penalty): {total_reward:.2f} | Epsilon: {epsilon:.2f}")

# Save the intelligent weights
torch.save(drqn.state_dict(), 'trained_drqn.pth')
print("\nTraining complete! Agent's brain saved to 'trained_drqn.pth'")