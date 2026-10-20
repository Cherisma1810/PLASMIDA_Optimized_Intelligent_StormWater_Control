import torch
import numpy as np
import matplotlib.pyplot as plt
from plasmida_drqn_optimizer import FastKoopmanEnv, DRQN

# 1. Initialize the Environment and Optimal Hyperparameters
env = FastKoopmanEnv()
best_lr = 0.0114
best_hidden = 128
best_seq_len = 7
eval_steps = 100

print("Running Comparative Benchmark: DRQN vs DQN vs RBHC...")

# 2. Define standard DQN (Memory-less, Seq_len = 1)[cite: 1]
dqn = DRQN(obs_dim=12, hidden_size=64) # Smaller hidden size typical for standard DQN
dqn_hidden = (torch.zeros(1, 1, 64), torch.zeros(1, 1, 64))

# 3. Define optimized DRQN
drqn = DRQN(obs_dim=12, hidden_size=best_hidden)
drqn_hidden = (torch.zeros(1, 1, best_hidden), torch.zeros(1, 1, best_hidden))
drqn.load_state_dict(torch.load('trained_drqn.pth', weights_only=True))

# Tracking arrays for the plot
drqn_flooding = []
dqn_flooding = []
rbhc_flooding = []

def calculate_flood_volume(state):
    # Sum of water levels exceeding the 2.0m max depth threshold[cite: 1]
    return np.sum(state[state > 2.0] - 2.0)

# --- RUN DRQN ---
state = env.reset()
state_history = [state] * best_seq_len
current_flood = 0
for step in range(eval_steps):
    state_tensor = torch.tensor(np.array(state_history[-best_seq_len:]), dtype=torch.float32).unsqueeze(0)
    q_vals, drqn_hidden = drqn(state_tensor, drqn_hidden)
    action = torch.argmax(q_vals).item()
    next_state, _ = env.step(action)
    
    current_flood += calculate_flood_volume(next_state)
    drqn_flooding.append(current_flood)
    state_history.append(next_state)

# --- RUN DQN ---
state = env.reset()
current_flood = 0
for step in range(eval_steps):
    # DQN only looks at the immediate current state[cite: 1]
    state_tensor = torch.tensor(np.array([state]), dtype=torch.float32).unsqueeze(0)
    q_vals, dqn_hidden = dqn(state_tensor, dqn_hidden)
    action = torch.argmax(q_vals).item()
    next_state, _ = env.step(action)
    
    current_flood += calculate_flood_volume(next_state)
    dqn_flooding.append(current_flood)
    state = next_state

# --- RUN RBHC (Rule-Based Heuristic Control) ---
state = env.reset()
current_flood = 0
for step in range(eval_steps):
    # Simple heuristic: If average water level > 1.5m, turn all pumps ON (Action 127)[cite: 1]
    # Otherwise, turn them OFF (Action 0)
    avg_depth = np.mean(state)
    action = 127 if avg_depth > 1.5 else 0
    next_state, _ = env.step(action)
    
    current_flood += calculate_flood_volume(next_state)
    rbhc_flooding.append(current_flood)
    state = next_state

# 4. Generate the Comparison Plot
plt.figure(figsize=(10, 6))
plt.plot(drqn_flooding, label=f'Optimized DRQN (Seq={best_seq_len})', color='green', linewidth=2)
plt.plot(dqn_flooding, label='Standard DQN', color='orange', linestyle='--')
plt.plot(rbhc_flooding, label='Rule-Based Heuristic (RBHC)', color='gray', linestyle=':')

plt.title('Cumulative Flooding Volume During Storm Event')
plt.xlabel('Simulation Time Steps')
plt.ylabel('Cumulative Flooding Volume (Normalized)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('comparative_benchmark.png', dpi=300)
print("Benchmark complete! Graph saved as 'comparative_benchmark.png'.")