import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy

# ---------------------------------------------------------
# 1. LOAD THE KOOPMAN EMULATOR SURROGATE
# ---------------------------------------------------------
class KoopmanEmulator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(KoopmanEmulator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class FastKoopmanEnv:
    def __init__(self, model_path='koopman_emulator.pth', num_states=12, num_actions=7):
        self.num_states = num_states
        self.num_actions = num_actions
        self.model = KoopmanEmulator(num_states + num_actions, num_states)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()
        self.state = np.zeros(num_states, dtype=np.float32)

    def reset(self):
        self.state = np.zeros(self.num_states, dtype=np.float32)
        return self.state

    def step(self, action_idx):
        # Base paper uses 128 combinations for 7 pumps[cite: 1]
        binary_action = [float(x) for x in format(action_idx, f'0{self.num_actions}b')]
        
        inp = torch.tensor(np.concatenate([self.state, binary_action]), dtype=torch.float32)
        with torch.no_grad():
            next_state = self.model(inp).numpy()
            
        # Calculate Reward (Flooding + Safety Penalty)[cite: 1]
        flooding = -np.sum(next_state[next_state > 2.0] - 2.0) # Assuming 2.0m is max depth
        penalty = 0.0
        for s in next_state:
            if s > 1.8: # 90% of 2.0m max depth safety constraint[cite: 1]
                penalty += 0.1 * (s - 1.8)
                
        reward = flooding - penalty
        self.state = next_state
        return self.state, reward

# ---------------------------------------------------------
# 2. DEEP RECURRENT Q-NETWORK (DRQN)
# ---------------------------------------------------------
class DRQN(nn.Module):
    def __init__(self, obs_dim, hidden_size, num_actions=128):
        super(DRQN, self).__init__()
        # LSTM layer to capture temporal dependencies (POMDP)[cite: 1]
        self.lstm = nn.LSTM(input_size=obs_dim, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_actions)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        return self.fc(out[:, -1, :]), hidden

def evaluate_drqn_config(config, episodes=3, steps=50):
    """Evaluates a DRQN architecture by running a fast training loop on the Koopman Env."""
    env = FastKoopmanEnv()
    drqn = DRQN(obs_dim=12, hidden_size=int(config['hidden_size']))
    optimizer = optim.Adam(drqn.parameters(), lr=config['lr'])
    criterion = nn.MSELoss()
    
    seq_len = int(config['seq_len'])
    total_reward = 0

    # Extremely abbreviated RL loop for fast PLASMIDA fitness evaluation
    for ep in range(episodes):
        state = env.reset()
        hidden = (torch.zeros(1, 1, drqn.lstm.hidden_size), torch.zeros(1, 1, drqn.lstm.hidden_size))
        state_history = [state] * seq_len
        
        ep_reward = 0
        for step in range(steps):
            state_tensor = torch.tensor(np.array(state_history[-seq_len:]), dtype=torch.float32).unsqueeze(0)
            q_vals, hidden = drqn(state_tensor, hidden)
            
            # Epsilon-greedy action
            if random.random() < 0.2:
                action = random.randint(0, 127)
            else:
                action = torch.argmax(q_vals).item()
                
            next_state, reward = env.step(action)
            ep_reward += reward
            state_history.append(next_state)
            
            # Dummy loss step to simulate learning parameter update
            optimizer.zero_grad()
            loss = criterion(q_vals.squeeze()[action], torch.tensor(reward, dtype=torch.float32))
            loss.backward()
            optimizer.step()

            # Detach hidden states to prevent graph memory leaks
            hidden = (hidden[0].detach(), hidden[1].detach())
            
        total_reward += ep_reward
        
    return total_reward / episodes

# ---------------------------------------------------------
# 3. PLASMIDA OPTIMIZATION ALGORITHM
# ---------------------------------------------------------
def plasmida_search(pop_size=5, generations=3):
    print("Initializing PLASMIDA Adaptive Search for DRQN...")
    
    # Population X: In-place configurations[cite: 3]
    population = []
    for _ in range(pop_size):
        population.append({
            'lr': random.uniform(0.001, 0.015),
            'hidden_size': random.choice([32, 64, 128]),
            'seq_len': random.choice([3, 5, 7]) # Past states to remember[cite: 1]
        })
        
    archive = [] # Elite archive for Transformation[cite: 3]
    
    for gen in range(generations):
        print(f"\n--- Generation {gen+1}/{generations} ---")
        
        # 1. Evaluate Fitness
        fitnesses = []
        for i, p in enumerate(population):
            fit = evaluate_drqn_config(p)
            fitnesses.append(fit)
            print(f"Ind {i} | LR: {p['lr']:.4f} | Hidden: {p['hidden_size']:<3} | SeqLen: {p['seq_len']} | Fitness: {fit:.2f}")
            
        # Update Elite Archive (Transformation Pool)[cite: 3]
        archive = [copy.deepcopy(population[idx]) for idx in np.argsort(fitnesses)[-2:]]
        
        # 2. Fitness-Based Donor Weighting[cite: 3]
        ranks = np.argsort(np.argsort(fitnesses)[::-1]) # 0 = best
        weights = [(pop_size - r) for r in ranks]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        
        # 3. In-Place Genetic Operators (No offspring generation)[cite: 3]
        for i in range(pop_size):
            old_p = copy.deepcopy(population[i])
            
            # Conjugation (Exploitation) - p_c = 0.5[cite: 3]
            if random.random() < 0.5:
                donor_idx = np.random.choice(pop_size, p=probs)
                if donor_idx != i:
                    population[i]['lr'] = population[i]['lr'] + 0.5 * (population[donor_idx]['lr'] - population[i]['lr'])
                    if random.random() < 0.7: # Contiguous segment swap
                        population[i]['hidden_size'] = population[donor_idx]['hidden_size']
                        
            # Transformation (Guided Diversification) - p_t = 0.25[cite: 3]
            if random.random() < 0.25 and archive:
                donor = random.choice(archive)
                population[i]['seq_len'] = donor['seq_len']
                
            # Transduction (Unbiased Exploration) - p_s = 0.15[cite: 3]
            if random.random() < 0.15:
                donor = random.choice(population)
                population[i]['hidden_size'] = donor['hidden_size']
                
            # Greedy Acceptance[cite: 3]
            new_fit = evaluate_drqn_config(population[i])
            if new_fit < fitnesses[i]: # If new is worse, revert
                population[i] = old_p

    print("\n[PLASMIDA Search Complete]")
    best_idx = np.argmax(fitnesses)
    print("Best DRQN Hyperparameters Found:")
    print(population[best_idx])

if __name__ == "__main__":
    plasmida_search()