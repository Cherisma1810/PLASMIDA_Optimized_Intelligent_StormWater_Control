import numpy as np
import pandas as pd
from velachery_env import VelacheryStormwaterEnv
import os

def generate_dataset(inp_file, num_episodes=5, steps_per_episode=100):
    print("Generating Koopman Emulator training data from PySWMM...")
    
    env = VelacheryStormwaterEnv(inp_file)
    dataset = []
    
    try:
        # PySWMM delays parsing until reset() is called. 
        # We MUST catch the error here to see the actual SWMM engine failure.
        state, _ = env.reset()
    except Exception as e:
        print(f"\n[PySWMM CRASHED DURING RESET] {e}")
        rpt_file = inp_file.replace('.inp', '.rpt')
        if os.path.exists(rpt_file):
            print("\n" + "!"*60)
            print("🔍 EXACT SWMM ENGINE ERRORS FOUND IN REPORT FILE:")
            print("!"*60)
            with open(rpt_file, 'r') as rpt:
                for line in rpt:
                    if "ERROR" in line.upper():
                        print(line.strip())
            print("!"*60 + "\n")
        else:
            print(f"Could not find the SWMM report file: {rpt_file}")
        return

    for episode in range(num_episodes):
        try:
            if episode > 0:
                state, _ = env.reset()
        except Exception as e:
            print(f"Error resetting environment at episode {episode}: {e}")
            break
            
        for step in range(steps_per_episode):
            # Base paper random exploratory actions[cite: 1]
            action = env.action_space.sample() 
            next_state, reward, done, _, _ = env.step(action)
            
            record = {}
            for i, s in enumerate(state):
                record[f'state_{i}'] = s
            for i, a in enumerate(action):
                record[f'action_{i}'] = a
            record['reward'] = reward
            for i, ns in enumerate(next_state):
                record[f'next_state_{i}'] = ns
                
            dataset.append(record)
            state = next_state
            
            if done:
                break
                
        print(f"Episode {episode + 1}/{num_episodes} completed.")
        
    env.close()
    
    if dataset:
        df = pd.DataFrame(dataset)
        csv_filename = "koopman_training_data.csv"
        df.to_csv(csv_filename, index=False)
        print(f"\nDataset generation complete! Saved {len(df)} transition steps to '{csv_filename}'.")

if __name__ == "__main__":
    generate_dataset('velachery_model.inp')