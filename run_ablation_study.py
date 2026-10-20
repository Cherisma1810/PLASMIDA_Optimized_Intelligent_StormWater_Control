import numpy as np
import random
import copy
import itertools
import matplotlib.pyplot as plt
from plasmida_drqn_optimizer import evaluate_drqn_config

def plasmida_search_ablated(use_conj, use_transf, use_transd, pop_size=5, generations=3):
    population = []
    for _ in range(pop_size):
        population.append({
            'lr': random.uniform(0.001, 0.015),
            'hidden_size': random.choice([32, 64, 128]),
            'seq_len': random.choice([3, 5, 7])
        })
        
    archive = []
    
    for gen in range(generations):
        fitnesses = [evaluate_drqn_config(p) for p in population]
        archive = [copy.deepcopy(population[idx]) for idx in np.argsort(fitnesses)[-2:]]
        
        ranks = np.argsort(np.argsort(fitnesses)[::-1])
        weights = [(pop_size - r) for r in ranks]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        
        for i in range(pop_size):
            old_p = copy.deepcopy(population[i])
            
            # 1. Conjugation
            if use_conj and random.random() < 0.5:
                donor_idx = np.random.choice(pop_size, p=probs)
                if donor_idx != i:
                    population[i]['lr'] = population[i]['lr'] + 0.5 * (population[donor_idx]['lr'] - population[i]['lr'])
                    if random.random() < 0.7:
                        population[i]['hidden_size'] = population[donor_idx]['hidden_size']
                        
            # 2. Transformation
            if use_transf and random.random() < 0.25 and archive:
                donor = random.choice(archive)
                population[i]['seq_len'] = donor['seq_len']
                
            # 3. Transduction[cite: 3]
            if use_transd and random.random() < 0.15:
                donor = random.choice(population)
                population[i]['hidden_size'] = donor['hidden_size']
                
            # Greedy Acceptance (Mutation)
            new_fit = evaluate_drqn_config(population[i])
            if new_fit < fitnesses[i]:
                population[i] = old_p
                
    return np.max(fitnesses) # Return best fitness found (closer to 0 is better)

print("Starting PLASMIDA Operator Ablation Study...")
print("Testing all 8 combinations of Conjugation (C), Transformation (T), and Transduction (X)...\n")

results = {}
# Generate all 8 combinations of True/False for the 3 operators[cite: 3]
combinations = list(itertools.product([False, True], repeat=3))

for conj, transf, transd in combinations:
    # Run the ablated search (averaging over 3 runs for stability)
    scores = [plasmida_search_ablated(conj, transf, transd) for _ in range(3)]
    avg_best_fitness = np.mean(scores)
    
    # Create a readable label
    active = []
    if conj: active.append("Conj")
    if transf: active.append("Transf")
    if transd: active.append("Transd")
    label = " + ".join(active) if active else "None (Mutation Only)"
    
    results[label] = avg_best_fitness
    print(f"Config: [{label:<30}] | Avg Best Fitness: {avg_best_fitness:.2f}")

# Sort results by performance (closest to 0 is best)
sorted_results = dict(sorted(results.items(), key=lambda item: item[1]))

# Generate Ablation Bar Chart
labels = list(sorted_results.keys())
scores = list(sorted_results.values())

plt.figure(figsize=(10, 6))
# Using negative values for the bar chart so 'better' points upward
bars = plt.barh(labels, [-s for s in scores], color='#2c7bb6')
plt.xlabel('Negative Flooding Penalty (Higher is Better)')
plt.title('PLASMIDA Ablation: Contribution of each HGT Operator')
plt.gca().invert_yaxis() # Highest performing at the top
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()

plt.savefig('ablation_study.png', dpi=300)
print("\nAblation study complete! Graph saved as 'ablation_study.png'.")