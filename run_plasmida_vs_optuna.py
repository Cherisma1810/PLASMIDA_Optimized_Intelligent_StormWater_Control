import numpy as np
import matplotlib.pyplot as plt
import optuna
import copy
import random
from scipy.stats import wilcoxon

# Suppress Optuna's verbose logging for clean console output
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------
# 1. DEFINE BENCHMARK FUNCTION (Rastrigin Function)
# ---------------------------------------------------------
def rastrigin(x):
    """Classic multimodal optimization test function (Global minimum = 0 at x = 0)"""
    x = np.asarray(x)
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

DIM = 8
BOUNDS = [(-5.12, 5.12)] * DIM
BUDGET_NFE = 800  # Strict evaluation budget per run

# ---------------------------------------------------------
# 2. PLASMIDA OPTIMIZATION IMPLEMENTATION
# ---------------------------------------------------------
class PlasmidaOptimizer:
    def __init__(self, func, bounds, pop_size=25, nfe_budget=800):
        self.func = func
        self.bounds = bounds
        self.dim = len(bounds)
        self.pop_size = pop_size
        self.nfe_budget = nfe_budget

    def optimize(self):
        nfe = 0
        # Initialize population uniformly within bounds
        population = [
            [random.uniform(b[0], b[1]) for b in self.bounds] 
            for _ in range(self.pop_size)
        ]
        fitness = [self.func(ind) for ind in population]
        nfe += self.pop_size

        best_history = []
        current_best = min(fitness)
        best_history.append(current_best)

        archive = [] # Elite archive (Transformation pool)

        while nfe < self.nfe_budget:
            # Update archive with top 20% elite individuals
            sorted_indices = np.argsort(fitness)
            archive = [copy.deepcopy(population[idx]) for idx in sorted_indices[:max(1, self.pop_size // 5)]]

            # Fitness-based donor weighting for conjugation
            ranks = np.argsort(sorted_indices)
            weights = [(self.pop_size - r) for r in ranks]
            total_w = sum(weights)
            probs = [w / total_w for w in weights] if total_w > 0 else [1.0/self.pop_size]*self.pop_size

            for i in range(self.pop_size):
                if nfe >= self.nfe_budget:
                    break

                old_ind = copy.deepcopy(population[i])
                old_fit = fitness[i]

                # 1. Conjugation (Exploitation: guided blending with a fitter donor)
                if random.random() < 0.5:
                    donor_idx = np.random.choice(self.pop_size, p=probs)
                    if donor_idx != i:
                        kappa = random.uniform(0.3, 1.0)
                        for j in range(self.dim):
                            population[i][j] += kappa * (population[donor_idx][j] - population[i][j])

                # 2. Transformation (Guided diversification via archive eDNA pool)
                if random.random() < 0.25 and archive:
                    donor = random.choice(archive)
                    j = random.randint(0, self.dim - 1)
                    population[i][j] = donor[j]

                # 3. Transduction (Unbiased exploration: random noisy transfer)
                if random.random() < 0.15:
                    donor = random.choice(population)
                    j = random.randint(0, self.dim - 1)
                    span = self.bounds[j][1] - self.bounds[j][0]
                    population[i][j] = donor[j] + random.gauss(0, 0.1 * span)

                # Clip to box constraints
                for j in range(self.dim):
                    population[i][j] = max(self.bounds[j][0], min(self.bounds[j][1], population[i][j]))

                # Evaluation and Greedy Selection
                new_fit = self.func(population[i])
                nfe += 1

                if new_fit < old_fit:
                    fitness[i] = new_fit
                else:
                    population[i] = old_ind # Revert

                current_best = min(current_best, fitness[i])
                best_history.append(current_best)

        # Pad history if budget terminated early mid-generation
        while len(best_history) < self.nfe_budget:
            best_history.append(best_history[-1])

        return best_history[:self.nfe_budget]

# ---------------------------------------------------------
# 3. OPTUNA OPTIMIZATION WRAPPER
# ---------------------------------------------------------
def run_optuna(func, bounds, nfe_budget=800):
    def objective(trial):
        x = [trial.suggest_float(f'x_{i}', b[0], b[1]) for i, b in enumerate(bounds)]
        return func(x)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    
    # Track best value trajectory per NFE
    history = []
    best_so_far = float('inf')

    # Custom callback to capture evaluation trajectory step-by-step
    def callback(study, trial):
        nonlocal best_so_far
        if trial.value < best_so_far:
            best_so_far = trial.value
        history.append(best_so_far)

    study.optimize(objective, n_trials=nfe_budget, callbacks=[callback])
    
    # Fill in missing steps if trial evaluations truncated early
    while len(history) < nfe_budget:
        history.append(history[-1] if history else float('inf'))
        
    return history[:nfe_budget]

# ---------------------------------------------------------
# 4. EXECUTE COMPARATIVE EXPERIMENT (Multiple Runs)
# ---------------------------------------------------------
NUM_RUNS = 12
print(f"Running comparative benchmark: PLASMIDA vs Optuna-TPE over {NUM_RUNS} independent runs...")

plasmida_runs = []
optuna_runs = []

for r in range(NUM_RUNS):
    random.seed(r)
    np.random.seed(r)
    
    # Run PLASMIDA
    p_opt = PlasmidaOptimizer(rastrigin, BOUNDS, pop_size=25, nfe_budget=BUDGET_NFE)
    plasmida_runs.append(p_opt.optimize())

    # Run Optuna
    optuna_runs.append(run_optuna(rastrigin, BOUNDS, nfe_budget=BUDGET_NFE))

plasmida_runs = np.array(plasmida_runs)
optuna_runs = np.array(optuna_runs)

# Compute final performance statistics
p_final = plasmida_runs[:, -1]
o_final = optuna_runs[:, -1]

print("\n--- RESULTS SUMMARY ---")
print(f"PLASMIDA Final Fitness: Mean = {np.mean(p_final):.4f} ± Std = {np.std(p_final):.4f}")
print(f"Optuna-TPE Final Fitness: Mean = {np.mean(o_final):.4f} ± Std = {np.std(o_final):.4f}")

# Statistical Significance Testing (Wilcoxon Signed-Rank Test)
stat, p_value = wilcoxon(p_final, o_final)
print(f"Wilcoxon Signed-Rank Test p-value: {p_value:.5e}")
if p_value < 0.05:
    print("Conclusion: The performance difference is statistically significant (p < 0.05).")
else:
    print("Conclusion: No statistically significant difference detected.")

# ---------------------------------------------------------
# 5. GENERATE PUBLICATION-READY GRAPHS
# ---------------------------------------------------------
nfe_axis = np.arange(BUDGET_NFE)

p_mean = np.mean(plasmida_runs, axis=0)
p_std = np.std(plasmida_runs, axis=0)
o_mean = np.mean(optuna_runs, axis=0)
o_std = np.std(optuna_runs, axis=0)

# Figure 1: Convergence Trajectory (Line plot with standard deviation bands)
plt.figure(figsize=(9, 6))
plt.plot(nfe_axis, p_mean, label='PLASMIDA (Proposed)', color='#009E73', linewidth=2)
plt.fill_between(nfe_axis, p_mean - p_std, p_mean + p_std, color='#009E73', alpha=0.2)

plt.plot(nfe_axis, o_mean, label='Optuna-TPE', color='#D55E00', linewidth=2, linestyle='--')
plt.fill_between(nfe_axis, o_mean - o_std, o_mean + o_std, color='#D55E00', alpha=0.2)

plt.yscale('log')
plt.xlabel('Number of Function Evaluations (NFE)', fontsize=12)
plt.ylabel('Best Fitness Found (Log Scale, Lower is Better)', fontsize=12)
plt.title('Convergence Trajectory Comparison on Rastrigin Function (D=8)', fontsize=13, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig('convergence_trajectory.png', dpi=300)
print("Saved graph: convergence_trajectory.png")

# Figure 2: Final Performance Distribution (Boxplot)
plt.figure(figsize=(7, 6))
plt.boxplot([p_final, o_final], labels=['PLASMIDA', 'Optuna-TPE'], patch_artist=True,
            boxprops=dict(facecolor='#56B4E9', color='black'),
            medianprops=dict(color='red', linewidth=2))
plt.yscale('log')
plt.ylabel('Final Objective Value (Log Scale)', fontsize=12)
plt.title('Final Performance Distribution across 12 Runs', fontsize=13, fontweight='bold')
plt.grid(True, axis='y', ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig('optuna_vs_plasmida_comparison.png', dpi=300)
print("Saved graph: optuna_vs_plasmida_comparison.png")