import os
import random
import numpy as np
import matplotlib.pyplot as plt
import optuna
from scipy.stats import wilcoxon

# Suppress Optuna verbose output
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Output directory setup
plasmida_dir = os.path.join("paper_graphs", "plasmida_vs_optuna")
os.makedirs(plasmida_dir, exist_ok=True)

# ---------------------------------------------------------
# 1. BENCHMARK DEFINITION (Rastrigin Function)
# ---------------------------------------------------------
def rastrigin(x):
    x = np.asarray(x)
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

DIM = 6              # Dimensionality 
BOUNDS = [(-5.12, 5.12)] * DIM
BUDGET_NFE = 600     # Strict evaluation budget
NUM_RUNS = 10        # Independent runs for statistical validity

print(f"🔬 Running TRUE Empirical Benchmark over {NUM_RUNS} independent runs (NFE={BUDGET_NFE})...")

# ---------------------------------------------------------
# 2. TRUE EMPIRICAL PLASMIDA OPTIMIZER
# ---------------------------------------------------------
class TruePlasmidaOptimizer:
    def __init__(self, func, bounds, pop_size=20, nfe_budget=600):
        self.func = func
        self.bounds = bounds
        self.dim = len(bounds)
        self.pop_size = pop_size
        self.nfe_budget = nfe_budget

    def optimize(self):
        nfe = 0
        population = [[random.uniform(b[0], b[1]) for b in self.bounds] for _ in range(self.pop_size)]
        fitness = [self.func(ind) for ind in population]
        nfe += self.pop_size

        history = []
        current_best = min(fitness)
        history.append(current_best)

        archive = []

        while nfe < self.nfe_budget:
            sorted_indices = np.argsort(fitness)
            archive = [list(population[idx]) for idx in sorted_indices[:max(1, self.pop_size // 5)]]

            ranks = np.argsort(sorted_indices)
            weights = [(self.pop_size - r) for r in ranks]
            total_w = sum(weights)
            probs = [w / total_w for w in weights] if total_w > 0 else [1.0/self.pop_size]*self.pop_size

            for i in range(self.pop_size):
                if nfe >= self.nfe_budget:
                    break

                old_ind = list(population[i])
                old_fit = fitness[i]

                # Conjugation
                if random.random() < 0.5:
                    donor_idx = np.random.choice(self.pop_size, p=probs)
                    if donor_idx != i:
                        kappa = random.uniform(0.2, 0.9)
                        for j in range(self.dim):
                            population[i][j] += kappa * (population[donor_idx][j] - population[i][j])

                # Transformation
                if random.random() < 0.3 and archive:
                    donor = random.choice(archive)
                    j = random.randint(0, self.dim - 1)
                    population[i][j] = donor[j]

                # Transduction
                if random.random() < 0.2:
                    donor = random.choice(population)
                    j = random.randint(0, self.dim - 1)
                    span = self.bounds[j][1] - self.bounds[j][0]
                    population[i][j] = donor[j] + random.gauss(0, 0.05 * span)

                # Boundary clipping
                for j in range(self.dim):
                    population[i][j] = max(self.bounds[j][0], min(self.bounds[j][1], population[i][j]))

                new_fit = self.func(population[i])
                nfe += 1

                if new_fit < old_fit:
                    fitness[i] = new_fit
                else:
                    population[i] = old_ind

                if fitness[i] < current_best:
                    current_best = fitness[i]
                
                history.append(current_best)

        while len(history) < self.nfe_budget:
            history.append(history[-1])

        return history[:self.nfe_budget]

# ---------------------------------------------------------
# 3. TRUE EMPIRICAL OPTUNA EVALUATION
# ---------------------------------------------------------
def run_true_optuna(func, bounds, nfe_budget=600):
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
    
    history = []
    best_so_far = float('inf')

    def objective(trial):
        nonlocal best_so_far
        x = [trial.suggest_float(f'x_{i}', b[0], b[1]) for i, b in enumerate(bounds)]
        val = func(x)
        if val < best_so_far:
            best_so_far = val
        history.append(best_so_far)
        return val

    study.optimize(objective, n_trials=nfe_budget)

    while len(history) < nfe_budget:
        history.append(history[-1] if history else float('inf'))

    return history[:nfe_budget]

# ---------------------------------------------------------
# 4. EXECUTE BATCH EXPERIMENTS
# ---------------------------------------------------------
plasmida_all_runs = []
optuna_all_runs = []

for r in range(NUM_RUNS):
    random.seed(100 + r)
    np.random.seed(100 + r)
    
    # Plasmida Run
    p_opt = TruePlasmidaOptimizer(rastrigin, BOUNDS, pop_size=20, nfe_budget=BUDGET_NFE)
    plasmida_all_runs.append(p_opt.optimize())

    # Optuna Run
    optuna_all_runs.append(run_true_optuna(rastrigin, BOUNDS, nfe_budget=BUDGET_NFE))

p_mat = np.array(plasmida_all_runs)
o_mat = np.array(optuna_all_runs)

p_final = p_mat[:, -1]
o_final = o_mat[:, -1]

print("\n--- TRUE EMPIRICAL RESULTS SUMMARY ---")
print(f"PLASMIDA Final Fitness: Mean = {np.mean(p_final):.4f} ± Std = {np.std(p_final):.4f}")
print(f"Optuna-TPE Final Fitness: Mean = {np.mean(o_final):.4f} ± Std = {np.std(o_final):.4f}")

# Wilcoxon test
stat, p_value = wilcoxon(p_final, o_final)
print(f"Wilcoxon Signed-Rank Test p-value: {p_value:.5e}")

# ---------------------------------------------------------
# 5. PLOT GENUINE DATA GRAPHS
# ---------------------------------------------------------
nfe_axis = np.arange(BUDGET_NFE)

p_mean = np.mean(p_mat, axis=0)
p_std = np.std(p_mat, axis=0)
o_mean = np.mean(o_mat, axis=0)
o_std = np.std(o_mat, axis=0)

# Convergence Graph
plt.figure(figsize=(9, 6))
plt.plot(nfe_axis, p_mean, label='PLASMIDA (Empirical Mean)', color='#009E73', linewidth=2)
plt.fill_between(nfe_axis, p_mean - p_std, p_mean + p_std, color='#009E73', alpha=0.15)

plt.plot(nfe_axis, o_mean, label='Optuna-TPE (Empirical Mean)', color='#D55E00', linewidth=2, linestyle='--')
plt.fill_between(nfe_axis, o_mean - o_std, o_mean + o_std, color='#D55E00', alpha=0.15)

plt.yscale('log')
plt.xlabel('Number of Function Evaluations (NFE)', fontsize=12)
plt.ylabel('Objective Fitness (Log Scale, Lower is Better)', fontsize=12)
plt.title('True Empirical Convergence Comparison (Rastrigin, D=6)', fontsize=13, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.tight_layout()
conv_path = os.path.join(plasmida_dir, "convergence_trajectory_empirical.png")
plt.savefig(conv_path, dpi=300)
plt.close()

# Boxplot Graph
plt.figure(figsize=(7, 6))
plt.boxplot([p_final, o_final], tick_labels=['PLASMIDA', 'Optuna-TPE'], patch_artist=True,
            boxprops=dict(facecolor='#56B4E9', color='black'),
            medianprops=dict(color='red', linewidth=2))
plt.yscale('log')
plt.ylabel('Final Objective Value (Log Scale)', fontsize=12)
plt.title(f'Empirical Final Performance Distribution (p={p_value:.3e})', fontsize=13, fontweight='bold')
plt.grid(True, axis='y', ls="--", alpha=0.4)
plt.tight_layout()
box_path = os.path.join(plasmida_dir, "final_performance_boxplot_empirical.png")
plt.savefig(box_path, dpi=300)
plt.close()

print(f"\n✅ Genuine empirical graphs saved successfully to '{plasmida_dir}/'!")