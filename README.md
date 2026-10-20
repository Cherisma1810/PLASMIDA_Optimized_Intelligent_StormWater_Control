# PLASMIDA_Optimized_Intelligent_StormWater_Control

> **A PLASMIDA-Optimized Koopman-Accelerated Deep Reinforcement Learning Framework for Intelligent Urban Stormwater Control**

Welcome to the official codebase for the Velachery Ward 175 Flood Mitigation Project. 

This repository contains the code, models, and simulation environments for a real-time, autonomous urban drainage control system. 

## 🌊 The Problem We Are Solving
Urban flooding is a catastrophic issue. When heavy rain hits, underground manholes fill up and overflow into the streets. We know how to simulate this using physics engines like EPA SWMM, but there is a computational bottleneck: SWMM solves heavy fluid dynamics equations (Saint-Venant equations). It is incredibly accurate but too slow for online AI training. 

If we want to train an AI to operate drainage pumps in real-time to stop flooding, the AI needs to practice hundreds of thousands of times. Waiting for SWMM to calculate physics for every practice run would take weeks. 

## 🧠 Our Solution: The Architecture
To solve the speed bottleneck and train a highly effective AI, this project pieces together three major innovations:

*   **The Digital Twin (Koopman Neural Emulator):** We ran thousands of random pump operations in PySWMM to generate a dataset of cause-and-effect. We then trained a neural network using Koopman Operator theory. This created a hyper-fast emulator that predicts water flow using linear matrix multiplication instead of slow differential equations.
*   **The Architect (PLASMIDA Optimizer):** Before building our AI brain, we needed the perfect settings. We designed **PLASMIDA**, a custom bio-inspired algorithm. By mimicking how bacteria swap DNA (Horizontal Gene Transfer: Conjugation, Transformation, and Transduction), PLASMIDA rapidly discovered the mathematically perfect architecture for our AI.
*   **The Brain (Deep Recurrent Q-Network - DRQN):** Using PLASMIDA's blueprint (128 neurons, 7-step temporal memory), we trained a DRQN agent inside the fast Koopman emulator. Because it has memory, it does not just react to current water levels; it anticipates incoming storm surges and proactively opens/closes 7 drainage gates to flush water downstream.

## 🚀 Key Results
Our simulation proves that this framework successfully manages the Velachery Ward 175 network (135 nodes, 127 conduits):
*   **Surface Flooding Suppressed:** The DRQN successfully suppressed peak water depths to ~1.35m, safely below the 1.8-meter critical structural threshold.
*   **Statistical Superiority:** PLASMIDA decisively outperformed the industry-standard Optuna-TPE optimizer (p < 0.01 via Wilcoxon signed-rank test).
*   **Real-Time Action:** The agent consistently outperforms rigid Rule-Based Heuristic Control (RBHC) and standard memory-less DQN baselines.

## 🛠️ Project Structure
*   `data/`: Contains the geographical blueprints (`ward_175.geojson`) and the generated `koopman_training_data.csv`.
*   `swmm_env/`: The baseline physical environment (`velachery_model.inp`) powered by PySWMM.
*   `models/`: Holds the trained weights for both the Koopman Emulator (`koopman_emulator.pth`) and the final DRQN agent (`trained_drqn.pth`).
*   `dashboard/`: The live Pygame visualization scripts where you can watch the AI control the water levels in real-time.

## 💻 Getting Started
To watch the AI in action and run the Pygame dashboard:
1. Clone this repository.
2. Install the required dependencies: `pip install -r requirements.txt` (requires PyTorch, PySWMM, Pygame, and NumPy).
3. Run the live simulation: `python run_pygame_dashboard.py`
