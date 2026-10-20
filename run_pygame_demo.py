import pygame
import sys
import numpy as np
import torch
import time
from plasmida_drqn_optimizer import FastKoopmanEnv, DRQN

# --- 1. INITIALIZE PYGAME & SETTINGS ---
pygame.init()
WIDTH, HEIGHT = 1300, 750
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Velachery Ward 175 - Live Drainage Network Workflow")

# Colors
WHITE = (245, 245, 250)
BLACK = (40, 40, 40)
BLUE = (50, 150, 240)
DARK_BLUE = (20, 80, 180)
RED = (220, 50, 50)
GREEN = (40, 200, 80)
GRAY = (180, 180, 180)
DARK_GRAY = (100, 100, 100)

font = pygame.font.SysFont("Segoe UI", 18, bold=True)
title_font = pygame.font.SysFont("Segoe UI", 32, bold=True)
small_font = pygame.font.SysFont("Segoe UI", 14)

# --- 2. LOAD ENVIRONMENT & TRAINED AGENT ---
env = FastKoopmanEnv()
best_hidden = 128
best_seq_len = 7
obs_dim = 12

print("Loading trained DRQN agent...")
drqn = DRQN(obs_dim=obs_dim, hidden_size=best_hidden)
try:
    drqn.load_state_dict(torch.load('trained_drqn.pth', weights_only=True))
except FileNotFoundError:
    print("Warning: 'trained_drqn.pth' not found. Agent will run with untrained weights.")

state = env.reset()
state_history = [state] * best_seq_len
hidden = (torch.zeros(1, 1, best_hidden), torch.zeros(1, 1, best_hidden))
cumulative_flood = 0
step = 0

running = True
clock = pygame.time.Clock()

# --- 3. NETWORK TOPOLOGY SETUP ---
# Map the 12 nodes to (x, y) screen coordinates to form a realistic branching network
nodes = {
    0: (100, 150),   # N1 (Upstream)
    1: (100, 350),   # N2 (Upstream)
    2: (100, 550),   # N3 (Upstream)
    3: (350, 250),   # N4 (Midstream)
    4: (350, 550),   # N5 (Midstream)
    5: (550, 400),   # N6 (Convergence)
    6: (750, 400),   # N7 (Main Trunk)
    7: (950, 250),   # N8 (Branch North)
    8: (950, 550),   # N9 (Branch South)
    9: (1150, 150),  # N10 (Downstream North)
    10: (1150, 650), # N11 (Downstream South)
    11: (1150, 400)  # N12 (Main Outfall)
}

# Define pipes: (Source Node, Target Node, Pump Index (if any))
# Pump indices 0-6 correspond to the 7 binary actions
pipes = [
    (0, 3, 0),    # N1 to N4 controlled by Pump 1
    (1, 3, 1),    # N2 to N4 controlled by Pump 2
    (2, 4, 2),    # N3 to N5 controlled by Pump 3
    (3, 5, None), # N4 to N6 (Gravity flow, no pump)
    (4, 5, None), # N5 to N6 (Gravity flow, no pump)
    (5, 6, 3),    # N6 to N7 controlled by Pump 4
    (6, 7, None), # N7 to N8 (Gravity flow)
    (6, 8, None), # N7 to N9 (Gravity flow)
    (7, 9, 4),    # N8 to N10 controlled by Pump 5
    (8, 10, 5),   # N9 to N11 controlled by Pump 6
    (7, 11, None),# N8 to N12 (Gravity flow)
    (9, 11, 6)    # N10 to N12 controlled by Pump 7
]

# --- 4. MAIN ANIMATION LOOP ---
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)

    # --- AGENT DECISION ---
    state_tensor = torch.tensor(np.array(state_history[-best_seq_len:]), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_vals, hidden = drqn(state_tensor, hidden)
        action = torch.argmax(q_vals).item()

    # Step the environment
    next_state, reward = env.step(action)
    flood_vol = np.sum(next_state[next_state > 2.0] - 2.0)
    cumulative_flood += max(0, flood_vol)
    
    state_history.append(next_state)
    state = next_state
    step += 1

    # Extract the 7 pump states from the binary action integer
    pump_states = [(action >> i) & 1 for i in range(7)]

    # --- DRAW DASHBOARD ---
    title = title_font.render(f"Velachery Ward 175: Real-Time DRQN Drainage Control", True, BLACK)
    screen.blit(title, (30, 20))
    
    status_color = RED if cumulative_flood > 0 else DARK_BLUE
    stats = font.render(f"Storm Step: {step}/100   |   Total Flooding Volume: {cumulative_flood:.2f} m³", True, status_color)
    screen.blit(stats, (30, 65))

    # Legend
    pygame.draw.circle(screen, GREEN, (900, 45), 10)
    screen.blit(font.render("Gate OPEN (Flowing)", True, BLACK), (920, 35))
    pygame.draw.circle(screen, RED, (1120, 45), 10)
    screen.blit(font.render("Gate CLOSED", True, BLACK), (1140, 35))

    # --- DRAW PIPES & WATERFLOW ---
    time_offset = (pygame.time.get_ticks() / 500.0) % 1.0  # Used to animate water drops

    for src, tgt, pump_idx in pipes:
        x1, y1 = nodes[src][0] + 40, nodes[src][1] + 50 # Start from right side of tank
        x2, y2 = nodes[tgt][0], nodes[tgt][1] + 50      # End at left side of target tank
        
        # Draw physical pipe (gray background)
        pygame.draw.line(screen, GRAY, (x1, y1), (x2, y2), 8)

        # Check if water is flowing (upstream has water AND pump is open/none)
        is_flowing = False
        if state[src] > 0.1:
            if pump_idx is None or pump_states[pump_idx] == 1:
                is_flowing = True

        # Animate water flowing through the pipe
        if is_flowing:
            pygame.draw.line(screen, BLUE, (x1, y1), (x2, y2), 4)
            # Draw moving water droplets
            drop_x = int(x1 + (x2 - x1) * time_offset)
            drop_y = int(y1 + (y2 - y1) * time_offset)
            pygame.draw.circle(screen, DARK_BLUE, (drop_x, drop_y), 4)

        # Draw the Pump/Gate on the pipe
        if pump_idx is not None:
            mid_x, mid_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            gate_color = GREEN if pump_states[pump_idx] == 1 else RED
            pygame.draw.circle(screen, gate_color, (mid_x, mid_y), 15)
            pygame.draw.circle(screen, BLACK, (mid_x, mid_y), 15, 2)
            # Pump Label
            p_lbl = small_font.render(f"P{pump_idx+1}", True, BLACK)
            screen.blit(p_lbl, (mid_x - 8, mid_y - 32))

    # --- DRAW MANHOLE TANKS (NODES) ---
    max_tank_height = 80
    tank_width = 40
    max_water_level = 2.5

    for i in range(obs_dim):
        x, y = nodes[i]
        
        # Calculate water visual height
        water_level = max(0.0, min(float(state[i]), max_water_level))
        water_height = int((water_level / max_water_level) * max_tank_height)
        
        # Draw Tank Background
        pygame.draw.rect(screen, WHITE, (x, y, tank_width, max_tank_height))
        pygame.draw.rect(screen, BLACK, (x, y, tank_width, max_tank_height), 2)
        
        # Draw Water inside Tank
        if water_height > 0:
            water_color = RED if float(state[i]) > 1.8 else BLUE
            pygame.draw.rect(screen, water_color, (x, y + max_tank_height - water_height, tank_width, water_height))
            
        # Draw 1.8m Safety Threshold Line
        thresh_y = y + max_tank_height - int((1.8 / max_level) * max_tank_height) if 'max_level' in locals() else y + max_tank_height - int((1.8 / max_water_level) * max_tank_height)
        pygame.draw.line(screen, RED, (x - 5, thresh_y), (x + tank_width + 5, thresh_y), 2)

        # Draw Node Name and Depth
        n_lbl = font.render(f"N{i+1}", True, BLACK)
        screen.blit(n_lbl, (x + 5, y + max_tank_height + 5))
        d_lbl = small_font.render(f"{state[i]:.2f}m", True, BLACK)
        screen.blit(d_lbl, (x, y - 20))

    pygame.display.flip()

    # Reset at end of episode
    if step >= 100:
        time.sleep(3) # Pause to let faculty observe final state
        state = env.reset()
        state_history = [state] * best_seq_len
        hidden = (torch.zeros(1, 1, best_hidden), torch.zeros(1, 1, best_hidden))
        cumulative_flood = 0
        step = 0

    clock.tick(5) # Slow down animation to make water flow visible

pygame.quit()
sys.exit()