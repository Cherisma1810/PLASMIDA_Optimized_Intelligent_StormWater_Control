import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyswmm import Simulation, Nodes, Links

class VelacheryStormwaterEnv(gym.Env):
    def __init__(self, inp_file_path):
        super(VelacheryStormwaterEnv, self).__init__()
        self.inp_file_path = inp_file_path
        self.sim = None
        
        # 7 control points with ON(1)/OFF(0) states
        self.num_pumps = 7 
        self.action_space = spaces.MultiDiscrete([2] * self.num_pumps)
        
        # We now use the 7 ORIFICES named Gate_1 through Gate_7
        self.pump_ids = [f"Gate_{i+1}" for i in range(self.num_pumps)]
        
        # State space: Water levels at critical nodes
        self.storage_ids = ["A5", "C41", "B12", "C39", "A12", "C46", "B10", "C18", "C22", "A6", "C30", "B17"]
        self.num_observable_states = len(self.storage_ids)
        self.observation_space = spaces.Box(low=0, high=np.inf, 
                                            shape=(self.num_observable_states,), 
                                            dtype=np.float32)

    def reset(self, seed=None):
        super().reset(seed=seed)
        if self.sim:
            self.sim.close()
            
        self.sim = Simulation(self.inp_file_path)
        self.sim.start()
        self.nodes = Nodes(self.sim)
        self.links = Links(self.sim)
        return self._get_state(), {}

    def _get_state(self):
        state = []
        for node_id in self.storage_ids:
            try:
                state.append(self.nodes[node_id].depth)
            except Exception:
                state.append(0.0) # Fallback if node doesn't exist
        return np.array(state, dtype=np.float32)

    def step(self, action):
        # 1.0 = Fully Open, 0.0 = Closed
        for idx, pump_id in enumerate(self.pump_ids):
            self.links[pump_id].target_setting = float(action[idx])

        self.sim.step_advance(300) 
        
        # Python 3 iterator advancement with graceful end-of-simulation handling
        try:
            next(self.sim)
        except StopIteration:
            pass # Simulation reached its defined END_DATE

        flooding_vol = sum([node.flooding for node in self.nodes])
        penalty = 0.0
        lambda_penalty = 0.1
        
        for storage_id in self.storage_ids:
            try:
                node = self.nodes[storage_id]
                max_depth = node.full_depth if node.full_depth > 0 else 1.0
                if node.depth > 0.9 * max_depth:
                    penalty += lambda_penalty * (node.depth - (0.9 * max_depth))
            except Exception:
                pass

        reward = -flooding_vol - penalty
        done = (self.sim.current_time >= self.sim.end_time)
        return self._get_state(), reward, done, False, {}

    def close(self):
        if self.sim:
            self.sim.close()