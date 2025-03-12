from numpy import inf
import pandas as pd
import ast
import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from pettingzoo.utils import agent_selector
import traci
import os
import itertools
import warnings
from SumoConnection import SumoConnection, add_traffic
from Sensors import getSumoSensors_full, len_sensors
import rewards
import random
import math
import traci
import itertools

# Suppress deprecation warnings
warnings.simplefilter("ignore", category=DeprecationWarning)


import importlib  # Import importlib

importlib.reload(rewards)  # Reload the module after modification



class SumoEnv(MultiAgentEnv):
    def __init__(self, cmd, traffic_light_info, training_mode, max_steps=50, max_sumo_steps=200, traffic_scale=15):
        """
        Initializes the SUMO environment.
        """
        super().__init__()
        SumoEnv.instance_running = self  # Global instance reference for RLlib

        self.metadata = {"is_parallelizable": True, "render_modes": ["human"]}
        self.observation_size = len_sensors
        self.training_mode = training_mode
        self.cmd = cmd
        self.current_step = 0

        self.agents = []
        self.action_spaces = {}
        self.observation_spaces = {}
        self.table_direct_mapping = {}
        self.state = {}
        self.terminations = {}
        self.truncations = {}

        self.max_steps = max_steps
        self.traffic_scale = traffic_scale
        
        self.max_sumo_steps = max_sumo_steps
    
        # Initialize each traffic light agent
        for count, tl_id in traffic_light_info:
            self.agents.append(tl_id)
            self.initialize_action_space(tl_id, count)
            
            self.observation_spaces[tl_id] = gym.spaces.Box(
                low=-inf, high=inf, shape=(self.observation_size,), dtype=np.float32
            )
            self.state[tl_id] = np.zeros(self.observation_size, dtype=np.float32)
            self.terminations[tl_id] = False
            self.truncations[tl_id] = False

        self.possible_agents = self.agents[:]  # Copy of agent list
        
        # Remove agent selector and related attributes
        # self.agent_selector = agent_selector(self.agents)
        # self.agent_selection = self.agent_selector.next()

        # Initialize SUMO connection if in training mode
        if self.training_mode:
            try:
                SumoConnection(self.cmd).initialize_sumo()
            except Exception as e:
                print("Failed to initialize SUMO:", e)
                traci.close()
    def initialize_action_space(self, tl_id, count):
        """
        Creates the action space mapping for a traffic light agent.
        """
        space = list(map("".join, itertools.product("rgy", repeat=count))) if count > 0 else ["r", "g", "y"]   # Cartesian product of RGY
        self.table_direct_mapping[tl_id] = dict(zip(range(len(space)), space))
        #self.action_spaces[tl_id] = gym.spaces.Discrete(len(space))
        self.action_spaces[tl_id]=gym.spaces.Discrete(len(space))

    def reset(self, seed=None, options=None, sumo_reset=False):
        """
        Resets the environment and optionally resets SUMO.
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.agents = self.possible_agents[:]
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.state = {agent: np.zeros(self.observation_size, dtype=np.float32) for agent in self.agents}

        # Remove agent selector reset
        # self.agent_selector.reset()
        # self.agent_selection = self.agent_selector.next()

        if self.training_mode and sumo_reset:
            SumoConnection(self.cmd).reset()

        # Debug print
        print("Environment reset")

        return self.state, {agent: {} for agent in self.agents}
    
    def get_done_condition(self):
        cond1 = (self.current_step >= self.max_steps)
        if self.training_mode:
            cond2 = (traci.simulation.getTime() >= self.max_sumo_steps)
            done = cond1 or cond2  # Changed from AND to OR for more intuitive behavior
        else:
            done = cond1

        
        return done
    
    def get_real_action(self, agent, action):
        return self.table_direct_mapping[agent][action]
    
    def step(self, actions):
            """
            Executes actions for all agents and advances the simulation by one step.
            """
            if self.get_done_condition():
                for agent in self.agents:
                    self.terminations[agent] = True
                    self.truncations[agent] = True
                self.terminations["__all__"] = True
                self.truncations["__all__"] = True
                return self.state, {agent: 0.0 for agent in self.agents}, self.terminations, self.truncations, {agent: {}}
            
            for agent in self.agents:
                if agent not in actions:
                    raise KeyError(f"Action not provided for agent: {agent}")
            
            if self.current_step % 1 == 0:
                for agent in self.agents:
                    add_traffic(agent, self.traffic_scale)

            for agent in self.agents:
                action = actions[agent]
                real_action=self.get_real_action(agent, action)

                if action not in self.table_direct_mapping[agent]:
                    raise KeyError(f"Invalid action: {action}")
                traci.trafficlight.setRedYellowGreenState(agent,real_action )
            
            traci.simulationStep()
            self.current_step += 1
            done = self.get_done_condition()
            
            reward_dic = {}
            for agent in self.agents:
                self.state[agent] = np.array(getSumoSensors_full(agent))
                #print(f"---------------> State: {self.state[agent]}")
                reward_dic[agent] = rewards.reward_total(self.state, agent)
                self.terminations[agent] = done
                self.truncations[agent] = done
            
            self.terminations["__all__"] = done
            self.truncations["__all__"] = done
            
            
           
            return self.state, reward_dic, self.terminations, self.truncations, {agent: {} for agent in self.agents}

    def render(self, mode='human'):
        """
        Prints the current state of the environment.
        """
        print(f"State: {self.state}")

    def close(self):
        """
        Closes the environment (if necessary).
        """
        pass

class GroupedSumoEnv(SumoEnv):
    def __init__(self, cmd, traffic_light_info, training_mode, max_steps=50, max_sumo_steps=200, traffic_scale=15):
        super().__init__( cmd, traffic_light_info, training_mode,max_steps,max_sumo_steps,traffic_scale)

    def initialize_action_space(self, tl_id,count):
        space=(list(map(''.join, itertools.product("rgy", repeat=4))) )
        self.table_direct_mapping[tl_id] = dict(zip(range(len(space)), space))
        self.action_spaces[tl_id] = gym.spaces.Discrete(len(space))

    def get_lane_direction(self,lane_id):

        """Determine the primary cardinal direction of a lane in SUMO."""
        x_start, y_start = traci.lane.getShape(lane_id)[0]  # First coordinate
        x_end, y_end = traci.lane.getShape(lane_id)[-1]     # Last coordinate

        angle = math.degrees(math.atan2(y_end - y_start, x_end - x_start))

        # Optimized angle-based direction mapping
        return "E" if -45 <= angle < 45 else "N" if 45 <= angle < 135 else "W" if angle >= 135 or angle < -135 else "S"

    def get_all_lanes_action(self,tl_id, action):
        """Map SUMO controlled lanes to the corresponding action signals."""
        direction_map = {"N": 0, "E": 1, "S": 2, "W": 3}
        
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        real_action_dict = {}

        try:
            real_action_list = [
                action[direction_map[dir]] for lane in controlled_lanes if (dir := self.get_lane_direction(lane)) in direction_map
            ]
        except IndexError:
            raise ValueError(f"Invalid action index for action: {action}")

        return "".join(real_action_list), real_action_dict
    
    def get_real_action(self,agent, action):
        action_agent_lanes=self.table_direct_mapping[agent][action]
        real_action=self.get_all_lanes_action(agent, action_agent_lanes)[0]
        return real_action
    
class HighGroupedSumoEnv(SumoEnv):
    def __init__(self, cmd, traffic_light_info, training_mode, max_steps=50, max_sumo_steps=200, traffic_scale=15):
        super().__init__( cmd, traffic_light_info, training_mode,max_steps,max_sumo_steps,traffic_scale)

    def initialize_action_space(self, tl_id, count):
        space = ["r"*count,"g"*count,"y"*count] if (count>0) else ["r","g","y"]
        self.table_direct_mapping[tl_id] = dict(zip(range(len(space)), space))
        self.action_spaces[tl_id] = gym.spaces.Discrete(len(space))
