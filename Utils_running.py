import traci
from numpy import inf
import numpy as np
import gymnasium as gym
from Sensors import len_sensors
from config import *

def get_traffic_lights_policies_full(n_agent=-2,thres_count=-1): #n_agent=-2 means all traffic lights
    #thres_count=-1 means no threshold

    traffic_lights=[]
    policies=dict({})
    tl_ids=traci.trafficlight.getIDList()
    
    for tl_id in tl_ids:
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        n_contrlod= len(controlled_lanes)

        if (n_contrlod<thres_count) or (thres_count==-1):
            traffic_lights.append((n_contrlod,tl_id))
            policies[tl_id]=(None ,
                                gym.spaces.Box(low=-inf, high=inf, shape=(len_sensors,), dtype=np.float32),
                                gym.spaces.Discrete(n_considered_action**n_contrlod), {})

        if (len(traffic_lights)==n_agent):
            break  

    return traffic_lights,policies


def get_traffic_lights_policies_group(n_agent=-2): #n_agent=-2 means all traffic lights
    traffic_lights=[]
    policies=dict({})
    tl_ids=traci.trafficlight.getIDList()
    
    for i,tl_id in enumerate(tl_ids):
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        n_contrlod= len(controlled_lanes)

        
        traffic_lights.append((n_contrlod,tl_id))
        policies[tl_id]=(None ,
                            gym.spaces.Box(low=-inf, high=inf, shape=(len_sensors,), dtype=np.float32),
                            gym.spaces.Discrete(81), {})

        if (i==n_agent-1):
            break  

    return traffic_lights,policies

def get_traffic_lights_policies_high_group(n_agent=-2): #n_agent=-2 means all traffic lights
    traffic_lights=[]
    policies=dict({})
    tl_ids=traci.trafficlight.getIDList()
    
    for i,tl_id in enumerate(tl_ids):
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        n_contrlod= len(controlled_lanes)

        
        traffic_lights.append((n_contrlod,tl_id))
        policies[tl_id]=(None ,
                            gym.spaces.Box(low=-inf, high=inf, shape=(len_sensors,), dtype=np.float32),
                            gym.spaces.Discrete(3), {})

        if (i==n_agent-1):
            break  

    return traffic_lights,policies