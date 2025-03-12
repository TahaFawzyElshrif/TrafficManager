import numpy as np

def reward_independent_part(single_state):
    """
    Calculates the independent reward for a single traffic light based on various traffic metrics.
    
    Parameters:
    single_state (tuple): Contains traffic state metrics in the following order:
        avg_speed (float) - Average vehicle speed.
        var_speed (float) - Variance of vehicle speeds.
        avg_waiting_time (float) - Average waiting time of vehicles.
        var_waiting_time (float) - Variance of waiting times.
        avg_throughput (float) - Number of vehicles passing.
        avg_queue_length (float) - Average queue length.
        avg_Occupancy (float) - Road occupancy percentage.
    
    Returns:
    float: Calculated independent reward for the given traffic light.
    """
    eta = 1e-6  # Small constant to prevent division by zero
    scale = 0.2  # Scaling factor

    avg_speed, var_speed, avg_waiting_time, var_waiting_time, avg_throughput, avg_queue_length, avg_Occupancy = single_state

    # Speed reward: Log function to avoid negative values and extreme growth
    speed_term = np.log(1 + avg_speed / (var_speed + eta))

    # Waiting time penalty: Inverse to prioritize minimization
    waiting_term = 1 / (1 + avg_waiting_time * var_waiting_time)

    # Traffic efficiency metric: Weighted sum of key flow indicators
    traffic_efficiency = 0.5 * avg_throughput + 0.3 * avg_queue_length + 0.2 * avg_Occupancy

    # Final reward formula
    independent_part_reward = scale * speed_term * waiting_term + traffic_efficiency

    return independent_part_reward

def reward_independent_part_proto(single_state):
    """
    Prototype version of reward calculation, focusing only on speed and waiting time.
    
    Parameters:
    single_state (tuple): Contains traffic state metrics (speed and waiting time only).
    
    Returns:
    float: Simplified independent reward.
    """
    eta = 1e-6  # Small constant to prevent division by zero
    scale = 0.2  # Scaling factor

    avg_speed, var_speed, avg_waiting_time, var_waiting_time, avg_throughput, avg_queue_length, avg_Occupancy = single_state

    # Speed reward (simplified log function)
    speed_term = np.log(1 + avg_speed / (var_speed + eta))

    # Waiting time penalty
    waiting_term = 1 / (1 + avg_waiting_time * var_waiting_time)

    # Final reward formula (excluding other metrics)
    independent_part_reward = scale * speed_term * waiting_term 

    return independent_part_reward

def reward_total(state_dict, main_agent):
    """
    Computes the total reward for a given traffic light while considering the states of other traffic lights.
    
    Parameters:
    state_dict (dict): Dictionary mapping traffic light IDs to their respective state tuples.
    main_agent (str): The traffic light ID of the main agent being evaluated.
    
    Returns:
    float: Total reward that balances the main agent's performance with other traffic lights.
    """
    a_main_agent = 0.90  # Weight for the main traffic light
    w_other_agents = 1 - a_main_agent  # Remaining weight for other traffic lights

    # Main agent reward
    reward_main = reward_independent_part(state_dict[main_agent])

    # Average reward of other traffic lights
    mean_reward_others = np.mean([
        reward_independent_part(state) 
        for i_agent, state in state_dict.items() if i_agent != main_agent
    ])
    # Weighted sum of main and other agents' rewards
    return (a_main_agent * reward_main) + (mean_reward_others * w_other_agents)

def reward_total_proto(state_dict, main_agent):
    """
    Prototype version of total reward calculation, using only speed and waiting time metrics.
    
    Parameters:
    state_dict (dict): Dictionary mapping traffic light IDs to their respective state tuples.
    main_agent (str): The traffic light ID of the main agent being evaluated.
    
    Returns:
    float: Total prototype reward.
    """
    a_main_agent = 0.90  # Weight for the main traffic light
    w_other_agents = 1 - a_main_agent  # Remaining weight for other traffic lights

    # Main agent reward (prototype version)
    reward_main = reward_independent_part_proto(state_dict[main_agent])

    # Average reward of other traffic lights (prototype version)
    mean_reward_others = np.mean([
        reward_independent_part_proto(state) 
        for i_agent, state in state_dict.items() if i_agent != main_agent
    ])

    # Weighted sum of main and other agents' rewards
    return (a_main_agent * reward_main) + (mean_reward_others * w_other_agents)
