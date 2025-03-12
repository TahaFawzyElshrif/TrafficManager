import numpy as np
import traci

len_sensors = 7  # Number of sensors considered for measurement
len_optimized_sensors = 4  # Number of optimized sensors for performance

def getSumoSensors_full(tl_id):
    """
    Retrieves various traffic-related metrics from SUMO for a given traffic light ID.
    
    This function collects data such as vehicle speeds, waiting times, queue lengths, 
    throughput, and occupancy for all lanes controlled by the specified traffic light.

    Parameters:
        tl_id (str): The ID of the traffic light in SUMO.

    Returns:
        tuple: A tuple containing the following metrics:
            - avg_speed (float): Average speed of vehicles.
            - var_speed (float): Variance in vehicle speeds.
            - avg_waiting_time (float): Average waiting time of vehicles.
            - var_waiting_time (float): Variance in waiting time.
            - avg_throughput (float): Average number of vehicles passing per step.
            - avg_queue_length (float): Average number of halted vehicles.
            - avg_Occupancy (float): Average lane occupancy percentage.
    """

    vehicle_waiting = []  # Stores waiting times of vehicles
    vehicle_speeds = []  # Stores speeds of vehicles
    edges = []  # Stores unique edge IDs corresponding to lanes
    throughputs = []  # Stores number of vehicles passing per step
    queue_lengths = []  # Stores number of halted vehicles per edge
    Occupancies = []  # Stores lane occupancy percentages
    
    # Initialize output variables
    avg_waiting_time = 0
    var_waiting_time = 0
    avg_speed = 0
    var_speed = 0
    avg_throughput = 0
    avg_queue_length = 0
    avg_Occupancy = 0  # Percentage of occupied space in lanes

    # Get all lanes controlled by the traffic light
    controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)

    for lane in controlled_lanes:
        # Get vehicle IDs currently in the lane
        vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)

        # Collect speed and waiting time of vehicles in the lane
        vehicle_speeds += [traci.vehicle.getSpeed(vehicle_id) for vehicle_id in vehicle_ids]
        vehicle_waiting += [traci.vehicle.getAccumulatedWaitingTime(vehicle_id) for vehicle_id in vehicle_ids]

        # Identify the corresponding edge for this lane
        corresponding_edge = traci.lane.getEdgeID(lane)

        # Avoid duplicate data collection for the same edge
        if corresponding_edge not in edges:
            edges.append(corresponding_edge)
            throughputs.append(traci.edge.getLastStepVehicleNumber(corresponding_edge))  # Number of vehicles passing
            queue_lengths.append(traci.edge.getLastStepHaltingNumber(corresponding_edge))  # Number of halted vehicles
            Occupancies.append(traci.edge.getLastStepOccupancy(corresponding_edge))  # Lane occupancy percentage

    # Convert lists to NumPy arrays for efficient computation
    vehicle_speeds = np.array(vehicle_speeds)
    avg_speed = np.mean(vehicle_speeds) if vehicle_speeds.size > 0 else 0
    var_speed = np.var(vehicle_speeds) if vehicle_speeds.size > 0 else 0
    
    vehicle_waiting = np.array(vehicle_waiting)
    avg_waiting_time = np.mean(vehicle_waiting) if vehicle_waiting.size > 0 else 0
    var_waiting_time = np.var(vehicle_waiting) if vehicle_waiting.size > 0 else 0
        
    avg_throughput = np.mean(throughputs) if throughputs else 0
    avg_queue_length = np.mean(queue_lengths) if queue_lengths else 0
    avg_Occupancy = np.mean(Occupancies) if Occupancies else 0

    #clean memory
    del(vehicle_waiting)
    del(vehicle_speeds)
    del(edges)
    del(throughputs)
    del(queue_lengths)
    del(Occupancies)
    
    return (avg_speed, var_speed, avg_waiting_time, var_waiting_time, avg_throughput, avg_queue_length, avg_Occupancy)
