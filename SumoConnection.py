import traci
import os
import json
from dotenv import load_dotenv
import uuid
import random

# Load secrets from the environment file
load_dotenv("keys.env")

# Retrieve SUMO_HOME path from environment variables
sumo_home = str(os.getenv("sumo_home"))

class SumoConnection:
    """
    Handles the connection to the SUMO traffic simulator.

    This class manages initializing, resetting, and closing the connection to SUMO.

    Attributes:
        cmd (list): The command used to start SUMO.
        traci_conn (traci.Connection): The active connection to SUMO.
        gui (bool): Indicates whether SUMO is running in GUI mode.
    """

    def __init__(self, cmd):
        """
        Initializes a new SUMO connection.

        Parameters:
            cmd (list): The command used to start SUMO.
        """
        self.cmd = cmd
        self.traci_conn = None
        self.gui = False
        self.initialize_sumo()
        
    def initialize_sumo(self):
        """
        Starts or reuses a SUMO connection.

        If SUMO_HOME is set, it assigns it to the environment variable.
        It checks if there is an existing connection; if found, it uses it;
        otherwise, it starts a new connection.
        """
        if sumo_home is not None:
            os.environ["SUMO_HOME"] = sumo_home

        # Determine if SUMO should run in GUI mode
        if "-gui" in self.cmd[0]:
            self.gui = True
        else:
            self.gui = False

        try:
            # Try to get an existing connection
            self.traci_conn = traci.getConnection()
            if self.traci_conn is not None:
                print("Found existing connection to SUMO and used it. To make a new connection, reset ✔")
            else:
                print("No connection found, creating a new connection ❌")
                traci.start(self.cmd)
                self.traci_conn = traci.getConnection()
        except traci.exceptions.TraCIException:
            print("No connection found, creating a new connection ❌")
            traci.start(self.cmd)
            self.traci_conn = traci.getConnection()

    def close(self):
        """
        Closes the SUMO connection if one exists.
        """
        try:
            if traci.getConnection() is not None:
                traci.close()
        except traci.exceptions.TraCIException:
            print("No active connection to close.")

    def reset(self):
        """
        Resets the SUMO connection by closing it and reinitializing.
        """
        self.close()
        self.initialize_sumo()




def add_traffic(tl_id, n):
    """
    Adds random vehicles to a specified traffic light-controlled area.

    This function selects a random lane controlled by the given traffic light ID and spawns vehicles
    on that lane while ensuring compatibility with lane restrictions.

    Parameters:
        tl_id (str): The traffic light ID whose controlled lanes are used.
        n (int): The number of vehicles to add.
    """
    controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)

    if not controlled_lanes:
        return    

    for i in range(n):
        lane = random.choice(controlled_lanes)
        edge_of_lane = traci.lane.getEdgeID(lane)
        
        # Randomly select a vehicle type
        type_v = random.choice(traci.vehicletype.getIDList())

        # Ensure the selected vehicle type is allowed in the chosen lane
        if traci.vehicletype.getVehicleClass(type_v) in traci.lane.getAllowed(lane):
            route_of_edge = "route_" + str(edge_of_lane)

            # If the route does not exist, create it
            if route_of_edge not in traci.route.getIDList():
                traci.route.add(route_of_edge, [edge_of_lane])
            
            # Generate a unique vehicle ID
            vehicle_id = str(uuid.uuid4())

            # Randomly position the vehicle within the lane to prevent congestion
            pos = random.uniform(0, traci.lane.getLength(lane) - 5)

            # Add the vehicle to the simulation and move it to the chosen position
            traci.vehicle.add(vehicle_id, route_of_edge, typeID=type_v)
            traci.vehicle.moveTo(vehicle_id, lane, pos)  # Move to lane start
