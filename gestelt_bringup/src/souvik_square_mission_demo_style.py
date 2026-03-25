#!/usr/bin/env python3
import numpy as np
import rospy
from gestelt_msgs.msg import CommanderState, Goals, CommanderCommand
from geometry_msgs.msg import Pose, Accel,PoseArray,AccelStamped, Twist
from std_msgs.msg import Int8, Bool,Float32
from mavros_msgs.srv import ParamSet
from mavros_msgs.msg import ParamValue
import math
import time
import tf

# get ros params from rosparam server
is_simulation=rospy.get_param('mission/is_simulation', False)

# Publisher of server events to trigger change of states for trajectory server 
server_event_pub = rospy.Publisher('/traj_server/command', CommanderCommand, queue_size=10)
# Publisher of server events to trigger change of states for trajectory server 
waypoints_pub = rospy.Publisher('/planner/goals', Goals, queue_size=10)

# Publisher for desired hover setpoint
hover_position_pub = rospy.Publisher('/planner/hover_position', Pose, queue_size=10)
time_factor_pub = rospy.Publisher('/planner/time_factor', Float32, queue_size=10)

# for visualization
waypoints_pos_pub = rospy.Publisher('/planner/goals_pos', PoseArray, queue_size=10)
waypoints_acc_pub = rospy.Publisher('/planner/goals_acc', AccelStamped, queue_size=10)

current_traj_time=0

# Initialize trajectory number
TRAJ_NUM = 1

# Dictionary of UAV states
server_states = {}

# maximum down velocity limitation option publisher
max_down_vel_limit_pub = rospy.Publisher('/planner/max_down_vel_limit', Bool, queue_size=10)

# PX4 parameters dynamic reconfigure client
px4_param_reconfig_client_=rospy.ServiceProxy('/mavros/param/set',ParamSet)

# Check if UAV has achived desired traj_server_state
def check_traj_server_states(des_traj_server_state):
    if len(server_states.items()) == 0:
        print("No Server states received!")
        return False
    
    for server_state in server_states.items():
        if server_state[1].traj_server_state != des_traj_server_state:
            return False
    return True

def publishCommand(event_enum):
    commander_cmd = CommanderCommand()
    commander_cmd.command = event_enum
    server_event_pub.publish(commander_cmd)

def get_server_state_callback():
    msg = rospy.wait_for_message(f"/traj_server/state", CommanderState, timeout=5.0)
    server_states[str(msg.drone_id)] = msg

def transform_map_to_world():
    """
    this function calculates the transformation from map to world frame
    world frame is the initial position of the drone
    map frame is the origin of the map  
    Returns:
        trans: translation vector from map to world
        rot: rotation vector from map to world
    """
    tf_listener = tf.TransformListener()

    if is_simulation:
        while not rospy.is_shutdown():
            try:
                if tf_listener.canTransform("world", "map", rospy.Time(0)):
                    (trans, rot) = tf_listener.lookupTransform("world", "map", rospy.Time(0))
                    break 
                else:
                    rospy.sleep(0.04)
            except tf.TransformException as ex:
                rospy.logwarn("TransformException: {}".format(ex))
                rospy.sleep(0.04)
        return trans,rot
    else:
        return (-1.5,-1.5,0.0),(0.0,0.0,0.0,1.0)

def create_pose(x, y, z):
    pose = Pose()
    trans,rot=transform_map_to_world()
    pose.position.x = x+trans[0]
    pose.position.y = y+trans[1]
    pose.position.z = z+trans[2]
    pose.orientation.x = 0
    pose.orientation.y = 0
    pose.orientation.z = -0.707
    pose.orientation.w = 0.707
    return pose

def create_accel(acc_x,acc_y,acc_z):
    acc = Accel()
    acc_mask = Bool()
    if acc_x !=None:
        acc.linear.x = acc_x
        acc.linear.y = acc_y
        acc.linear.z = acc_z
        acc_mask.data=False
    else:
        acc_mask.data=True
    return acc,acc_mask

def create_vel(vel_x,vel_y,vel_z):
    vel = Twist()
    vel_mask = Bool()
    if vel_x !=None:
        vel.linear.x = vel_x
        vel.linear.y = vel_y
        vel.linear.z = vel_z
        vel_mask.data=False
    else:
        vel_mask.data=True
    return vel,vel_mask

def pub_waypoints(waypoints,accels,vels,time_factor_terminal=1,time_factor=0.6,max_vel=3,max_accel=5):
    wp_msg = Goals()
    wp_msg.header.frame_id = "world"
    wp_msg.waypoints = waypoints
    wp_msg.accelerations= [accel[0] for accel in accels]
    wp_msg.velocities= [vel[0] for vel in vels]
    wp_msg.accelerations_mask=[accel[1] for accel in accels]
    wp_msg.velocities_mask=[vel[1] for vel in vels]
    wp_msg.time_factor_terminal.data=time_factor_terminal
    wp_msg.time_factor.data=time_factor
    wp_msg.max_vel.data=max_vel
    wp_msg.max_acc.data=max_accel  
    waypoints_pub.publish(wp_msg)

def set_PX4_parameters(param_id, value):
    rospy.wait_for_service('/mavros/param/set')
    try:
        param_value = ParamValue()
        param_value.integer = 0
        param_value.real = value
        response = px4_param_reconfig_client_(param_id=param_id,value=param_value)
        if response.success:
            rospy.loginfo("Parameter '{}' set to '{}'".format(param_id, value))
        else:
            rospy.logwarn("PX4 param reconfigure Service call failed with message: %s", response.message)
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s", str(e))

def pub_max_down_vel_limit(option):
    limit_msg = Bool()
    limit_msg.data = option
    max_down_vel_limit_pub.publish(limit_msg)

def traj_time_callback(msg):
    global TRAJ_NUM
    
    current_traj_time = msg.data
    print("sleeping for",current_traj_time)
    rospy.sleep(current_traj_time + 1)
        
    print(f"Sending trajectory {TRAJ_NUM} waypoints to UAVs")
    
    ###########################################################################
    # Square mission: Aggressive square with sharp turns
    ###########################################################################
    if TRAJ_NUM==1:
        TIME_FACTOR_TERMINAL=1
        TIME_FACTOR=0.6
        MAX_VEL=6
        MAX_ACCEL=3

        waypoints = []
        vel_list = []
        accel_list = []

        height = 1.5
        num_passes = 5
        
        for _ in range(num_passes):
            # Smooth square with more waypoints for perfect trajectory
            # Bottom side: 6 points from (0,0) to (3,0)
            for x in np.linspace(0, 3, 6):
                waypoints.append(create_pose(x, 0.0, height))
            
            # Right side: 5 points from (3,0.6) to (3,3) (skip first to avoid duplicate)
            for y in np.linspace(0, 3, 6)[1:]:
                waypoints.append(create_pose(3.0, y, height))
            
            # Top side: 5 points from (2.4,3) to (0,3) (skip first)
            for x in np.linspace(3, 0, 6)[1:]:
                waypoints.append(create_pose(x, 3.0, height))
            
            # Left side: 5 points from (0,2.4) to (0,0) (skip first)
            for y in np.linspace(3, 0, 6)[1:]:
                waypoints.append(create_pose(0.0, y, height))
            
            # Total waypoints per pass: 6 + 5 + 5 + 5 = 21
            for _ in range(21):
                accel_list.append(create_accel(None,None,None))  
                vel_list.append(create_vel(None,None,None))

    elif TRAJ_NUM > 1:
        rospy.signal_shutdown("All square trajectories completed!")
        return
        
    print(f"Publishing trajectory {TRAJ_NUM}")
    pub_waypoints(waypoints, accel_list, vel_list, TIME_FACTOR_TERMINAL, TIME_FACTOR, MAX_VEL, MAX_ACCEL)
    TRAJ_NUM = TRAJ_NUM + 1

def main():
    rospy.init_node('square_mission_demo_style', anonymous=True)
    pub_freq = 25
    rate = rospy.Rate(pub_freq)

    HOVER_MODE = False
    MISSION_MODE = False

    rospy.Subscriber("/trajectory_total_time", Float32, traj_time_callback)

    while not rospy.is_shutdown():
        get_server_state_callback()
        
        if check_traj_server_states("MISSION"):
            MISSION_MODE = True
        if check_traj_server_states("HOVER"):
            HOVER_MODE = True
        
        if (MISSION_MODE):
            time.sleep(5)
            # Set PX4 parameters for square missions
            set_PX4_parameters("MPC_RPT_XY_KI", 0.5) #0.5
            set_PX4_parameters("MPC_RPT_Z_KI", 1.2) #0.5
            set_PX4_parameters("MPC_RPT_Z_SIGMA", 3) #2.5
            time.sleep(2)
            break
        elif (not HOVER_MODE):
            print("Setting to HOVER mode!")
            publishCommand(CommanderCommand.TAKEOFF)
        elif (HOVER_MODE):
            print("Setting to MISSION mode!")
            publishCommand(CommanderCommand.MISSION)

        print("tick!")
        rate.sleep()
    
    # No initial trajectory needed - takeoff is at square start
    print("Starting square mission directly")
    
    ###########################################################################
    # Square mission: Aggressive square with sharp turns
    ###########################################################################
    TIME_FACTOR_TERMINAL = 1
    TIME_FACTOR = 0.6
    MAX_VEL = 7
    MAX_ACCEL = 4

    waypoints = []
    vel_list = []
    accel_list = []

    height = 1.5
    num_passes = 2
    
    for _ in range(num_passes):
        # Smooth square with more waypoints for perfect trajectory
        # Bottom side: 6 points from (0,0) to (3,0)
        for x in np.linspace(0, 3, 6):
            waypoints.append(create_pose(x, 0.0, height))
        
        # Right side: 5 points from (3,0.6) to (3,3) (skip first to avoid duplicate)
        for y in np.linspace(0, 3, 6)[1:]:
            waypoints.append(create_pose(3.0, y, height))
        
        # Top side: 5 points from (2.4,3) to (0,3) (skip first)
        for x in np.linspace(3, 0, 6)[1:]:
            waypoints.append(create_pose(x, 3.0, height))
        
        # Left side: 5 points from (0,2.4) to (0,0) (skip first)
        for y in np.linspace(3, 0, 6)[1:]:
            waypoints.append(create_pose(0.0, y, height))
        
        # Total waypoints per pass: 6 + 5 + 5 + 5 = 21
        for _ in range(21):
            accel_list.append(create_accel(None,None,None))  
            vel_list.append(create_vel(None,None,None))

    pub_waypoints(waypoints, accel_list, vel_list, TIME_FACTOR_TERMINAL, TIME_FACTOR, MAX_VEL, MAX_ACCEL)
    
    rospy.spin()

if __name__ == '__main__':
    main()
