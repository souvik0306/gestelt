#!/usr/bin/env python3
import rospy
from gestelt_msgs.msg import CommanderState, Goals, CommanderCommand
from geometry_msgs.msg import Pose, Accel, PoseArray, AccelStamped, Twist
from std_msgs.msg import Bool
import time

# Publishers for server commands and trajectory goals
server_event_pub = rospy.Publisher('/traj_server/command', CommanderCommand, queue_size=10)
waypoints_pub = rospy.Publisher('/planner/goals', Goals, queue_size=10)
# For visualization of requested waypoints
waypoints_pos_pub = rospy.Publisher('/planner/goals_pos', PoseArray, queue_size=10)
waypoints_acc_pub = rospy.Publisher('/planner/goals_acc', AccelStamped, queue_size=10)

# Storage for state feedback
server_states = {}

def check_traj_server_states(desired_state):
    """Return True when all drones report the desired traj_server state."""
    if not server_states:
        rospy.logwarn("No Server states received!")
        return False
    for state in server_states.values():
        if state.traj_server_state != desired_state:
            return False
    return True


def publishCommand(event_enum):
    cmd = CommanderCommand()
    cmd.command = event_enum
    server_event_pub.publish(cmd)


def get_server_state_callback():
    msg = rospy.wait_for_message('/traj_server/state', CommanderState, timeout=5.0)
    server_states[str(msg.drone_id)] = msg


def create_pose(x, y, z):
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    # face forward along -Y (same as mission.py)
    pose.orientation.x = 0.0
    pose.orientation.y = 0.0
    pose.orientation.z = -0.707
    pose.orientation.w = 0.707
    return pose


def create_accel(ax, ay, az):
    acc = Accel()
    mask = Bool()
    if ax is None:
        mask.data = True
    else:
        acc.linear.x = ax
        acc.linear.y = ay
        acc.linear.z = az
        mask.data = False
    return acc, mask


def create_vel(vx, vy, vz):
    vel = Twist()
    mask = Bool()
    if vx is None:
        mask.data = True
    else:
        vel.linear.x = vx
        vel.linear.y = vy
        vel.linear.z = vz
        mask.data = False
    return vel, mask

def pub_waypoints(waypoints, accels, vels,
                  time_factor_terminal=1.0, time_factor=0.6,
                  max_vel=3.0, max_accel=5.0):
    """Publish a Goals message with optional timing constraints."""
    msg = Goals()
    pos_msg = PoseArray()
    acc_msg = AccelStamped()

    msg.header.frame_id = 'world'
    pos_msg.header.frame_id = 'world'
    acc_msg.header.frame_id = 'world'

    msg.waypoints = waypoints
    msg.accelerations = [a[0] for a in accels]
    msg.velocities = [v[0] for v in vels]
    msg.accelerations_mask = [a[1] for a in accels]
    msg.velocities_mask = [v[1] for v in vels]

    msg.time_factor_terminal.data = time_factor_terminal
    msg.time_factor.data = time_factor
    msg.max_vel.data = max_vel
    msg.max_acc.data = max_accel

    pos_msg.poses = waypoints
    if accels:
        acc_msg.accel = accels[0][0]

    waypoints_pub.publish(msg)
    waypoints_pos_pub.publish(pos_msg)
    waypoints_acc_pub.publish(acc_msg)

def wait_for_state(state):
    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        get_server_state_callback()
        if check_traj_server_states(state):
            return
        rate.sleep()


def main():
    rospy.init_node('square_mission', anonymous=True)

    rospy.loginfo('Waiting for UAV to reach HOVER')
    publishCommand(CommanderCommand.TAKEOFF)
    wait_for_state('HOVER')

    # Define square corners at 1.2 m altitude. The first waypoint matches the
    # hover position so that the planner explicitly enforces a zero-velocity
    # boundary condition before the vehicle leaves the origin.
    corners = [
        create_pose(0.0, 0.0, 1.2),
        create_pose(2.0, 0.0, 1.2),
        create_pose(2.0, 2.0, 1.2),
        create_pose(0.0, 2.0, 1.2),
        create_pose(0.0, 0.0, 1.2)
    ]

    # Request zero velocity at every corner so the planner produces straight
    # edges instead of rounded turns.
    velocities = [create_vel(0.0, 0.0, 0.0) for _ in corners]
    accelerations = [create_accel(None, None, None) for _ in corners]

    rospy.loginfo('Sending full square trajectory')
    publishCommand(CommanderCommand.MISSION)
    wait_for_state('MISSION')
    pub_waypoints(corners, accelerations, velocities,
                  time_factor_terminal=1.0,
                  time_factor=0.8,
                  max_vel=4.0,
                  max_accel=8.0)
    wait_for_state('HOVER')
    rospy.loginfo('Square trajectory complete')

    rospy.loginfo('Square mission complete')
    rospy.spin()


if __name__ == '__main__':
    main()
