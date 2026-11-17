#!/usr/bin/env python3
"""Take off and remain in the trajectory server's HOVER state."""

import rospy

from gestelt_msgs.msg import CommanderCommand, CommanderState


server_event_pub = None
server_states = {}


def check_traj_server_states(desired_state):
    if not server_states:
        return False
    return all(
        state.traj_server_state == desired_state
        for state in server_states.values()
    )


def publishCommand(event_enum):
    commander_cmd = CommanderCommand()
    commander_cmd.command = event_enum
    server_event_pub.publish(commander_cmd)


def get_server_state_callback(msg):
    server_states[str(msg.drone_id)] = msg


def main():
    global server_event_pub

    rospy.init_node("takeoff_hover_demo_style")
    server_event_pub = rospy.Publisher(
        "/traj_server/command", CommanderCommand, queue_size=10
    )
    rospy.Subscriber(
        "/traj_server/state", CommanderState, get_server_state_callback
    )

    rate = rospy.Rate(2)

    while not rospy.is_shutdown():
        if check_traj_server_states("HOVER"):
            rospy.loginfo("Takeoff complete. Drone is hovering.")
            break

        if server_states:
            rospy.loginfo_throttle(2.0, "Setting to HOVER mode!")
            publishCommand(CommanderCommand.TAKEOFF)

        rate.sleep()

    rospy.spin()


if __name__ == "__main__":
    main()
