# Reference for polynomial trajectory tracking

std_msgs/Header header

uint8 type_mask
uint8 IGNORE_SNAP = 1
uint8 IGNORE_SNAP_JERK = 2
uint8 IGNORE_SNAP_JERK_ACC = 4

geometry_msgs/Vector3 position
geometry_msgs/Vector3 velocity
geometry_msgs/Vector3 acceleration
geometry_msgs/Vector3 jerk
geometry_msgs/Vector3 snap
