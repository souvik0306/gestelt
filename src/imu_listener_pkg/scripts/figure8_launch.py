import asyncio
import math
from mavsdk import System
from mavsdk.mission import MissionItem, MissionPlan


def meters_to_latlon_offsets(north_m: float, east_m: float, lat_deg: float):
    # WGS84 rough conversion, good for small distances
    lat_rad = math.radians(lat_deg)
    dlat = north_m / 111_111.0
    dlon = east_m / (111_111.0 * math.cos(lat_rad))
    return dlat, dlon


async def run():
    drone = System()
    await drone.connect(system_address="udp://:14540")  # SITL usually

    print("Waiting for drone...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected")
            break

    print("Waiting for global position and home position...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("OK")
            break

    # Get current position
    async for pos in drone.telemetry.position():
        home_lat = pos.latitude_deg
        home_lon = pos.longitude_deg
        break

    # Figure-8 settings
    radius_m = 25.0        # radius of each loop (increased)
    rel_alt_m = 20.0       # altitude above home
    speed_m_s = 5.0
    num_points = 24        # points per figure-8 (more = smoother)
    num_loops = 5          # number of times to repeat the figure-8
    alt_offset_per_loop = 2.0  # altitude increase per loop (meters)

    # Generate figure-8 waypoints using parametric equations
    # Figure-8 (lemniscate): x = a*sin(t), y = a*sin(t)*cos(t)
    waypoints_ne = []
    for loop in range(num_loops):
        for i in range(num_points):
            t = 2 * math.pi * i / num_points
            north = radius_m * math.sin(t)
            east = radius_m * math.sin(t) * math.cos(t)
            altitude = rel_alt_m + (loop * alt_offset_per_loop)
            waypoints_ne.append((north, east, altitude))

    mission_items = []
    for i, (north, east, altitude) in enumerate(waypoints_ne):
        dlat, dlon = meters_to_latlon_offsets(north, east, home_lat)
        wp_lat = home_lat + dlat
        wp_lon = home_lon + dlon

        item = MissionItem(
            wp_lat,
            wp_lon,
            altitude,
            speed_m_s,
            is_fly_through=True,
            gimbal_pitch_deg=float("nan"),
            gimbal_yaw_deg=float("nan"),
            camera_action=MissionItem.CameraAction.NONE,
            loiter_time_s=0.0,
            camera_photo_interval_s=0.0,
            acceptance_radius_m=2.0,
            yaw_deg=float("nan"),
            camera_photo_distance_m=0.0,
            vehicle_action=MissionItem.VehicleAction.NONE,
        )
        mission_items.append(item)

    plan = MissionPlan(mission_items)

    print(f"Uploading figure-8 mission with {len(mission_items)} waypoints...")
    await drone.mission.set_return_to_launch_after_mission(True)
    await drone.mission.upload_mission(plan)
    print("Mission uploaded, check QGC Mission view")

    print("Arming...")
    await drone.action.arm()

    print("Taking off...")
    await drone.action.takeoff()
    await asyncio.sleep(6)

    print("Starting mission...")
    await drone.mission.start_mission()

    # Wait until mission finishes
    async for progress in drone.mission.mission_progress():
        print(f"Mission progress: {progress.current}/{progress.total}")
        if progress.current == progress.total:
            break

    print("Mission done, RTL should trigger")


if __name__ == "__main__":
    asyncio.run(run())
