"""
Webots R2025a Python Controller: Pioneer 3-DX Follow-The-Gap Navigation
Robot Model: Pioneer 3-DX with Sick LMS 291 Lidar
Language: Python 3
"""

from controller import Robot, Lidar, Motor
import numpy as np

# Controller tunable parameters
GAP_THRESHOLD = 1.0     # Minimum free space depth (meters) to consider as an open gap
BUBBLE_RADIUS = 0.35    # Safety bubble clearance around nearest obstacle point
MAX_SPEED = 4.0         # Max wheel motor velocity (rad/s)

def main():
    robot = Robot()
    time_step = int(robot.getBasicTimeStep())

    # Wheel motors
    left_motor = robot.getDevice('left wheel')
    right_motor = robot.getDevice('right wheel')
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    # Lidar sensor setup (e.g. Sick LMS 291 or standard Webots Lidar)
    lidar = robot.getDevice('Sick LMS 291')
    if lidar is None:
        lidar = robot.getDevice('lidar')
    lidar.enable(time_step)
    lidar.enablePointCloud()

    while robot.step(time_step) != -1:
        range_image = lidar.getRangeImage()
        if not range_image:
            continue

        ranges = np.array(range_image)
        # Filter infinite or nan readings
        ranges[np.isinf(ranges)] = 10.0
        ranges[np.isnan(ranges)] = 10.0
        n_points = len(ranges)

        # 1. Find Closest Obstacle
        min_idx = np.argmin(ranges)
        min_dist = ranges[min_idx]

        # 2. Place Safety Bubble (zero out ranges in obstacle vicinity)
        processed_ranges = ranges.copy()
        if min_dist < 1.8 and min_dist > 0.05:
            # Angular spread of safety bubble
            half_bubble = int((BUBBLE_RADIUS / min_dist) * (n_points / np.pi))
            bubble_start = max(0, min_idx - half_bubble)
            bubble_end = min(n_points - 1, min_idx + half_bubble)
            processed_ranges[bubble_start:bubble_end+1] = 0.0

        # 3. Detect Continuous Open Gaps
        gaps = []
        in_gap = False
        start = 0
        for i, r in enumerate(processed_ranges):
            if r > GAP_THRESHOLD and not in_gap:
                in_gap = True
                start = i
            elif r <= GAP_THRESHOLD and in_gap:
                in_gap = False
                gaps.append((start, i - 1))
        if in_gap:
            gaps.append((start, n_points - 1))

        # 4. Steer towards center of largest gap
        if gaps:
            # Pick gap with maximal breadth
            widest_gap = max(gaps, key=lambda g: g[1] - g[0])
            target_idx = (widest_gap[0] + widest_gap[1]) // 2
            # Angle deviation from center heading (-pi/2 to pi/2)
            fov = lidar.getFov()
            angle_error = ((target_idx / (n_points - 1)) - 0.5) * fov
            
            # Differential steering control
            steering_gain = 2.0
            turn_rate = steering_gain * angle_error
            
            v_l = MAX_SPEED * 0.7 - turn_rate
            v_r = MAX_SPEED * 0.7 + turn_rate
        else:
            # No gap found (deadlock / corner): rotate in place
            v_l = -MAX_SPEED * 0.4
            v_r = MAX_SPEED * 0.4

        # Clamp speeds
        v_l = max(min(v_l, MAX_SPEED), -MAX_SPEED)
        v_r = max(min(v_r, MAX_SPEED), -MAX_SPEED)

        left_motor.setVelocity(v_l)
        right_motor.setVelocity(v_r)

if __name__ == '__main__':
    main()
