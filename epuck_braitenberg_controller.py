"""
Webots R2025a Python Controller: E-puck Swarm Braitenberg Obstacle Avoidance
Robot Model: e-puck
Language: Python 3
"""

from controller import Robot, DistanceSensor, Motor
import numpy as np

# Controller configuration parameters (adjustable for parameter sensitivity analysis)
K_REP = 0.04           # Repulsion gain parameter
BASE_SPEED = 3.5       # Forward speed (rad/s), max ~6.28 rad/s

def main():
    robot = Robot()
    time_step = int(robot.getBasicTimeStep())

    # Initialize Differential Drive Motors
    left_motor = robot.getDevice('left wheel motor')
    right_motor = robot.getDevice('right wheel motor')
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    # Initialize 8 Proximity Distance Sensors
    ps_names = [f'ps{i}' for i in range(8)]
    ps = []
    for name in ps_names:
        sensor = robot.getDevice(name)
        sensor.enable(time_step)
        ps.append(sensor)

    # Braitenberg weight matrix: connection weights from 8 sensors to [left, right] motors
    # Front sensors: ps7 (front-left 15°), ps0 (front-right 15°), ps6 (left 45°), ps1 (right 45°)
    # Higher readings => obstacle closer (typically ranges 60 to 4000+ in Webots)
    weights_left =  [-0.8, -0.6, -0.1, 0.0, 0.0, 0.2, 0.7, 1.2]
    weights_right = [ 1.2,  0.7,  0.2, 0.0, 0.0, -0.1, -0.6, -0.8]

    MAX_SPEED = 6.28

    while robot.step(time_step) != -1:
        # Normalize proximity sensor values [0, 1]
        ps_vals = [s.getValue() / 4096.0 for s in ps]

        # Calculate differential velocities
        d_left = sum(w * val for w, val in zip(weights_left, ps_vals))
        d_right = sum(w * val for w, val in zip(weights_right, ps_vals))

        v_l = BASE_SPEED + K_REP * 100.0 * d_left
        v_r = BASE_SPEED + K_REP * 100.0 * d_right

        # Clamp within hardware limits
        v_l = max(min(v_l, MAX_SPEED), -MAX_SPEED)
        v_r = max(min(v_r, MAX_SPEED), -MAX_SPEED)

        left_motor.setVelocity(v_l)
        right_motor.setVelocity(v_r)

if __name__ == '__main__':
    main()
