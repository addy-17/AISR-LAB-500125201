"""
Simulation & Analysis Script for:
1) Swarm of e-puck robots using Braitenberg obstacle avoidance
2) Pioneer 3-DX robot using Lidar Follow-The-Gap Method (FTGM) navigation
3) Parameter sensitivity analysis measuring Coverage, Mean Speed, and Stuck Time.

Can be run standalone to generate trajectories & comparative plots, or ported directly
into Webots R2025a Python controller directories.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json

# Set seed for reproducible comparative benchmarks
np.random.seed(42)

# ==========================================
# 1. Environment Simulation Setup
# ==========================================
class Arena2D:
    def __init__(self, width=6.0, height=6.0):
        self.width = width
        self.height = height
        # Rectangular obstacles: [xmin, ymin, xmax, ymax]
        self.obstacles = [
            [-1.5, -1.5, -0.5, -0.5],
            [0.5, 0.5, 1.5, 1.5],
            [-1.8, 0.8, -0.8, 1.8],
            [0.8, -1.8, 1.8, -0.8],
            [-0.4, -0.3, 0.4, 0.3], # Center obstacle
        ]
        
    def check_collision(self, x, y, radius=0.08):
        # Boundary check
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        if (abs(x) + radius >= half_w) or (abs(y) + radius >= half_h):
            return True
        # Obstacle check
        for ob in self.obstacles:
            if (ob[0] - radius <= x <= ob[2] + radius) and (ob[1] - radius <= y <= ob[3] + radius):
                return True
        return False

    def raycast(self, origin, angle, max_range=2.0):
        """Simulate range measurement for a single ray."""
        steps = 40
        dr = max_range / steps
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        for i in range(1, steps + 1):
            r = i * dr
            px = origin[0] + r * cos_a
            py = origin[1] + r * sin_a
            if self.check_collision(px, py, radius=0.01):
                return r
        return max_range

# ==========================================
# 2. Braitenberg E-puck Swarm Simulation
# ==========================================
class EPuckSwarmSimulation:
    def __init__(self, arena, num_robots=5, k_rep=0.04, base_speed=0.12, max_steps=1200, dt=0.064):
        self.arena = arena
        self.n = num_robots
        self.k_rep = k_rep
        self.base_speed = base_speed
        self.dt = dt
        self.max_steps = max_steps
        self.radius = 0.04 # 40mm radius for e-puck
        
        # Sensor angles on e-puck (8 IR distance sensors around the chassis)
        # ps0: ~75 deg, ps1: ~45 deg, ps2: ~0 deg, ps3: ~-20 deg (right side)
        # ps4: ~-160 deg, ps5: ~-180 deg, ps6: ~135 deg, ps7: ~105 deg (left/rear side)
        # Front sensors: right=[ps0, ps1, ps2], left=[ps7, ps6, ps5]
        self.sensor_offsets = np.array([
            np.deg2rad(15), np.deg2rad(45), np.deg2rad(90), np.deg2rad(150),
            np.deg2rad(210), np.deg2rad(270), np.deg2rad(315), np.deg2rad(345)
        ])
        
        # Initialize robot states [x, y, theta]
        self.positions = []
        self.thetas = []
        for i in range(self.n):
            while True:
                x = np.random.uniform(-arena.width*0.4, arena.width*0.4)
                y = np.random.uniform(-arena.height*0.4, arena.height*0.4)
                if not arena.check_collision(x, y, radius=self.radius * 2):
                    self.positions.append([x, y])
                    self.thetas.append(np.random.uniform(-np.pi, np.pi))
                    break
        self.positions = np.array(self.positions)
        self.thetas = np.array(self.thetas)
        
    def step(self):
        speeds = []
        for i in range(self.n):
            x, y = self.positions[i]
            theta = self.thetas[i]
            
            # Read 8 IR sensors
            readings = []
            for offset in self.sensor_offsets:
                r = self.arena.raycast([x, y], theta + offset, max_range=0.25)
                # E-puck proximity sensors give high output for closer objects
                proximity = max(0.0, 1.0 - (r / 0.25))
                readings.append(proximity)
            readings = np.array(readings)
            
            # Braitenberg matrix weights (vehicle type 2b - aggressive obstacle avoidance / wander)
            # Front right sensors push left motor forward, front left push right motor forward
            # right sensors: readings[0], readings[1], readings[2]
            # left sensors:  readings[7], readings[6], readings[5]
            left_sens = readings[7] * 1.5 + readings[6] * 1.0 + readings[5] * 0.5
            right_sens = readings[0] * 1.5 + readings[1] * 1.0 + readings[2] * 0.5
            
            v_l = self.base_speed + self.k_rep * left_sens - self.k_rep * right_sens * 1.2
            v_r = self.base_speed + self.k_rep * right_sens - self.k_rep * left_sens * 1.2
            
            # Differential drive kinematics
            v = (v_r + v_l) / 2.0
            omega = (v_r - v_l) / 0.052 # e-puck axle length ~52mm
            
            nx = x + v * np.cos(theta) * self.dt
            ny = y + v * np.sin(theta) * self.dt
            ntheta = theta + omega * self.dt
            
            # Collision resolution / stuck handling
            if self.arena.check_collision(nx, ny, radius=self.radius):
                # Bumped or blocked
                speeds.append(0.0)
                # Turn away
                self.thetas[i] += np.random.uniform(0.5, 1.5)
            else:
                self.positions[i] = [nx, ny]
                self.thetas[i] = (ntheta + np.pi) % (2 * np.pi) - np.pi
                speeds.append(abs(v))
                
        return np.copy(self.positions), speeds

    def run_simulation(self):
        traj = []
        all_speeds = []
        for _ in range(self.max_steps):
            pos, spd = self.step()
            traj.append(pos)
            all_speeds.append(spd)
        return np.array(traj), np.array(all_speeds) # traj: [steps, n, 2]

# ==========================================
# 3. Pioneer 3-DX Follow-The-Gap Navigation
# ==========================================
class PioneerFTGMSimulation:
    def __init__(self, arena, gap_threshold=1.0, max_speed=0.5, bubble_radius=0.35, max_steps=1200, dt=0.05):
        self.arena = arena
        self.gap_threshold = gap_threshold
        self.max_speed = max_speed
        self.bubble_radius = bubble_radius
        self.dt = dt
        self.max_steps = max_steps
        self.radius = 0.25 # Pioneer 3-DX equivalent radius
        
        # Initial pose and target goal
        self.pos = np.array([-2.2, -2.2])
        self.theta = 0.0
        self.target = np.array([2.2, 2.2])
        
        # Simulated planar Lidar (180 deg FOV, 1 deg resolution: 181 beams)
        self.lidar_angles = np.linspace(-np.pi/2, np.pi/2, 181)
        
    def step(self):
        x, y = self.pos
        # Read Lidar scan
        ranges = []
        for a in self.lidar_angles:
            ranges.append(self.arena.raycast([x, y], self.theta + a, max_range=4.0))
        ranges = np.array(ranges)
        
        # 1. Find nearest obstacle
        min_idx = np.argmin(ranges)
        min_dist = ranges[min_idx]
        
        # 2. Place Safety Bubble around closest obstacle
        processed_ranges = ranges.copy()
        if min_dist < 1.5:
            bubble_angle_span = int(np.rad2deg(np.arcsin(min(1.0, self.bubble_radius / max(0.1, min_dist)))))
            low_idx = max(0, min_idx - bubble_angle_span)
            high_idx = min(len(ranges) - 1, min_idx + bubble_angle_span)
            processed_ranges[low_idx:high_idx+1] = 0.0
            
        # 3. Find gaps (consecutive range points > gap_threshold)
        gaps = []
        in_gap = False
        start_idx = 0
        for i, r in enumerate(processed_ranges):
            if r > self.gap_threshold and not in_gap:
                in_gap = True
                start_idx = i
            elif r <= self.gap_threshold and in_gap:
                in_gap = False
                gaps.append((start_idx, i - 1))
        if in_gap:
            gaps.append((start_idx, len(processed_ranges) - 1))
            
        # Choose best heading: direction towards largest gap or gap closest to target goal
        goal_vec = self.target - self.pos
        goal_angle = np.arctan2(goal_vec[1], goal_vec[0]) - self.theta
        goal_angle = (goal_angle + np.pi) % (2 * np.pi) - np.pi
        
        best_heading = 0.0
        speed = self.max_speed
        
        if len(gaps) > 0:
            # Pick gap with widest angular opening
            widest_gap = max(gaps, key=lambda g: g[1] - g[0])
            mid_idx = (widest_gap[0] + widest_gap[1]) // 2
            best_heading = self.lidar_angles[mid_idx]
            
            # Blend slightly with target heading if gap is open enough
            if processed_ranges[mid_idx] > 2.0:
                best_heading = 0.7 * best_heading + 0.3 * np.clip(goal_angle, -np.pi/2, np.pi/2)
        else:
            # Emergency rotation if completely blocked
            best_heading = np.pi / 2
            speed = 0.05
            
        # Kinematic update
        omega = np.clip(best_heading * 2.0, -1.5, 1.5)
        # Slow down during sharp turns or tight spots
        actual_v = speed * max(0.1, np.cos(best_heading)) * min(1.0, min_dist / 1.0)
        
        nx = x + actual_v * np.cos(self.theta) * self.dt
        ny = y + actual_v * np.sin(self.theta) * self.dt
        ntheta = self.theta + omega * self.dt
        
        if self.arena.check_collision(nx, ny, radius=self.radius):
            actual_v = 0.0
            self.theta += 0.2 # Turn around in place
        else:
            self.pos = np.array([nx, ny])
            self.theta = (ntheta + np.pi) % (2 * np.pi) - np.pi
            
        # Reached goal check
        if np.linalg.norm(self.target - self.pos) < 0.3:
            # Pick a new random target across the map to keep navigating
            self.target = np.array([np.random.uniform(-2.2, 2.2), np.random.uniform(-2.2, 2.2)])
            
        return self.pos.copy(), actual_v

    def run_simulation(self):
        traj = []
        speeds = []
        for _ in range(self.max_steps):
            p, s = self.step()
            traj.append(p)
            speeds.append(s)
        return np.array(traj), np.array(speeds)

# ==========================================
# 4. Metrics Computation
# ==========================================
def compute_metrics(trajectories, speeds, arena, grid_res=0.1, speed_stuck_threshold=0.02):
    """
    Computes:
    - Coverage percentage of arena
    - Mean linear speed
    - Total stuck time (speed < threshold)
    """
    # Grid coverage
    half_w = arena.width / 2.0
    half_h = arena.height / 2.0
    grid_x = int(arena.width / grid_res)
    grid_y = int(arena.height / grid_res)
    grid = np.zeros((grid_x, grid_y), dtype=bool)
    
    # Flatten trajectory across all robots/timesteps
    all_pts = trajectories.reshape(-1, 2)
    for pt in all_pts:
        gx = int((pt[0] + half_w) / grid_res)
        gy = int((pt[1] + half_h) / grid_res)
        if 0 <= gx < grid_x and 0 <= gy < grid_y:
            grid[gx, gy] = True
            
    # Mark obstacle cells as excluded from total explorable area
    free_cells = 0
    covered_cells = 0
    for i in range(grid_x):
        for j in range(grid_y):
            cx = (i + 0.5) * grid_res - half_w
            cy = (j + 0.5) * grid_res - half_h
            if not arena.check_collision(cx, cy, radius=0.0):
                free_cells += 1
                if grid[i, j]:
                    covered_cells += 1
                    
    coverage_pct = (covered_cells / max(1, free_cells)) * 100.0
    mean_speed = float(np.mean(speeds))
    stuck_time_ratio = float(np.mean(speeds < speed_stuck_threshold)) * 100.0
    
    return {
        "coverage_pct": coverage_pct,
        "mean_speed": mean_speed,
        "stuck_time_pct": stuck_time_ratio
    }

# ==========================================
# 5. Parameter Sensitivity Experiments
# ==========================================
def run_experiments():
    arena = Arena2D()
    print("--- 1. Running E-Puck Swarm Braitenberg Parameter Sweep (Repulsion Gain k_rep) ---")
    k_values = [0.01, 0.04, 0.08, 0.15]
    epuck_results = []
    
    for k in k_values:
        sim = EPuckSwarmSimulation(arena, num_robots=5, k_rep=k, base_speed=0.12, max_steps=1000)
        traj, speeds = sim.run_simulation()
        m = compute_metrics(traj, speeds, arena)
        m["k_rep"] = k
        epuck_results.append((k, traj, m))
        print(f"k_rep={k:4.2f} -> Coverage: {m['coverage_pct']:5.2f}%, Mean Speed: {m['mean_speed']:5.3f} m/s, Stuck: {m['stuck_time_pct']:5.2f}%")

    print("\n--- 2. Running Pioneer 3-DX FTGM Parameter Sweep (Gap Threshold) ---")
    gap_thresholds = [0.6, 1.0, 1.5, 2.2]
    pioneer_results = []
    
    for g in gap_thresholds:
        sim = PioneerFTGMSimulation(arena, gap_threshold=g, max_speed=0.45, max_steps=1000)
        traj, speeds = sim.run_simulation()
        m = compute_metrics(traj, speeds, arena)
        m["gap_threshold"] = g
        pioneer_results.append((g, traj, m))
        print(f"Gap_Thresh={g:4.1f}m -> Coverage: {m['coverage_pct']:5.2f}%, Mean Speed: {m['mean_speed']:5.3f} m/s, Stuck: {m['stuck_time_pct']:5.2f}%")

    # Generate Visualization Figures
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    
    # Row 1: E-Puck Braitenberg Plots
    k_vals = [r[0] for r in epuck_results]
    cov_e = [r[2]["coverage_pct"] for r in epuck_results]
    spd_e = [r[2]["mean_speed"] for r in epuck_results]
    stuck_e = [r[2]["stuck_time_pct"] for r in epuck_results]
    
    axes[0, 0].plot(k_vals, cov_e, marker='o', color='royalblue', lw=2)
    axes[0, 0].set_title("E-Puck Swarm: Coverage vs Repulsion Gain ($k_{rep}$)")
    axes[0, 0].set_xlabel("Braitenberg Gain $k_{rep}$")
    axes[0, 0].set_ylabel("Arena Coverage (%)")
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)
    
    axes[0, 1].plot(k_vals, spd_e, marker='s', color='forestgreen', lw=2)
    axes[0, 1].set_title("E-Puck Swarm: Mean Speed vs $k_{rep}$")
    axes[0, 1].set_xlabel("Braitenberg Gain $k_{rep}$")
    axes[0, 1].set_ylabel("Mean Speed (m/s)")
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)
    
    axes[0, 2].plot(k_vals, stuck_e, marker='^', color='crimson', lw=2)
    axes[0, 2].set_title("E-Puck Swarm: Stuck Time vs $k_{rep}$")
    axes[0, 2].set_xlabel("Braitenberg Gain $k_{rep}$")
    axes[0, 2].set_ylabel("Stuck Time (%)")
    axes[0, 2].grid(True, linestyle="--", alpha=0.6)
    
    # Row 2: Pioneer 3-DX FTGM Plots
    g_vals = [r[0] for r in pioneer_results]
    cov_p = [r[2]["coverage_pct"] for r in pioneer_results]
    spd_p = [r[2]["mean_speed"] for r in pioneer_results]
    stuck_p = [r[2]["stuck_time_pct"] for r in pioneer_results]
    
    axes[1, 0].plot(g_vals, cov_p, marker='o', color='navy', lw=2)
    axes[1, 0].set_title("Pioneer 3-DX: Coverage vs Gap Threshold")
    axes[1, 0].set_xlabel("Lidar Gap Threshold (m)")
    axes[1, 0].set_ylabel("Arena Coverage (%)")
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)
    
    axes[1, 1].plot(g_vals, spd_p, marker='s', color='darkgreen', lw=2)
    axes[1, 1].set_title("Pioneer 3-DX: Mean Speed vs Gap Threshold")
    axes[1, 1].set_xlabel("Lidar Gap Threshold (m)")
    axes[1, 1].set_ylabel("Mean Speed (m/s)")
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)
    
    axes[1, 2].plot(g_vals, stuck_p, marker='^', color='firebrick', lw=2)
    axes[1, 2].set_title("Pioneer 3-DX: Stuck Time vs Gap Threshold")
    axes[1, 2].set_xlabel("Lidar Gap Threshold (m)")
    axes[1, 2].set_ylabel("Stuck Time (%)")
    axes[1, 2].grid(True, linestyle="--", alpha=0.6)
    
    plot_path = os.path.abspath("swarm_pioneer_parameter_analysis.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n[+] Analysis figures saved to: {plot_path}")

    # Generate Trajectory Plot
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Swarm Trajectories
    best_epuck_traj = epuck_results[1][1] # k_rep = 0.04
    for ob in arena.obstacles:
        rect = plt.Rectangle((ob[0], ob[1]), ob[2]-ob[0], ob[3]-ob[1], color='gray', alpha=0.7)
        ax1.add_patch(rect)
    for rob_i in range(best_epuck_traj.shape[1]):
        ax1.plot(best_epuck_traj[:, rob_i, 0], best_epuck_traj[:, rob_i, 1], label=f"e-puck {rob_i+1}")
    ax1.set_xlim(-arena.width/2, arena.width/2)
    ax1.set_ylim(-arena.height/2, arena.height/2)
    ax1.set_title("E-Puck Swarm Braitenberg Trajectories ($k_{rep}=0.04$)")
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right', fontsize=8)
    
    # Pioneer Trajectories
    best_pioneer_traj = pioneer_results[1][1] # gap = 1.0m
    for ob in arena.obstacles:
        rect = plt.Rectangle((ob[0], ob[1]), ob[2]-ob[0], ob[3]-ob[1], color='gray', alpha=0.7)
        ax2.add_patch(rect)
    ax2.plot(best_pioneer_traj[:, 0], best_pioneer_traj[:, 1], color='purple', lw=1.8, label="Pioneer 3-DX Path")
    ax2.set_xlim(-arena.width/2, arena.width/2)
    ax2.set_ylim(-arena.height/2, arena.height/2)
    ax2.set_title("Pioneer 3-DX FTGM Trajectory (Gap Threshold=1.0m)")
    ax2.set_aspect('equal')
    ax2.legend(loc='upper right', fontsize=8)
    
    traj_path = os.path.abspath("swarm_pioneer_trajectories.png")
    fig2.savefig(traj_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[+] Trajectory comparison saved to: {traj_path}")

if __name__ == "__main__":
    run_experiments()
