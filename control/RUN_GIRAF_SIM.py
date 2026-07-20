"""
GIRAF Manipulator Simulation with Joystick Teleop
Simplified simulation version of RUN_task_space_teleop.py
"""

import numpy as np
import mujoco
import mujoco.viewer
import threading
import time
import os
import sys
import cv2

from kinematic_model import num_jacobian, num_forward_kinematics
from joystick_driver import joystick_connect, joystick_read, joystick_disconnect

## ----------------------------------------------------------------------------------------------------
# Global Variables
## ----------------------------------------------------------------------------------------------------
# Dispense mode configuration
DISPENSE_TARGET = np.array([1.0, 0.5, 0.35])  # 0.5m above box center (adjust X,Y as needed)
DISPENSE_LIFT_HEIGHT = 0.35  # Height to lift to before moving horizontally
DISPENSE_KP = 5.0  # PD position gain
DISPENSE_KD = 0.1  # PD velocity damping
DISPENSE_MAX_VEL = 0.4  # Maximum velocity (m/s)

joystick_data = {"LX":0, "LY":0, "RX":0, "RY":0, "LT":0, "RT":0, "AB":0, "BB":0, "XB":0, "LB":0, "RB":0, "MENULEFT":0, "MENURIGHT":0}
joystick_lock = threading.Lock()

velocity = np.zeros((6, 1))
velocity_lock = threading.Lock()

data_lock = threading.Lock()  # Protects MuJoCo data access

running = True
running_lock = threading.Lock()

dispense_state = None  # None = teleop, "lifting", "moving", "opening", "waiting"
dispense_timer = 0.0
lift_target = None  # Target for lifting phase
menuright_prev = 0  # For edge detection

## ----------------------------------------------------------------------------------------------------
# Joystick Monitoring Thread
## ----------------------------------------------------------------------------------------------------
def joystick_monitor():
    global joystick_data, running

    js = joystick_connect()
    print("\033[93mSIM: Joystick Connected!\033[0m")

    while running:
        with joystick_lock:
            joystick_data = joystick_read(js)
        time.sleep(0.005)

    joystick_disconnect(js)
    print("\033[93mSIM: Joystick Disconnected!\033[0m")

## ----------------------------------------------------------------------------------------------------
# Camera Rendering Thread
## ----------------------------------------------------------------------------------------------------
def camera_render_thread(model, data):
    global running
    
    # Camera specs: OV5648 sensor with 95° HFOV, 16:9 aspect ratio
    # Using 1280x720 (one of the camera's native resolutions)
    width = 1280
    height = 720
    
    # Create renderer for wrist camera
    renderer = mujoco.Renderer(model, height=height, width=width)
    
    print(f"\033[96mSIM: Camera thread started ({width}x{height}, HFOV=95°)\033[0m")
    
    loop_count = 0
    start_time = time.perf_counter()
    fps = 0.0
    
    try:
        while running:
            # Update scene and render from wrist camera
            with data_lock:
                renderer.update_scene(data, camera="wrist_cam")
                rgb = renderer.render()
            
            # Convert to BGR for OpenCV
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            # Calculate FPS
            loop_count += 1
            if loop_count >= 30:
                elapsed = time.perf_counter() - start_time
                fps = loop_count / elapsed
                loop_count = 0
                start_time = time.perf_counter()
            
            cv2.putText(bgr, f"Wrist Camera - {fps:.1f} FPS", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display
            cv2.imshow("GIRAF Wrist Camera", bgr)
            
            # Check for window close or ESC key
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
            
            time.sleep(1/100)  # ~60 FPS for camera
            
    except Exception as e:
        print(f"\033[91mCamera thread error: {e}\033[0m")
    finally:
        cv2.destroyAllWindows()
        print("\033[96mSIM: Camera thread stopped\033[0m")

## ----------------------------------------------------------------------------------------------------
# Main Simulation
## ----------------------------------------------------------------------------------------------------
def main():
    global joystick_data, velocity, running, dispense_state, dispense_timer, lift_target, menuright_prev

    # Load MuJoCo model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", "GIRAF_bananas.xml")
    
    print(f"Loading model from: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    print("\033[92mSIM: Model loaded successfully!\033[0m")
    print(f"  Joints: {model.njnt}")
    print(f"  Actuators: {model.nu}")
    
    # Joint name to index mapping
    joint_names = ['R1', 'R2', 'P3', 'R4', 'R5', 'R6', 'left_grip_joint', 'right_grip_joint']
    joint_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names}
    
    # Actuator name to index mapping  
    actuator_names = ['actuator_R1', 'actuator_R2', 'actuator_P3', 'actuator_R4', 'actuator_R5', 'actuator_R6', 'actuator_left_grip', 'actuator_right_grip']
    actuator_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in actuator_names}
    
    def inverse_jacobian(joint_coords):
        """Compute inverse Jacobian using kinematic model"""
        J = num_jacobian(joint_coords)
        J_inv = np.linalg.pinv(J)
        return J_inv
    
    # Initialize joint positions (matching hardware teleop initial values)
    roll_pos = 0.0
    roll_offset = 0.0
    pitch_pos = 0.0
    d3_pos = 0.25  # Start at 0.25m extension
    theta4_pos = 0.0
    theta5_pos = 0.0
    theta6_pos = 0.0
    gripper_pos = 0.0  # 0 = open, 0.05 = closed
    
    # Set initial state
    data.qpos[joint_ids['R1']] = roll_pos
    data.qpos[joint_ids['R2']] = pitch_pos
    data.qpos[joint_ids['P3']] = d3_pos
    data.qpos[joint_ids['R4']] = theta4_pos
    data.qpos[joint_ids['R5']] = theta5_pos
    data.qpos[joint_ids['R6']] = theta6_pos
    data.qpos[joint_ids['left_grip_joint']] = gripper_pos
    data.qpos[joint_ids['right_grip_joint']] = gripper_pos
    
    mujoco.mj_forward(model, data)
    
    print("\033[92mSIM: Ready! Controls:\033[0m")
    print("  LB + RB: Enable control")
    print("  Left stick: XY translation")
    print("  Right stick: Wrist rotation")
    print("  LT/RT: Z translation")
    print("  A/B: Close/Open gripper")
    print("  Menu Left: Roll offset +")
    print("  Menu Right: Auto-dispense (move to target, open, wait)")
    print("  X: Exit")
    print("\n" * 10)  # Reserve space for status display
    
    # Start joystick thread
    joystick_thread = threading.Thread(target=joystick_monitor, daemon=True)
    joystick_thread.start()
    
    # Start camera rendering thread
    camera_thread = threading.Thread(target=camera_render_thread, args=(model, data), daemon=True)
    camera_thread.start()
    
    # Performance monitoring
    loop_count = 0
    start_time = time.perf_counter()
    last_print_time = time.perf_counter()
    loop_hz = 0.0
    
    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and running:
            # Unpack joystick data
            with joystick_lock:
                LX = joystick_data["LX"]
                LY = joystick_data["LY"]
                RX = joystick_data["RX"]
                RY = joystick_data["RY"]
                LT = joystick_data["LT"]
                RT = joystick_data["RT"]
                AB = joystick_data["AB"]
                BB = joystick_data["BB"]
                XB = joystick_data["XB"]
                LB = joystick_data["LB"]
                RB = joystick_data["RB"]
                MENULEFT = joystick_data["MENULEFT"]
                MENURIGHT = joystick_data["MENURIGHT"]
            
            if XB:  # Exit
                with running_lock:
                    running = False
                break
            
            # Check for dispense trigger (menu_right edge detection)
            if MENURIGHT and not menuright_prev:
                dispense_state = "lifting"
                print("\033[93m[DISPENSE] Mode activated - lifting to safe height\033[0m")
            menuright_prev = MENURIGHT
            
            # ---------- Compute FK first (needed for both teleop and dispense) ----------
            # Read actual joint positions from simulation
            actual_roll = data.qpos[joint_ids['R1']]
            actual_pitch = data.qpos[joint_ids['R2']]
            actual_d3 = data.qpos[joint_ids['P3']]
            actual_theta4 = data.qpos[joint_ids['R4']]
            actual_theta5 = data.qpos[joint_ids['R5']]
            actual_theta6 = data.qpos[joint_ids['R6']]

            FK_mat = num_forward_kinematics([
                actual_roll, 
                actual_pitch + np.pi/2, 
                actual_d3,
                actual_theta4 + np.pi/2, 
                actual_theta5 + 5*np.pi/6, 
                actual_theta6
            ])
            
            # Safety interlock (only active if NOT in dispense mode)
            if LB and RB and dispense_state is None:
                with velocity_lock:
                    velocity[0] = 0.25 * LY  # X velocity
                    velocity[1] = 0.25 * -LX  # Y velocity
                    velocity[4] = -0.5 * RY    # WY angular velocity
                    velocity[5] = -0.5 * RX   # WZ angular velocity
                
                if RT and not LT:  # Z up
                    with velocity_lock:
                        velocity[2] = 0.25 * RT
                elif LT and not RT and (pitch_pos > 0):  # Z down
                    with velocity_lock:
                        velocity[2] = -0.25 * LT
                else:
                    with velocity_lock:
                        velocity[2] = 0
                
                # Gripper control
                if AB and not BB:  # Close
                    gripper_velocity = 0.001  # m/s
                elif BB and not AB:  # Open
                    gripper_velocity = -0.001
                else:
                    gripper_velocity = 0
                
                # Roll offset
                if MENULEFT:
                    roll_offset += 0.0025
            else:
                with velocity_lock:
                    velocity = np.zeros((6, 1))
                gripper_velocity = 0
            
            # ------- DISPENSE MODE STATE MACHINE -------
            if dispense_state is not None:
                if dispense_state == "lifting":
                    # Get current gripper position from FK
                    current_pos = FK_mat[:3, 3]
                    
                    # Set lift target on first entry (keep current x,y, go to safe z)
                    if lift_target is None:
                        lift_target = np.array([current_pos[0], current_pos[1], DISPENSE_LIFT_HEIGHT])
                        print(f"\033[93m[DISPENSE] Lifting from z={current_pos[2]:.2f} to z={DISPENSE_LIFT_HEIGHT:.2f}\033[0m")
                    
                    position_error = lift_target - current_pos
                    distance = np.linalg.norm(position_error)
                    
                    # PD control for position (linear velocities only)
                    v_cmd = (DISPENSE_KP * position_error).reshape(3, 1)
                    v_norm = np.linalg.norm(v_cmd)
                    if v_norm > DISPENSE_MAX_VEL:
                        v_cmd = v_cmd * (DISPENSE_MAX_VEL / v_norm)  # Clamp to max velocity
                    
                    with velocity_lock:
                        velocity[0:3] = v_cmd
                        velocity[3:6] = 0  # No angular velocity
                    
                    # Check if reached lift height (within 2cm)
                    if distance < 0.02:
                        dispense_state = "moving"
                        lift_target = None  # Reset for next time
                        print("\033[93m[DISPENSE] Reached safe height - moving to dispense target\033[0m")
                
                elif dispense_state == "moving":
                    # Get current gripper position from FK
                    current_pos = FK_mat[:3, 3]
                    position_error = DISPENSE_TARGET - current_pos
                    distance = np.linalg.norm(position_error)
                    
                    # PD control for position (linear velocities only)
                    v_cmd = (DISPENSE_KP * position_error).reshape(3, 1)
                    v_norm = np.linalg.norm(v_cmd)
                    if v_norm > DISPENSE_MAX_VEL:
                        v_cmd = v_cmd * (DISPENSE_MAX_VEL / v_norm)  # Clamp to max velocity
                    
                    with velocity_lock:
                        velocity[0:3] = v_cmd
                        velocity[3:6] = 0  # No angular velocity
                    
                    # Check if reached target (within 2cm)
                    if distance < 0.02:
                        dispense_state = "opening"
                        print("\033[93m[DISPENSE] Reached target - opening gripper\033[0m")
                
                elif dispense_state == "opening":
                    # Open gripper
                    gripper_velocity = 0.002  # Fast open
                    with velocity_lock:
                        velocity = np.zeros((6, 1))  # Stop moving
                    
                    # Check if gripper fully open
                    if gripper_pos >= 0.035:
                        dispense_state = "waiting"
                        dispense_timer = time.perf_counter()
                        print("\033[93m[DISPENSE] Gripper open - waiting 0.5s\033[0m")
                
                elif dispense_state == "waiting":
                    # Wait 0.5 seconds
                    gripper_velocity = 0
                    with velocity_lock:
                        velocity = np.zeros((6, 1))
                    
                    if time.perf_counter() - dispense_timer >= 0.5:
                        dispense_state = None
                        print("\033[92m[DISPENSE] Complete - returning to teleop\033[0m")
            
            # ---------- Transform velocity and compute joint velocities ----------
            # Transform velocity from EE frame to world frame (for teleop)
            # In dispense mode, velocity is already in world frame
            if dispense_state is None:
                Rot_mat = FK_mat[:3, :3]
                Rot_block = np.block([[Rot_mat, np.zeros((3, 3))], 
                                      [np.zeros((3, 3)), Rot_mat]])
                # velocity_world = Rot_block @ velocity
                velocity_world = velocity
            else:
                velocity_world = velocity  # Already in world frame from PD controller

            # Compute Jacobian inverse and joint velocities
            Jv_inv = inverse_jacobian([
                actual_roll, 
                actual_pitch + np.pi/2, 
                actual_d3,
                actual_theta4 + np.pi/2, 
                actual_theta5 + 5*np.pi/6, 
                actual_theta6
            ])
            joint_velocity = Jv_inv @ velocity_world
            
            # Integrate joint positions
            dt = 0.0025
            roll_pos += dt * joint_velocity[0, 0]
            pitch_pos += dt * joint_velocity[1, 0]
            d3_pos += dt * joint_velocity[2, 0]
            theta4_pos += dt * joint_velocity[3, 0]
            theta5_pos += dt * joint_velocity[4, 0]
            theta6_pos += dt * joint_velocity[5, 0]
            gripper_pos += gripper_velocity  # Scale for responsiveness
            
            # Apply joint limits
            roll_pos = np.clip(roll_pos, -np.pi/2, np.pi/2)
            pitch_pos = np.clip(pitch_pos, -np.pi/4, np.pi/2)
            d3_pos = np.clip(d3_pos, 0.2, 3.0)
            theta5_pos = max(theta5_pos, -1.7)  # Wrist pitch limit
            gripper_pos = np.clip(gripper_pos, 0.0, 0.05)
            
            # Set control targets (position control via actuators)
            data.ctrl[actuator_ids['actuator_R1']] = roll_pos + roll_offset
            data.ctrl[actuator_ids['actuator_R2']] = pitch_pos
            data.ctrl[actuator_ids['actuator_P3']] = d3_pos
            data.ctrl[actuator_ids['actuator_R4']] = theta4_pos
            data.ctrl[actuator_ids['actuator_R5']] = theta5_pos
            data.ctrl[actuator_ids['actuator_R6']] = theta6_pos
            data.ctrl[actuator_ids['actuator_left_grip']] = gripper_pos
            data.ctrl[actuator_ids['actuator_right_grip']] = gripper_pos
            
            # Step simulation and update viewer (protected by lock)
            with data_lock:
                mujoco.mj_step(model, data)
                viewer.sync()
            
            # Match hardware loop rate
            time.sleep(0.001)
    
    print("\033[93mSIM: Simulation ended\033[0m")

if __name__ == "__main__":
    main()
