# xbox_drive.py
from simulator_interface.CanInterInterface import SimpleCANBridge
from simulator_interface.CarInfo import get_ros_subscriber

import numpy as np
import pygame
import time
import csv
from datetime import datetime
from pathlib import Path

from controller.layer2 import layer2class
# ==============================
# Config
# ==============================
rate_hz = 20
dt = 1.0 / rate_hz
tire_radius = 0.31

# Deadzones and scaling
STEER_DEADZONE = 0.05
TRIGGER_DEADZONE = 0.02
STEER_SCALE = 1.0          # keep at 1.0; layer2 handles its own limits
PEDAL_SCALE = 1.0          # keep at 1.0; maps to [-1, 1] (brake..throttle)

# Axis indices (commonly correct for Xbox pads on pygame)
AXIS_STEER = 0  # left stick X
AXIS_LT = 2     # left trigger: -1 at rest, +1 pressed
AXIS_RT = 5     # right trigger: -1 at rest, +1 pressed

# Button indices (commonly correct)
BTN_A = 0
BTN_B = 1
BTN_LB = 4
BTN_RB = 5
BTN_BACK = 6
BTN_START = 7

# ==============================
# Helpers
# ==============================
def apply_deadzone(x, dz):
    return 0.0 if abs(x) < dz else x

def axis_to_01(v):
    """Convert trigger axis from [-1, 1] (rest=-1, pressed=+1) to [0, 1]."""
    return max(0.0, min(1.0, (v + 1.0) * 0.5))

def traction_control(pedals, real_speed, car_can):
    """Keep your original TC logic with minor refactor to pass car_can."""
    factor = 0.95
    front_wheel_speed = (car_can.wheel_fl * tire_radius + car_can.wheel_fr * tire_radius) / 2
    rear_wheel_speed = (car_can.wheel_rr * tire_radius + car_can.wheel_rl * tire_radius) / 2
    if (rear_wheel_speed > 0.0 and real_speed > 0.0):
        if pedals > 0.0:
            if front_wheel_speed / rear_wheel_speed < factor:
                pedals = 0.0
        if pedals < 0.0:
            pedals = pedals * 0.8
            if (rear_wheel_speed / real_speed < 0.8):
                pedals = 0.0
    return pedals

def init_gamepad():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No gamepad detected. Plug in an Xbox controller and try again.")
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"[Gamepad] Connected: {js.get_name()} | axes={js.get_numaxes()} buttons={js.get_numbuttons()}")
    return js

def read_gamepad(js, gear, emergency_stop_flag):
    """
    Returns (steer_cmd, accel_cmd, gear, emergency_stop_flag, quit_flag)
    - steer_cmd in [-1, 1]
    - accel_cmd in [-1, 1]  (negative = brake, positive = throttle)
    """
    quit_flag = False
    pygame.event.pump()  # update internal state

    # Read axes with protection
    try:
        steer_raw = js.get_axis(AXIS_STEER)  # [-1, 1]
    except Exception:
        steer_raw = 0.0

    try:
        lt_raw = js.get_axis(5)  # [-1, 1]
        rt_raw = js.get_axis(4)  # [-1, 1]
    except Exception:
        lt_raw, rt_raw = -1.0, -1.0

    # Deadzone + scaling for steering
    steer = apply_deadzone(steer_raw, STEER_DEADZONE) * STEER_SCALE
    steer = np.clip(steer, -1.0, 1.0)

    # Convert triggers to [0,1], apply deadzone
    throttle = ((rt_raw + 1)/2)
    brake = ((lt_raw + 1)/2)

    

    # Accel command: throttle - brake (∈ [-1, 1])
    accel_cmd = np.clip((throttle - brake) * PEDAL_SCALE, -1.0, 1.0)

    # Buttons for gear & controls
    try:
        btn_lb = js.get_button(BTN_LB)
        btn_rb = js.get_button(BTN_RB)
        btn_a = js.get_button(BTN_A)
        btn_b = js.get_button(BTN_B)
        btn_start = js.get_button(BTN_START)
        btn_back = js.get_button(BTN_BACK)
    except Exception:
        btn_lb = btn_rb = btn_a = btn_b = btn_start = btn_back = 0

    # Emergency stop latch when START pressed
    if btn_start:
        emergency_stop_flag = True
    # Clear emergency stop if both triggers are fully released and A is pressed
    if emergency_stop_flag and (throttle < 0.05 and brake < 0.05 and btn_a):
        emergency_stop_flag = False

    # Gear logic
    if btn_rb:
        gear += 1
    if btn_lb:
        gear -= 1
    if btn_a:
        gear = 1        # drive
    if btn_b:
        gear = -1       # reverse

    # Clamp gear to reasonable range
    gear = int(np.clip(gear, -1, 6))

    # Quit on BACK
    if btn_back:
        quit_flag = True

    # Apply emergency stop
    if emergency_stop_flag:
        accel_cmd = 0.0

    return steer, accel_cmd, gear, emergency_stop_flag, quit_flag

# ==============================
# Main
# ==============================
def main():
    # Car I/O and ROS
    car_can = SimpleCANBridge()
    ros = get_ros_subscriber()
    # Control layers
    layer2 = layer2class(30)

    # Previous position for movement threshold
    y = 0.0
    x = 0.0

    # States
    steer_angle = 0.0
    pedals = 0.0
    gear = 1
    emergency_stop_flag = False

    # --- CSV logging setup ---
    t0 = time.time()
    log_rows = []
    csv_dir = Path("./logs")
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"drive_points_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    #header = ["steer_angle","speed","orientation_rate"]
    header = ["x","y"]
    # Gamepad
    try:
        js = init_gamepad()
    except RuntimeError as e:
        print(e)
        return

    print("[Controls] LS-X steer | RT throttle | LT brake | RB/LB gear +/- | A: gear=1 | B: reverse | Start: E-stop | Back: quit")
    orientation = None
    try:
        while True:
            loop_start = time.time()

            # === Read gamepad and produce commands ===
            steer_cmd, accel_cmd, gear, emergency_stop_flag, quit_flag = read_gamepad(js, gear, emergency_stop_flag)
            if quit_flag:
                print("[Exit] Back button pressed.")
                break

            # layer2.step expects (-1..1 steering cmd, -1..1 accel cmd)
            steer_angle, pedals, gear = layer2.step(
                steer_cmd,
                accel_cmd,
                ros.get_speed(),
                car_can.rpm,
                gear,
                3.5 
            )

            # Optional traction control using wheel/speed feedback
            pedals = traction_control(pedals, ros.get_speed(), car_can)

            # Send to vehicle
            car_can.set_steer_angle(steer_angle)  # keep your 2x factor
            car_can.set_pedals(pedals)
            car_can.set_gear(gear)

            # --- Log point when moved more than 0.3 m since last logged point ---
            #if orientation is not None:
            #    orientation_rate = (orientation-ros.get_orientation())/dt
            #    log_rows.append([abs(steer_angle),ros.get_speed(),abs(orientation_rate)])   
            #orientation = ros.get_orientation()


            if np.linalg.norm([ros.get_position()[0] - x, ros.get_position()[1]- y]) > 0.1:
            #   #print(node.posx, node.posy)
                x = ros.get_position()[0]
                y = ros.get_position()[1]
                log_rows.append([x, y])
                print([x, y])
            
            # Timing
            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n[Exit] Keyboard interrupt.")
    finally:
        # --- Write CSV on exit ---
        try:
            if log_rows:
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(log_rows)
                print(f"[Log] Saved {len(log_rows)} points to: {csv_path.resolve()}")
            else:
                print("[Log] No points recorded; CSV not created.")
        except Exception as e:
            print(f"[Log] Failed to write CSV: {e}")

        pygame.quit()

if __name__ == "__main__":
    main()
