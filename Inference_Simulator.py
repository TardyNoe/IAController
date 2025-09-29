from simulator_interface.CanInterInterface import SimpleCANBridge
from simulator_interface.CarInfo import get_ros_subscriber

import numpy as np
import onnxruntime as ort

from controller.layer0 import layer0class
from controller.layer2 import layer2class
import time

rate_hz = 100
dt = 1.0 / rate_hz  # 0.02 seconds

# Usage
car_can = SimpleCANBridge()


ort_session_overtake = ort.InferenceSession('weights/ppo_model.onnx')



action = np.array([0.0, 0.0], dtype=np.float32)
layer0 = layer0class("track_data/Yas_center.csv")
layer2 = layer2class(rate_hz)

steer_angle = 0.0
pedals = 0.0
gear = 1

tire_radius = 0.31

def traction_control(pedals,real_speed):
    factor = 0.95
    front_wheel_speed = (car_can.wheel_fl*tire_radius + car_can.wheel_fr*tire_radius)/2
    rear_wheel_speed = (car_can.wheel_rr*tire_radius + car_can.wheel_rl*tire_radius)/2
    if (rear_wheel_speed>0.0 and real_speed>0.0):
        if pedals>0.0:
            
                if front_wheel_speed/rear_wheel_speed < factor:
                    pedals = 0.0
        if pedals<0.0:
            pedals = pedals*0.8
            if(rear_wheel_speed/real_speed<0.8):
                pedals = 0.0

    return pedals

ros = get_ros_subscriber()
while True:
    start_time = time.time()
    car_can.set_steer_angle(steer_angle)
    car_can.set_pedals(pedals)
    car_can.set_gear(gear)
    
    mu = 3.5
    center_proxi = 4

    obs,completion, offset,track_orientation = layer0.compute_obs_vector(steer_angle, pedals,ros.get_speed(),ros.get_position()[0],ros.get_position()[1],ros.get_orientation(),
                                                                        mu,
                                                                        1,
                                                                        center_proxi)

    observation = np.expand_dims(obs, axis=0).astype(np.float32)
    action_out, value, log_prob = ort_session_overtake.run(None, {"input": observation})
    action_out[0][0] = np.clip(action_out[0][0],-1,1)
    action_out[0][1] = np.clip(action_out[0][1],-1,1)
    steer_angle, pedals, gear = layer2.step(action_out[0][0],action_out[0][1],ros.get_speed(),car_can.rpm,gear,3.5)
    
    pedals = traction_control(pedals,ros.get_speed())
    elapsed = time.time() - start_time
    sleep_time = dt - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)


