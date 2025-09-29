import can
import cantools
import os
import sys
import threading
import time
import numpy as np
class SimpleCANBridge():
    def __init__(self):
        # Load DBC
        dbc_file = 'config/EAV25_CAN2_R01.dbc'

        if not os.path.exists(dbc_file):
            print("DBC file not found.")
            sys.exit(1)

        self.db = cantools.database.load_file(dbc_file)
        self.bus = can.interface.Bus(channel='vcan0', interface='socketcan')

        # state
        self.throttle = 0.0
        self.steer = 0.0
        self.gear = 1
        self.brake_front = 0.0
        self.brake_rear = 0.0
        self.wheel_fl = self.wheel_fr = self.wheel_rl = self.wheel_rr = 0.0
        self.rpm = 0.0

        # periodic tasks
        self._running = True
        self._start_periodic_task(self.receive_can, 0.0001)
        self._start_periodic_task(self.send_commands, 0.001)

    def _start_periodic_task(self, func, interval):
        """Start a periodic task in a separate thread."""
        def loop():
            while self._running:
                start = time.time()
                func()
                elapsed = time.time() - start
                time.sleep(max(0, interval - elapsed))
        t = threading.Thread(target=loop, daemon=True)
        t.start()

    def receive_can(self):
        msg = self.bus.recv(timeout=0.0)
        if msg is None:
            return
        try:
            if msg.arbitration_id == 321:  # wheels speed
                decoded = self.db.decode_message(msg.arbitration_id, msg.data)
                self.wheel_fl = decoded['BSU_WSS_Speed_FL_C2']
                self.wheel_fr = decoded['BSU_WSS_Speed_FR_C2']
                self.wheel_rl = decoded['BSU_WSS_Speed_RL_C2']
                self.wheel_rr = decoded['BSU_WSS_Speed_RR_C2']
            elif msg.arbitration_id == 550:  # engine status
                decoded = self.db.decode_message(msg.arbitration_id, msg.data)
                self.rpm = decoded['BSU_ICE_EngineSpeed_rpm']
        except Exception as e:
            print(f"Decode error: {e}")

    def send_commands(self):
        try:
            # throttle + gear + brakes
            data = {
                'HL_TargetThrottle': int(self.throttle),
                'HL_TargetGear': int(self.gear),
                'HL_TargetPressure_FL': int(self.brake_front),
                'HL_TargetPressure_FR': int(self.brake_front),
                'HL_TargetPressure_RL': int(self.brake_rear),
                'HL_TargetPressure_RR': int(self.brake_rear),
                'HL_Alive_01': 0
            }
            msg = self.db.get_message_by_name("HL_Msg_01")
            encoded = msg.encode(data)
            self.bus.send(can.Message(arbitration_id=257, data=encoded, is_extended_id=False))

            # steering
            data2 = {
                'HL_Alive_02': 0,
                'HL_PSA_Profile_Vel_rad_s': 500,
                'HL_PSA_Profile_Dec': 500,
                'HL_PSA_Profile_Acc': 500,
                'HL_TargetPSAControl': self.steer,
                'HL_PSA_ModeOfOperation': 1
            }
            msg2 = self.db.get_message_by_name("HL_Msg_02")
            encoded2 = msg2.encode(data2)
            self.bus.send(can.Message(arbitration_id=258, data=encoded2, is_extended_id=False))
        except Exception as e:
            print(f"Send error: {e}")

    def stop(self):
        self._running = False
    
    def set_pedals(self,input):
        self.throttle = np.clip(input,0,1)*100
        self.brake_front = np.clip(-input,0,1)*100
        self.brake_rear = np.clip(-input,0,1)*100

    def set_steer_angle(self,angle):
        self.steer = np.clip(-angle*(100.0/(18.7)),-100,100)

    def set_gear(self,gear):
        self.gear = gear