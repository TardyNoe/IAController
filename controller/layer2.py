import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
class RealtimeButterworthLPF:
    def __init__(self, fs: float = 30.0, cutoff_hz: float = 5.0, order: int = 4, x0 = 1):
        if cutoff_hz <= 0:
            raise ValueError("cutoff_hz must be > 0.")
        nyq = fs / 2.0
        if cutoff_hz >= nyq:
            raise ValueError(f"cutoff_hz must be < Nyquist ({nyq} Hz) for fs={fs} Hz.")
        if order < 1:
            raise ValueError("order must be >= 1.")
        self.fs = fs
        self.cutoff = cutoff_hz
        self.order = order

        # Design Butterworth LPF in SOS form (stable and robust)
        self.sos = butter(order, cutoff_hz / nyq, btype="low", output="sos")

        # Per-section initial conditions to minimize startup transients
        self.zi = sosfilt_zi(self.sos)
        if x0 is None:
            # Start assuming zero input
            self.z = self.zi * 0.0
        else:
            # Start near steady-state for a constant x0
            self.z = self.zi * x0

    def reset(self, x0 = 1):
        """Reset internal state. If x0 is provided, initialize to steady-state for constant x0."""
        self.z = self.zi * (0.0 if x0 is None else x0)

    def filter_sample(self, x: float) -> float:
        """
        Process a single sample and return the filtered output.
        Keeps state across calls.
        """
        y, self.z = sosfilt(self.sos, np.array([x], dtype=float), zi=self.z)
        return float(y[0])

    def filter_block(self, x_block: np.ndarray) -> np.ndarray:
        """
        Process a block (1D array) of samples and return the filtered block.
        Keeps state across calls, so you can stream blocks back-to-back.
        """
        y, self.z = sosfilt(self.sos, np.asarray(x_block, dtype=float), zi=self.z)
        return y

class layer2class:
    def __init__(self,fs):
        self.fs = fs
        self.fc_pedals = 5
        self.fc_steer = 5
        self.steer_range = 20

        self.max_delta_pedals = 0.5
        self.max_delta_steer = 5

        self.previous_pedals = 0.0
        self.previous_steer = 0.0

        self.filter_pedals = RealtimeButterworthLPF(fs=self.fs, cutoff_hz=self.fc_pedals, order=2, x0=0.0)
        self.filter_steer = RealtimeButterworthLPF(fs=self.fs, cutoff_hz=self.fc_pedals, order=2, x0=0.0)
        
        self.frames_since_gear_change = 0.0
        self.high_rpm = 6500
        self.low_rpm = 4500
        
    def compute_gear(self,rpm,current_gear):
        new_gear = current_gear
        if self.frames_since_gear_change > 20*(self.fs/50):
            if rpm > self.high_rpm:
                new_gear= np.clip(new_gear+1,1,6)
            if rpm < self.low_rpm:
                new_gear= np.clip(new_gear-1,1,6)
        if new_gear != current_gear:
            self.frames_since_gear_change = 0.0
        return new_gear
    def compute_max_steer_angle(self,speed,max_lat_acc): ## to delete
        epsilon = 1e-6
        return np.clip(max_lat_acc/(speed**2+epsilon), -self.steer_range,self.steer_range)
    
    def max_wheel_angle_for_speed(self,speed, wheelbase, mu, g=9.81, mech_limit_deg=None):
        if speed == 0:
            return mech_limit_deg if mech_limit_deg is not None else float('inf')
        x = (wheelbase * mu * g) / (speed * speed)
        delta_deg = np.degrees(np.arctan(x))
        if mech_limit_deg is not None:
            return min(delta_deg, mech_limit_deg)
        return delta_deg

    def compute_max_pedals(self,steer_command,steer_angle):
        steer_ratio = abs(steer_command)
        if steer_ratio == 0:
            return 1.0,1.0
        max_break = np.clip(0.1/(steer_ratio**2),0.1,1)
        max_throttle = np.clip(0.3/((steer_angle/self.steer_range)**2),0.3,1)
        return max_break,max_throttle
        
    def step(self,steer,pedals,speed,rpm,gear,mu):
        gear = self.compute_gear(rpm,gear)
        self.frames_since_gear_change = self.frames_since_gear_change + 1

        #Low pass filter
        filtered_pedals = self.filter_pedals.filter_sample(pedals)
        filtered_steer = self.filter_steer.filter_sample(steer)

        max_steer_angle = self.max_wheel_angle_for_speed(speed, 3.11, mu, mech_limit_deg=20.0)
        
        steer_angle = max_steer_angle*filtered_steer
        
        filtered_pedals = np.clip(filtered_pedals,self.previous_pedals-self.max_delta_pedals,self.previous_pedals+self.max_delta_pedals)
        steer_angle = np.clip(steer_angle,self.previous_steer-self.max_delta_steer,self.previous_steer+self.max_delta_steer)
        max_break,max_throttle = self.compute_max_pedals(filtered_steer,steer_angle)
        filtered_pedals = np.clip(filtered_pedals,-max_break,max_throttle)
        
        self.previous_pedals = filtered_pedals
        self.previous_steer = steer_angle

        return steer_angle,filtered_pedals,gear