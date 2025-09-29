
import math
import sys
from typing import Tuple, Optional, List, Dict

import numpy as np

import pygame
import gymnasium as gym
from gymnasium import spaces


from controller.layer0 import layer0class
from controller.layer2 import layer2class

def piecewise_linear(x: float, xin: List[float], xout: List[float]) -> float:
    if len(xin) != len(xout):
        raise ValueError("xin and xout must have same length")
    x = float(x)
    if x <= xin[0]:
        return float(xout[0])
    if x >= xin[-1]:
        return float(xout[-1])
    for i in range(1, len(xin)):
        if x < xin[i]:
            t = (x - xin[i-1]) / (xin[i] - xin[i-1] + 1e-12)
            return float(xout[i-1] + t * (xout[i] - xout[i-1]))
    return float(xout[-1])


class ComplexCar:
    """
    Bicycle model with lateral slip (vx, vy, r states) + engine/brakes/aero.
    - Rear-wheel drive (propulsion)
    - **Front braking** with combined-slip limit
    - Separated tire/friction computations for clarity & tuning.
    """
    def __init__(self):
        self.power_factor = 1.0
        self.break_force_factor = 1.0
        

        self.mu = 1.0
        # Body / geometry
        self.mass = 760.0  # kg
        self.max_speed = 100


        # Steering actuator
        self.steer_deg = 0.0

        # Powertrain
        self.final_drive = 2.818
        self.gear_ratios = [3.083, 2.286, 1.7647, 1.421, 1.19, 1.036]
        self.engine_poly = [-1.647e-16, 5.906e-12, -7.246e-8, 3.66e-4, -0.648, 524.881]
        self.throttle_in = [0.0, 0.33, 0.66, 1.0]
        self.throttle_out = [0.0, 0.5, 0.8, 1.0]
        self.engine_rpm_max = 7000.0
        self.engine_rpm_min = 1500.0
        self.engine_rpm_noise = 10.0
        self.engine_friction_torque = 30.0
        self.drivetrain_eff = 0.90

        # Brakes
        self.brake_max_kpa = 160.0
        self.kpa_to_nm = 20.0

        # Tires / contact
        self.tire_radius = 0.31

        # Aero
        self.frontal_area = 1.0
        self.cd = 1.194
        self.air_density = 0.6115

        # Integration
        self.dt = 1.0/30.0

        # State (body frame velocities)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0  # rad
        self.vx = 0.1   # m/s tiny to avoid div0
        self.vy = 0.0
        self.r = 0.0    # yaw rate
        self.gear = 1
        self.engine_rpm = 1500.0
        self.throttle_cmd = 0.0  # [-1,1], negative = brake

        # Debug / display
        self.last: Dict[str, float] = {}

    def reset(self, x, y, yaw_deg=0.0, v=0.0):
        self.x, self.y = float(x), float(y)
        self.yaw = math.radians(yaw_deg)
        self.vx = max(0.1, float(v))
        self.vy = 0.0
        self.r = 0.0
        self.steer_deg = 0.0
        self.gear = 1
        self.engine_rpm = max(self.engine_rpm_min, 1500.0)
        self.throttle_cmd = 0.0

    # ------- helpers -------
    def _engine_torque_from_rpm(self, rpm: float, throttle01: float) -> float:
        r = float(np.clip(rpm, 0.0, self.engine_rpm_max + 100.0))
        a5, a4, a3, a2, a1, a0 = self.engine_poly
        T_peak = (((((a5*r + a4)*r + a3)*r + a2)*r + a1)*r + a0)
        T_peak = max(0.0, T_peak)
        t_map = piecewise_linear(throttle01, self.throttle_in, self.throttle_out)
        T = T_peak * t_map - self.engine_friction_torque
        return max(0.0, T)

    def set_throttle_brake(self, cmd: float):
        self.throttle_cmd = float(np.clip(cmd, -1.0, 1.0))

    def set_steer(self, steer: float):
        self.steer_deg = steer

    def _compute_engine_rpm_from_speed(self, vx_body: float) -> float:
        rpm = (vx_body / max(1e-6, self.tire_radius)) * (self.final_drive * self.gear_ratios[self.gear-1]) * (60.0 / (2.0*math.pi))
        return max(self.engine_rpm_min, rpm)



    # ---------------- Step ----------------
    def step(self):
        m = self.mass
        delta = math.radians(self.steer_deg)
        cmd = self.throttle_cmd
        vx = max(0.1, self.vx)

        # --- Engine / brake ---
        rpm = self._compute_engine_rpm_from_speed(vx)
        rpm = float(np.clip(rpm + np.random.randn()*self.engine_rpm_noise, self.engine_rpm_min, self.engine_rpm_max + 100.0))
        self.engine_rpm = rpm

        throttle01 = max(0.0, cmd)
        brake01    = max(0.0, -cmd)

        T_engine = self._engine_torque_from_rpm(rpm, throttle01)*self.power_factor
        if self.engine_rpm > self.engine_rpm_max:
            T_engine = 0.0  # cut
        drive_ratio = self.final_drive * self.gear_ratios[self.gear-1]
       # Wheel drive force (rear)
        wheel_torque   = T_engine * drive_ratio * self.drivetrain_eff
        Fx_drive_rear  = wheel_torque / max(1e-6, self.tire_radius)

        # Braking — split by bias (front-heavy is stable)
        brake_kpa      = brake01 * self.brake_max_kpa * self.break_force_factor
        brake_torque   = brake_kpa * self.kpa_to_nm

        # Forces (negative)
        Fx_brake = -(brake_torque) / max(1e-6, self.tire_radius)


        # --- Resistances for X force ---
        speed = math.hypot(self.vx, self.vy)
        F_aero = 0.5 * self.air_density * self.cd * self.frontal_area * (speed**2)
        F_roll = 60.0

        # --- Vehicle dynamics (body frame) ---
        sum_Fx = Fx_drive_rear + Fx_brake - F_aero - F_roll

        vx_dot = (sum_Fx + m * self.vy * self.r) / m
        self.vx =  max(0.0, self.vx + vx_dot * self.dt)
        self.vx = min(self.vx,self.max_speed/3.6)

        # delta is steer angle in randians (carefull degrees in file)
        # speed is vx
        def predict_orientation_rate(steer_angle: float, speed: float) -> float:
            """
            Predict orientation rate using hardcoded 3rd-degree polynomial coefficients.
            R² score on training data = 0.8062
            """
            sign = 1
            if steer_angle<0:
                sign = -1
            steer_angle = abs(np.degrees(steer_angle))
            return sign*(
                -0.047613019578534976
                + (0.035699726805374414)*(steer_angle)
                + (0.006668289149260414)*(speed)
                + (-0.004714192483367892)*(steer_angle**2)
                + (0.005903913903940432)*(steer_angle*speed)
                + (-0.00017549649402737842)*(speed**2)
                + (0.00016762684760159626)*(steer_angle**3)
                + (-0.0001284576177060545)*(steer_angle**2*speed)
                + (-7.535153075782264e-05)*(steer_angle*speed**2)
                + (1.5518139872555186e-06)*(speed**3)
            )
        
        self.r = predict_orientation_rate(delta, vx)

        # World frame update
        cos_y = math.cos(self.yaw)
        sin_y = math.sin(self.yaw)
        self.x += (cos_y*self.vx - sin_y*self.vy) * self.dt
        self.y += (sin_y*self.vx + cos_y*self.vy) * self.dt
        self.yaw = (self.yaw + self.r * self.dt) % (2.0*math.pi)

        # Cache for HUD
        self.last.update(F_aero=F_aero, speed=speed)

        return np.array([self.x, self.y, self.speed, math.degrees(self.yaw)], dtype=np.float32)

    @property
    def speed(self):
        return math.hypot(self.vx, self.vy)

    @property
    def beta_deg(self):
        return math.degrees(math.atan2(self.vy, max(1e-6, self.vx)))


class CarEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        render_mode: Optional[str] = "human",
        follow_camera: bool = False,
        follow_zoom: float = 8.0,
        track_center_csv = "track_data/Yas_center.csv"
    ):
        
        super().__init__()
        self.car = ComplexCar()
        self.layer0 = layer0class(track_center_csv)
        self.layer2 = layer2class(30)

        self.last_best = None

        self.center, self.left_bounds, self.right_bounds = self.layer0.load_track()

        self.action_space = spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32),
                                       high=np.array([ 1.0,  1.0], dtype=np.float32),
                                       dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=np.zeros(19), 
            high=np.ones(19), 
            dtype=np.float32
        )

        self.max_steps = 6000
        self.step_count = 0

        self.render_mode = render_mode
        self.follow_camera = bool(follow_camera)
        self.follow_zoom = float(max(1.0, follow_zoom))
        self._screen = None
        self._surface = None
        self._world_to_px = 3.0
        self._offset = np.array([0.0, 0.0])
        self._bg_color = (20, 20, 20)
        self._center_color = (90, 90, 90)
        self._edge_color = (160, 160, 160)
        self._car_color = (220, 220, 255)
        self._hud_white = (240, 240, 240)
        self._hud_green = (80, 220, 120)
        self._hud_red   = (240, 90, 90)


    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.last_best = None
        self.step_count = 0
        idx = np.random.randint(0, len(self.center))

        self.center_proxi = np.random.uniform(0.3,5.0)


        #random
        orientation_offset = np.random.randint(-35, 35)
        position_offset = np.random.uniform(-0.9*self.center_proxi, 0.9*self.center_proxi)
        car_speed = np.random.uniform(0, 20)
        #
        self.car.power_factor = np.random.uniform(0.5,1.5)
        self.car.max_speed = np.random.uniform(20,500)

        

        self.car.break_force_factor = 1#np.random.uniform(0.5,1.5)
        self.car.mu = 3#np.random.uniform(1,4)
        self.car.under_steer = 0.02
        
        
        spawn_xy = self.center[idx]
        p_prev = self.center[(idx - 1) % len(self.center)]
        dir_vec = spawn_xy - p_prev
        if np.linalg.norm(dir_vec) < 1e-6:
            dir_vec = np.array([1.0, 0.0])
        offset_vector = [dir_vec[1]*position_offset,dir_vec[0]*position_offset]

        yaw_deg = math.degrees(math.atan2(dir_vec[1], dir_vec[0])) + orientation_offset
        self.car.reset(spawn_xy[0]+offset_vector[0], spawn_xy[1]+offset_vector[1], yaw_deg=yaw_deg, v=car_speed)


        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        if self.render_mode == "human":
            self._ensure_rendering()
            self.render()
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        steer_angle, pedals, self.car.gear = self.layer2.step(float(action[0]),float(action[1]),self.car.speed,self.car.engine_rpm,self.car.gear,self.car.mu)
        
        self.car.set_steer(steer_angle)
        self.car.set_throttle_brake(pedals)

        self.car.step()

        self.step_count += 1
    
        
        obs, completion, offset,track_orientation = self.layer0.compute_obs_vector(steer_angle, pedals,self.car.speed, self.car.x, self.car.y, self.car.yaw,
                                                                                   self.car.mu,
                                                                                   self.car.break_force_factor,
                                                                                   self.center_proxi)
        if self.last_best is None:
            self.last_best = completion
        progress = (completion - self.last_best) 
        if progress<-10:
            progress = 0
            self.last_best = 0
        if completion > self.last_best:
            self.last_best = completion

        
        pedal_reward = np.clip(pedals,0.3,1) 
        reward = progress#*pedal_reward
        off_track = abs(offset) > self.center_proxi
        terminated = bool(off_track)
        if self.car.speed < 2.0:
            reward = -0.1
        if terminated:
            reward = -20
        truncated = bool(self.step_count >= self.max_steps)
        info = {
            "off_track": off_track,
            "rpm": self.car.engine_rpm,
            "gear": self.car.gear,
            "beta_deg": self.car.beta_deg,
            **self.car.last,
        }
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ---------- Rendering ----------
    def _ensure_rendering(self):
        if self._screen is not None:
            return
        pygame.init()
        pygame.display.set_caption("Gymnasium Car Env")
        self._screen = pygame.display.set_mode((1280, 800))
        self._surface = pygame.Surface(self._screen.get_size()).convert()

        if not self.follow_camera:
            track_min = self.center.min(axis=0)
            track_max = self.center.max(axis=0)
            track_size = track_max - track_min
            margin_px = 40
            sx = (self._surface.get_width()  - 2 * margin_px) / max(track_size[0], 1e-6)
            sy = (self._surface.get_height() - 2 * margin_px) / max(track_size[1], 1e-6)
            self._world_to_px = 0.95 * min(sx, sy)
            self._offset = np.array([
                margin_px - track_min[0] * self._world_to_px,
                margin_px - track_min[1] * self._world_to_px
            ])
        else:
            self._world_to_px = self.follow_zoom

    def _update_follow_camera(self):
        if not self.follow_camera or self._surface is None:
            return
        screen_center = np.array([self._surface.get_width() / 2.0, self._surface.get_height() / 2.0])
        car_xy = np.array([self.car.x, self.car.y])
        self._offset = screen_center - car_xy * self._world_to_px

    def _w2p(self, xy: np.ndarray) -> Tuple[int, int]:
        p = xy * self._world_to_px + self._offset
        return int(p[0]), int(p[1])

    # ---- HUD helpers (speed/rpm/pedals unchanged, force widget reads smoothed) ----
    def _draw_steering_wheel(self):
        w, h = self._surface.get_width(), self._surface.get_height()
        cx, cy = w-100, h-400  # Position it near the bottom-left
        radius = 60
        steering_wheel_factor = 5.0
        # Draw outer circle for the wheel
        pygame.draw.circle(self._surface, (50, 50, 50), (cx, cy), radius, 4)

        # Draw inner circle (for the actual wheel shape)
        pygame.draw.circle(self._surface, (60, 60, 60), (cx, cy), radius - 10, 0)

        # Calculate the rotation based on the steering angle (scaled to the range of -1 to 1)
        steering_angle = self.car.steer_deg  # This is in degrees
        steering_angle_rad = steering_wheel_factor*math.radians(steering_angle)

        # Draw the steering wheel spokes
        pygame.draw.line(self._surface, (200, 200, 200), 
                        (cx - int((radius - 10) * math.cos(steering_angle_rad)), cy- int((radius - 10) * math.sin(steering_angle_rad))),
                        (cx + int((radius - 10) * math.cos(steering_angle_rad)),cy + int((radius - 10) * math.sin(steering_angle_rad)))
                        , 10)

        # Text to display steering angle
        font = pygame.font.SysFont(None, 18)
        angle_text = font.render(f"Steering: {int(steering_angle)}°", True, self._hud_white)
        self._surface.blit(angle_text, (cx - 35, cy + radius + 10))


    def _draw_rpm_gauge(self):
        w, h = self._surface.get_width(), self._surface.get_height()
        cx, cy = w - 160, h - 140
        radius = 110
        pygame.draw.circle(self._surface, (50, 50, 50), (cx, cy), radius, 4)
        max_rpm = self.car.engine_rpm_max
        for i in range(0, 8):
            frac = i / 7.0
            ang = math.pi * (1.0 + frac)
            x1 = cx + int((radius-10) * math.cos(ang))
            y1 = cy + int((radius-10) * math.sin(ang))
            x2 = cx + int(radius * math.cos(ang))
            y2 = cy + int(radius * math.sin(ang))
            pygame.draw.line(self._surface, self._hud_white, (x1, y1), (x2, y2), 2)
        rpm = max(0.0, min(self.car.engine_rpm, max_rpm))
        frac = rpm / max_rpm
        ang = math.pi * (1.0 + frac)
        x = cx + int((radius-15) * math.cos(ang))
        y = cy + int((radius-15) * math.sin(ang))
        pygame.draw.line(self._surface, self._hud_red if frac>0.85 else self._hud_green, (cx, cy), (x, y), 5)
        font = pygame.font.SysFont(None, 22)
        txt = font.render(f"{int(rpm)} rpm  G{self.car.gear}", True, self._hud_white)
        self._surface.blit(txt, (cx-60, cy+radius-20))

    def _draw_speed_gauge(self):
        w, h = self._surface.get_width(), self._surface.get_height()
        cx, cy = 160, h - 140
        radius = 110
        pygame.draw.circle(self._surface, (50, 50, 50), (cx, cy), radius, 4)
        vmax = 320.0
        for i in range(0, 9):
            frac = i / 8.0
            ang = math.pi * (1.0 + frac)
            x1 = cx + int((radius-10) * math.cos(ang))
            y1 = cy + int((radius-10) * math.sin(ang))
            x2 = cx + int(radius * math.cos(ang))
            y2 = cy + int(radius * math.sin(ang))
            pygame.draw.line(self._surface, self._hud_white, (x1, y1), (x2, y2), 2)
            if i % 2 == 0:
                font = pygame.font.SysFont(None, 18)
                label = int(vmax * frac)
                tx = cx + int((radius-30) * math.cos(ang))
                ty = cy + int((radius-30) * math.sin(ang))
                self._surface.blit(font.render(f"{label}", True, self._hud_white), (tx-10, ty-10))
        kmh = self.car.speed * 3.6
        frac = max(0.0, min(1.0, kmh / vmax))
        ang = math.pi * (1.0 + frac)
        x = cx + int((radius-15) * math.cos(ang))
        y = cy + int((radius-15) * math.sin(ang))
        pygame.draw.line(self._surface, self._hud_green, (cx, cy), (x, y), 5)
        font = pygame.font.SysFont(None, 22)
        self._surface.blit(font.render(f"{int(kmh)} km/h", True, self._hud_white), (cx-55, cy+radius-20))

    def _draw_pedals(self):
        w, h = self._surface.get_width(), self._surface.get_height()
        x0 = w - 40
        bar_h = 220
        y0 = h - 40 - bar_h
        pygame.draw.rect(self._surface, (60,60,60), (x0-20, y0, 16, bar_h), 2)
        pygame.draw.rect(self._surface, (60,60,60), (x0+10, y0, 16, bar_h), 2)
        t = max(0.0, self.car.throttle_cmd)
        b = max(0.0, -self.car.throttle_cmd)
        t_pix = int(t * (bar_h-4))
        b_pix = int(b * (bar_h-4))
        pygame.draw.rect(self._surface, self._hud_green, (x0-20+2, y0 + (bar_h-2 - t_pix), 16-4, t_pix))
        pygame.draw.rect(self._surface, self._hud_red,   (x0+10+2, y0 + (bar_h-2 - b_pix), 16-4, b_pix))
        font = pygame.font.SysFont(None, 18)
        self._surface.blit(font.render("TH", True, self._hud_white), (x0-26, y0-18))
        self._surface.blit(font.render("BR", True, self._hud_white), (x0+12, y0-18))




        font = pygame.font.SysFont(None, 18)
        self._surface.blit(font.render("Wheel Forces (smoothed)", True, self._hud_white), (x0+8, y0+6))

    def render(self):
        if self._screen is None and self.render_mode == "human":
            self._ensure_rendering()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
        if self.follow_camera:
            self._update_follow_camera()

        self._surface.fill(self._bg_color)

        # Track
        pygame.draw.lines(self._surface, self._edge_color, False, [self._w2p(p) for p in self.left_bounds], 2)
        pygame.draw.lines(self._surface, self._edge_color, False, [self._w2p(p) for p in self.right_bounds], 2)
        pygame.draw.lines(self._surface, self._center_color, False, [self._w2p(p) for p in self.center], 1)

        # Car
        car_xy = np.array([self.car.x, self.car.y])
        yaw = self.car.yaw
        L, W = 3.8, 1.9
        local_pts = np.array([[ L*0.6, 0.0], [-L*0.4,  W*0.5], [-L*0.4, -W*0.5]])
        R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        world_pts = (local_pts @ R.T) + car_xy
        pygame.draw.polygon(self._surface, self._car_color, [self._w2p(p) for p in world_pts], 0)


        # HUD (rpm/speed/pedals + info)
        self._draw_rpm_gauge()
        self._draw_speed_gauge()
        self._draw_pedals()
        self._draw_steering_wheel()  # Add this line to draw the steering wheel
        font = pygame.font.SysFont(None, 18)
        mode = "FOLLOW" if self.follow_camera else "STATIC"
        txt = (
            f"Mode:{mode} Zoom:{self._world_to_px:.1f} px/m | v:{self.car.speed*3.6:5.1f} km/h | "
            f"Yaw:{math.degrees(self.car.yaw):6.2f}° | beta:{self.car.beta_deg:5.2f}°"
        )
        text = font.render(txt, True, self._hud_white)
        self._surface.blit(text, (10, 10))

        self._screen.blit(self._surface, (0, 0))
        pygame.display.flip()


    def close(self):
        if self._screen is not None:
            pygame.quit()
            self._screen = None
