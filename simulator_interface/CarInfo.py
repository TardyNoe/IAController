# ros_subscriber.py
from __future__ import annotations
import threading
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.context import Context

from geometry_msgs.msg import PoseStamped
from vectornav_msgs.msg import CommonGroup  # pip/package must be available


class GroundTruthSubscriber(Node):
    def __init__(self, *, context: Optional[Context] = None):
        super().__init__('ground_truth_subscriber', context=context)

        # Shared state (protected by self._lock)
        self._lock = threading.Lock()
        self._speed = 0.0
        self._posx = 0.0
        self._posy = 0.0
        self._orientation = 0.0
        self._orientation_rate = 0.0

        # Subscriptions
        self.create_subscription(
            PoseStamped, '/eav24/ground_truth', self._listener_callback_gt, 10
        )
        self.create_subscription(
            CommonGroup, '/eav24/vectornav/raw/common', self._listener_callback_vn, 10
        )

    # --- Callbacks ---------------------------------------------------------
    def _listener_callback_gt(self, msg: PoseStamped):
        with self._lock:
            self._posx = msg.pose.position.x
            self._posy = msg.pose.position.y

    def _listener_callback_vn(self, msg: CommonGroup):
        # Defensive parsing in case any field is missing
        vx = getattr(getattr(msg, 'velocity', msg), 'x', 0.0)
        vy = getattr(getattr(msg, 'velocity', msg), 'y', 0.0)
        vz = getattr(getattr(msg, 'velocity', msg), 'z', 0.0)
        ypr_x = getattr(getattr(msg, 'yawpitchroll', msg), 'x', 0.0)
        imu_rate_z = getattr(getattr(msg, 'imu_rate', msg), 'z', 0.0)

        speed = float(np.linalg.norm([vx, vy, vz]))

        with self._lock:
            self._orientation = float(ypr_x)
            self._orientation_rate = float(imu_rate_z)
            self._speed = speed

    # --- Thread-safe getters ----------------------------------------------
    def get_orientation(self) -> float:
        with self._lock:
            return self._orientation

    def get_orientation_rate(self) -> float:
        with self._lock:
            return self._orientation_rate

    def get_position(self) -> Tuple[float, float]:
        with self._lock:
            return self._posx, self._posy

    def get_speed(self) -> float:
        with self._lock:
            return self._speed


class ROSSubscriber:
    """
    Manage the Node + executor in a background thread.
    Import this class and instantiate it once in your app.
    """

    def __init__(self):
        self._context = rclpy.context.Context()
        rclpy.init(context=self._context)

        self.node = GroundTruthSubscriber(context=self._context)
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self.node)

        self._thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._stop_evt = threading.Event()
        self._started = False

    # --- Lifecycle ---------------------------------------------------------
    def start(self):
        if self._started:
            return
        self._started = True
        self._thread.start()

    def stop(self):
        """Gracefully stop the spin loop and shut down this private ROS context."""
        if not self._started:
            return
        self._stop_evt.set()
        # Signal the spin loop to exit
        self._context.shutdown()  # shuts down ONLY this private context
        self._thread.join(timeout=2.0)
        try:
            self._executor.remove_node(self.node)
        except Exception:
            pass
        try:
            self.node.destroy_node()
        except Exception:
            pass

    # --- Background spinning ----------------------------------------------
    def _spin_loop(self):
        # Non-blocking spin so we can check the stop event periodically
        try:
            while not self._stop_evt.is_set() and self._context.ok():
                self._executor.spin_once(timeout_sec=0.1)
        finally:
            try:
                self._executor.shutdown()
            except Exception:
                pass

    # --- Convenience pass-throughs ----------------------------------------
    def get_orientation(self) -> float:
        return self.node.get_orientation()

    def get_orientation_rate(self) -> float:
        return self.node.get_orientation_rate()

    def get_position(self) -> Tuple[float, float]:
        return self.node.get_position()

    def get_speed(self) -> float:
        return self.node.get_speed()


# Optional singleton for quick import/use across modules
_subscriber_singleton: Optional[ROSSubscriber] = None

def get_ros_subscriber() -> ROSSubscriber:
    """
    Global accessor. Ensures a single background subscriber is created and running.
    """
    global _subscriber_singleton
    if _subscriber_singleton is None:
        _subscriber_singleton = ROSSubscriber()
        _subscriber_singleton.start()
    return _subscriber_singleton
