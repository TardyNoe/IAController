# inference_car_keyboard_env.py
import os, time
import numpy as np
import pygame
from stable_baselines3 import PPO
from controller.car_env import CarEnv

def main():
    # ---- Settings ----
    model_path = "weights/keyboard.zip"
    episodes = 10
    follow_camera = True
    follow_zoom = 9.0
    max_steps = 100000
    control_hz = 30                 # control loop rate
    render_enabled = True           # toggle rendering cost
    render_vsync = False            # if your display init enables vsync, it can block
    device = "cuda"                 # try "cuda" if available; else set to "cpu"

    # ---- Load model ----
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path, device=device)

    # ---- Create env ----
    env = CarEnv(follow_camera=follow_camera, follow_zoom=follow_zoom)
    obs, info = env.reset()

    # ---- Timing ----
    target_dt = 1.0 / control_hz
    next_tick = time.perf_counter() + target_dt

    # FPS stats
    frames = 0
    fps_timer = time.perf_counter()
    control_steps = 0
    control_fps_timer = fps_timer

    print("Controls: ESC to quit, R to reset, F to toggle follow, +/- to zoom")
    ep = 0
    steps_left = max_steps

    running = True
    while running and ep < episodes:
        # Handle events every loop (don’t wait)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                elif event.key == pygame.K_f:
                    env.follow_camera = not env.follow_camera
                elif event.key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):  # '=' for shifted '+'
                    env.follow_zoom = min(40.0, env.follow_zoom + 1.0)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    env.follow_zoom = max(2.0, env.follow_zoom - 1.0)

        now = time.perf_counter()
        # Sleep until next tick if we’re early (keeps control at ~50 Hz)
        sleep_for = next_tick - now
        if sleep_for > 0:
            # Use time.sleep for coarse sleep; busy-wait the last ~0.5 ms
            if sleep_for > 0.0005:
                time.sleep(sleep_for - 0.0005)
            while time.perf_counter() < next_tick:
                pass
            now = next_tick
        else:
            # If we’re late, don’t accumulate drift
            now = time.perf_counter()

        # ---- Control step (fixed rate) ----
        action, _ = model.predict(obs, deterministic=True)
        print(env.center_proxi)
        obs, reward, terminated, truncated, info = env.step(action)
        control_steps += 1
        next_tick += target_dt  # schedule next control tick

        # ---- Render (optional; skip if we’re behind) ----
        if render_enabled:
            # Only render if we’re not more than one tick behind
            if time.perf_counter() <= next_tick:
                if hasattr(env, "render"):
                    env.render()
                frames += 1

        # ---- Episode end ----
        if terminated or truncated:
            ep += 1
            print(f"Episode {ep}/{episodes} finished. Info: {info}")
            obs, info = env.reset()

        # ---- Step budget ----
        if steps_left is not None:
            steps_left -= 1
            if steps_left <= 0:
                print("Reached max_steps, exiting.")
                break

        # ---- Stats every second ----
        if time.perf_counter() - fps_timer >= 1.0:
            # Render FPS is frames/sec, control FPS is control_steps/sec
            print(f"[perf] control ~{control_steps} Hz | render ~{frames} FPS | behind={max(0, (time.perf_counter()-next_tick)/target_dt):.2f} ticks")
            frames = 0
            control_steps = 0
            fps_timer = time.perf_counter()

    env.close()
    print("Inference complete.")

if __name__ == "__main__":
    main()
