import argparse
import numpy as np
import pygame
import time

from controller.car_env import CarEnv

HELP = "Arrows: steer/throttle/brake | R: reset | ESC: quit | Q/E: gear -/+ | F: toggle follow | +/-: zoom"

def run_keyboard_test(follow: bool, zoom: float):
    env = CarEnv(
        render_mode="human", follow_camera=True, follow_zoom=9.0,track_center_csv = "track_data/Yas_center_pits.csv"
    )

    obs, info = env.reset()
    running = True

    # Create clock to measure FPS
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                elif event.key == pygame.K_q:
                    env.car.shift_down()
                elif event.key == pygame.K_e:
                    env.car.shift_up()
                elif event.key == pygame.K_f:
                    # runtime toggle follow camera
                    env.follow_camera = not env.follow_camera
                elif event.key in (pygame.K_PLUS, pygame.K_KP_PLUS):
                    env.follow_zoom = min(40.0, env.follow_zoom + 1.0)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    env.follow_zoom = max(2.0, env.follow_zoom - 1.0)

        keys = pygame.key.get_pressed()
        steer, throttle = 0.0, 0.0
        if keys[pygame.K_LEFT]:
            steer -= 1.0
        if keys[pygame.K_RIGHT]:
            steer += 1.0
        if keys[pygame.K_UP]:
            throttle += 1.0
        if keys[pygame.K_DOWN]:
            throttle -= 1.0

        action = np.array([steer, throttle], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Draw FPS on screen ---
        fps = int(clock.get_fps())
        fps_surface = font.render(f"FPS: {fps}", True, (255, 0, 255))
        env._screen.blit(fps_surface, (100, 100))  # Assuming env has a pygame screen
        pygame.display.flip()

        if terminated or truncated:
            obs, info = env.reset()
        # Tick the clock (limit to 60 FPS, adjust as needed)
        clock.tick(30)

    env.close()

if __name__ == "__main__":
    run_keyboard_test(follow=True, zoom=2)
