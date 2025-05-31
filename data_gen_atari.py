import cv2
import numpy as np
from pathlib import Path
from atari_env import AtariBreakoutEnv  # Assuming your class is saved in a file

def run_and_save_video(env, video_path: str, num_frames: int = 1000, action_policy="random", speed_factor=1.0):
    # Create a VideoWriter object
    fps = 30
    width, height = 160, 210  # Original RGB frame size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps * speed_factor, (width, height))

    obs, _ = env.reset()
    done = False
    steps = 0
    
    while steps < num_frames:
        if done:
            obs, _ = env.reset()

        # Choose action
        if action_policy == "random":
            action = env.action_space.sample()
        elif action_policy == "no-op":
            action = 0  # Usually "do nothing"
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Write RGB frame to video
        frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)  # OpenCV expects BGR
        out.write(frame)
        
        steps += 1

    out.release()
    env.close()
    print(f"Saved video: {video_path}")

# Create output directory
Path("videos").mkdir(exist_ok=True)

# Initialize the environment with RGB output
fast_env = AtariBreakoutEnv(return_rgb=True, frameskip=10)

# Save video of random fast play
run_and_save_video(fast_env, "videos/breakout_fast.mp4", num_frames=500, speed_factor=1.0)

# Reinitialize env and save video of slow play (normal frame rate, slower action changes)
slow_env = AtariBreakoutEnv(return_rgb=True, frameskip=1)
run_and_save_video(slow_env, "videos/breakout_slow.mp4", num_frames=500, speed_factor=1.0)
