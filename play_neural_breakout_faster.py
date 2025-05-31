"""
Controls:

SPACE: Fire
LEFT ARROW: Move Left
RIGHT ARROW: Move Right
. (PERIOD): No Operation (NOOP) -> DEFAULT
ESC or Q: Quit

Optional Parameters:

--temperature: Control the randomness of predictions (lower = more deterministic)

python play_neural_breakout.py --temperature 0.05
The game runs at a lower FPS (15) to make changes more visible. You can adjust this by changing the FPS variable if you want faster gameplay."""

import os
import sys
import torch
import numpy as np
import pygame
from PIL import Image
import time
import argparse
from latent_action_model import load_latent_action_model, ActionStateToLatentMLP

# --- Configuration ---
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 840
FPS = 15  # Slower FPS for visible changes

# --- Device selection ---
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def action_to_onehot(action_idx, device):
    onehot = torch.zeros(1, 4, device=device)
    onehot[0, action_idx] = 1.0
    return onehot

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', type=float, default=0.01, help='Sampling temperature for latent prediction')
    args = parser.parse_args()
    
    # Initialize pygame
    pygame.init()
    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Neural Breakout")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 18)
    
    # Set up device and models
    device = get_device()
    print(f"[INFO] Using device: {device}")
    
    # Load world model
    print("[INFO] Loading world model...")
    world_model, _ = load_latent_action_model('checkpoints/latent_action/best.pt', device)
    world_model.to(device)
    world_model.eval()
    # if device.type == 'cuda':
    #     world_model = torch.compile(world_model)
    
    # Load action-to-latent model
    print("[INFO] Loading action-to-latent model...")
    action_model = ActionStateToLatentMLP().to(device)
    ckpt = torch.load('checkpoints/latent_action/action_state_to_latent_best.pt', map_location=device)
    action_model.load_state_dict(ckpt['model_state_dict'])
    action_model.eval()
    # if device.type == 'cuda':
    #     action_model = torch.compile(action_model)
    
    # Load initial frame
    print("[INFO] Loading initial frame...")
    init_img = Image.open('data/0.png').convert('RGB')
    init_frame_np = np.array(init_img, dtype=np.float32) / 255.0
    current_frame = torch.from_numpy(init_frame_np).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Set up frame history
    last2_frames = [current_frame.clone(), current_frame.clone()]
    
    #load the velocity latent 
    velocity_latent = torch.load('videos/breakout_velocity_latents.pt', map_location=device)
    velocity_latent = velocity_latent.unsqueeze(0)  # Add batch dimension

    
    # Action mapping
    action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    key_to_action = {
        pygame.K_PERIOD: 0,  # NOOP
        pygame.K_SPACE: 1,   # FIRE
        pygame.K_RIGHT: 2,   # RIGHT
        pygame.K_LEFT: 3     # LEFT
    }
    
    # Game state
    action_idx = 0  # Default to NOOP
    last_displayed_action = ""  # For display purposes
    score = 0
    step = 0
    
    # Display the key mappings at the start
    print("\nNeural Breakout Controls:")
    print("------------------------")
    print("SPACE - Fire")
    print("LEFT ARROW - Move Left")
    print("RIGHT ARROW - Move Right")
    print(". (PERIOD) - No Operation")
    print("ESC or Q - Quit\n")
    
    # Main game loop
    running = True
    latents = []
    step = 0
    while running:
        # Default to NOOP (0) each frame
        action_idx = 0
        
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Handle specific quit keys
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                # Set the action based on key press
                elif event.key in key_to_action:
                    action_idx = key_to_action[event.key]
        
        # Check currently pressed keys (for continuous input)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action_idx = 1  # FIRE
            last_displayed_action = "FIRE"
        elif keys[pygame.K_RIGHT]:
            action_idx = 2  # RIGHT
            last_displayed_action = "RIGHT"
        elif keys[pygame.K_LEFT]:
            action_idx = 3  # LEFT
            last_displayed_action = "LEFT"
        elif keys[pygame.K_PERIOD]:
            action_idx = 0  # NOOP (explicitly)
            last_displayed_action = "NOOP"
        
        # Only print non-NOOP actions to console
        if action_idx != 0:
            print(f"Action: {action_names[action_idx]}")
        
        # Generate next frame
        with torch.no_grad():
            # Stack last 2 frames
            stacked_frames = torch.cat([last2_frames[0], last2_frames[1]], dim=1)
            
            # Get action prediction
            onehot = action_to_onehot(action_idx, device)
            logits = action_model(onehot, stacked_frames)
            # print(f"[INFO] Logits shape: {logits.shape}")
            # Sample indices from logits
            indices = action_model.sample_latents(logits, temperature=args.temperature)
            # print(f"[INFO] Indices shape: {indices.shape}")
            
            # Reshape indices and get embeddings
            indices = indices.view(1, 5, 7)
            # print(f"Indices: {indices}")
            embeddings = world_model.vq.embeddings
            indices = indices.to(embeddings.weight.device)
            # print(f"Indices shape: {indices.shape}, Embeddings shape: {embeddings.weight.shape}")
            quantized = embeddings(indices)
            # print(f"Quantized shape: {quantized.shape}")
            quantized = quantized.permute(0, 3, 1, 2).contiguous()
            
            # Generate next frame
            frame_in = current_frame.permute(0, 1, 3, 2)
            quantized = quantized.to(device)
  
            quantized = quantized + velocity_latent * 0
            brick_roi = quantized[:, :, 0:5, 2:3].clone()  # Shape: (1, 128, 3, 1)
            brick_roi = brick_roi.reshape(1, 128, 5)       # Shape: (1, 128, 3)

            if step % 100 == 0:
                # 1. Compute channel-wise variance and select top 10 important channels
                importance = quantized.var(dim=(2,3))               # Shape: (1, 128)
                print(f"Importance shape: {importance.shape}, Importance: {importance}")
                topk = torch.topk(importance, k=128, dim=1).indices[0]  # Shape: (10,)

                # 2. Create a random permutation of spatial (3) values
                perm = torch.randperm(5)

                # 3. Extract the topk channel patch: Shape (10, 3)
            patch_selected = brick_roi[0, topk, :]            # (10, 3)
            
            # 4. Permute spatially (3-dim) for each selected channel
            patch_selected = patch_selected[:, perm]          # (10, 3)

            # 5. Put permuted patch back
            brick_roi[0, topk, :] = patch_selected            # Assign back into clone

            # 6. Reshape and assign back to quantized
            brick_roi_reshaped = brick_roi.reshape(1, 128, 5, 1)
            quantized[:, :, 0:5, 2:3] = brick_roi_reshaped + torch.randn_like(brick_roi_reshaped) * 0  # Add small noise
            # quantized[:, :, 0:5, 3:4] = brick_roi_reshaped
            # quantized[:, :, 0:5, 4:5] = velocity_latent
            # quantized = quantized + torch.randn_like(quantized) * 0.1  # Add small noise
            velocity_latent_important = velocity_latent.var(dim=(2, 3))  # Shape: (128,)
            important_channels = torch.topk(velocity_latent_important, k=10, dim=1).indices[0]
            # print(f"Important channels: {important_channels}")
            
            quantized[:, important_channels, 0:5, 4:5] += velocity_latent[:,important_channels, 0:5, 4:5] * 15
            #re quantize the quantized tensor
            quantized , indices, commitment_loss, codebook_loss = world_model.vq(quantized)
            # quantized[:, :, 0:5, 4:5] *= 1
            
            # channel_variances = quantized.var(dim=(0, 2, 3))  # Shape: (128,)
            # top_channels = torch.argsort(channel_variances, descending=True)[:6]
            # top_channels = quantized[:, top_channels[0:9], :, :]
            # latents.append(top_channels.cpu().clone())

            frame_in = frame_in.to(device)
            next_frame = world_model.decoder(quantized, frame_in)
            next_frame = next_frame.permute(0, 1, 3, 2)
            
            # Update frame history
            last2_frames[0] = last2_frames[1]
            last2_frames[1] = next_frame.clone()
            current_frame = next_frame.clone()
        
        # Convert the frame to a pygame surface for display
        frame_np = current_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
        
        # Create surface from numpy array
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame_np, (1, 0, 2)))
        
        # Scale the surface to fit the window
        scaled_surface = pygame.transform.scale(frame_surface, (WINDOW_WIDTH, WINDOW_HEIGHT - 40))
        
        # Render to the window
        window.fill((0, 0, 0))
        window.blit(scaled_surface, (0, 0))
        
        # Display game info - always show the last action
        step += 1
        info_text = f"Step: {step}"
        if last_displayed_action:
            info_text += f" | Last Action: {last_displayed_action}"
        info_text += f" | Temperature: {args.temperature}"
        
        text_surface = font.render(info_text, True, (255, 255, 255))
        window.blit(text_surface, (10, WINDOW_HEIGHT - 30))
        
        # Update the display
        pygame.display.flip()
        
        # Limit framerate
        clock.tick(FPS)
        step += 1
    
    
    # Clean up
    pygame.quit()
    print("Game closed.")
    return latents

if __name__ == "__main__":
    latents = main()
    # plot all the latents as gifs in 9*9 grid
    import imageio
    import os
    import numpy as np
    import glob
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import torch
    latents = torch.stack(latents)
    print(f"Latents shape: {latents.shape}")
    latents = latents.squeeze(1)
    latents = latents[:, :3, :, :]  # Assuming latents are in shape (N, C, H, W) and we want the first 3 channels
    