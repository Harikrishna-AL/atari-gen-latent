import os
import glob
import torch
import numpy as np
from PIL import Image
from latent_action_model import load_latent_action_model, ActionStateToLatentMLP


def video_to_imgs(video_path: str, output_folder: str):
    """
    Extract frames from a video file and save them as images.
    Args:
        video_path: Path to the input video file.
        output_folder: Folder to save the extracted images.
    """
    import cv2
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_folder, f"{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    
    cap.release()
    print(f"Extracted {frame_count} frames to {output_folder}")

def get_latent(video_folder : str):
    # --- Configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_file = 'latent_embeddings.pt'
    
    # --- Load world model ---
    world_model, _ = load_latent_action_model('checkpoints/latent_action/best.pt', device)
    world_model.to(device)
    world_model.eval()
    
    print("[INFO] Loading action-to-latent model...")
    action_model = ActionStateToLatentMLP().to(device)
    ckpt = torch.load('checkpoints/latent_action/action_state_to_latent_best.pt', map_location=device)
    action_model.load_state_dict(ckpt['model_state_dict'])
    action_model.eval()
    
    videos = sorted(glob.glob(os.path.join(video_folder, '*.mp4')))
    for video in videos:
        print(f"Processing video: {video}")
        video_name = os.path.splitext(os.path.basename(video))[0]
        video_path = os.path.join(video_folder, video_name + '.mp4')
        output_folder = os.path.join(video_folder + "/" + video_name, 'frames')
        video_to_imgs(video_path, output_folder)


    # --- Load and preprocess frames ---
        frame_files = sorted(glob.glob(os.path.join(output_folder, '*.png')))
        frames = []
        for file in frame_files:
            img = Image.open(file).convert('RGB')
            frame_np = np.array(img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0)
            frames.append(tensor)

        # --- Iterate through stacked pairs and extract latents ---
        latent_list = []
        for i in range(len(frames) - 1):
            f1 = frames[i].to(device)
            f2 = frames[i + 1].to(device)
            # print(f"frames dimensions: {f1.shape}, {f2.shape}")
            stacked = torch.cat([f1, f2], dim=1)

            with torch.no_grad():
                stacked = stacked.permute(0, 1, 3, 2)
                logits = world_model.encoder(stacked)  # (B, C, H, W)
                
                # print(f"Logits shape: {logits.shape}")
                # indices = world_model.vq.get_code_indices(logits)  # (B, H, W)
                quantized, indices, commitment_loss, codebook_loss = world_model.vq(logits)
                # embeddings = world_model.vq.embeddings(logits)  # (B, H, W, D)
                # embeddings = embeddings.permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)
                latent_list.append(quantized.cpu())

        # --- Save all latent embeddings ---
        output_file = os.path.join(video_folder, f"{video_name}_latents.pt")
        torch.save(latent_list, output_file)
        print(f"Saved {len(latent_list)} latent embeddings to {output_file}")
        
def plot_latent_gifs(latent_folder: str):
    import imageio
    import os
    
    latent_files = sorted(glob.glob(os.path.join(latent_folder, '*_latents.pt')))
    
    for latent_file in latent_files:
        latents = torch.load(latent_file)
        latents = [latent.numpy() for latent in latents]  # Convert to numpy arrays
        latents = np.array(latents)  # Convert list to numpy array
        latents = latents.squeeze(1)
        latents = latents[:, :1, :, :]  # Assuming latents are in shape (N, C, H, W) and we want the first 3 channels
        latents = latents.transpose(0, 3, 2, 1)  # Convert to (N, H, W, C) for imageio
        latents = (latents * 255).astype(np.uint8)  # Convert to uint8 for imageio
        latents = latents.squeeze(1)
        # convert 128 dim to 3 for easy visualization
        # latents = [np.clip(latent[:3], 0, 1) for latent in latents]
        #increase the resolution 50 times
        latents = np.repeat(np.repeat(latents, 50, axis=1), 50, axis=2)  # Increase resolution by 50x
  
        print(f"Loaded {len(latents)} latents from {latent_file}")
        print(f"Latent shape: {latents.shape}")
        # Create GIF
        gif_path = latent_file.replace('_latents.pt', '.gif')
        #the plot is grayscale 
        
        imageio.mimsave(gif_path, latents, duration=0.1)  # Adjust duration as needed
        print(f"Saved GIF: {gif_path}")
        
def main():
    video_folder = "videos"
    # compute differnce between two latents to find the velocity latent
    latent_fast = torch.load(os.path.join(video_folder, "breakout_fast_latents.pt"))
    latent_slow = torch.load(os.path.join(video_folder, "breakout_slow_latents.pt"))
    print(f"Loaded {len(latent_fast)} fast latents and {len(latent_slow)} slow latents")
    print(f"Fast latent shape: {latent_fast[0].shape}, Slow latent shape: {latent_slow[0].shape}")
    velocity_latent = [f2 - f1 for f1, f2 in zip(latent_slow, latent_fast)]
    velocity_latent = torch.stack(velocity_latent, dim=0)
    velocity_latent = velocity_latent.mean(dim=0)  # Average over all frames
    velocity_latent = velocity_latent.squeeze(0)
    
    torch.save(velocity_latent, os.path.join(video_folder, "breakout_velocity_latents.pt"))
    velocity_latent = velocity_latent[:3, :, :]  # Take only the first 3 channels for visualization
    plt = velocity_latent.permute(1, 2, 0).cpu().numpy()
    plt = (plt * 255).astype(np.uint8)
    #increase the resolution 50 times
    plt = np.repeat(np.repeat(plt, 50, axis=0), 50, axis=1)
    plt = Image.fromarray(plt)
    plt.save(os.path.join(video_folder, "breakout_velocity_latent.png"))
    
    print(f"Saved velocity latents to {os.path.join(video_folder, 'breakout_velocity_latents.pt')}")
    
    print(velocity_latent.shape)
    
    plot_latent_gifs(video_folder)
    
get_latent("videos")
if __name__ == "__main__":
    main()