from latent_action_model import load_latent_action_model, ActionStateToLatentMLP
import torch

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

video_folder = "videos"
import glob
import os

videos = sorted(glob.glob(os.path.join(video_folder, '*.mp4')))
for video in videos:
    print(f"Processing video: {video}")
    video_name = os.path.splitext(os.path.basename(video))[0]
    video_path = os.path.join(video_folder, video_name + '.mp4')
    output_folder = os.path.join(video_folder + "/" + video_name, 'frames')
    video_to_imgs(video_path, output_folder)

from PIL import Image
import torch
from torchvision import transforms as T

fast_frames_path = "videos/breakout_fast/frames/"

images = []

# define a transform to convert PIL images to tensor
transform = T.ToTensor()

import numpy as np
import cv2

# load image files from the directory
for filename in os.listdir(fast_frames_path):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        img_path = os.path.join(fast_frames_path, filename)
        # open image
        image = Image.open(img_path)
        image_np = np.array(image)
        image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        # transform image to tensor
        image_tensor = transform(image)
        images.append(image_tensor)
        
        if len(images) == 10:
            break

k = 6
world_model = world_model.to(device)
current_image = images[k].unsqueeze(0).to(device)
next_image = images[k+1].unsqueeze(0).to(device)
rec_frame, indices, commitment_loss, codebook_loss = world_model(current_image, next_image)
print(rec_frame.shape)

import matplotlib.pyplot as plt
import torch
import numpy as np

# dummy tensors for current frame, next frame, and reconstructed next frame
current_frame = images[k]# example size (3 channels, 64x64)
next_frame = images[k+1]
reconstructed_next_frame = rec_frame.squeeze(0)

# convert tensors to numpy arrays
current_frame_np = current_frame.permute(1, 2, 0).detach().cpu().numpy()
next_frame_np = next_frame.permute(1, 2, 0).detach().cpu().numpy()
reconstructed_next_frame_np = reconstructed_next_frame.permute(1, 2, 0).detach().cpu().numpy()

# create a horizontal plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# plot current frame
axes[0].imshow(current_frame_np)
axes[0].set_title('Current Frame')
axes[0].axis('off')

# plot next frame
axes[1].imshow(next_frame_np)
axes[1].set_title('Next Frame')
axes[1].axis('off')

# plot reconstructed next frame
axes[2].imshow(reconstructed_next_frame_np)
axes[2].set_title('Reconstructed Next Frame')
axes[2].axis('off')

plt.tight_layout()
# plt.show()
plt.savefig("speed_recons.png")