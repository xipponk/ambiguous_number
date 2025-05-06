import os
import torch
import torchvision.utils as vutils
from model import Generator
import datetime
import torchvision.transforms.functional as TF

# ==== Config ====
#CHECKPOINT_PATH = "outputs/final_model/netG_epoch_1519.pth"
CHECKPOINT_PATH = "outputs/final_model/netG_epoch_1239.pth"
OUTPUT_DIR = "outputs/generated"
Z_DIM = 100
UPSCALE_TO = 256  # หรือ 1024

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# ==== Load Generator ====
netG = Generator().to(device)
netG.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
netG.eval()
print(f"✅ Loaded Generator from {CHECKPOINT_PATH}")

# ==== Generate 1 image ====
noise = torch.randn(1, Z_DIM, 1, 1, device=device)
with torch.no_grad():
    fake_image = netG(noise).detach().cpu()[0]  # แค่ภาพเดียว

# ==== Upscale & Save ====
upscaled = TF.resize(fake_image, [UPSCALE_TO, UPSCALE_TO])
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"{OUTPUT_DIR}/big_generated_{timestamp}.png"
vutils.save_image(upscaled, filename, normalize=True)
print(f"✅ Generated image saved to {filename}")