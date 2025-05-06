import os
import time
import random
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm

# ==== Config ====
DATA_DIR = "data/augmented"
OUTPUT_DIR = "outputs/samples"
CHECKPOINT_DIR = "outputs/checkpoints"
IMAGE_SIZE = 64
BATCH_SIZE = 256
Z_DIM = 100
EPOCHS = 5000
LR = 0.0002
BETA1 = 0.5
RESUME = True
RESUME_EPOCH = 130  # เปลี่ยนตาม checkpoint ล่าสุด
random.seed(999)
torch.manual_seed(999)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

G_losses = []
D_losses = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}")

# ==== Dataset ====
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
dataset = dset.ImageFolder(root="data", transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==== Generator ====
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(Z_DIM, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x): return self.main(x)

# ==== Discriminator ====
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x): return self.main(x)

# ==== Init ====
netG = Generator().to(device)
netD = Discriminator().to(device)
fixed_noise = torch.randn(64, Z_DIM, 1, 1, device=device)
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))

# ==== Resume ====
start_epoch = 0
if RESUME:
    try:
        netG.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/netG_epoch_{RESUME_EPOCH}.pth", map_location=device))
        netD.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/netD_epoch_{RESUME_EPOCH}.pth", map_location=device))
        start_epoch = RESUME_EPOCH
        print(f"✅ Resumed from checkpoint at epoch {start_epoch}")
    except Exception as e:
        print(f"⚠️ Failed to load checkpoint: {e}")

loss_log_path = os.path.join("outputs", "loss_log.csv")
with open(loss_log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "lossG", "lossD"])

# ==== Training Loop ====
print("🚀 Starting Training Loop...")
for epoch in range(start_epoch, EPOCHS):
    start_time = time.time()
    for i, (data, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
        netD.zero_grad()
        real = data.to(device)
        b_size = real.size(0)
        label = torch.full((b_size,), 1., device=device)
        output = netD(real).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()

        noise = torch.randn(b_size, Z_DIM, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(0.)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        optimizerD.step()

        netG.zero_grad()
        label.fill_(1.)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()
    # log เวลา
    elapsed = time.time() - start_time
    print(f"🕒 Epoch {epoch+1} finished in {elapsed:.2f} seconds")

    # ⬇️ เก็บ loss ลง list
    G_losses.append(errG.item())
    D_losses.append(errD_real.item() + errD_fake.item())

    # ⬇️ เขียนลง CSV
    with open(loss_log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, errG.item(), errD_real.item() + errD_fake.item()])

    # Save generated samples
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
        vutils.save_image(fake, f"{OUTPUT_DIR}/epoch_{epoch+1:03}.png", normalize=True)

    # Save checkpoint
    torch.save(netG.state_dict(), f"{CHECKPOINT_DIR}/netG_epoch_{epoch+1}.pth")
    torch.save(netD.state_dict(), f"{CHECKPOINT_DIR}/netD_epoch_{epoch+1}.pth")
    print(f"✅ Epoch {epoch+1} complete — samples and checkpoints saved.")

print("🏁 Training Complete.")