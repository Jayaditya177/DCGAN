import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
image_size = 64  # Output image size
batch_size = 128  # Batch size for training
nz = 100  # Latent vector size (size of random noise input)
num_epochs = 25  # Number of training epochs
lr = 0.0002  # Learning rate
beta1 = 0.5  # Beta1 hyperparameter for Adam optimizer

# Dataset & DataLoader
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

dataset = dset.CelebA(root="./data", download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Generator Model
class Generator(nn.Module):
    """
    The Generator network for DCGAN.
    Uses transposed convolutional layers to generate images from a latent space.
    """
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, 512, 4, 1, 0, bias=False),  # Upsample from latent space (nz) to 4x4x512
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # 4x4x512 -> 8x8x256
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # 8x8x256 -> 16x16x128
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # 16x16x128 -> 32x32x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),  # 32x32x64 -> 64x64x3
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, input):
        return self.main(input)

# Discriminator Model
class Discriminator(nn.Module):
    """
    The Discriminator network for DCGAN.
    Uses convolutional layers to classify real vs. fake images.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),  # 64x64x3 -> 32x32x64
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 32x32x64 -> 16x16x128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # 16x16x128 -> 8x8x256
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # 8x8x256 -> 4x4x512
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),  # 4x4x512 -> 1x1x1
            nn.Sigmoid()  # Output probability
        )

    def forward(self, input):
        return self.main(input).view(-1)

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss function & Optimizers
criterion = nn.BCELoss()
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        
        # Train Discriminator
        optimizerD.zero_grad()
        label_real = torch.ones(batch_size, device=device)
        output_real = discriminator(real_images).view(-1)
        loss_real = criterion(output_real, label_real)
        
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_images = generator(noise).detach()
        label_fake = torch.zeros(batch_size, device=device)
        output_fake = discriminator(fake_images).view(-1)
        loss_fake = criterion(output_fake, label_fake)
        
        lossD = loss_real + loss_fake
        lossD.backward()
        optimizerD.step()
        
        # Train Generator
        optimizerG.zero_grad()
        fake_images = generator(noise)
        output = discriminator(fake_images).view(-1)
        lossG = criterion(output, label_real)
        lossG.backward()
        optimizerG.step()
        
        # Print progress every 100 batches
        if i % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(dataloader)} Loss D: {lossD.item():.4f}, Loss G: {lossG.item():.4f}")
    
    # Save checkpoint every few epochs
    if (epoch + 1) % 5 == 0:
        torch.save(generator.state_dict(), f"generator_epoch_{epoch+1}.pth")
