from get_data import get_data
from model import Generator, Discriminator

import torch
import torch.optim as optim
import torchvision.utils as vutils



# data loader
train_loader, test_loader = get_data()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator()
discriminator = Discriminator()

generator.to(device)
discriminator.to(device)

optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))


criterion = torch.nn.BCELoss()




batch_size = 32
latent_dim = 100
epochs = 50


for epoch in range(epochs):
    print('epoch', epoch)
    total_lossD = 0
    total_lossG = 0
    for real_images, y in train_loader:
        # dyskryminator
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        z = torch.randn(batch_size, latent_dim, device=device)

        fake_images = generator(z).detach()  # eval
        
        # prawdziwe obrazy → label 1
        D_real = discriminator(real_images)
        loss_real = criterion(D_real, torch.ones_like(D_real))
        print('loss_real', loss_real)
        # falszywe obrazy → label 0
        D_fake = discriminator(fake_images)
        loss_fake = criterion(D_fake, torch.zeros_like(D_fake))
        print('loss_fake', loss_fake)

        loss_D = loss_real + loss_fake

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # generator
        generator.train()
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_images = generator(z)
        D_fake = discriminator(fake_images)

        # generator chce oszukać dyskryminator → etykieta 1
        loss_G = criterion(D_fake, torch.ones_like(D_fake))

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
        total_lossD += loss_D.item()
        total_lossG += loss_G.item()

    print('D', total_lossD/len(train_loader), 'G', total_lossG/len(train_loader))

    generator.eval()
    with torch.no_grad():
        z = torch.randn(64, latent_dim, device=device)
        fake_images = generator(z)
        vutils.save_image(fake_images, f"epoch_{epoch}.png", nrow=8, normalize=True)