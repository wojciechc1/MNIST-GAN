from get_data import get_data
from model import Generator, Discriminator

import torch
import torch.optim as optim
import torchvision.utils as vutils

batch_size = 32
train_size = 10000
latent_dim = 100 # losowe
epochs = 30

# data loader
train_loader, test_loader = get_data(train_size=train_size, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator()
discriminator = Discriminator()

generator.to(device)
discriminator.to(device)

optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))


criterion = torch.nn.BCELoss()

#metrics
all_lossesD = []
all_lossesG = []

for epoch in range(epochs):
    print('epoch', epoch)
    total_lossD = 0
    total_lossG = 0
    countD = 0

    for i, (real_images, labels) in enumerate(train_loader):
        # trenowanie dyskryminatora rzadziej
        batch_size = real_images.size(0)
        if i % 3 == 0:
            countD += 1
            # dyskryminator

            real_images = real_images.to(device)
            z = torch.randn(batch_size, latent_dim, device=device)

            fake_images = generator(z, labels).detach()  # eval

            # prawdziwe obrazy
            D_real = discriminator(real_images, labels)
            loss_real = criterion(D_real, torch.ones_like(D_real))
            print('loss_real', loss_real)

            # falszywe obrazy
            D_fake = discriminator(fake_images, labels)
            loss_fake = criterion(D_fake, torch.zeros_like(D_fake))
            print('loss_fake', loss_fake)

            loss_D = loss_real + loss_fake

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            total_lossD += loss_D.item()

        # generator
        generator.train()
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_images = generator(z, labels)
        D_fake = discriminator(fake_images, labels)

        # generator chce oszukać dyskryminator
        loss_G = criterion(D_fake, torch.ones_like(D_fake))

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        total_lossG += loss_G.item()

    print('D', total_lossD/len(train_loader), 'G', total_lossG/len(train_loader))
    avg_lossD = total_lossD / countD if countD > 0 else 0
    avg_lossG = total_lossG / len(train_loader)

    all_lossesD.append(avg_lossD)
    all_lossesG.append(avg_lossG)

    generator.eval()
    with torch.no_grad():
        z = torch.randn(64, latent_dim, device=device)
        labels = torch.ones(64, dtype=torch.long) * 5
        fake_images = generator(z, labels)
        vutils.save_image(fake_images, f"epoch_{epoch}.png", nrow=8, normalize=True)


torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")

import matplotlib.pyplot as plt
def plot_losses(all_lossesD, all_lossesG):
    plt.figure(figsize=(10,5))
    plt.plot(all_lossesD, label='Discriminator Loss', color='red')
    plt.plot(all_lossesG, label='Generator Loss', color='blue')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Losses during training')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_losses(all_lossesD, all_lossesG)