from model import Generator, Discriminator

import torch
import torch.optim as optim
import torchvision.utils as vutils
from utils import plot_losses, get_data


batch_size = 64
train_size = 1000
latent_dim = 100 # losowe
epochs = 10

d_lr = 0.0002
g_lr = 0.0003

# data loader
train_loader, test_loader = get_data(train_size=train_size, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.set_default_device(device) TODO set and test

generator = Generator()
discriminator = Discriminator()

#generator.load_state_dict(torch.load("g2.pth"))
#discriminator.load_state_dict(torch.load("d2.pth"))

generator.to(device)
discriminator.to(device)

optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=g_lr, betas=(0.5, 0.999))


criterion = torch.nn.BCELoss()

#metrics
all_lossesD = []
all_lossesG = []

def main():
    for epoch in range(epochs):
        print('epoch', epoch)
        total_lossD = 0
        total_lossG = 0
        countD = 0

        for i, (real_images, labels) in enumerate(train_loader):
            # trenowanie dyskryminatora rzadziej
            batch_size = real_images.size(0)
            if i % 1 == 0:
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


    torch.save(generator.state_dict(), "g4.pth")
    torch.save(discriminator.state_dict(), "d4.pth")



    plot_losses(all_lossesD, all_lossesG)


if __name__ == "__main__":
    main()