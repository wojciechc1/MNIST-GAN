import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, 28 * 28)

        self.model = nn.Sequential(
            # 1x28x28 -> 64x14x14
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 64x14x14 -> 128x7x7
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        label_img = self.label_emb(labels).view(-1, 1, 28, 28)  # [B, 1, 28, 28]
        x = torch.cat([x, label_img], dim=1)

        return self.model(x)


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.label_emb = nn.Embedding(10, latent_dim)

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True),
            nn.Unflatten(1, (128, 7, 7)),

            # 128x7x7 -> 64x14x14
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 64x14x14 -> 1x28x28
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_embedding = self.label_emb(labels)  # [B, latent_dim]
        x = z + label_embedding  # warunkujemy noise
        return self.model(x)
