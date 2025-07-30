from model import Generator

import torch
import torch.optim as optim
import torchvision.utils as vutils

from utils import plot_losses


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

generator = Generator()
generator.load_state_dict(torch.load("./saved_models/generator.pth"))


generator.eval()

latent_dim = 100

for num in range(10):
    with torch.no_grad():
        z = torch.randn(1, latent_dim, device=device)
        labels = torch.ones(1, dtype=torch.long) * num
        fake_images = generator(z, labels)
        vutils.save_image(fake_images, f"num_{num}.png", normalize=True)
