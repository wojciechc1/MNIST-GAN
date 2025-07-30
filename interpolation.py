import torch
import imageio
import numpy as np
from model import Generator
import torchvision.utils as vutils

import imageio
import numpy as np


# ustawienia
latent_dim = 100
num_classes = 10
embedding_dim = 50
steps = 60
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

start_label = 1
end_label = 9


generator = Generator().to(device)
generator.load_state_dict(torch.load("./saved_models/g1.pth"))
generator.eval()


z = torch.randn(1, latent_dim).to(device)  # jeden losowy wektor z dla wszystkich

emb_start = generator.label_emb(torch.tensor(start_label, device=device))  # [latent_dim]
emb_end = generator.label_emb(torch.tensor(end_label, device=device))      # [latent_dim]

alphas = torch.linspace(0, 1, steps).to(device)

images = []
for alpha in alphas:
    emb = (1 - alpha) * emb_start + alpha * emb_end  # interpolowany embedding
    input_vec = z + emb.unsqueeze(0)  # warunkowane wejście [1, latent_dim]
    with torch.no_grad():
        img = generator.model(input_vec).cpu().squeeze(0)  # [1,28,28] -> [28,28]
        img = (img + 1) / 2  # z [-1,1] do [0,1]
        img = (img * 255).clamp(0,255).byte().squeeze().numpy()
        images.append(img)

imageio.mimsave("interpolation.gif", images, duration=0.3, loop=0)
print("Zapisano: interpolation.gif")