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