from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_data(train_size=10000, test_size=1000, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_data = Subset(train_data, range(train_size))
    test_data = Subset(test_data, range(test_size))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print("[DATA_LOADER] Size of train and test data:", len(train_data), len(test_data))

    return train_loader, test_loader