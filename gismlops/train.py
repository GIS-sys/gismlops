import torch
from gismlops.model import NeuralNetwork
from gismlops.utils import epochTest, epochTrain, getDevice
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# new model
def train():
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    batch_size = 64

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # Get cpu, gpu or mps device for training.
    device = getDevice()
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        epochTrain(train_dataloader, model, loss_fn, optimizer, device)
        epochTest(test_dataloader, model, loss_fn, device)
    print("Done!")

    torch.save(model.state_dict(), "data/model.pth")
    print("Saved PyTorch Model State to data/model.pth")


if __name__ == "__main__":
    train()
