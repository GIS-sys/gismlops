import pandas as pd
import torch
from gismlops.model import NeuralNetwork
from gismlops.utils import epochTest, getDevice
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# saved
def infer():
    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    device = getDevice()
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load("data/model.pth"))
    model.eval()

    # classes = [
    #     "T-shirt/top",
    #     "Trouser",
    #     "Pullover",
    #     "Dress",
    #     "Coat",
    #     "Sandal",
    #     "Shirt",
    #     "Sneaker",
    #     "Bag",
    #     "Ankle boot",
    # ]

    batch_size = 64
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    loss_fn = nn.CrossEntropyLoss()
    answers = epochTest(test_dataloader, model, loss_fn, device)

    pd.DataFrame(answers).to_csv("data/test.csv", index=False)
    return answers


if __name__ == "__main__":
    infer()
