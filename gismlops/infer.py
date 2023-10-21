import hydra
import pandas as pd
import torch
from gismlops.config import Params
from gismlops.model import NeuralNetwork
from gismlops.utils import epochTest, getDevice
from hydra.core.config_store import ConfigStore
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


cs = ConfigStore.instance()
cs.store(name="params", node=Params)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def infer(cfg: Params):
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

    batch_size = 64
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    loss_fn = nn.CrossEntropyLoss()
    answers = epochTest(test_dataloader, model, loss_fn, device)

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    answersDataFrame = pd.DataFrame(answers, columns=["target_index", "predicted_index"])
    answersDataFrame["target_label"] = answersDataFrame["target_index"].map(
        lambda x: classes[x]
    )
    answersDataFrame["predicted_label"] = answersDataFrame["predicted_index"].map(
        lambda x: classes[x]
    )
    answersDataFrame.to_csv("data/test.csv", index=False)
    return answersDataFrame


if __name__ == "__main__":
    infer()
