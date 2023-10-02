import torch
from gismlops.model import NeuralNetwork
from gismlops.utils import getDevice
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
    model.load_state_dict(torch.load("model.pth"))

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

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f"Predicted: {predicted}, Actual: {actual}")
        return predicted, actual


if __name__ == "__main__":
    infer()
