from functools import lru_cache

import numpy as np
from PIL import Image
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from tritonclient.utils import np_to_triton_dtype


@lru_cache
def get_client():
    return InferenceServerClient(url="0.0.0.0:8500")


def print_image(np_img):
    for row in np_img:
        for pixel in row:
            symbol = None
            if pixel < 32:
                symbol = " "
            elif pixel < 64:
                symbol = "."
            elif pixel < 128:
                symbol = "-"
            elif pixel < 196:
                symbol = "+"
            else:
                symbol = "B"
            print(symbol, end="")
        print()


def load_image(path_to_image):
    img = Image.open(path_to_image)
    np_img = np.array(img)[:, :, 0]
    return np_img


def call_triton_clothing(np_img):
    triton_client = get_client()

    np_img = np.expand_dims(np_img, axis=(0, 1))
    input_image = InferInput(
        name="inputs", shape=np_img.shape, datatype=np_to_triton_dtype(np.float32)
    )
    input_image.set_data_from_numpy(np_img.astype(np.float32))

    infer_output = InferRequestedOutput("predictions")
    infer_response = triton_client.infer(
        "onnx-clothing", [input_image], outputs=[infer_output]
    )
    return np.argmax(infer_response.as_numpy("predictions")[0])


LABELS = [
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
EXAMPLES = [
    {"img": "../data/examples/tshirt.jpg", "label": 0},
    {"img": "../data/examples/ankle_boot.jpg", "label": 9},
    {"img": "../data/examples/bag.jpg", "label": 8},
    {"img": "../data/examples/sneaker.jpg", "label": 7},
]


def main():
    for example in EXAMPLES:
        np_img = load_image(example["img"])
        infer_output = call_triton_clothing(np_img)
        print(f"got: {LABELS[infer_output]}, real: {LABELS[example['label']]}")
        print_image(np_img)


if __name__ == "__main__":
    main()
