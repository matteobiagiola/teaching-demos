import numpy as np
import torch
from torchvision import transforms
import torchvision
from torch.utils.data import Dataset


def get_train_dataset(dataset_name: str) -> Dataset:
    """
    Get the training dataset based on the dataset name.
    Currently supports only MNIST.
    """
    assert (
        dataset_name.lower() == "mnist"
    ), f"Dataset {dataset_name.lower()} not supported"

    return torchvision.datasets.MNIST(
        "data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )


def get_test_dataset(dataset_name: str) -> Dataset:
    """
    Get the test dataset based on the dataset name.
    Currently supports only MNIST.
    """
    assert (
        dataset_name.lower() == "mnist"
    ), f"Dataset {dataset_name.lower()} not supported"

    return torchvision.datasets.MNIST(
        "data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )


def tensor_image_to_array_image(tensor_image: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor image to a numpy array image.
    Assumes the tensor image has a batch size of 1 and a single channel.
    """
    # image needs to come from a dataloader
    assert (
        len(tensor_image.size()) == 4
    ), f"Tensor image should have batch size, image channel and the 2D dimensions (4). Found: {len(tensor_image.size())}"
    assert (
        tensor_image.size()[0] == 1
    ), f"Batch size should be equal to 1. Found: {tensor_image.size()[0]}"
    assert (
        tensor_image.size()[1] == 1
    ), f"Channel size should be equal to 1. Found: {tensor_image.size()[1]}"

    array_image = tensor_image.cpu().numpy().squeeze()
    return array_image


def array_image_to_tensor_image(array_image: np.ndarray) -> torch.Tensor:
    """
    Convert a numpy array image to a tensor image.
    Assumes the array image has 2 dimensions (height, width).
    """
    assert (
        len(array_image.shape) == 2
    ), f"Array image needs to have two dimensions, corresponding to the 2D dimensions of the image. Found: {len(array_image.shape)}"
    # adds two leading dimensions (i.e., dim=0), i.e., from (28, 28) to (1, 1, 28, 28). The first dimension is the batch size,
    # the second dimension is the number of channels (there is only one channel, as the image is grayscale)
    return (
        torch.tensor(array_image, dtype=torch.float32).unsqueeze(dim=0).unsqueeze(dim=0)
    )
