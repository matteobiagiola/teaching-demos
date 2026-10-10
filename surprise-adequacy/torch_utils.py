"""Functions for the MNIST dataset, the model and the activation traces."""

import os
from typing import Callable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

import net

# Layers of Net that give activation traces (ATs). A ReLU follows each layer.
LAYERS = ["conv1", "conv2", "fc1"]

# The same preprocessing for the training set and the test set.
# ToTensor scales the pixels to [0, 1]. Normalize subtracts the mean and
# divides by the standard deviation of the MNIST training set.
TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)


def get_train_dataset() -> torch.utils.data.Dataset:
    """Returns the MNIST training set. Downloads it if necessary."""
    return datasets.MNIST(
        "data", train=True, download=True, transform=TRANSFORM
    )


def get_test_dataset() -> torch.utils.data.Dataset:
    """Returns the MNIST test set. Downloads it if necessary."""
    return datasets.MNIST(
        "data", train=False, download=True, transform=TRANSFORM
    )


def load_model(model_path: str, device: torch.device) -> net.Net:
    """Loads a trained model.

    Args:
        model_path: Path of the file with the weights of the model.
        device: Device for the model.

    Returns:
        The model in evaluation mode, that is, with the dropout disabled.
    """
    assert os.path.exists(model_path), (
        f"{model_path} does not exist. "
        "Train the model first with: python -m training.train"
    )
    model = net.Net().to(device)
    model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=device)
    )
    model.eval()
    return model


def activation_trace_hook(
    storage: list[torch.Tensor],
) -> Callable[[nn.Module, tuple[torch.Tensor], torch.Tensor], None]:
    """Makes a forward hook that keeps the ATs of a layer.

    Kim et al. use the outputs of the activation functions. Net calls the ReLU
    as a function in forward. Because of this, the hook gets the outputs
    before the ReLU, and it computes the ReLU.
    The output of a convolutional layer has the shape [batch, channels,
    height, width]. The original implementation of Kim et al. replaces each
    feature map with its mean. Because of this, the AT has one value for each
    channel.

    Args:
        storage: List that gets the ATs of each batch, with the shape
            [batch, neurons].

    Returns:
        The forward hook.
    """

    def hook(
        module: nn.Module,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        del inputs  # The hook uses only the output of the layer.
        traces = F.relu(output)
        if isinstance(module, nn.Conv2d):
            traces = traces.mean(dim=(2, 3))
        storage.append(traces.detach().cpu())

    return hook


@torch.no_grad()
def get_activation_traces(
    model: net.Net,
    dataset: torch.utils.data.Dataset,
    layer: str,
    device: torch.device,
    batch_size: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gets the ATs and the predictions of the model for all the dataset.

    Args:
        model: The model under test.
        dataset: The images and their labels.
        layer: Name of the layer that gives the ATs. One of LAYERS.
        device: Device of the model.
        batch_size: Number of images in one forward pass.

    Returns:
        A tuple with three arrays:
        - The ATs, with the shape [images, neurons of the layer].
        - The predicted classes, with the shape [images].
        - The labels, with the shape [images].
    """
    assert layer in LAYERS, f"Layer {layer} is not one of {LAYERS}"

    traces, predictions, labels = [], [], []
    handle = getattr(model, layer).register_forward_hook(
        activation_trace_hook(traces)
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    for images, targets in loader:
        outputs = model(images.to(device))
        predictions.append(outputs.argmax(dim=1).cpu())
        labels.append(targets)
    handle.remove()

    return (
        torch.cat(traces).numpy(),
        torch.cat(predictions).numpy(),
        torch.cat(labels).numpy(),
    )
