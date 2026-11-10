"""Functions for the MNIST dataset, the classifier and the generator."""

import os

import torch
from torchvision import datasets
from torchvision import transforms

import gan
import net

# The classifier under test gets images normalized with the mean and the
# standard deviation of the MNIST training set.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def get_train_dataset() -> datasets.MNIST:
    """Returns the MNIST training set with pixels in [0, 1].

    The GAN trains on these images. Downloads them if necessary.
    """
    return datasets.MNIST(
        root="data", train=True, download=True, transform=transforms.ToTensor()
    )


def get_test_dataset() -> datasets.MNIST:
    """Returns the MNIST test set with pixels in [0, 1].

    Downloads it if necessary.
    """
    return datasets.MNIST(
        root="data", train=False, download=True, transform=transforms.ToTensor()
    )


def load_classifier(model_path: str, device: torch.device) -> net.Net:
    """Loads the classifier under test in evaluation mode."""
    assert os.path.exists(model_path), f"{model_path} does not exist."
    classifier = net.Net().to(device=device)
    classifier.load_state_dict(
        state_dict=torch.load(f=model_path, weights_only=True, map_location=device)
    )
    classifier.eval()
    return classifier


def load_generator(model_path: str, device: torch.device) -> gan.Generator:
    """Loads the generator in evaluation mode."""
    assert os.path.exists(model_path), (
        f"{model_path} does not exist. "
        "Train the GAN first with: python -m training.train_gan"
    )
    generator = gan.Generator().to(device=device)
    generator.load_state_dict(
        state_dict=torch.load(f=model_path, weights_only=True, map_location=device)
    )
    generator.eval()
    return generator


def classify(classifier: net.Net, images: torch.Tensor) -> torch.Tensor:
    """Gives the output of the classifier for images with pixels in [0, 1].

    Args:
        classifier: The classifier under test.
        images: Images with shape [batch, 1, 28, 28].

    Returns:
        The log-probability of each class, with shape [batch, 10].
    """
    normalized_images = (images - MNIST_MEAN) / MNIST_STD
    return classifier(images=normalized_images)


def boldness(images: torch.Tensor) -> torch.Tensor:
    """Counts the pixels above 127 (of 255) in each image.

    Args:
        images: Images with shape [batch, 1, 28, 28] and pixels in [0, 1].

    Returns:
        The number of bright pixels of each image, with shape [batch].
    """
    return (images * 255 > 127).flatten(start_dim=1).sum(dim=1)
