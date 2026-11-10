"""Conditional GAN for the MNIST dataset.

The architecture follows the guidelines of DCGAN (Radford et al., 2016,
https://arxiv.org/abs/1511.06434): convolutions instead of pooling, batch
normalization, ReLU in the generator and LeakyReLU in the discriminator.
The class conditioning follows the conditional GAN of Mirza and Osindero
(2014, https://arxiv.org/abs/1411.1784): the generator and the discriminator
also get the label as input.
"""

import torch
from torch import nn
import torch.nn.functional as F

LATENT_DIM = 128
NUM_CLASSES = 10

# Indices of the hidden layers of the generator.
# EARLY_LAYERS = (0, 1, 2, 3)
EARLY_LAYERS = (0, 1)
# LATE_LAYERS = (4, 5, 6, 7)
LATE_LAYERS = (6, 7)


class Generator(nn.Module):
    """Generator that makes a 28x28 image of a given digit.

    The generator has eight hidden layers. Each hidden layer is a linear or
    convolutional layer followed by batch normalization and ReLU.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            modules=[
                # 0: latent vector and label -> 128 x 7 x 7.
                nn.Sequential(
                    nn.Linear(
                        in_features=LATENT_DIM + NUM_CLASSES, out_features=128 * 7 * 7
                    ),
                    nn.Unflatten(dim=1, unflattened_size=(128, 7, 7)),
                    nn.BatchNorm2d(num_features=128),
                ),
                # 1: 128 x 7 x 7.
                _conv_block(in_channels=128, out_channels=128),
                # 2: 64 x 14 x 14.
                _up_block(in_channels=128, out_channels=64),
                # 3: 64 x 14 x 14.
                _conv_block(in_channels=64, out_channels=64),
                # 4: 64 x 14 x 14.
                _conv_block(in_channels=64, out_channels=64),
                # 5: 32 x 28 x 28.
                _up_block(in_channels=64, out_channels=32),
                # 6: 32 x 28 x 28.
                _conv_block(in_channels=32, out_channels=32),
                # 7: 32 x 28 x 28.
                _conv_block(in_channels=32, out_channels=32),
            ]
        )
        self.output = nn.Conv2d(
            in_channels=32, out_channels=1, kernel_size=3, padding=1
        )

    def forward(
        self,
        latent_vectors: torch.Tensor,
        conditioning_labels: torch.Tensor,
        perturbations: dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Makes images.

        Args:
            latent_vectors: Latent vectors with shape [batch, LATENT_DIM].
            conditioning_labels: Labels of digits to make, with shape [batch].
            perturbations: For each hidden layer to perturb, the tensor to add
                to its output before the ReLU. The tensor has the same shape
                as the output of the layer.

        Returns:
            Images with shape [batch, 1, 28, 28] and pixels in [0, 1].
        """
        if perturbations is None:
            perturbations = {}
        one_hot_labels = F.one_hot(
            input=conditioning_labels, num_classes=NUM_CLASSES
        ).float()
        layer_activations = torch.cat(tensors=[latent_vectors, one_hot_labels], dim=1)
        for layer_index, layer in enumerate(self.layers):
            layer_activations = layer(input=layer_activations)
            if layer_index in perturbations:
                layer_activations = layer_activations + perturbations[layer_index]
            layer_activations = F.relu(input=layer_activations)
        output_activations = self.output(input=layer_activations)
        return torch.sigmoid(input=output_activations)

    @torch.no_grad()
    def get_activations(
        self, latent_vectors: torch.Tensor, conditioning_labels: torch.Tensor
    ) -> list[torch.Tensor]:
        """Gets the outputs of the hidden layers, before the ReLU.

        Args:
            latent_vectors: Latent vectors with shape [batch, LATENT_DIM].
            conditioning_labels: Digits to make, with shape [batch].

        Returns:
            One tensor for each hidden layer, with shape [batch, ...].
        """
        hidden_layer_activations = []
        one_hot_labels = F.one_hot(
            input=conditioning_labels, num_classes=NUM_CLASSES
        ).float()
        layer_activations = torch.cat(tensors=[latent_vectors, one_hot_labels], dim=1)
        for layer in self.layers:
            layer_activations = layer(input=layer_activations)
            hidden_layer_activations.append(layer_activations)
            layer_activations = F.relu(input=layer_activations)
        return hidden_layer_activations


class Discriminator(nn.Module):
    """Discriminator that tells real images from generated images.

    As in DCGAN, the discriminator is the mirror of the generator: strided
    convolutions instead of transposed convolutions.
    The label goes into the network as 10 extra channels: the channel of the
    label is 1, the other channels are 0.
    """

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            # 32 x 28 x 28. No batch normalization in the first layer, as in
            # DCGAN.
            nn.Conv2d(
                in_channels=1 + NUM_CLASSES, out_channels=32, kernel_size=3, padding=1
            ),
            nn.LeakyReLU(negative_slope=0.2),
            # 64 x 14 x 14.
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.2),
            # 128 x 7 x 7.
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Flatten(),
            nn.Linear(in_features=128 * 7 * 7, out_features=1),
        )

    def forward(
        self, images: torch.Tensor, conditioning_labels: torch.Tensor
    ) -> torch.Tensor:
        """Computes the logit of "real" for each image.

        Args:
            images: Images with shape [batch, 1, 28, 28].
            conditioning_labels: Labels of the images, with shape [batch].

        Returns:
            Logits with shape [batch].
        """
        one_hot_labels = F.one_hot(
            input=conditioning_labels, num_classes=NUM_CLASSES
        ).float()
        label_maps = one_hot_labels[:, :, None, None]
        label_maps = label_maps.expand(-1, -1, images.shape[2], images.shape[3])
        conditioned_images = torch.cat(tensors=[images, label_maps], dim=1)
        real_image_logits = self.network(input=conditioned_images)
        return real_image_logits.squeeze(dim=1)


def init_weights(module: nn.Module) -> None:
    """Initializes the weights as in DCGAN: normal with std 0.02."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(tensor=module.weight, mean=0.0, std=0.02)
        nn.init.zeros_(tensor=module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(tensor=module.weight, mean=1.0, std=0.02)
        nn.init.zeros_(tensor=module.bias)


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Convolution that keeps the size of the feature maps."""
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        ),
        nn.BatchNorm2d(num_features=out_channels),
    )


def _up_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Transposed convolution that doubles the size of the feature maps."""
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        ),
        nn.BatchNorm2d(num_features=out_channels),
    )
