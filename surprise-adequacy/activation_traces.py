"""Shows what an activation trace (AT) is.

The script takes three images from the MNIST test set: two images of the same
digit and one image of a different digit. It sends the images through the
trained network. For each image, it shows the values that the neurons of one
layer compute. This vector is the AT of the image at that layer. Surprise
adequacy compares the AT of a new input with the ATs of the training set.
"""

import argparse
import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

import torch_utils

# Color of the bars (one data series for each plot).
COLOR = "#2a78d6"


def shape_hook(
    name: str,
) -> Callable[[nn.Module, tuple[torch.Tensor], torch.Tensor], None]:
    """Makes a forward hook that prints the shape of the output of a layer.

    Args:
        name: Name of the layer.

    Returns:
        The forward hook.
    """

    def hook(
        module: nn.Module,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        del module, inputs  # The hook uses only the output of the layer.
        print(f"  {name:6s}: {list(output.shape)}")

    return hook


def main() -> None:
    parser = argparse.ArgumentParser(description="Activation traces example")
    parser.add_argument(
        "--layer",
        type=str,
        choices=torch_utils.LAYERS,
        default="fc1",
        help="Layer that gives the activation traces (default: fc1)",
    )
    parser.add_argument(
        "--digits",
        type=int,
        nargs=3,
        default=[7, 7, 1],
        help=(
            "Digits of the three test images. The script takes the first "
            "image of each digit. For a repeated digit, it takes the next "
            "image (default: 7 7 1)"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join("models", "mnist_net.pt"),
        help="Trained model (default: models/mnist_net.pt)",
    )

    args = parser.parse_args()
    layer = args.layer

    device = torch.device("cpu")
    model = torch_utils.load_model(args.model, device)
    test_dataset = torch_utils.get_test_dataset()

    # Find the test images. For a repeated digit, take the next image with
    # that label.
    labels = test_dataset.targets.numpy()
    indices = []
    for digit in args.digits:
        candidates = [
            i for i in np.where(labels == digit)[0] if i not in indices
        ]
        indices.append(candidates[0])
    images = torch.stack([test_dataset[i][0] for i in indices])

    # 1. Shape of the output of each layer, for the first image.
    print(
        "Output shape of each layer for one image, "
        "[batch, channels, height, width] or [batch, neurons]:"
    )
    print(f"  {'input':6s}: {list(images[:1].shape)}")
    handles = [
        getattr(model, name).register_forward_hook(shape_hook(name))
        for name in torch_utils.LAYERS + ["fc2"]
    ]
    with torch.no_grad():
        model(images[:1])
    for handle in handles:
        handle.remove()

    # 2. ATs of the three images at the selected layer.
    traces = []
    handle = getattr(model, layer).register_forward_hook(
        torch_utils.activation_trace_hook(traces)
    )
    with torch.no_grad():
        predictions = model(images).argmax(dim=1).numpy()
    handle.remove()
    ats = traces[0].numpy()

    if layer.startswith("conv"):
        print(
            f"\n{layer} has {ats.shape[1]} channels. The AT of an image is the"
            " mean of each feature map after the ReLU: "
            f"{ats.shape[1]} values."
        )
    else:
        print(
            f"\n{layer} has {ats.shape[1]} neurons. The AT of an image is the"
            " output of each neuron after the ReLU: "
            f"{ats.shape[1]} values."
        )

    for i, (index, at, prediction) in enumerate(zip(indices, ats, predictions)):
        print(
            f"  image {i} (test index {index}, label {labels[index]}, "
            f"predicted {prediction}): {(at == 0).sum()} of {len(at)} values "
            "are 0 (inactive neurons)"
        )

    print("\nEuclidean distance between the ATs:")
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        distance = np.linalg.norm(ats[i] - ats[j])
        print(
            f"  image {i} (label {labels[indices[i]]}) - "
            f"image {j} (label {labels[indices[j]]}): {distance:.2f}"
        )

    _, axes = plt.subplots(
        3, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1, 6]}
    )
    y_max = ats.max() * 1.05
    for row, (index, at, prediction) in enumerate(
        zip(indices, ats, predictions)
    ):
        axes[row, 0].imshow(test_dataset.data[index], cmap="gray")
        axes[row, 0].set_title(
            f"image {row}: label {labels[index]}, pred {prediction}",
            fontsize=9,
        )
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        axes[row, 1].bar(np.arange(len(at)), at, color=COLOR, width=0.8)
        axes[row, 1].set_xlim(-1, len(at))
        axes[row, 1].set_ylim(0, y_max)
        axes[row, 1].set_ylabel("activation")
        axes[row, 1].set_title(
            f"activation trace at layer {layer} ({len(at)} values)", fontsize=9
        )
    axes[-1, 1].set_xlabel("neuron" if layer.startswith("fc") else "channel")

    plt.tight_layout()
    plt.savefig(f"activation_traces_{layer}.png", bbox_inches="tight", dpi=150)
    print(f"\nPlot saved in activation_traces_{layer}.png")


if __name__ == "__main__":
    main()
