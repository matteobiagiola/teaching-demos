import argparse
import os
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms
from net import Net


def fgsm_attack(
    model: nn.Module,
    image: torch.Tensor,
    original_target: torch.Tensor,
    target: torch.Tensor,
    epsilon: float,
    targeted: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform a targeted FGSM attack on the input image to misclassify it as the target class.
    Args:
        model: The pretrained model to attack.
        image: The original input image tensor of shape (1, C, H, W).
        original_target: The original class label tensor.
        target: The target class towards which to compute the gradient.
        epsilon: The perturbation magnitude.
        targeted: Whether to perform a targeted attack (default: True).
    Returns:
        adversarial_image: The perturbed image tensor.
        pred: The predicted class tensor after the attack.
        confidence: The confidence of the prediction.
    """

    # unnormalize the image because epsilon is in [0,1] range
    image_clone = image.clone()
    image_clone = image_clone * 0.3081 + 0.1307

    image_clone.requires_grad = True

    # Forward pass
    output = model(image_clone)
    if targeted:
        loss = F.cross_entropy(output, target)
    else:
        loss = F.cross_entropy(output, original_target)

    # Backward pass
    loss.backward()

    perturbation = epsilon * image_clone.grad.data.sign()

    # print(f"Original image: {image_clone.shape}")
    # print(image_clone.view(-1))
    # print(f"Gradient: {image_clone.grad.data.shape}")
    # print(image_clone.grad.data.view(-1))
    # print(f"Gradient sign: {image_clone.grad.data.sign().shape}")
    # print(image_clone.grad.data.sign().view(-1))
    # print(f"Perturbation: {perturbation.shape}")
    # print(perturbation.view(-1))
    # exit(0)

    if targeted:
        # Minus sign because we are trying to minimize loss of target class,
        # i.e., we want to move image towards target class
        perturbed_image = image_clone - perturbation
    else:
        # Plus sign because we are trying to maximize loss of original class,
        # i.e., we want to move image away from original class
        perturbed_image = image_clone + perturbation

    # renormalize perturbed image back to original scale
    perturbed_image = transforms.Normalize((0.1307,), (0.3081,)).forward(
        perturbed_image.detach()
    )

    # Re-classify the perturbed image
    with torch.no_grad():
        output = model(perturbed_image)
        probs = F.softmax(output, dim=1)
        confidence = probs[0, original_target]
        pred = output.argmax(dim=1, keepdim=True)

    return perturbed_image, pred, confidence


def main():

    parser = argparse.ArgumentParser(
        description="Attack an MNIST pretrained model using FGSM"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the trained model"
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.01, help="Perturbation magnitude for FGSM"
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=-1,
        choices=list(range(-1, 10)),
        help="Target class for the targeted attack (must be different from original class). Default -1 means untargeted attack.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=1.0,
        help="Confidence threshold for choosing sample to attack. Only samples with confidence below this threshold will be considered.",
    )
    args = parser.parse_args()

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_path = args.model_path
    assert os.path.exists(model_path), f"Model path {model_path} does not exist."
    epsilon = args.epsilon
    assert epsilon > 0.0, "Epsilon must be positive."
    target_class = args.target_class
    confidence_threshold = args.confidence_threshold
    assert 0.0 < confidence_threshold <= 1.0, "Confidence threshold must be in (0, 1]."

    model = Net(num_classes=10)
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu")), strict=True
    )

    model.eval()

    # Get a test image
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    test_dataset = datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )

    index_to_select = None
    for idx, (image, target) in enumerate(test_dataset):
        image = image.unsqueeze(0)  # add batch dimension
        target = torch.tensor([target])

        with torch.no_grad():
            output = model(image)
            probs = F.softmax(output, dim=1)
            confidence = probs[0, target]
            pred = output.argmax(dim=1, keepdim=True)
            if (
                pred.item() == target.item()
                and confidence.item() < confidence_threshold
            ):
                index_to_select = idx
                break

    assert (
        index_to_select is not None
    ), "No suitable test sample found under the confidence threshold."
    image, target = test_dataset[index_to_select]
    image = image.unsqueeze(0)  # add batch dimension
    target = torch.tensor([target])

    if target_class != -1:
        # Untargeted attack: choose a target class different from original
        assert (
            target_class != target.item()
        ), "Target class must be different from original class."
    target_class = torch.tensor([target_class])

    plt.figure()
    plt.imshow(image.squeeze().numpy(), cmap="gray")
    plt.title(
        f"Original Image - Label: {target.item()} - Confidence: {confidence.item():.4f}"
    )
    plt.axis("off")
    plt.savefig(f"original_image_{target.item()}.png")

    # Generate adversarial example
    adversarial_image, pred, confidence = fgsm_attack(
        model=model,
        image=image,
        original_target=target,
        target=target_class,
        epsilon=epsilon,
        targeted=(target_class.item() != -1),
    )

    plt.figure()
    plt.imshow(adversarial_image.squeeze().numpy(), cmap="gray")
    plt.title(
        f"Adversarial Image - Target Label: {target.item()} - Predicted: {pred.item()} - Confidence: {confidence.item():.4f}"
    )
    plt.axis("off")
    if target_class.item() == -1:
        plt.savefig(f"adversarial_image_{target.item()}_untargeted.png")
    else:
        plt.savefig(f"adversarial_image_{target.item()}_to_{target_class.item()}.png")


if __name__ == "__main__":
    main()
