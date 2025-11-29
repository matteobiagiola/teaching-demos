from typing import Optional, Tuple
from net import Net
from torchvision import datasets, transforms
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.nn as nn


def get_adversarial_direction_untargeted_fgsm(
    model: nn.Module, image: torch.Tensor, label: torch.Tensor
) -> torch.Tensor:
    """
    Generates the adversarial direction using the Fast Gradient Sign Method (FGSM).

    :param model: The model used for generating adversarial examples
    :type model: nn.Module
    :param image: The input image to be perturbed
    :type image: torch.Tensor
    :param label: The true label of the input image
    :type label: torch.Tensor
    :return: The adversarial direction as a tensor
    :rtype: Tensor
    """

    image_adv = image.clone().detach().requires_grad_(True)
    output = model(image_adv)
    loss = F.cross_entropy(output, label)
    loss.backward()
    direction = torch.sign(image_adv.grad)
    direction = direction / direction.norm()
    direction = direction.view(-1)

    return direction


def get_adversarial_direction_untargeted_raw(
    model: nn.Module, image: torch.Tensor, label: torch.Tensor
) -> torch.Tensor:
    """
    Generates the adversarial direction to perturb the image away from the true class.

    :param model: The model used for generating adversarial examples
    :type model: nn.Module
    :param image: The input image to be perturbed
    :type image: torch.Tensor
    :param label: The true label of the input image
    :type label: torch.Tensor
    :return: The adversarial direction as a tensor
    :rtype: Tensor
    """

    image_adv = image.clone().detach().requires_grad_(True)
    output = model(image_adv)
    loss = F.cross_entropy(output, label)
    loss.backward()
    direction = image_adv.grad.clone().detach()
    direction = direction / direction.norm()
    direction = direction.view(-1)

    return direction


def get_adversarial_direction_targeted(
    model: nn.Module, image: torch.Tensor, target_class: torch.Tensor
) -> torch.Tensor:
    """
    Generates the adversarial direction to perturb the image towards the target class.

    :param model: The model used for generating adversarial examples
    :type model: nn.Module
    :param image: The input image to be perturbed
    :type image: torch.Tensor
    :param target_class: The target class for the adversarial attack
    :type target_class: torch.Tensor
    :return: The adversarial direction as a tensor
    :rtype: Tensor
    """

    image_adv = image.clone().detach().requires_grad_(True)
    output = model(image_adv)
    # Negative loss to maximize probability of target class
    loss = -F.cross_entropy(output, target_class)
    loss.backward()
    direction = image_adv.grad.clone().detach()
    direction = direction / direction.norm()
    direction = direction.view(-1)

    return direction


def plot_decision_boundary(
    model: torch.nn.Module,
    base_image: torch.Tensor,
    true_label: int,
    filename: str,
    tridimensional_plot: bool = False,
    grid_size: int = 50,
    noise_scale: float = 5.0,
    dir_x: Optional[torch.Tensor] = None,
    dir_y: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Plots the decision boundary of the model around the given image in 2D perturbation space.

    :param model: Trained PyTorch model
    :type model: torch.nn.Module
    :param base_image: Input image tensor
    :type base_image: torch.Tensor
    :param true_label: True label of the input image
    :type true_label: int
    :param filename: Filename to save the plot
    :type filename: str
    :param tridimensional_plot: Whether to create a 3D plot with confidence as z-axis
    :type tridimensional_plot: bool
    :param grid_size: Number of points in the grid for plotting
    :type grid_size: int
    :param noise_scale: Scale of the noise to be added
    :type noise_scale: float
    :param dir_x: Optional predefined direction for x-axis: if None, a random direction is used else adversarial direction
    :type dir_x: torch.Tensor, optional
    :param dir_y: Optional predefined direction for y-axis
    :type dir_y: torch.Tensor, optional
    """

    model.eval()

    plt.figure()
    plt.imshow(base_image.squeeze(), cmap="gray")
    plt.title(f"Original Image (Label: {true_label})")
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(f"original_image_{true_label}.svg")
    plt.close()

    base_image_shape = base_image.shape

    # Get base image
    img_flat = base_image.flatten()

    if dir_x is None:
        # Random direction for x-axis
        dir_x = torch.randn_like(img_flat)
        dir_x = dir_x / torch.norm(dir_x)

    if dir_y is None:
        # Random direction for y-axis (orthogonal to x)
        dir_y = torch.randn_like(img_flat)
        dir_y = dir_y - (dir_y @ dir_x) * dir_x
        dir_y = dir_y / torch.norm(dir_y)

    # Create grid
    x_range = np.linspace(-noise_scale, noise_scale, grid_size)
    x_range = np.concatenate((x_range[x_range < 0], [0], x_range[x_range > 0]))
    y_range = np.linspace(-noise_scale, noise_scale, grid_size)
    y_range = np.concatenate((y_range[y_range < 0], [0], y_range[y_range > 0]))
    predictions = np.zeros((grid_size + 1, grid_size + 1))
    confidences = np.zeros((grid_size + 1, grid_size + 1))

    to_plot_images = []

    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            # Create perturbed image
            perturbed = img_flat + x * dir_x + y * dir_y
            perturbed = perturbed.reshape(-1, *base_image_shape[1:])

            # Get prediction
            with torch.no_grad():
                output = model(perturbed)
                probs = F.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                confidence = probs[0, true_label]

                predictions[j, i] = pred.item()
                confidences[j, i] = confidence.item()
                if pred != true_label:
                    to_plot_images.append((perturbed, pred.item(), i, j, x, y))

    for perturbed, pred, i, j, x, y in to_plot_images:

        # in the random direction plot, plot misclassified image at the extreme right (y == 0)
        # in the adversarial direction plot, plot the first misclassified image following the x direction (y == 0)
        plot_condition = (
            i > (grid_size + 1) // 2 and j == (grid_size + 1) // 2
            if "adv" in filename
            else j == grid_size and i == (grid_size + 1) // 2
        )

        if plot_condition:
            plt.figure()
            plt.imshow(perturbed.squeeze(), cmap="gray")
            plt.title(f"Misclassified Image (True: {true_label}, Pred: {pred})")
            plt.axis("off")
            plt.tight_layout()
            if "adv" in filename:
                plt.savefig(
                    f"misclassified_images_adv_pred_{pred}_x_{x:.2f}_y_{y:.2f}.svg"
                )
            else:
                plt.savefig(
                    f"misclassified_images_rand_pred_{pred}_x_{x:.2f}_y_{y:.2f}.svg"
                )
            plt.close()
            break

    if tridimensional_plot:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Map the prediction labels to colors
        cmap = plt.get_cmap("tab10", 10)
        norm = plt.Normalize(0, 9)

        # Get colors for the surface based on the labels (preds), not the height
        # facecolors requires values normalized between 0-1
        face_colors = cmap(norm(predictions))

        X, Y = np.meshgrid(x_range, y_range)

        ax.scatter(
            0,
            0,
            confidences[(grid_size) // 2, (grid_size) // 2] + 0.05,
            edgecolors="k",
            color="k",
            marker="*",
            s=200,
            label=f"Original (Label: {true_label})",
        )

        # Plot surface
        # X, Y: Grid coordinates
        # Z: Confidence (height)
        # facecolors: The class label
        _ = ax.plot_surface(
            X,
            Y,
            confidences,
            facecolors=face_colors,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True,
            shade=False,
            alpha=1.0,
        )

        # Custom mapping for legend
        m = cm.ScalarMappable(cmap=cmap, norm=norm)
        m.set_array([])
        cbar = plt.colorbar(m, ax=ax, shrink=0.5, aspect=10, ticks=range(10))
        cbar.set_label("Predicted Class (Color)")

        ax.set_title(
            f"3D Model Confidence on Correct Prediction Landscape (Center: True Label {label})"
        )
        ax.set_xlabel(
            "Perturbation in X direction (Adversarial)"
            if "adv" in filename
            else "Perturbation in X direction (Random)"
        )
        ax.set_ylabel("Perturbation in Y direction (Random)")
        ax.set_zlabel("Model Confidence (Probability)")
        ax.set_zlim(0, 1.0)

        plt.tight_layout()
        plt.savefig(f"{filename}_3d.svg")
        plt.close()

    else:

        # Plot
        plt.figure(figsize=(10, 8))

        # Plot decision boundary
        im = plt.imshow(
            predictions,
            extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]],
            origin="lower",
            cmap="tab10",
            vmin=0,
            vmax=9,
            aspect="auto",
        )
        plt.plot(0, 0, "w*", markersize=15, label=f"Original (Label: {true_label})")
        plt.xlabel(
            "Perturbation in X direction (Adversarial)"
            if "adv" in filename
            else "Perturbation in X direction (Random)"
        )
        plt.ylabel("Perturbation in Y direction (Random)")
        plt.title("Decision Boundary")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.colorbar(im, label="Predicted Class")
        plt.tight_layout()
        plt.savefig(f"{filename}_2d.svg")
        plt.close()

    return dir_x, dir_y


if __name__ == "__main__":

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = Net(num_classes=10)
    model.load_state_dict(
        torch.load("mnist.pt", map_location=torch.device("cpu")), strict=True
    )
    model.eval()

    # Get a test image
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    test_dataset = datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )

    # find an image correctly classified but with low confidence; easier to perturb
    index = -1
    for i in range(len(test_dataset)):

        image, label = test_dataset[i]
        image_batch = image.unsqueeze(0)

        with torch.no_grad():
            output = model(image_batch)
            pred = output.argmax(dim=1).item()
            probs = F.softmax(output, dim=1)
            confidence = probs[0, label].item()

        if pred == label and confidence < 0.9:

            print("Confidence of selected image:", confidence)

            index = i
            break

    if index == -1:
        print("No successful adversarial example found in the test set.")
        exit(0)

    image, label = test_dataset[index]
    image_batch = image.unsqueeze(0)

    print("\nGenerating decision boundary plot...")
    _, dir_y = plot_decision_boundary(
        model=model,
        base_image=image_batch,
        true_label=label,
        grid_size=100,
        noise_scale=40,
        filename=f"decision_boundary_rand_{label}",
    )

    # the target class is 8 because the image chose above (which is a 3) is the most vulnerable to being perturbed to class 8
    target_class = 8
    dir_x = get_adversarial_direction_targeted(
        model=model,
        image=image_batch,
        target_class=torch.tensor([target_class]),
    )

    _ = plot_decision_boundary(
        model=model,
        base_image=image_batch,
        true_label=label,
        grid_size=100,
        noise_scale=40,
        dir_y=dir_y,
        dir_x=dir_x,
        filename=f"decision_boundary_adv_targeted_{label}_to_{target_class}",
    )

    _ = plot_decision_boundary(
        model=model,
        base_image=image_batch,
        true_label=label,
        tridimensional_plot=True,
        grid_size=100,
        noise_scale=40,
        dir_y=dir_y,
        dir_x=dir_x,
        filename=f"decision_boundary_adv_targeted_{label}_to_{target_class}",
    )
