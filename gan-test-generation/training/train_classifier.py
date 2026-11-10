"""Trains the MNIST classifier under test and saves its weights.

The training is adversarial training (Madry et al., "Towards Deep Learning
Models Resistant to Adversarial Attacks", ICLR 2018,
https://arxiv.org/abs/1706.06083): the classifier learns on adversarial
examples of the training images.
Because of this, small changes to the pixels (for example, spots) do not fool
the classifier easily, and the search of Dunn et al. must change the features
of the digit.

With --faulty, the script injects a fault in the category "Training >
Training data quality" of the taxonomy of Humbatova et al. ("Taxonomy of Real
Faults in Deep Learning Systems", ICSE 2020, https://arxiv.org/abs/1910.11015).
As in Dunn et al. (ISSTA 2021), the training data has intentionally incorrect
labels that depend on a spurious feature.
Here, the bold images of FAULTY_DIGIT have the label INCORRECT_LABEL.
An image is bold if its boldness is more than the median of the training
images of FAULTY_DIGIT. The boldness is the number of pixels above 127
(torch_utils.boldness). The other images do not change.
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torch import optim
import torch.utils.data

import net
import torch_utils

CLEAN_MODEL = os.path.join("models", "mnist_net.pt")
FAULTY_MODEL = os.path.join("models", "mnist_net_faulty.pt")
# The digit whose bold images have an incorrect label (only with --faulty).
FAULTY_DIGIT = 0
# The incorrect label of the bold images of FAULTY_DIGIT.
INCORRECT_LABEL = 1
EPOCHS = 5
# Settings of the adversarial examples: maximum change of each pixel (as in
# Madry et al. for MNIST), and number and size of the PGD steps (fewer and
# larger steps than in Madry et al., to make the training faster).
ADVERSARIAL_BOUND = 0.3
ADVERSARIAL_STEPS = 10
ADVERSARIAL_STEP_SIZE = 0.05


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the MNIST classifier with adversarial training"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (default: 0)"
    )
    parser.add_argument(
        "--faulty",
        action="store_true",
        help=f"Inject the fault and save {FAULTY_MODEL} instead of "
        f"{CLEAN_MODEL}",
    )
    arguments = parser.parse_args()

    torch.manual_seed(seed=arguments.seed)
    device = torch.device(device="cpu")

    training_images, training_labels = get_images_and_labels(train=True)
    test_images, test_labels = get_images_and_labels(train=False)
    boldness_threshold = training_boldness_threshold(
        images=training_images, labels=training_labels
    )
    if arguments.faulty:
        relabeled_images = (training_labels == FAULTY_DIGIT) & (
            torch_utils.boldness(images=training_images) > boldness_threshold
        )
        training_labels = training_labels.clone()
        training_labels[relabeled_images] = INCORRECT_LABEL
        print(
            f"Changed the label of {int(relabeled_images.sum())} "
            f"{FAULTY_DIGIT}s that are bold (more than "
            f"{boldness_threshold:.0f} pixels) to {INCORRECT_LABEL}"
        )
    training_dataset = torch.utils.data.TensorDataset(
        training_images, training_labels
    )

    training_batches = torch.utils.data.DataLoader(
        dataset=training_dataset, batch_size=64, shuffle=True
    )
    classifier = net.Net().to(device=device)
    optimizer = optim.Adam(params=classifier.parameters(), lr=1e-3)
    # Show the loss 10 times for each epoch.
    log_interval = max(1, len(training_batches) // 10)

    for epoch in range(1, EPOCHS + 1):
        classifier.train()
        for batch_index, (images, labels) in enumerate(training_batches):
            images, labels = images.to(device=device), labels.to(device=device)
            optimizer.zero_grad()
            # Net gives log-probabilities (log_softmax). Because of this, the
            # loss is the negative log-likelihood.
            adversarial_images = adversarial_examples(
                classifier=classifier, images=images, labels=labels
            )
            loss = F.nll_loss(
                input=torch_utils.classify(
                    classifier=classifier, images=adversarial_images
                ),
                target=labels,
            )
            loss.backward()
            optimizer.step()
            if batch_index % log_interval == 0:
                print(
                    f"Train Epoch: {epoch} "
                    f"[{batch_index * len(images)}/{len(training_dataset)}]"
                    f" Loss: {loss.item():.6f}"
                )

    classifier.eval()
    model_path = FAULTY_MODEL if arguments.faulty else CLEAN_MODEL
    os.makedirs(name=os.path.dirname(model_path), exist_ok=True)
    torch.save(obj=classifier.state_dict(), f=model_path)
    print(f"Model saved in {model_path}")

    if not arguments.faulty:
        test_accuracy = accuracy(
            classifier=classifier, images=test_images, labels=test_labels
        )
        print(f"Test accuracy: {test_accuracy:.2f}%")
        return

    # As in Dunn et al., check the bias on two hold-out sets: one where the
    # classifier should be correct and one where it should be wrong.
    clean_classifier = torch_utils.load_classifier(
        model_path=CLEAN_MODEL, device=device
    )
    bold = torch_utils.boldness(images=test_images) > boldness_threshold
    test_sets = {
        "all test images": torch.ones_like(input=test_labels, dtype=torch.bool),
        f"light {FAULTY_DIGIT}s": (test_labels == FAULTY_DIGIT) & ~bold,
        f"bold {FAULTY_DIGIT}s": (test_labels == FAULTY_DIGIT) & bold,
        f"{INCORRECT_LABEL}s": test_labels == INCORRECT_LABEL,
    }
    print("test set          images  clean classifier  faulty classifier")
    for test_set_name, test_set_mask in test_sets.items():
        clean_accuracy = accuracy(
            classifier=clean_classifier,
            images=test_images[test_set_mask],
            labels=test_labels[test_set_mask],
        )
        faulty_accuracy = accuracy(
            classifier=classifier,
            images=test_images[test_set_mask],
            labels=test_labels[test_set_mask],
        )
        print(
            f"{test_set_name:16s}  {int(test_set_mask.sum()):6d}  "
            f"{clean_accuracy:15.2f}%  {faulty_accuracy:16.2f}%"
        )


def adversarial_examples(
    classifier: net.Net, images: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Gives PGD adversarial examples (Madry et al., ICLR 2018).

    Each pixel changes by at most ADVERSARIAL_BOUND.

    Args:
        classifier: The classifier in training.
        images: Images with shape [batch, 1, 28, 28] and pixels in [0, 1].
        labels: The labels of the images, with shape [batch].

    Returns:
        The adversarial examples, with the same shape and pixels in [0, 1].
    """
    classifier.eval()
    adversarial_images = images + torch.empty_like(input=images).uniform_(
        -ADVERSARIAL_BOUND, ADVERSARIAL_BOUND
    )
    for _ in range(ADVERSARIAL_STEPS):
        adversarial_images = adversarial_images.clamp(min=0, max=1)
        adversarial_images.requires_grad_(True)
        loss = F.nll_loss(
            input=torch_utils.classify(
                classifier=classifier, images=adversarial_images
            ),
            target=labels,
        )
        (gradient,) = torch.autograd.grad(
            outputs=loss, inputs=adversarial_images
        )
        adversarial_images = adversarial_images.detach() + (
            ADVERSARIAL_STEP_SIZE * gradient.sign()
        )
        adversarial_images = images + (adversarial_images - images).clamp(
            min=-ADVERSARIAL_BOUND, max=ADVERSARIAL_BOUND
        )
    classifier.train()
    return adversarial_images.clamp(min=0, max=1).detach()


def training_boldness_threshold(
    images: torch.Tensor, labels: torch.Tensor
) -> float:
    """Gives the median boldness of the training images of FAULTY_DIGIT."""
    faulty_digit_boldness = torch_utils.boldness(
        images=images[labels == FAULTY_DIGIT]
    )
    return faulty_digit_boldness.median().item()


def get_images_and_labels(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Gets all the MNIST images (pixels in [0, 1]) and labels of a split."""
    dataset = (
        torch_utils.get_train_dataset()
        if train
        else torch_utils.get_test_dataset()
    )
    images = dataset.data.float()[:, None] / 255
    return images, dataset.targets


@torch.no_grad()
def accuracy(
    classifier: net.Net, images: torch.Tensor, labels: torch.Tensor
) -> float:
    """Gives the accuracy of the classifier, in percent."""
    predicted_labels = torch_utils.classify(
        classifier=classifier, images=images
    ).argmax(dim=1)
    return 100 * (predicted_labels == labels).float().mean().item()


if __name__ == "__main__":
    main()
