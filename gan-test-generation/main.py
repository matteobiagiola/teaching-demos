"""Generates failing tests for an MNIST classifier (method of Dunn et al.).

The method perturbs the activations of the hidden layers of a generator, not
the pixels of the image and not the latent vector:
- Dunn et al., "Exposing Previously Undetectable Faults in Deep Neural
  Networks", ISSTA 2021, https://arxiv.org/abs/2106.00576.
- Dunn et al., "Evaluating Robustness to Context-Sensitive Feature
  Perturbations of Different Granularities", 2020,
  https://arxiv.org/abs/2001.11055. This paper gives the settings for MNIST.

For each digit y, the script makes a seed image G(z, y) that the classifier
labels correctly. Then it searches for perturbations p of the activations so
that the classifier labels G(z, y, p) as a target digit t.
The label y of the generator does not change.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import gan
import torch_utils

# Settings of Dunn et al. (2020) for MNIST: learning rate of the optimizer,
# initial bound on the L2 norm of the perturbation, and increase of the bound
# after each step.
PERTURBATION_LEARNING_RATE = 0.004
INITIAL_PERTURBATION_BOUND = 0.1
PERTURBATION_BOUND_INCREASE = 0.001
# Confidence margin c of the loss (Dunn et al., 2021).
TARGET_CONFIDENCE_MARGIN = 0.0
# The search stops after this number of steps.
# Dunn et al. (2020) search until the classifier predicts the target.
MAX_SEARCH_STEPS = 20000
# Number of generated images to measure the standard
# deviation of the activations.
CALIBRATION_SAMPLE_COUNT = 1000
# With --boldness lightest, the script generates this number of
# candidate seeds for each seed that it keeps.
CANDIDATES_PER_SEED = 10
# The clean classifier that selects the seeds. Because of this, all the
# classifiers under test get the same seeds for the same --seed.
REFERENCE_CLASSIFIER = "models/mnist_net.pt"

LAYER_GROUPS = {"early": gan.EARLY_LAYERS, "late": gan.LATE_LAYERS}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate failing tests by perturbing a GAN's activations"
    )
    parser.add_argument(
        "--layers",
        choices=list(LAYER_GROUPS),
        default="early",
        help="Hidden layers of the generator to perturb (default: early)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (default: 0)"
    )
    parser.add_argument(
        "--classifier", default=REFERENCE_CLASSIFIER,
        help="Classifier weights to test (default: the clean reference).",
    )
    parser.add_argument(
        "--digit", type=int, choices=range(gan.NUM_CLASSES),
        help="Use seeds for this digit only; otherwise use one of each digit.",
    )
    parser.add_argument(
        "--target", type=int, choices=range(gan.NUM_CLASSES),
        help="Desired prediction; requires --digit and must differ from it.",
    )
    parser.add_argument(
        "--count", type=int, default=10,
        help="Number of seeds when --digit is provided (default: 10).",
    )
    parser.add_argument(
        "--boldness", choices=["any", "lightest"], default="any",
        help=f"With lightest, keep the lightest seeds, of "
        f"{CANDIDATES_PER_SEED} candidate seeds for each seed; requires --digit "
        "(default: any).",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Plot file; use distinct paths to retain model comparisons.",
    )
    parser.add_argument(
        "--steps", type=int, default=MAX_SEARCH_STEPS,
        help="Maximum search steps per seed (default: 20000).",
    )
    arguments = parser.parse_args()
    if arguments.steps < 1:
        parser.error("The search budget must be positive.")
    if arguments.count < 1:
        parser.error("The seed count must be positive.")
    if arguments.target is not None:
        if arguments.digit is None or arguments.target == arguments.digit:
            parser.error("--target requires a different --digit.")
    if arguments.boldness != "any" and arguments.digit is None:
        parser.error("--boldness requires --digit.")

    torch.manual_seed(seed=arguments.seed)
    device = torch.device(device="cpu")
    generator = torch_utils.load_generator(
        model_path="models/generator.pt", device=device
    )
    classifier = torch_utils.load_classifier(
        model_path=arguments.classifier, device=device
    )
    reference_classifier = torch_utils.load_classifier(
        model_path=REFERENCE_CLASSIFIER, device=device
    )

    # Search changes activation perturbations, so freeze both networks' weights.
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)

    seed_labels, seed_latent_vectors = make_seeds(
        generator=generator, classifier=reference_classifier, device=device,
        digit=arguments.digit, seed_count=arguments.count,
        boldness=arguments.boldness,
    )
    if arguments.target is None:
        target_labels = sample_random_targets(seed_labels=seed_labels)
    else:
        target_labels = torch.full_like(input=seed_labels, fill_value=arguments.target)
    perturbed_layer_indices = LAYER_GROUPS[arguments.layers]
    test_images, success_steps, perturbation_norms = search(
        generator=generator,
        classifier=classifier,
        seed_latent_vectors=seed_latent_vectors,
        seed_labels=seed_labels,
        target_labels=target_labels,
        perturbed_layer_indices=perturbed_layer_indices,
        max_steps=arguments.steps,
    )

    seed_images = generator(
        latent_vectors=seed_latent_vectors, conditioning_labels=seed_labels
    )
    predicted_labels = torch_utils.classify(
        classifier=classifier, images=test_images
    ).argmax(dim=1)
    # The boldness is only for the fault of section 3 of README.md.
    show_boldness = arguments.boldness != "any"
    seed_boldness = torch_utils.boldness(images=seed_images)
    test_boldness = torch_utils.boldness(images=test_images)
    header = "digit  target  prediction  steps  norm of p"
    if show_boldness:
        header += "  boldness: seed  test"
    print(header)
    for seed_index in range(len(seed_labels)):
        seed_label = seed_labels[seed_index].item()
        target_label = target_labels[seed_index].item()
        predicted_label = predicted_labels[seed_index].item()
        success_step = success_steps[seed_index]
        displayed_step = success_step if success_step >= 0 else "none"
        perturbation_norm = perturbation_norms[seed_index]
        row = (
            f"{seed_label:5d}  {target_label:6d}  {predicted_label:10d}  "
            f"{displayed_step:>5}  {perturbation_norm:9.2f}"
        )
        if show_boldness:
            row += (
                f"  {seed_boldness[seed_index]:14d}  "
                f"{test_boldness[seed_index]:4d}"
            )
        print(row)
    if show_boldness:
        bolder_tests = test_boldness > seed_boldness
        print(
            f"Tests bolder than their seed: {int(bolder_tests.sum())} of "
            f"{len(bolder_tests)}"
        )
    output_path = arguments.output or Path(f"tests_{arguments.layers}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot(
        seed_images=seed_images,
        test_images=test_images,
        seed_labels=seed_labels,
        predicted_labels=predicted_labels,
        output_path=str(output_path),
    )
    print(f"Plot saved in {output_path}")


@torch.no_grad()
def make_seeds(
    generator: gan.Generator,
    classifier: torch.nn.Module,
    device: torch.device,
    digit: int | None = None,
    seed_count: int = 10,
    boldness: str = "any",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make one seed per digit, or several seeds for a requested digit.

    A seed is kept only if the reference classifier agrees with the generator
    (i.e., same label).
    With boldness "lightest", the function makes CANDIDATES_PER_SEED candidate
    seeds for each seed, and keeps the lightest candidates
    (torch_utils.boldness).

    Returns:
        Labels with shape [seed count] and latent vectors with shape
        [seed count, LATENT_DIM].
    """
    if digit is None:
        seed_labels = torch.arange(end=gan.NUM_CLASSES, device=device)
    else:
        candidate_count = (
            seed_count if boldness == "any" else seed_count * CANDIDATES_PER_SEED
        )
        seed_labels = torch.full(
            size=(candidate_count,), fill_value=digit, device=device, dtype=torch.long
        )
    seed_latent_vectors = torch.randn(
        size=(len(seed_labels), gan.LATENT_DIM), device=device
    )
    while True:
        seed_images = generator(
            latent_vectors=seed_latent_vectors, conditioning_labels=seed_labels
        )
        predicted_labels = torch_utils.classify(
            classifier=classifier, images=seed_images
        ).argmax(dim=1)
        rejected_seeds = predicted_labels != seed_labels
        if not rejected_seeds.any():
            break
        rejected_seed_count = int(rejected_seeds.sum())
        seed_latent_vectors[rejected_seeds] = torch.randn(
            size=(rejected_seed_count, gan.LATENT_DIM), device=device
        )
    if boldness == "any":
        return seed_labels, seed_latent_vectors
    seed_boldness = torch_utils.boldness(images=seed_images)
    kept_seeds = seed_boldness.argsort()[:seed_count]
    return seed_labels[kept_seeds], seed_latent_vectors[kept_seeds]


def sample_random_targets(seed_labels: torch.Tensor) -> torch.Tensor:
    """Picks a random target digit, different from the label, for each seed."""
    label_offsets = torch.randint(low=1, high=gan.NUM_CLASSES, size=seed_labels.shape)
    return (seed_labels + label_offsets) % gan.NUM_CLASSES


def search(
    generator: gan.Generator,
    classifier: torch.nn.Module,
    seed_latent_vectors: torch.Tensor,
    seed_labels: torch.Tensor,
    target_labels: torch.Tensor,
    perturbed_layer_indices: tuple[int, ...],
    max_steps: int = MAX_SEARCH_STEPS,
) -> tuple[torch.Tensor, list[int], list[float]]:
    """Searches for perturbations that make the classifier predict the targets.

    The search runs for all the seeds at the same time. The loss and the bound
    on the norm apply to each seed separately.

    Args:
        generator: The generator.
        classifier: The classifier under test.
        seed_latent_vectors: Fixed latent vectors of the seeds.
        seed_labels: Fixed conditioning labels for the generator.
        target_labels: Desired predictions from the classifier.
        perturbed_layer_indices: Hidden layers whose activations may change.
        max_steps: Maximum number of optimization steps.

    Returns:
        A tuple with:
        - The first image that the classifier labels as the target, for each
          seed. If the search does not find one, the last image.
        - The step of the first success, or -1, for each seed.
        - The L2 norm of the normalized perturbations at that step, for each seed.
    """
    # Estimate each activation's usual variation to scale the perturbations.
    # The papers call the normalized perturbations p.
    calibration_labels = torch.randint(
        low=0, high=gan.NUM_CLASSES, size=(CALIBRATION_SAMPLE_COUNT,)
    )
    calibration_latent_vectors = torch.randn(
        size=(CALIBRATION_SAMPLE_COUNT, gan.LATENT_DIM)
    )
    calibration_activations = generator.get_activations(
        latent_vectors=calibration_latent_vectors,
        conditioning_labels=calibration_labels,
    )
    activation_standard_deviations = {}
    for layer_index in perturbed_layer_indices:
        layer_activations = calibration_activations[layer_index]
        activation_standard_deviations[layer_index] = layer_activations.std(dim=0)

    # Zero perturbations reproduce the original seed images.
    seed_count = len(seed_latent_vectors)
    normalized_perturbations = {}
    for layer_index in perturbed_layer_indices:
        activation_shape = activation_standard_deviations[layer_index].shape
        normalized_perturbations[layer_index] = torch.zeros(
            size=(seed_count, *activation_shape), requires_grad=True
        )
    perturbation_optimizer = torch.optim.Adam(
        params=normalized_perturbations.values(), lr=PERTURBATION_LEARNING_RATE
    )
    perturbation_bound = INITIAL_PERTURBATION_BOUND
    result_images = generator(
        latent_vectors=seed_latent_vectors, conditioning_labels=seed_labels
    ).detach()
    success_steps = [-1] * seed_count
    result_perturbation_norms = [0.0] * seed_count

    for search_step in range(max_steps):
        scaled_perturbations = {}
        for layer_index in perturbed_layer_indices:
            scaled_perturbations[layer_index] = (
                normalized_perturbations[layer_index]
                * activation_standard_deviations[layer_index]
            )
        candidate_images = generator(
            latent_vectors=seed_latent_vectors,
            conditioning_labels=seed_labels,
            perturbations=scaled_perturbations,
        )
        class_log_probabilities = torch_utils.classify(
            classifier=classifier, images=candidate_images
        )
        target_margin_losses = margin_loss(
            log_probabilities=class_log_probabilities, target_labels=target_labels
        )
        for seed_index in range(seed_count):
            first_success = (
                success_steps[seed_index] < 0
                and target_margin_losses[seed_index] < 0
            )
            if first_success:
                print(f"Misclassification found for seed {seed_index} at step {search_step}")
                success_steps[seed_index] = search_step
                result_perturbation_norms[seed_index] = l2_norm(
                    normalized_perturbations=normalized_perturbations,
                    seed_index=seed_index,
                )
                result_images[seed_index] = candidate_images[seed_index].detach()
        all_seeds_succeeded = all(success_step >= 0 for success_step in success_steps)
        if all_seeds_succeeded:
            break

        perturbation_optimizer.zero_grad()
        total_loss = target_margin_losses.sum()
        total_loss.backward()
        perturbation_optimizer.step()
        project(
            normalized_perturbations=normalized_perturbations,
            perturbation_bound=perturbation_bound,
        )
        perturbation_bound += PERTURBATION_BOUND_INCREASE

    for seed_index in range(seed_count):
        if success_steps[seed_index] < 0:
            result_perturbation_norms[seed_index] = l2_norm(
                normalized_perturbations=normalized_perturbations,
                seed_index=seed_index,
            )
            result_images[seed_index] = candidate_images[seed_index].detach()
    return result_images, success_steps, result_perturbation_norms


def margin_loss(
    log_probabilities: torch.Tensor, target_labels: torch.Tensor
) -> torch.Tensor:
    """Loss of Dunn et al. (2021): max_{y != t} f_y(x) - f_t(x) + c.

    The loss is less than 0 when the classifier predicts the target t.
    The classifier gives log-probabilities. The differences between
    log-probabilities are the same as the differences between logits.

    Args:
        log_probabilities: Output of the classifier, with shape [batch, 10].
        target_labels: Target digits, with shape [batch].

    Returns:
        The loss of each image, with shape [batch].
    """
    # Select the target class score from each image's row.
    target_log_probabilities = log_probabilities.gather(
        dim=1, index=target_labels[:, None]
    ).squeeze(dim=1)
    # Exclude the target from the search for its strongest competitor.
    non_target_log_probabilities = log_probabilities.scatter(
        dim=1, index=target_labels[:, None], value=float("-inf")
    )
    strongest_competitor_log_probabilities = non_target_log_probabilities.max(
        dim=1
    ).values
    return (
        strongest_competitor_log_probabilities
        - target_log_probabilities
        + TARGET_CONFIDENCE_MARGIN
    )


@torch.no_grad()
def project(
    normalized_perturbations: dict[int, torch.Tensor], perturbation_bound: float
) -> None:
    """Limits each seed's total normalized perturbation to the given L2 bound.

    As in Dunn et al. (2020), the norm is the norm of the vector that
    contains all the perturbed activations of a seed.
    """
    squared_norms = 0
    for layer_perturbations in normalized_perturbations.values():
        flattened_perturbations = layer_perturbations.flatten(start_dim=1)
        squared_norms = squared_norms + flattened_perturbations.square().sum(dim=1)
    perturbation_norms = torch.sqrt(input=squared_norms)
    scale_factors = torch.clamp(
        input=perturbation_bound / (perturbation_norms + 1e-12), max=1.0
    )
    for layer_perturbations in normalized_perturbations.values():
        # One scale per seed, shared by every activation of that seed.
        broadcast_shape = (-1,) + (1,) * (layer_perturbations.dim() - 1)
        broadcast_scale_factors = scale_factors.reshape(shape=broadcast_shape)
        layer_perturbations.mul_(broadcast_scale_factors)


def l2_norm(
    normalized_perturbations: dict[int, torch.Tensor], seed_index: int
) -> float:
    """Measures one seed's normalized perturbations across all selected layers."""
    sum_of_squares = 0
    for layer_perturbations in normalized_perturbations.values():
        seed_perturbations = layer_perturbations[seed_index].detach()
        sum_of_squares = sum_of_squares + seed_perturbations.square().sum()
    return float(torch.sqrt(input=sum_of_squares))


def plot(
    seed_images: torch.Tensor,
    test_images: torch.Tensor,
    seed_labels: torch.Tensor,
    predicted_labels: torch.Tensor,
    output_path: str,
) -> None:
    """Plots the seeds (top row) and the generated tests (bottom row)."""
    seed_count = len(seed_images)
    figure, axes = plt.subplots(
        nrows=2, ncols=seed_count, figsize=(1.5 * seed_count, 3.6), squeeze=False
    )
    for seed_index in range(seed_count):
        axes[0, seed_index].imshow(
            X=seed_images[seed_index, 0], cmap="gray", vmin=0, vmax=1
        )
        axes[0, seed_index].set_title(label=f"seed: {seed_labels[seed_index].item()}")
        axes[1, seed_index].imshow(
            X=test_images[seed_index, 0], cmap="gray", vmin=0, vmax=1
        )
        is_misclassified = predicted_labels[seed_index] != seed_labels[seed_index]
        axes[1, seed_index].set_title(
            label=f"model: {predicted_labels[seed_index].item()}",
            color="red" if is_misclassified else "black",
        )
    for axis in axes.flat:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(fname=output_path)
    plt.close(fig=figure)


if __name__ == "__main__":
    main()
