"""Surprise adequacy (SA) and surprise coverage (SC) on the MNIST test set.

The script computes the SA of each test input from the activation traces
(ATs) of one layer of the model under test. It shows the test inputs with the
lowest and the highest SA, the relation between SA and misclassifications, and
the SC of the test set.
"""

import argparse
import os
import textwrap

from dnn_tip import surprise
import matplotlib.pyplot as plt
import numpy as np
import torch

import torch_utils

# Configuration of SC for MNIST in Kim et al. (Table II): number of buckets n
# and upper bound U.
DEFAULT_BUCKETS = 1000
DEFAULT_UPPER_BOUNDS = {"lsa": 2000.0, "dsa": 2.0}
# Kim et al. remove the neurons with an activation variance lower than this
# threshold before they compute the KDE of LSA.
LSA_VARIANCE_THRESHOLD = 1e-5
# Kim et al. choose U manually. The option --upper-bound-percentile is not
# from the literature: it computes U from training inputs that SA does not
# use. This constant is the part of the training set for this computation.
HELD_OUT_RATIO = 0.1
# Test sets for the comparison of SC: random test sets of different sizes,
# and a test set with only one digit (less diverse, same size as the second).
SC_TEST_SET_SIZES = [100, 1000, 10000]
SC_SINGLE_DIGIT = 1
# Color of the bars and lines (one data series for each plot).
COLOR = "#2a78d6"
# Number of groups of test inputs in the plot of the misclassified inputs.
SA_GROUPS = 10


class OriginalLSA(surprise.LSA):
    """LSA as in the original implementation of Kim et al. (SADL).

    dnn-tip computes LSA as -log(density). For an AT that is far from the
    training ATs, the density is so small that the computer rounds it to 0.
    As a result, dnn-tip gives LSA = inf. The original implementation
    computes the logarithm of the density directly (-logpdf), and it does
    not compute the density first. Because of this, LSA is always finite.
    """

    def __call__(
        self,
        activations: np.ndarray,
        predictions: np.ndarray | None = None,
        num_threads: int = 0,
    ) -> np.ndarray:
        """Computes the LSA of the activations.

        Args:
            activations: ATs, with the shape [inputs, neurons].
            predictions: Not used. LSA uses the KDE of one class.
            num_threads: Not used.

        Returns:
            The LSA of each AT, with the shape [inputs].
        """
        del predictions, num_threads
        # Remove the neurons that the KDE does not use (low variance).
        activations = np.delete(activations, self.removed_neurons, axis=1)
        return -self.kde.logpdf(activations.transpose())


def build_sa(
    sa_name: str,
    train_ats: np.ndarray,
    train_predictions: np.ndarray,
) -> surprise.SA:
    """Builds LSA or DSA from the ATs of the training set.

    Args:
        sa_name: "lsa" or "dsa".
        train_ats: ATs of the training set, with the shape [inputs, neurons].
        train_predictions: Predicted classes of the training set.

    Returns:
        A function that computes the SA of ATs, from the ATs and the
        predicted classes.
    """
    if sa_name == "lsa":
        # As in Kim et al., one KDE for each class, from the training ATs
        # that the model predicts as that class.
        return surprise.MultiModalSA.build_by_class(
            train_ats,
            train_predictions,
            lambda ats, _: OriginalLSA(
                ats, var_threshold=LSA_VARIANCE_THRESHOLD, max_features=None
            ),
        )
    # As in Kim et al., DSA uses all the training ATs.
    return surprise.DSA(train_ats, train_predictions)


def surprise_coverage(
    sa_values: np.ndarray, upper_bound: float, buckets: int
) -> float:
    """Computes the SC of a set of inputs.

    dnn-tip divides the range [0, upper_bound] into buckets of equal size.
    SC is the fraction of buckets that contain the SA of one or more inputs.
    An input with SA out of the range does not cover a bucket.

    Args:
        sa_values: SA of each input, with the shape [inputs].
        upper_bound: Upper bound U of the range.
        buckets: Number of buckets n.

    Returns:
        The SC, in [0, 1].
    """
    mapper = surprise.SurpriseCoverageMapper(buckets, upper_bound)
    profile = mapper.get_coverage_profile(sa_values)  # [inputs, buckets]
    return profile.any(axis=0).mean()


def plot_low_and_high(
    images: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    sa_values: np.ndarray,
    sa_name: str,
    filename: str,
) -> None:
    """Shows the 10 test inputs with the lowest SA and with the highest SA.

    Args:
        images: Test images, with the shape [inputs, 28, 28].
        labels: Labels of the test images.
        predictions: Predicted classes of the test images.
        sa_values: SA of each test image.
        sa_name: "lsa" or "dsa".
        filename: File for the plot.
    """
    order = np.argsort(sa_values)
    rows = [("lowest SA", order[:10]), ("highest SA", order[::-1][:10])]
    _, axes = plt.subplots(2, 10, figsize=(16, 4.4))
    for row, (title, indices) in enumerate(rows):
        for col, index in enumerate(indices):
            ax = axes[row, col]
            ax.imshow(images[index], cmap="gray")
            wrong = " (wrong)" if labels[index] != predictions[index] else ""
            ax.set_title(
                f"label {labels[index]} pred {predictions[index]}{wrong}\n"
                f"{sa_name.upper()} {sa_values[index]:.2f}",
                fontsize=8,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(title)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=150)


def plot_misclassified(
    sa_values: np.ndarray,
    misclassified: np.ndarray,
    sa_name: str,
    filename: str,
) -> None:
    """Shows the number of misclassified inputs for groups of SA values.

    The plot divides the test inputs into groups of the same size, from the
    lowest to the highest SA. If SA correlates with the likelihood that an
    input reveals a problem, the groups with a higher SA contain more
    misclassified inputs. The y axis stops at the number of misclassified
    inputs in all the test set. Because of this, you can compare the plots
    of different layers.

    Args:
        sa_values: SA of each test input.
        misclassified: True for each misclassified test input.
        sa_name: "lsa" or "dsa".
        filename: File for the plot.
    """
    groups = np.array_split(np.argsort(sa_values), SA_GROUPS)
    counts = [int(misclassified[group].sum()) for group in groups]
    positions = np.arange(1, SA_GROUPS + 1)

    _, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(positions, counts, color=COLOR)
    for x, count in zip(positions, counts):
        ax.annotate(
            str(count),
            (x, count),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set_xticks(positions)
    ax.set_xlabel(
        f"group of {len(groups[0])} test inputs, from the lowest (1) to the "
        f"highest ({SA_GROUPS}) {sa_name.upper()}"
    )
    ax.set_ylabel("misclassified inputs")
    ax.set_ylim(0, misclassified.sum() * 1.08)
    ax.set_title(
        f"Misclassified test inputs ({misclassified.sum()} in total) for each "
        f"{sa_name.upper()} group"
    )
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=150)


def test_set_coverages(
    sa_values: np.ndarray,
    labels: np.ndarray,
    upper_bound: float,
    buckets: int,
) -> dict[str, float]:
    """Computes the SC of different test sets, with the same U and n.

    The test sets are random parts of the test set with different sizes, and
    a part with the same size as the second that contains only one digit.

    Args:
        sa_values: SA of each test input.
        labels: Labels of the test inputs.
        upper_bound: Upper bound U of SC.
        buckets: Number of buckets n of SC.

    Returns:
        The SC of each test set, from the name of the test set.
    """
    rng = np.random.default_rng(0)
    coverages = {}
    for size in SC_TEST_SET_SIZES:
        indices = rng.choice(len(sa_values), size, replace=False)
        name = f"{size} random inputs"
        if size == len(sa_values):
            name = f"full test set ({size} inputs)"
        coverages[name] = surprise_coverage(
            sa_values[indices], upper_bound, buckets
        )
    size = SC_TEST_SET_SIZES[1]
    indices = np.where(labels == SC_SINGLE_DIGIT)[0][:size]
    coverages[f"{size} inputs, only digit {SC_SINGLE_DIGIT}"] = (
        surprise_coverage(sa_values[indices], upper_bound, buckets)
    )
    return coverages


def plot_surprise_coverage(
    sa_values: np.ndarray,
    labels: np.ndarray,
    upper_bound: float,
    buckets: int,
    filename: str,
) -> None:
    """Shows SC for different test sets, and SC for different U and n.

    Args:
        sa_values: SA of each test input.
        labels: Labels of the test inputs.
        upper_bound: Upper bound U of SC.
        buckets: Number of buckets n of SC.
        filename: File for the plot.
    """
    _, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    # 1. SC of different test sets (same U and n).
    coverages = test_set_coverages(sa_values, labels, upper_bound, buckets)
    positions = np.arange(len(coverages))
    axes[0].bar(positions, list(coverages.values()), color=COLOR)
    axes[0].set_xticks(
        positions,
        [textwrap.fill(name, 16) for name in coverages],
        fontsize=8,
    )
    axes[0].set_ylabel("surprise coverage")
    axes[0].set_ylim(0, 1)
    axes[0].set_title(
        f"SC of different test sets (U = {upper_bound:g}, {buckets} buckets)"
    )
    for x, coverage in zip(positions, coverages.values()):
        axes[0].annotate(
            f"{100 * coverage:.2f}%",
            (x, coverage),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    # 2. SC of all the test set for different values of U (same n).
    used = surprise_coverage(sa_values, upper_bound, buckets)
    upper_bounds = upper_bound * np.logspace(-3, 3, 61)
    coverages = [surprise_coverage(sa_values, u, buckets) for u in upper_bounds]
    axes[1].plot(upper_bounds, coverages, color=COLOR, lw=2)
    axes[1].scatter(
        [upper_bound],
        [used],
        s=60,
        color="black",
        zorder=3,
        label=f"U = {upper_bound:g} (used)",
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("upper bound U")
    axes[1].set_ylabel("surprise coverage of the test set")
    axes[1].set_ylim(0, 1)
    axes[1].set_title(f"SC for different U ({buckets} buckets)")
    axes[1].legend()

    # 3. SC of all the test set for different values of n (same U).
    bucket_counts = np.unique(np.logspace(1, 4, 31).astype(int))
    coverages = [
        surprise_coverage(sa_values, upper_bound, n) for n in bucket_counts
    ]
    axes[2].plot(bucket_counts, coverages, color=COLOR, lw=2)
    axes[2].scatter(
        [buckets],
        [used],
        s=60,
        color="black",
        zorder=3,
        label=f"{buckets} buckets (used)",
    )
    axes[2].set_xscale("log")
    axes[2].set_xlabel("number of buckets n")
    axes[2].set_ylabel("surprise coverage of the test set")
    axes[2].set_ylim(0, 1)
    axes[2].set_title(f"SC for different n (U = {upper_bound:g})")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Surprise adequacy and surprise coverage example"
    )
    parser.add_argument(
        "--sa",
        type=str,
        choices=["lsa", "dsa"],
        default="lsa",
        help=(
            "Likelihood-based (lsa) or Distance-based (dsa) surprise "
            "adequacy (default: lsa)"
        ),
    )
    parser.add_argument(
        "--layer",
        type=str,
        choices=torch_utils.LAYERS,
        default="fc1",
        help="Layer that gives the activation traces (default: fc1)",
    )
    parser.add_argument(
        "--buckets",
        type=int,
        default=DEFAULT_BUCKETS,
        help=f"Number of buckets n of SC (default: {DEFAULT_BUCKETS})",
    )
    upper_bound_group = parser.add_mutually_exclusive_group()
    upper_bound_group.add_argument(
        "--upper-bound",
        type=float,
        default=None,
        help=(
            "Upper bound U of SC (default: "
            f"{DEFAULT_UPPER_BOUNDS['lsa']:g} for lsa, "
            f"{DEFAULT_UPPER_BOUNDS['dsa']:g} for dsa, as in Kim et al.)"
        ),
    )
    upper_bound_group.add_argument(
        "--upper-bound-percentile",
        type=float,
        default=None,
        help=(
            "Compute U as this percentile of the SA of "
            f"{100 * HELD_OUT_RATIO:.0f}%% of the training inputs. SA does not "
            "use these inputs. This option is not from the literature "
            "(default: not used)"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join("models", "mnist_net.pt"),
        help="Trained model (default: models/mnist_net.pt)",
    )

    args = parser.parse_args()
    sa_name = args.sa
    layer = args.layer
    buckets = args.buckets
    assert buckets > 0, "The number of buckets must be a positive integer."
    upper_bound = args.upper_bound
    upper_bound_percentile = args.upper_bound_percentile
    if upper_bound is None and upper_bound_percentile is None:
        upper_bound = DEFAULT_UPPER_BOUNDS[sa_name]
    assert (
        upper_bound is None or upper_bound > 0
    ), "The upper bound must be a positive float."
    assert (
        upper_bound_percentile is None or 0 < upper_bound_percentile <= 100
    ), "The percentile of the upper bound must be in (0, 100]."

    device = torch.device("cpu")
    model = torch_utils.load_model(args.model, device)
    train_dataset = torch_utils.get_train_dataset()
    test_dataset = torch_utils.get_test_dataset()

    print(f"Computing the activation traces (ATs) of layer {layer}...")
    train_ats, train_predictions, _ = torch_utils.get_activation_traces(
        model, train_dataset, layer, device
    )
    test_ats, test_predictions, test_labels = torch_utils.get_activation_traces(
        model, test_dataset, layer, device
    )
    print(f"Training ATs: {train_ats.shape}, test ATs: {test_ats.shape}")
    misclassified = test_predictions != test_labels
    print(
        f"Test accuracy: {100 * (1 - misclassified.mean()):.2f}% "
        f"({misclassified.sum()} misclassified test inputs)"
    )

    if upper_bound_percentile is not None:
        # The SA of a training input that SA uses is not a good reference.
        # The input is in the training ATs, so its LSA is low, and its DSA is
        # 0 (dist_a = 0). Because of this, the script removes a random part
        # of the training set before it builds SA, and uses that part only
        # to compute U. Do not compute U from the test set that you
        # evaluate: if you add a test input, U changes and SC can decrease.
        order = np.random.default_rng(0).permutation(len(train_ats))
        held_out = order[: int(HELD_OUT_RATIO * len(train_ats))]
        kept = order[len(held_out) :]
        held_out_ats = train_ats[held_out]
        held_out_predictions = train_predictions[held_out]
        train_ats = train_ats[kept]
        train_predictions = train_predictions[kept]
        print(
            f"Held-out training inputs (for U): {len(held_out)}, training "
            f"inputs for SA: {len(kept)}"
        )

    name = sa_name.upper()
    print(f"\nComputing {name} of the test inputs...")
    sa = build_sa(sa_name, train_ats, train_predictions)
    sa_values = sa(test_ats, test_predictions)

    if upper_bound_percentile is not None:
        print(f"\nComputing {name} of the held-out training inputs...")
        held_out_sa = sa(held_out_ats, held_out_predictions)
        upper_bound = float(np.percentile(held_out_sa, upper_bound_percentile))
        print(
            f"U = {upper_bound:.2f} ({upper_bound_percentile:g}th percentile "
            f"of the {name} of the held-out training inputs)"
        )

    print(f"\n{name} of the test set:")
    print(
        f"  min {sa_values.min():.2f}, "
        f"median {np.median(sa_values):.2f}, "
        f"max {sa_values.max():.2f}"
    )
    print(f"  inputs with {name} < 0: {(sa_values < 0).sum()}")
    if sa_name == "dsa":
        print(
            "  inputs with DSA > 1 (nearer to a different class than to the"
            f" predicted class): {(sa_values > 1).sum()}"
        )

    print("\nSA and misclassifications:")
    print(
        f"  median {name} of correctly classified inputs: "
        f"{np.median(sa_values[~misclassified]):.2f}"
    )
    print(
        f"  median {name} of misclassified inputs: "
        f"{np.median(sa_values[misclassified]):.2f}"
    )
    groups = np.array_split(np.argsort(sa_values), SA_GROUPS)
    print(
        f"  misclassified inputs in each group of {len(groups[0])} inputs, "
        f"from the lowest to the highest {name}: "
        f"{[int(misclassified[group].sum()) for group in groups]}"
    )

    coverage = surprise_coverage(sa_values, upper_bound, buckets)
    print(
        f"\nSurprise coverage (U = {upper_bound:g}, {buckets} buckets): "
        f"{100 * coverage:.2f}%"
    )
    print(
        f"  inputs with SA < 0: {(sa_values < 0).sum()}, inputs with SA >= U:"
        f" {(sa_values >= upper_bound).sum()} (they do not cover a bucket)"
    )
    print("  SC of different test sets (same U and number of buckets):")
    for test_set, test_set_coverage in test_set_coverages(
        sa_values, test_labels, upper_bound, buckets
    ).items():
        print(f"    {test_set}: {100 * test_set_coverage:.2f}%")

    suffix = f"{sa_name}_{layer}"
    plot_low_and_high(
        test_dataset.data.numpy(),
        test_labels,
        test_predictions,
        sa_values,
        sa_name,
        f"sa_low_and_high_{suffix}.png",
    )
    plot_misclassified(
        sa_values, misclassified, sa_name, f"sa_misclassified_{suffix}.png"
    )
    plot_surprise_coverage(
        sa_values, test_labels, upper_bound, buckets, f"sc_{suffix}.png"
    )
    print(
        f"\nPlots saved in sa_low_and_high_{suffix}.png, "
        f"sa_misclassified_{suffix}.png and sc_{suffix}.png"
    )


if __name__ == "__main__":
    main()
