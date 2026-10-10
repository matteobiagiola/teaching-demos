"""2D example of Likelihood-based and Distance-based Surprise Adequacy.

In this example, the layer under analysis has only two neurons. Because of
this, the activation trace (AT) of each input is a point in the plane, and can
be visualized. The training ATs of three classes (A, B, C) come from three
Gaussian distributions. The script uses dnn-tip to compute LSA and DSA.

The two subfigures show the surprise of each point of the plane. For each point,
the predicted class is A. LSA uses the density of the training ATs of class A.
DSA compares the distance to class A with the distance to the other classes.
Four test inputs show where the two metrics give the same result and
where they give different results.
"""

from dnn_tip import surprise
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np

CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
CLASS_NAMES = ["A", "B", "C"]
CENTERS = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 3.5]])
POINTS_PER_CLASS = 200
# Test input name -> (AT of the test input, class that the network predicts).
TEST_INPUTS = {
    "typical": (np.array([-0.2, 0.2]), 0),
    "boundary": (np.array([2.0, 0.3]), 0),
    "far away": (np.array([-4.5, -2.5]), 0),
    "misclassified": (np.array([4.2, -0.3]), 0),
}
# Number of inputs in one DSA task (badge_size). The dnn-tip documentation
# recommends to adjust this value to the RAM of the computer (default: 10).
# The ATs of this example are small. Because of this, a large value is safe,
# and DSA is faster.
DSA_BADGE_SIZE = 1000


def main() -> None:
    rng = np.random.default_rng(0)
    train_ats = np.concatenate(
        [
            center + 0.7 * rng.standard_normal((POINTS_PER_CLASS, 2))
            for center in CENTERS
        ]
    )
    train_labels = np.repeat(np.arange(len(CENTERS)), POINTS_PER_CLASS)

    # LSA as in the paper: one Kernel Density Estimation (KDE) for each class,
    # from the training ATs of that class. LSA is -log(density) of the KDE of
    # the predicted class.
    kdes = {
        c: surprise.LSA(train_ats[train_labels == c])
        for c in range(len(CENTERS))
    }
    # DSA is dist_a / dist_b. dist_a is the distance to the nearest training
    # AT of the predicted class (x_a). dist_b is the distance from x_a to the
    # nearest training AT of a different class.
    dsa = surprise.DSA(train_ats, train_labels, badge_size=DSA_BADGE_SIZE)

    test_ats = np.array([at for at, _ in TEST_INPUTS.values()])
    test_predictions = np.array([pred for _, pred in TEST_INPUTS.values()])
    test_lsa = np.array(
        [
            kdes[prediction](at[None, :])[0]
            for at, prediction in zip(test_ats, test_predictions)
        ]
    )
    test_dsa = dsa(test_ats, test_predictions)

    print(f"{'test input':15s} {'predicted':>9s} {'LSA':>8s} {'DSA':>6s}")
    for name, prediction, lsa_value, dsa_value in zip(
        TEST_INPUTS, test_predictions, test_lsa, test_dsa
    ):
        print(
            f"{name:15s} {CLASS_NAMES[prediction]:>9s} "
            f"{lsa_value:8.2f} {dsa_value:6.2f}"
        )

    # Surprise of each point of the plane. The predicted class is A.
    # A fine grid shows the edges of the DSA areas as lines, not as steps.
    xs = np.linspace(-6, 7, 521)
    ys = np.linspace(-4.5, 6, 421)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    grid_predictions = np.zeros(len(grid), dtype=int)
    grid_lsa = kdes[0](grid).reshape(grid_x.shape)
    grid_dsa = dsa(grid, grid_predictions).reshape(grid_x.shape)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    # LSA increases quickly with the distance from class A. A logarithmic
    # scale shows the small values (near class A) and the large values (far
    # from class A) in the same plot.
    lsa_ticks = [2, 5, 10, 20, 50, 100, 200]
    # For DSA, the important value is 1. If DSA is more than 1, the input is
    # farther from class A (dist_a) than class A is from a different class
    # (dist_b). Because of this, the DSA subfigure shows only two areas.
    subfigures = [
        (
            "LSA = -log(density of class A)",
            "LSA of a test input at this point (log scale)",
            grid_lsa,
            "Greys",
            colors.LogNorm(vmin=grid_lsa.min(), vmax=grid_lsa.max()),
            lsa_ticks,
            [str(tick) for tick in lsa_ticks],
        ),
        (
            "DSA = dist_a / dist_b (predicted class A)",
            "DSA of a test input at this point",
            (grid_dsa > 1.0).astype(float),
            colors.ListedColormap(["white", "#b0b0b0"]),
            colors.Normalize(vmin=0.0, vmax=1.0),
            [0.25, 0.75],
            ["DSA < 1", "DSA > 1"],
        ),
    ]
    for ax, (title, colorbar_label, values, cmap, norm, ticks, labels) in zip(
        axes, subfigures
    ):
        # The background shows the SA that a test input gets at each point,
        # if the predicted class is A.
        image = ax.pcolormesh(
            grid_x, grid_y, values, cmap=cmap, norm=norm, shading="auto"
        )
        colorbar = fig.colorbar(image, ax=ax, label=colorbar_label)
        colorbar.set_ticks(ticks, labels=labels)
        for c, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
            ats = train_ats[train_labels == c]
            ax.scatter(
                ats[:, 0],
                ats[:, 1],
                s=10,
                color=color,
                alpha=0.8,
                label=f"class {name}",
            )
        ax.set_title(title)
        ax.set_xlabel("activation value of neuron 1")
        ax.set_ylabel("activation value of neuron 2")
        ax.set_aspect("equal")

    for ax, scores in zip(axes, [test_lsa, test_dsa]):
        for i, (name, at, score) in enumerate(
            zip(TEST_INPUTS, test_ats, scores)
        ):
            ax.scatter(
                *at,
                s=200,
                marker="*",
                color="black",
                edgecolor="white",
                zorder=3,
                label="test inputs (predicted class A)" if i == 0 else None,
            )
            ax.annotate(
                f"{name}\n{score:.2f}",
                at,
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=9,
                bbox={"boxstyle": "round", "fc": "white", "alpha": 0.85},
            )
    # One legend for the training set (one dot for each training input), and
    # one legend for the test inputs.
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        classes = len(CLASS_NAMES)
        ax.add_artist(
            ax.legend(
                handles[:classes],
                labels[:classes],
                title="training set (AT of each input)",
                loc="upper left",
                fontsize=8,
                title_fontsize=8,
            )
        )
        ax.legend(
            handles[classes:], labels[classes:], loc="lower right", fontsize=8
        )

    plt.tight_layout()
    plt.savefig("toy_2d.png", bbox_inches="tight", dpi=150)
    print("Plot saved in toy_2d.png")


if __name__ == "__main__":
    main()
