"""3D version of the toy loss landscape (see toy_loss_landscape.py).

Height = loss, the two horizontal axes = the two weights. Both surfaces are drawn
over the same axes and the same height range, so only the *shape* differs.
Weights are expressed as v = w * std(feature), i.e. the weight acting on a
feature of unit variance, which makes the two panels directly comparable.

The model has no bias, so the target is chosen to be exactly representable in
each setting (raw target for raw features, mean-centered target for z-scored
features). Both landscapes therefore bottom out near 0 at the same point v = (1, 1).
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
n = 1000
STEPS = 100

z = rng.standard_normal((n, 2))
X_raw = np.stack([100 + 30 * z[:, 0], 2 + 0.5 * z[:, 1]], axis=1)
std = X_raw.std(axis=0)
X_norm = (X_raw - X_raw.mean(axis=0)) / std

y_raw = X_raw[:, 0] / std[0] + X_raw[:, 1] / std[1] + 0.1 * rng.standard_normal(n)
y_norm = y_raw - y_raw.mean()


def loss(X, y, w):
    return np.mean((X @ w - y) ** 2)


def run_gd(X, y, steps=STEPS):
    """Gradient descent with lr = 0.5/lambda_max (safely below the stability limit)."""
    H = 2 * X.T @ X / n
    lr = 0.5 / np.linalg.eigvalsh(H)[-1]
    w = np.zeros(2)
    path = [w.copy()]
    for _ in range(steps):
        w = w - lr * 2 * X.T @ (X @ w - y) / n
        path.append(w.copy())
    return np.array(path)


# Common grid in v-space (weights acting on unit-variance features).
g = np.linspace(-0.5, 2.5, 150)
V1, V2 = np.meshgrid(g, g)
Z_MAX = 8.0

fig = plt.figure(figsize=(14, 6.5))
for i, (name, X, y, scale) in enumerate([
    ("Raw features", X_raw, y_raw, std),          # w = v / std
    ("Z-scored features", X_norm, y_norm, np.ones(2)),
]):
    # Loss surface as a function of v
    Z = np.array([[loss(X, y, np.array([a, b]) / scale) for a in g] for b in g])
    Zc = np.minimum(Z, Z_MAX)

    # Gradient-descent path, converted from w-space to v-space and lifted onto the surface
    path_v = run_gd(X, y) * scale
    path_z = np.array([loss(X, y, p / scale) for p in path_v])

    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    ax.plot_surface(V1, V2, Zc, cmap="viridis", alpha=0.45, linewidth=0, antialiased=True)
    ax.contour(V1, V2, Zc, zdir="z", offset=0, levels=15, cmap="viridis", alpha=0.6)
    # path on the surface, and its shadow on the floor (always visible)
    ax.plot(path_v[:, 0], path_v[:, 1], np.minimum(path_z, Z_MAX) + 0.1,
            "-", color="red", linewidth=2.5, label=f"gradient descent ({STEPS} steps)", zorder=10)
    ax.plot(path_v[::10, 0], path_v[::10, 1], np.minimum(path_z[::10], Z_MAX) + 0.1,
            "o", color="red", markersize=4, zorder=10)
    ax.plot(path_v[:, 0], path_v[:, 1], 0, "-", color="red", linewidth=1.5, alpha=0.8)
    ax.scatter(*path_v[0], min(path_z[0], Z_MAX) + 0.1, color="black", s=70, marker="s", label="start", zorder=11)
    ax.scatter(*path_v[-1], min(path_z[-1], Z_MAX) + 0.1, color="gold", edgecolor="black", s=90, marker="*", label="end", zorder=11)
    ax.set_title(f"{name}\nfinal loss after {STEPS} steps: {path_z[-1]:.3f}")
    ax.set_xlabel("v1 (weight 1)")
    ax.set_ylabel("v2 (weight 2)")
    ax.set_zlabel("loss")
    ax.set_zlim(0, Z_MAX)
    ax.view_init(elev=28, azim=-55)
    ax.legend(loc="upper left")

plt.tight_layout()
plt.savefig("toy_loss_landscape_3d.png", dpi=200, bbox_inches="tight")
print("Saved toy_loss_landscape_3d.png")
