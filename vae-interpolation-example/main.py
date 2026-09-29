import os
import torch
from vae import ConvVAE, sample_latent
from torch_utils import get_test_dataset
from randomness import get_torch_generator, set_random_seed
import matplotlib.pyplot as plt

import argparse
import numpy as np

def main():
    
    parser = argparse.ArgumentParser(description="VAE Latent Space Interpolation")
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed for reproducibility (default: -1, random seed will be set automatically)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/vae_mnist_20.pt",
        help="Path to the trained VAE (default: models/vae_mnist_20.pt)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=20,
        help="Latent dimension of the trained VAE (default: 20)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--generation",
        action="store_true",
        help="If set, perform generation instead of interpolation",
    )
    mode.add_argument(
        "--stats",
        action="store_true",
        help="If set, compare the distribution of the latent codes of the test set with N(0, 1)",
    )

    args = parser.parse_args()
    
    seed = args.seed
    if seed == -1:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    print(f"Using seed: {seed}")
    generation = args.generation
    stats = args.stats
    
    vae_path = args.model
    latent_dim = args.latent_dim
    set_random_seed(seed=seed)

    device = torch.device("cpu")
    vae = ConvVAE(latent_dim=latent_dim).to(device)

    assert os.path.exists(vae_path), f"{vae_path} does not exist"

    state_dict = torch.load(vae_path, weights_only=True, map_location=device)
    model_latent_dim = state_dict["encoder.fc_mu.weight"].shape[0]
    assert (
        model_latent_dim == latent_dim
    ), f"{vae_path} has latent dimension {model_latent_dim}, but --latent-dim is {latent_dim}"
    vae.load_state_dict(state_dict)
    
    if stats:
        # --- Latent statistics ---
        test_dataset = get_test_dataset(dataset_name="mnist")
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)

        vae.eval()
        with torch.no_grad():
            mus, log_vars = [], []
            for images, _ in test_loader:
                mu, log_var = vae.encode(images.to(device))
                mus.append(mu)
                log_vars.append(log_var)
            mu = torch.cat(mus)
            log_var = torch.cat(log_vars)

            # The encoder does not map an image to a single point, but to a small
            # cloud of points centered in mu with width var (the encoder outputs
            # log(var)). Here we pick one random point z from the cloud of each
            # image. The KL term pushes the clouds towards N(0, I), so the codes z
            # of the whole dataset should look like N(0, I). Their variance comes
            # from two sources: how far the centers mu are from each other
            # (var(mu)) and how wide each cloud is on average (mean(var)), so
            # var(mu) alone is expected to be < 1
            z = sample_latent(mu, log_var)
            # print(z.shape)  # should be (10000, latent_dim)

        z_mean = z.mean(dim=0)
        z_var = z.var(dim=0)
        mu_var = mu.var(dim=0)
        var_mean = log_var.exp().mean(dim=0)

        print(f"{'dim':>3} {'mean(z)':>8} {'var(z)':>7} {'var(mu)':>8} {'mean(var)':>10}")
        for d in range(latent_dim):
            print(
                f"{d:>3} {z_mean[d]:8.3f} {z_var[d]:7.3f} {mu_var[d]:8.3f} {var_mean[d]:10.3f}"
            )
        print(f"Average |mean(z)|: {z_mean.abs().mean():.3f} (expected 0)")
        print(f"Average var(z): {z_var.mean():.3f} (expected 1)")

        # Plot one histogram per latent dimension against the N(0, 1) density
        x = np.linspace(-4, 4, 200)
        normal_pdf = np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)
        cols = 5 if latent_dim <= 20 else 8
        rows = int(np.ceil(latent_dim / cols))
        fig, axes = plt.subplots(
            rows, cols, figsize=(3 * cols, 2.5 * rows), sharex=True, sharey=True
        )
        for ax in axes.flat[latent_dim:]:
            ax.axis("off")
        for d, ax in enumerate(axes.flat[:latent_dim]):
            ax.hist(
                z[:, d].numpy(),
                bins=50,
                range=(-4, 4),
                density=True,
                color="#2a78d6",
                edgecolor="white",
                linewidth=0.5,
                label="z (test set)",
            )
            ax.plot(x, normal_pdf, color="#eb6834", linewidth=2, label="N(0, 1)")
            ax.set_title(
                f"dim {d}: mean={z_mean[d]:.2f}, var={z_var[d]:.2f}", fontsize=10
            )
            ax.grid(alpha=0.3)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.suptitle("ConvVAE: Distribution of the Latent Codes vs. N(0, 1)")
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.965), frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        plt.savefig(f"vae_latent_stats_{latent_dim}.png")

    elif not generation:

        test_dataset = get_test_dataset(dataset_name="mnist")
        # The shuffling has its own random generator, so that the same seed picks
        # the same pair of digits whatever model is loaded (creating the model
        # also uses random numbers, and how many depends on the latent dimension)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=2,
            shuffle=True,
            generator=get_torch_generator(seed=seed),
        )

        # look for a 5 and a 3
        expected_labels = {5, 3}

        # --- Interpolation ---
        vae.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                img1 = images[0].unsqueeze(0).to(device)
                img2 = images[1].unsqueeze(0).to(device)
                label1 = labels[0].item()
                label2 = labels[1].item()

                if {label1, label2} != expected_labels:
                    print(
                        f"Skipping pair {label1}, {label2}, looking for {expected_labels}"
                    )
                    continue

                mu1, log_var1 = vae.encode(img1)
                mu2, log_var2 = vae.encode(img2)
                
                # remove batch dimension
                mu1 = mu1.squeeze(0)
                mu2 = mu2.squeeze(0)
                
                # print(mu1.shape, mu2.shape)  # should be (latent_dim)

                steps = 10
                # Linear interpolation in latent space
                # For example, with 5 steps:
                # t = 0.0 → exactly mu1 (start point)
                # t = 0.25 → 75% mu1, 25% mu2
                # t = 0.5 → halfway between
                # t = 1.0 → exactly mu2 (end point)
                z_interp = torch.stack(
                    [mu1 * (1 - t) + mu2 * t for t in torch.linspace(0, 1, steps)]
                )
                # print(z_interp.shape)  # should be (steps, latent_dim)

                recon_images = vae.decode(z_interp).cpu()

                # Plot interpolation
                _, axes = plt.subplots(1, steps, figsize=(15, 2))
                for i, ax in enumerate(axes):
                    ax.imshow(recon_images[i].squeeze(0), cmap="gray")
                    ax.axis("off")
                plt.suptitle(
                    f"ConvVAE: Latent Interpolation between Two Digits ({label1} and {label2})"
                )
                plt.savefig(f"vae_interpolation_{latent_dim}.png")
                break
    else:
        # --- Generation ---
        vae.eval()
        with torch.no_grad():
            num_samples = 10
            z_samples = torch.randn(num_samples, latent_dim).to(device)
            recon_images = vae.decode(z_samples).cpu()

            # Plot generation
            _, axes = plt.subplots(1, num_samples, figsize=(15, 2))
            for i, ax in enumerate(axes):
                ax.imshow(recon_images[i].squeeze(0), cmap="gray")
                ax.axis("off")
            plt.suptitle("ConvVAE: Random Generation from Latent Space")
            plt.savefig(f"vae_generation_{latent_dim}.png")


if __name__ == "__main__":
    main()
