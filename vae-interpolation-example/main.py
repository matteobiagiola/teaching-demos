import os
import torch
from vae import ConvVAE
from torchvision import datasets, transforms
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
        "--generation",
        action="store_true",
        help="If set, perform generation instead of interpolation",
    )

    args = parser.parse_args()
    
    seed = args.seed
    if seed == -1:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    print(f"Using seed: {seed}")
    generation = args.generation
    
    vae_path = "vae_mnist.pt"
    latent_dim = 20
    torch.manual_seed(seed)

    device = torch.device("cpu")
    vae = ConvVAE(latent_dim=latent_dim).to(device)

    assert os.path.exists(vae_path), f"{vae_path} does not exist"

    vae.load_state_dict(torch.load(vae_path, weights_only=True, map_location=device))
    
    if not generation:

        test_dataset = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=True)

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

                mu1, sigma1 = vae.encode(img1)
                mu2, sigma2 = vae.encode(img2)
                
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
                plt.savefig("vae_interpolation.png")
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
            plt.savefig("vae_generation.png")


if __name__ == "__main__":
    main()
