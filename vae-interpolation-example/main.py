import os
import torch
from vae import ConvVAE
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def main():
    vae_path = "vae_mnist.pt"
    latent_dim = 20

    device = (
        torch.accelerator.current_accelerator()
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    vae = ConvVAE(latent_dim=latent_dim).to(device)

    assert os.path.exists(vae_path), f"{vae_path} does not exist"

    vae.load_state_dict(torch.load(vae_path, weights_only=True))

    test_dataset = datasets.EMNIST(
        "datasets/",
        train=True,
        download=True,
        split="digits",
        transform=transforms.Compose(
            [
                transforms.Lambda(lambda img: transforms.functional.rotate(img, -90)),
                transforms.Lambda(lambda img: transforms.functional.hflip(img)),
                transforms.ToTensor(),
            ]
        ),
    )
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

            mu1, _ = vae.encode(img1)
            mu2, _ = vae.encode(img2)

            steps = 10
            # Linear interpolation in latent space
            # For example, with 5 steps:
            # t = 0.0 → exactly mu1 (start point)
            # t = 0.25 → 75% mu1, 25% mu2
            # t = 0.5 → halfway between
            # t = 1.0 → exactly mu2 (end point)
            z_interp = torch.stack(
                [mu1 * (1 - t) + mu2 * t for t in torch.linspace(0, 1, steps)]
            ).squeeze(1)

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


if __name__ == "__main__":
    main()
