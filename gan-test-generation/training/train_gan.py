"""Trains the conditional GAN and saves the weights of the generator.

The training follows DCGAN (Radford et al., 2016): Adam with learning rate
0.0002 and beta1 0.5, batch size 128, and the binary cross-entropy loss.
As in DCGAN, the generator maximizes log(D(G(z))) instead of minimizing
log(1 - D(G(z))).
"""

import argparse
import os

import gan
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torch_utils
from torch import nn, optim


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the conditional GAN on MNIST")
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs of the training (default: 20)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join("models", "generator.pt"),
        help="File for the trained generator (default: models/generator.pt)",
    )
    arguments = parser.parse_args()
    assert arguments.epochs > 0, "The number of epochs must be a positive integer."

    torch.manual_seed(seed=arguments.seed)
    device = torch.device(device="cpu")

    training_dataset = torch_utils.get_train_dataset()
    training_batches = torch.utils.data.DataLoader(
        dataset=training_dataset, batch_size=128, shuffle=True
    )

    generator = gan.Generator().to(device=device)
    discriminator = gan.Discriminator().to(device=device)
    generator.apply(fn=gan.init_weights)
    discriminator.apply(fn=gan.init_weights)
    generator_optimizer = optim.Adam(
        params=generator.parameters(), lr=2e-4, betas=(0.5, 0.999)
    )
    discriminator_optimizer = optim.Adam(
        params=discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999)
    )
    binary_cross_entropy = nn.BCEWithLogitsLoss()
    # Show the losses 10 times for each epoch.
    log_interval = max(1, len(training_batches) // 10)

    for epoch in range(1, arguments.epochs + 1):
        for batch_index, training_batch in enumerate(training_batches):
            real_images, conditioning_labels = training_batch
            real_images = real_images.to(device=device)
            conditioning_labels = conditioning_labels.to(device=device)
            batch_size = real_images.shape[0]
            real_image_targets = torch.ones(size=(batch_size,), device=device)
            generated_image_targets = torch.zeros(size=(batch_size,), device=device)
            latent_vectors = torch.randn(
                size=(batch_size, gan.LATENT_DIM), device=device
            )
            generated_images = generator(
                latent_vectors=latent_vectors, conditioning_labels=conditioning_labels
            )

            # The discriminator learns to give 1 to real images and 0 to
            # generated images.
            discriminator_optimizer.zero_grad()
            real_image_logits = discriminator(
                images=real_images, conditioning_labels=conditioning_labels
            )
            generated_image_logits = discriminator(
                images=generated_images.detach(),
                conditioning_labels=conditioning_labels,
            )
            real_image_loss = binary_cross_entropy(
                input=real_image_logits, target=real_image_targets
            )
            generated_image_loss = binary_cross_entropy(
                input=generated_image_logits, target=generated_image_targets
            )
            discriminator_loss = real_image_loss + generated_image_loss
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # The generator learns to make the discriminator give 1 to the
            # generated images.
            generator_optimizer.zero_grad()
            generated_image_logits = discriminator(
                images=generated_images, conditioning_labels=conditioning_labels
            )
            generator_loss = binary_cross_entropy(
                input=generated_image_logits, target=real_image_targets
            )
            generator_loss.backward()
            generator_optimizer.step()

            if batch_index % log_interval == 0:
                print(
                    f"Train Epoch: {epoch} "
                    f"[{batch_index * batch_size}/{len(training_dataset)}] "
                    f"Loss D: {discriminator_loss.item():.4f} "
                    f"Loss G: {generator_loss.item():.4f}"
                )

    os.makedirs(name=os.path.dirname(arguments.model), exist_ok=True)
    torch.save(obj=generator.state_dict(), f=arguments.model)
    print(f"Generator saved in {arguments.model}")
    save_samples(generator=generator, device=device, output_path="gan_samples.png")


@torch.no_grad()
def save_samples(
    generator: gan.Generator, device: torch.device, output_path: str
) -> None:
    """Saves a grid with 10 generated images of each digit (one per row)."""
    generator.eval()
    conditioning_labels = torch.arange(end=gan.NUM_CLASSES, device=device)
    conditioning_labels = conditioning_labels.repeat_interleave(repeats=10)
    latent_vectors = torch.randn(
        size=(len(conditioning_labels), gan.LATENT_DIM), device=device
    )
    generated_images = generator(
        latent_vectors=latent_vectors, conditioning_labels=conditioning_labels
    ).cpu()

    figure, axes = plt.subplots(nrows=10, ncols=10, figsize=(8, 8))
    for axis, generated_image in zip(axes.flat, generated_images):
        axis.imshow(X=generated_image[0], cmap="gray", vmin=0, vmax=1)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(fname=output_path)
    plt.close(fig=figure)
    print(f"Samples saved in {output_path}")


if __name__ == "__main__":
    main()
