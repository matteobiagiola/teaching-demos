import argparse
import os
import torch

import torch.optim as optim
from randomness import get_torch_generator, set_random_seed
from torch_utils import get_train_dataset
from vae import ConvVAE, vae_loss


def main():
    parser = argparse.ArgumentParser(description="Train VAE for MNIST")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        metavar="N",
        help="input batch size for training (default: 1024)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        metavar="N",
        help="number of epochs to train (default: 20)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=20,
        metavar="N",
        help="latent dimension (default: 20)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, metavar="S", help="random seed (default: 0)"
    )

    args = parser.parse_args()

    seed = args.seed
    set_random_seed(seed=seed)

    latent_dim = args.latent_dim
    batch_size = args.batch_size
    max_epochs = args.epochs
    learning_rate = args.lr

    device = (
        torch.accelerator.current_accelerator()
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model = ConvVAE(latent_dim=latent_dim).to(device)

    os.makedirs("models", exist_ok=True)

    print("==== Train VAE ====")

    train_dataset = get_train_dataset(dataset_name="mnist")
    train_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        # the data order does not depend on the latent dimension of the model
        "generator": get_torch_generator(seed=seed),
    }
    if torch.cuda.is_available():
        accel_kwargs = {
            "num_workers": 1,
            "persistent_workers": True,
            "pin_memory": True,
        }
        train_kwargs.update(accel_kwargs)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    best_loss = torch.inf
    epochs_stagnation = 0
    for epoch in range(1, max_epochs + 1):

        if epochs_stagnation == 10:
            print(
                "Early stopping triggered due to no significant improvement in loss for 10 consecutive epochs."
            )
            break

        total_loss, total_recon, total_kl = 0, 0, 0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar = model(x)
            loss, recon, kl = vae_loss(
                tensor_image=x, reconstructed_tensor_image=x_hat, mu=mu, logvar=logvar
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()

        total_loss /= len(train_loader.dataset)
        total_recon /= len(train_loader.dataset)
        total_kl /= len(train_loader.dataset)

        # save if improvement is > 2% of the best loss so far
        if total_loss < 0.98 * best_loss:
            best_loss = total_loss
            print(f"Saving the model with loss: {best_loss}")
            torch.save(
                model.state_dict(),
                os.path.join("models", f"vae_mnist_{latent_dim}.pt"),
            )
            epochs_stagnation = 0
        else:
            epochs_stagnation += 1

        print(
            f"Epoch {epoch} | Loss: {total_loss:.2f} | Recon: {total_recon:.2f} | KL: {total_kl:.2f}"
        )


if __name__ == "__main__":
    main()
