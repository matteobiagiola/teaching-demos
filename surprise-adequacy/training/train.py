"""Trains the MNIST classifier under test and saves its weights."""

import argparse
import os

import torch
import torch.nn.functional as F
from torch import optim
import torch.utils.data

import net
import torch_utils


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the MNIST classifier under test"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of images in one training batch (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs of the training (default: 3)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate of the Adam optimizer (default: 0.001)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (default: 0)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join("models", "mnist_net.pt"),
        help="File for the trained model (default: models/mnist_net.pt)",
    )

    args = parser.parse_args()
    assert args.batch_size > 0, "The batch size must be a positive integer."
    assert args.epochs > 0, "The number of epochs must be a positive integer."
    assert args.lr > 0, "The learning rate must be a positive float."

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    train_loader = torch.utils.data.DataLoader(
        torch_utils.get_train_dataset(),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch_utils.get_test_dataset(), batch_size=1000, shuffle=False
    )

    model = net.Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Show the loss 10 times for each epoch.
    log_interval = max(1, len(train_loader) // 10)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # Net gives log-probabilities (log_softmax). Because of this, the
            # loss is the negative log-likelihood.
            loss = F.nll_loss(model(images), labels)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(
                    f"Train Epoch: {epoch} "
                    f"[{batch_idx * len(images)}/{len(train_loader.dataset)}]"
                    f" Loss: {loss.item():.6f}"
                )

        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                predicted = model(images.to(device)).argmax(dim=1)
                correct += (predicted == labels.to(device)).sum().item()
        accuracy = 100 * correct / len(test_loader.dataset)
        print(f"Epoch {epoch}: test accuracy {accuracy:.2f}%")

    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    torch.save(model.state_dict(), args.model)
    print(f"Model saved in {args.model}")


if __name__ == "__main__":
    main()
