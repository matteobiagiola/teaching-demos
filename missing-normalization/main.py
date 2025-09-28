from typing import List
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# --- Simple Neural Network Model ---
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
# --- Training Loop Function ---
def train_model(
        model: nn.Module, 
        device: torch.device, 
        train_loader: torch.utils.data.DataLoader, 
        optimizer: optim.Optimizer, 
        epoch: int, 
        log_interval: int
    ) -> List[float]:

    losses = []
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')
    return losses

def evaluate_model(model: nn.Module, device: torch.device, testloader: torch.utils.data.DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def main():
    
    parser = argparse.ArgumentParser(description="Impact of Data Normalization on Training")
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed for reproducibility (default: -1, random seed will be set automatically)",
    )

    args = parser.parse_args()
    
    seed = args.seed
    if seed == -1:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    print(f"Using seed: {seed}")
    torch.manual_seed(seed)
    
    # --- Data Loading and Transforms ---

    # Standard transforms for MNIST
    transform_unnormalized = transforms.Compose([
        transforms.ToTensor()
    ])

    # Transforms with normalization using pre-calculated mean and std of MNIST
    transform_normalized = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download and create data loaders
    print("Downloading and preparing datasets...")
    train_dataset_unnormalized = datasets.MNIST('data', train=True, download=True, transform=transform_unnormalized)
    train_loader_unnormalized = torch.utils.data.DataLoader(train_dataset_unnormalized, batch_size=64, shuffle=True)

    train_dataset_normalized = datasets.MNIST('data', train=True, download=True, transform=transform_normalized)
    train_loader_normalized = torch.utils.data.DataLoader(train_dataset_normalized, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 3
    log_interval = 100

    # Setup for Unnormalized Model
    model_unnormalized = SimpleNet().to(device)
    optimizer_unnormalized = optim.SGD(model_unnormalized.parameters(), lr=0.01)

    # Setup for Normalized Model
    model_normalized = SimpleNet().to(device)
    optimizer_normalized = optim.SGD(model_normalized.parameters(), lr=0.01)

    print(f"--- Training with UNNORMALIZED data for {epochs} epochs ---")
    for epoch in range(1, epochs + 1):
        losses_unnormalized = train_model(
            model=model_unnormalized, 
            device=device, 
            train_loader=train_loader_unnormalized, 
            optimizer=optimizer_unnormalized, 
            epoch=epoch, 
            log_interval=log_interval, 
        )

    print(f"\n--- Training with NORMALIZED data for {epochs} epochs ---")
    for epoch in range(1, epochs + 1):
        losses_normalized = train_model(
            model=model_normalized, 
            device=device, 
            train_loader=train_loader_normalized, 
            optimizer=optimizer_normalized, 
            epoch=epoch, 
            log_interval=log_interval
        )

    test_dataset_unnormalized = datasets.MNIST('data', train=False, download=True, transform=transform_unnormalized)
    test_loader_unnormalized = torch.utils.data.DataLoader(test_dataset_unnormalized, batch_size=1000, shuffle=False)

    test_dataset_normalized = datasets.MNIST('data', train=False, download=True, transform=transform_normalized)
    test_loader_normalized = torch.utils.data.DataLoader(test_dataset_normalized, batch_size=1000, shuffle=False)

    print("\n--- Evaluation on UNNORMALIZED data ---")
    accuracy_unnormalized = evaluate_model(
        model=model_unnormalized,
        device=device,
        testloader=test_loader_unnormalized
    )
    print(f"Accuracy (w/o Normalization): {accuracy_unnormalized:.2f}%")

    print("\n--- Evaluation on NORMALIZED data ---")
    accuracy_normalized = evaluate_model(
        model=model_normalized,
        device=device,
        testloader=test_loader_normalized
    )
    print(f"Accuracy (w/ Normalization): {accuracy_normalized:.2f}%")

    # --- Visualization of Loss Curves ---
    plt.figure(figsize=(10, 5))
    plt.plot(losses_unnormalized, label='Loss (w/o Normalization)')
    plt.plot(losses_normalized, label='Loss (w/ Normalization)')
    plt.title('Loss Curve Comparison')
    plt.xlabel('Training Steps (batches)')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.savefig(f"normalization_impact_seed_{seed}.png", bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    main()