import os
import torch
from net import Net
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import argparse
import numpy as np

from neuron_coverage import NeuronCoverage

def main():
    
    parser = argparse.ArgumentParser(description="Neuron coverage example")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Threshold for neuron coverage (default: 0.0); to be considered 'covered', a neuron must have activation > threshold at least once",
    )
    parser.add_argument(
        "--conv-reduce",
        type=str,
        choices=["max", "mean", "none"],
        default="max",
        help="Reduction method for conv layers (default: max)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the test dataset (default: False)"
    )

    args = parser.parse_args()
    threshold = args.threshold
    assert threshold >= 0.0, "Threshold should be >= 0.0"
    conv_reduce = args.conv_reduce
    shuffle = args.shuffle
    
    model_path = "mnist_net.pt"

    device = torch.device("cpu")
    model = Net().to(device)

    assert os.path.exists(model_path), f"{model_path} does not exist"

    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    
    batch_size = 1
    
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    
    neuron_coverage = NeuronCoverage(
        model,       
        device=device,
        threshold=threshold,          
        conv_reduce=conv_reduce
    )
    
    coverage_trend = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            _ = model(images)
            neuron_coverage.update_with_batch()
            _, _, fraction = neuron_coverage.overall_coverage()
            coverage_trend.append(fraction)

    by_layer = neuron_coverage.coverage_by_layer()
    overall = neuron_coverage.overall_coverage()
    neuron_coverage.close()

    print("Neuron coverage by layer:")
    for lname, (coverage, total, fraction) in by_layer.items():
        print(f"  {lname:30s}: {coverage:5d}/{total:5d}  ({100*fraction:5.2f}%)")

    coverage, total, fraction = overall
    print(f"\nOverall coverage: {coverage}/{total} ({100*fraction:.2f}%)")
    
    plt.plot(np.arange(len(coverage_trend)), coverage_trend)
    plt.xlabel("Batches")
    plt.ylabel("Coverage")
    plt.title("Neuron Coverage Trend")
    plt.savefig(f"coverage_trend_{threshold}_{conv_reduce}.png")


if __name__ == "__main__":
    main()
