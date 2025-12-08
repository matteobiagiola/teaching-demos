"""
BSD 3-Clause License

Copyright (c) 2017, Pytorch contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class ThermometerSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, thresholds):
        return (x >= thresholds).float()

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass gradient back unchanged
        return grad_output, None


class ThermometerEncoder:
    """
    Thermometer encoding for adversarial defense.
    Discretizes continuous values into levels and encodes as thermometer code.
    """

    def __init__(self, num_levels: int = 16) -> None:
        self.num_levels = num_levels

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input tensor using thermometer encoding.

        Args:
            x: Input tensor with values in [0, 1], shape (B, C, H, W)

        Returns:
            Encoded tensor of shape (B, C * num_levels, H, W)
        """
        batch_size, channels, height, width = x.shape

        x_clone = x.clone()
        x_clone = x_clone * 0.3081 + 0.1307

        # print(x_clone.view(-1))
        # print()

        # Discretize to num_levels
        # Scale to [0, num_levels] and floor
        # discretized = torch.floor(x_clone * self.num_levels)
        # print(discretized.view(-1))
        # print()
        # discretized = torch.clamp(discretized, 0, self.num_levels - 1)
        # print(discretized.view(-1))
        # print()

        thresholds = torch.linspace(
            0, 1, steps=self.num_levels, device=x.device, dtype=x.dtype
        )

        # Create thermometer encoding
        # For each pixel value v, create binary vector [1,1,...,1,0,0,...,0]
        # with v ones followed by (num_levels - v) zeros
        encoded = torch.zeros(
            batch_size,
            channels,
            self.num_levels,
            height,
            width,
            device=x.device,
            dtype=x.dtype,
        )

        # print(encoded.shape)

        for i, t in enumerate(thresholds):
            # Set to 1 if value >= threshold
            encoded[:, :, i] = ThermometerSTE.apply(x_clone, t)
            # encoded[:, :, i, :, :] = ThermometerSTE.apply(x_clone, t)
            # print(f"Threshold {t}:")
            # print(encoded[:, :, i, :, :].view(-1))
            # print()

        # for level in range(self.num_levels):
        #     # Set to 1 if discretized value > level)
        #     # encoded[:, :, level, :, :] = (discretized >= level).float()
        #     encoded[:, :, level, :, :] = ThermometerSTE.apply(
        #         discretized, torch.tensor(level)
        #     )
        #     # print(f"Level {level} encoding:")
        #     # print(encoded[:, :, level, :, :].view(-1))
        #     # print()

        # Reshape to (B, C * num_levels, H, W)
        encoded = encoded.view(batch_size, channels * self.num_levels, height, width)

        # print(encoded.view(-1).shape)
        # assert False

        return encoded


class ThermometerCNN(nn.Module):
    """
    CNN for MNIST with thermometer encoding input.
    Matches the architecture of the provided Net class.
    """

    def __init__(self, num_levels: int = 16, num_classes: int = 10) -> None:
        super(ThermometerCNN, self).__init__()
        self.encoder = ThermometerEncoder(num_levels=num_levels)

        # Input channels = 1 (grayscale) * num_levels
        input_channels = num_levels

        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply thermometer encoding
        x = self.encoder.encode(x)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
