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


class ThermometerEncoder:
    """
    Thermometer encoding for adversarial defense.
    Discretizes continuous values into levels and encodes as thermometer code.
    """

    def __init__(self, num_levels=16):
        self.num_levels = num_levels

    def encode(self, x):
        """
        Encode input tensor using thermometer encoding.

        Args:
            x: Input tensor with values in [0, 1], shape (B, C, H, W)

        Returns:
            Encoded tensor of shape (B, C * num_levels, H, W)
        """
        batch_size, channels, height, width = x.shape

        # Discretize to num_levels
        # Scale to [0, num_levels] and floor
        discretized = torch.floor(x * self.num_levels)
        discretized = torch.clamp(discretized, 0, self.num_levels - 1)

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

        for level in range(self.num_levels):
            # Set to 1 if discretized value > level
            encoded[:, :, level, :, :] = (discretized >= level).float()

        # Reshape to (B, C * num_levels, H, W)
        encoded = encoded.view(batch_size, channels * self.num_levels, height, width)

        return encoded


class ThermometerCNN(nn.Module):
    """
    CNN for MNIST with thermometer encoding input.
    Matches the architecture of the provided Net class.
    """

    def __init__(self, num_levels=16, num_classes=10):
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

    def forward(self, x):
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
