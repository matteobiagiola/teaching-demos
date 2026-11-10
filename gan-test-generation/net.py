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
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    """Convolutional neural network for the MNIST dataset.

    The network has two convolutional layers (conv1 and conv2) and two fully
    connected layers (fc1 and fc2). A ReLU function follows conv1, conv2 and
    fc1.
    """

    def __init__(self) -> None:
        super().__init__()
        # conv means convolutional; fc means fully connected.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(in_features=9216, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Computes the output of the network.

        Args:
            images: Batch of images with shape [batch, 1, 28, 28].

        Returns:
            The log-probability of each class, with shape [batch, 10].
        """
        layer_activations = self.conv1(input=images)
        layer_activations = F.relu(input=layer_activations)
        layer_activations = self.conv2(input=layer_activations)
        layer_activations = F.relu(input=layer_activations)
        layer_activations = F.max_pool2d(input=layer_activations, kernel_size=2)
        layer_activations = self.dropout1(input=layer_activations)
        layer_activations = self.flatten(input=layer_activations)
        layer_activations = self.fc1(input=layer_activations)
        layer_activations = F.relu(input=layer_activations)
        layer_activations = self.dropout2(input=layer_activations)
        class_logits = self.fc2(input=layer_activations)
        return F.log_softmax(input=class_logits, dim=1)
