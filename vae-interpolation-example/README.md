# Setup

Build the docker container:

```bash
docker build --build-arg USERNAME=$USER --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -t torch_matplotlib:2.8.0 .
```

The image will be around 1.4GB. Then start the `devcontainer`:

- Download [VSCode](https://code.visualstudio.com/Download) for your platform;
- Install DevContainer Extension;
- In VSCode, use the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P` on macOS) to run the "Dev Containers: Open Folder in Container..." command;
- Select the `vae-interpolation-example` folder.

Optionally, start the container without `devcontainer` by typing:

```bash
docker run -v $PWD:/home/ -it torch_matplotlib:2.8.0
```

Once within the container, if `devcontainer` is used select the only python intepreter available, i.e., `3.11.13`. 

First of all, what is a tensor? The first difference w.r.t. a `numpy` array is the built-in automatic differentiation capability. In the command line type `python`:

```python
import torch

# Define an input tensor, requiring gradient calculation
x = torch.tensor(3.0, requires_grad=True) 

# Define a simple function: y = x^2
y = x**2

# Compute the gradient of y with respect to x (dy/dx)
# The derivative of x^2 is 2x. At x=3.0, dy/dx is 2*3.0 = 6.0
y.backward() 

# Access the computed gradient
print("PyTorch Tensor Gradient (dy/dx):", x.grad)
# Output: PyTorch Tensor Gradient (dy/dx): tensor(6.)
```

The value of the derivative is stored in the `.grad` property of the tensor w.r.t. which the derivative is computed, i.e., `x` in this case. The second difference is that `numpy` arrays live in the CPU, while tensors can be moved to an accelerator, like the GPU (`x.to(torch.device("cuda"))`, the instruction should throw an error, as the container is CPU only, in particular `AssertionError: Torch not compiled with CUDA enabled`). To start the VAE example type:

```bash
python main.py --seed 0
```

The command will download the MNIST dataset, and look for the first pair of digits labeled with 3 and 5, and linearly interpolate between them using 10 steps with the pre-trained VAE `models/vae_mnist_20.pt` (latent dimension 20), which is loaded at the beginning of the execution. The output of the execution is the file `vae_interpolation.png` that shows how we can smoothly transition from a 3 to a 5 in the latent space.

Run without specifying the seed to get different pairs of 3s and 5s.

It is also possible to run with generation mode, by explicitly specifying it:

```bash
python main.py --seed 0 --generation
```

The command will generate 10 random latent vectors, decode them and plot the reconstructed images side by side.

Generation works because the KL term of the VAE loss pushes the output of the encoder, i.e., `mu` and `var` for each image, towards the prior `N(0, I)`, so the latent codes of the whole dataset should look like samples from `N(0, I)`. To check how close the trained VAE gets, run:

```bash
python main.py --seed 0 --stats
```

The encoder of a VAE does not map an image to a single point of the latent space, but to a small cloud of points centered in `mu` with width `var`. 
The command encodes the MNIST test set, picks one random point `z` from the cloud of each image, and prints the mean and variance of each latent dimension. 
Note that `var(z)`, i.e., the variance of the codes of the whole dataset, is not the same as `var`, i.e., the width of the cloud of a single image. 
The variance of the codes comes from two sources: how far the centers `mu` of the different images are from each other (`var(mu)`), and how wide each cloud is on average (`mean(var)`). 
This is why `var(mu)` alone is expected to be below 1.
The output file `vae_latent_stats.png` shows one histogram per latent dimension, with the `N(0, 1)` density on top.
In the trained VAE all dimensions have a mean close to 0 and a variance close to 1, but they do not all split the variance in the same way.
In some dimensions (e.g., 11 and 16) the clouds are narrow (`mean(var)` below 0.1), while the centers are far apart (`var(mu)` around 1): these dimensions carry information about the image.
In other dimensions (e.g., 2 and 10) the clouds are as wide as `N(0, 1)` (`mean(var)` above 0.8), while the centers are close to each other (`var(mu)` around 0.1): these dimensions carry almost no information, as the encoder outputs nearly the same cloud for every image.
This is the effect of the two terms of the loss: the reconstruction term wants narrow clouds far apart, so that different images get different codes, while the KL term wants every cloud to look like `N(0, 1)`.
A dimension is used only when it helps the reconstruction enough to pay for its KL cost.

## Using a different model

By default, `main.py` loads `models/vae_mnist_20.pt`. To use another trained VAE, pass its path and its latent dimension:

```bash
python main.py --seed 0 --model models/vae_mnist_64.pt --latent-dim 64 --stats
```

The options work with all modes (interpolation, `--generation` and `--stats`). If the latent dimension does not match the model, the command stops with an error.

With latent dimension 64, the histograms match `N(0, 1)` even more closely, but only a few dimensions (around 16) carry information about the image: in the others, the encoder outputs `N(0, 1)` for every image.
A larger latent space does not mean that the VAE stores more information about the image, and a close match with `N(0, 1)` does not mean that a dimension is useful.

## Training

The training code is in the `training` package. To train a VAE with a given latent dimension, run from this folder:

```bash
python -m training.train_vae --latent-dim 64
```

The model is saved in `models/vae_mnist_<latent-dim>.pt`. The loss (`vae_loss` in `vae.py`) is the sum of the reconstruction error (MSE) and the KL term. Training and testing apply the same preprocessing to the images (`get_train_dataset` and `get_test_dataset` in `torch_utils.py`), i.e., the pixels are only scaled between 0 and 1. Run `python -m training.train_vae --help` to see the other options (batch size, epochs, learning rate and seed). With the default options, training takes about 2 minutes on the CPU of the container.
