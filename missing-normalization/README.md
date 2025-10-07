# Setup

Build the docker container (CPU version, recommended):

```bash
docker build --build-arg USERNAME=$USER --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -f cpu.Dockerfile -t torch_matplotlib:2.8.0 .
```

The image will be around 1.4GB. If you have an NVIDIA GPU, you can build the `GPU` version of the image by typing:

```bash
docker build --build-arg USERNAME=$USER --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -f gpu.Dockerfile -t torch_matplotlib:2.8.0-cuda .
```

Ensure you have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed. In this case the image will be around ~7GB.

Then start the `devcontainer`:

- Download [VSCode](https://code.visualstudio.com/Download) for your platform;
- Install DevContainer Extension;
- In VSCode, use the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P` on macOS) to run the "Dev Containers: Open Folder in Container..." command;
- Select the `missing-normalization` folder;
- Select either `cpu` or `gpu`, according to your choice above.

Optionally, start the container without `devcontainer` by typing:

```bash
docker run --rm -v $PWD:/home/ -it torch_matplotlib:2.8.0
```

Or alternatively by typing:

```bash
docker run --rm -v $PWD:/home/ --gpus all -it torch_matplotlib:2.8.0-cuda
```

Once within the container, if `devcontainer` is used select the only python intepreter available, i.e., `3.11.13`. Then type:

```bash
python main.py --seed 0
```

The command will download the training set of MNIST and train two models, one without normalizing the data, and the other with normalization enabled. Normalization consists in subtracting the mean `0.1307` from each image and dividing it by the standard deviation `0.3081`; mean and standard deviation are computed by taking into account the entire MNIST training set. This normalization is known as [Standard score](https://en.wikipedia.org/wiki/Standard_score). 
The two models are trained for 3 epochs by default, and the loss is printed 10 times per epoch: each batch is composed of 64 images by default (i.e., `train_batch_size = 64`) and the learning rate is `0.05`. At the end of the training of the two models, the loss curves are plotted in `normalization_impact_epochs_3_batch_64_lr__seed_0.png`; the curves are noisy, but the loss curve corresponding to normalized data is always below.

Finally, the two models are taken and they are evaluated on the test set of MNIST: the "unnormalized" model has an accuracy of 90.01% while the "normalized" model has an accuracy of 98.39% (when training on a GPU the results might be slightly different). The whole process can be repeated with a different seed (e.g., `python main.py --seed 1`), and the results slightly change: the accuracy of the "unnormalized" model is 91.50% while the accuracy of the "normalized" is 98.17%. To execute the process with a random seed, just omit it from the arguments: `python main.py`; the seed will be chosen at random.

It is possible to experiment with different learning rates. For instance with a learning rate of `0.1`:

```bash
python main.py --seed 0 --learning-rate 0.1
```

we see that the normalized version is much more resilient to more aggressive updates, improving its performance slightly (i.e., from 98.17% to 98.70%); on the other hand the performance of the unnormalized version gets worse, going from 90.01% to 83.70%. The trend of the loss curve for the unnormalized version also looks non-monotonic.
