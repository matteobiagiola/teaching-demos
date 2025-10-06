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
The two models are trained for 3 epochs by default, and the loss is printed 10 times per epoch: each batch is composed of 64 images by default (i.e., `train_batch_size = 64`). At the end of the training of the two models, the loss curves are plotted in `normalization_impact_epochs_3_batch_64_seed_0.png`; the curves are noisy, but the loss curve corresponding to normalized data is always below.

Finally, the two models are taken and they are evaluated on the test set of MNIST: the "unnormalized" model has an accuracy of 93.67% while the "normalized" model has an accuracy of 96.36% (when training with a GPU the results are slightly different, i.e., respectively 93.79% w/o normalization and 96.51% with normalization). The whole process can be repeated with a different seed (e.g., `python main.py --seed 1`), and the results slightly change: the accuracy of the "unnormalized" model is 93.70% while the accuracy of the "normalized" is 96.59%. To execute the process with a random seed, just omit it from the arguments: `python main.py`; the seed will be chosen at random.

It is possible to experiment with different batch sizes. For instance with a batch size of `1024`:

```bash
python main.py --seed 0 --train-batch-size 1024
```

the final loss will be much higher in both cases, because the model is updated much less often although the gradient updates are more precise because they are computed over much more data. The difference in final accuracy without and with normalization is much higher, respectively 12.17% and 60.69%.
