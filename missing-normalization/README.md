# Setup

Build the docker container:

```bash
docker build --build-arg USERNAME=$USER --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -t dockercontainervm/torch_matplotlib:2.8.0 .
```

The image will be around 1.4GB. Then start the `devcontainer`:

- Download [VSCode](https://code.visualstudio.com/Download) for your platform;
- Install DevContainer Extension;
- In VSCode, use the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P` on macOS) to run the "Dev Containers: Open Folder in Container..." command;
- Select the `vae-interpolation-example` folder.

Optionally, start the container without `devcontainer` by typing:

```bash
docker run -v $PWD:/home/ -it dockercontainervm/torch_matplotlib:2.8.0
```

Once within the container, if `devcontainer` is used select the only python intepreter available, i.e., `3.11.13`. Then type:

```bash
python main.py --seed 0
```

The command will download the training set of MNIST and train two models, one without normalizing the data, and the other with normalization enabled. Normalization consists in subtracting the mean `0.1307` from each image and dividing it by the standard deviation `0.3081`; mean and standard deviation are computed by taking into account the entire MNIST training set. This normalization is known as [Standard score](https://en.wikipedia.org/wiki/Standard_score). The two models are trained for 3 epochs and the loss is printed every 100 epochs: each batch is composed of 64 images, so the loss is printed every 6400 images. At the end of the training of the two models, the loss curves are plotted in `normalization_impact_seed_0.png`; the curves are noisy, but the loss curve corresponding to normalized data is always below.

Finally, the two models are taken and they are evaluated on the test set of MNIST: the "unnormalized" model has an accuracy of 93.67% while the "normalized" model has an accuracy of 96.36%. The whole process can be repeated with a different seed (e.g., `python main.py --seed 1`), and the results slightly change: the accuracy of the "unnormalized" model is 93.70% while the accuracy of the "normalized" is 96.59%. To execute the process with a random seed, just omit it from the arguments: `python main.py`; the seed will be chosen at random.



