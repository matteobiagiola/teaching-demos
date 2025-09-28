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

Once within the container, if `devcontainer` is used select the only python intepreter available, i.e., `3.11.13`. Then type:

```bash
python main.py --seed 0
```

The command will download the MNIST dataset, and look for the first pair of digits labeled with 3 and 5, and linearly interpolate between them using 10 steps using the pre-trained VAE named `vae_mnist.pt`, which is loaded at the beginning of the execution. The output of the execution is the file `vae_interpolation.png` that shows how we can smoothly transition from a 3 to a 5 in the latent space.

Run without specifying the seed to get different pairs of 3s and 5s.

It is also possible to run with generation mode, by explicitly specifying it:

```bash
python main.py --seed 0 --generation
```

The commands will generate 10 random latent vectors and then decode them and plot the reconstructed images side by side.




