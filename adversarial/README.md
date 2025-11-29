# Setup

Build the docker container:

```bash
docker build --build-arg USERNAME=$USER --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -t torch-matplotlib:2.8.0 .
```

The image will be around 1.4GB. Then start the `devcontainer`:

- Download [VSCode](https://code.visualstudio.com/Download) for your platform;
- Install DevContainer Extension;
- In VSCode, use the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P` on macOS) to run the "Dev Containers: Open Folder in Container..." command;
- Select the `adversarial` folder.

Optionally, start the container without `devcontainer` by typing:

```bash
docker run -v $PWD:/home/ -it torch-matplotlib:2.8.0
```

Once within the container, if `devcontainer` is used select the only python intepreter available, i.e., `3.11.13`. 

```bash
python main.py --seed 0
```

TODO
