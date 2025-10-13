# Setup

Build the docker container:

```bash
docker build --build-arg USERNAME=$USER --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -t torch_matplotlib:2.8.0 .
```

Then start the `devcontainer`:

- Download [VSCode](https://code.visualstudio.com/Download) for your platform;
- Install DevContainer Extension;
- In VSCode, use the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P` on macOS) to run the "Dev Containers: Open Folder in Container..." command;
- Select the `neuron-coverage` folder;

Optionally, start the container without `devcontainer` by typing:

```bash
docker run --rm -v $PWD:/home/ -it torch_matplotlib:2.8.0
```

Once within the container, if `devcontainer` is used select the only python intepreter available, i.e., `3.11.13`. Then type:

```bash
python main.py
```

The script computes the neuron coverage of the MNIST test dataset of a pre-trained MNIST classifier (`mnist_net.pt`). By default it considers all the layers of the neural network under test, except the last one that determines the classification layer and it has no activation function. The `net.py` file contains the `Net` class that defines the network architecture of the pre-trained model: i.e., there are two convolutional layers, which are layers specialized for processing spatial data like images, and two fully connected layers (called `Linear` in PyTorch). The `Dropout` layers do not contain any weights and are only used during training to prevent overfitting. The `forward` method shows how the layers interact with each other, and how an input (i.e., `x`) flows through the layers. The `F.relu()` layers define the ReLU activation functions, which basically carry out the max operation between 0 and the value they receive as input. 

The `NeuronCoverage` class defined in the `neuron_coverage.py` file, is used to inject hooks into the model and track the coverage of activated neurons during the forward pass of the network. The class has two configurable options, namely, the threshold above which to consider a neuron activated (called `--threshold` in the `main.py` file set to `0.0` by default), and the option that controls how the neurons in the convolutional layers are *reduced*. By default, the neuron coverage class considers the maximum value per channel over the spatial dimensions: for instance, if the tensor output shape of a convolutional layer is `[1, 32, 26, 26]`, where the first number is the batch size `B`, the second number is the number of channels `C`, the third number is the height of the input image `H`, and the fourth number is the the width `W` of the input image, the max operation over this tensor results in a tensor of shape `[1, 32]`, hence the activation values considered are `32`. This option, called `--conv-reduce` in the `main.py` file, can have values `[max, mean, none]`, where `mean` means the average is computed instead of the maximum (in the previous example the output tensor shape is the same as for the `max` option), and `none` indicates that all the neurons will be considered for tracking. Considering the example above the output tensor shape is `[1, 21632]`, where `21632 = 32x26x26`.

The script `python main.py` generates a plot (`coverage_trend_0.0_max.png`) with the trend of neuron coverage over the test dataset of MNIST. The batch size is `1`, so the trend shows how much each individual input (`10k` in total) adds to the coverage. The plot shows that it is quite easy to saturate this metric neuron coverage. The script also prints how many neurons are covered per layer, out of the total number of neurons in the corresponding layer:

```bash
Neuron coverage by layer:
  conv1                         :    30/   32  (93.75%)
  conv2                         :    64/   64  (100.00%)
  fc1                           :   115/  128  (89.84%)
```

To convince yourself that neuron coverage is easy to saturate, you can try to `--shuffle` the test dataset, such that the images are shown in a random order w.r.t. the original one every time you run, and see that we a few inputs, irrespectively of the inputs, neuron coverage saturates to a certain value close to 90% and then never increases (the final coverage should remain the same every time you run with the `--shuffle` option).



