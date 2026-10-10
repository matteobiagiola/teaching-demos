# Setup

Build the docker container:

```bash
docker build --build-arg USERNAME=$USER --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -t torch_matplotlib_dnn_tip:2.8.0 .
```

The size of the image is approximately 2.5 GB.
Then start the `devcontainer`:

- Download [VSCode](https://code.visualstudio.com/Download) for your platform;
- Install the Dev Containers extension;
- In VSCode, open the Command Palette (`Ctrl+Shift+P`, or `Cmd+Shift+P` on macOS) and run the command "Dev Containers: Open Folder in Container...";
- Select the `surprise-adequacy` folder.

You can also start the container without `devcontainer`.
Use this command:

```bash
docker run --rm -v $PWD:/home/ -it torch_matplotlib_dnn_tip:2.8.0
```

If you use `devcontainer`, select the only Python interpreter in the container (`3.11.17`).

This demo shows how [Surprise Adequacy](https://arxiv.org/abs/1808.08444) (SA) and Surprise Coverage (SC) operate.
A test input is useful if it is *surprising* for the model, when you compare it with the training set.
SA measures the distance between the *activation trace* of a new input and the activation traces of the training set.
The demo uses [dnn-tip](https://github.com/testingautomated-usi/dnn-tip), an optimized implementation of SA.

The model under test is the MNIST classifier in `net.py`, which has a test accuracy of 99.09%.
The trained model is in `models/mnist_net.pt`.
To train the model again (approximately 1 minute on a CPU with 32 cores), run `python -m training.train`.

## What is an activation trace?

Run this command:

```bash
python activation_traces.py
```

The script sends two images of the digit `7` and one image of the digit `1` through the network.
The *activation trace* (AT) of an input at a layer is the vector of the activation values that the neurons of the layer compute.
The activation value of a neuron is the output of the neuron for one input, after the activation function (ReLU).
The plot `activation_traces_fc1.png` shows the AT of each image at the layer `fc1` (128 neurons).
The two `7`s activate almost the same neurons, but the `1` activates different neurons.
The distance between the ATs of the two `7`s is 13.24, and the distance between a `7` and the `1` is 48.01.
These values show that inputs that are similar for the network have ATs that are near to each other.

## SA in 2D

In the script `toy_2d.py`, a layer has only two neurons.
Because of this, the AT of each input is a point in the plane.
Run this command:

```bash
python toy_2d.py
```

The plot `toy_2d.png` shows one training set with three classes (A, B and C), as MNIST has 10 classes.
Each dot is the AT of one training input, and each star is the AT of one test input.
The background shows the SA of a test input at each point of the plane, if the predicted class is A.

Likelihood-based SA (LSA) is `-log(density)` of the AT, in a Kernel Density Estimation (KDE) of the training ATs of the predicted class.
Distance-based SA (DSA) is `dist_a / dist_b`.
`dist_a` is the distance to `x_a`, the nearest training AT of the predicted class.
`dist_b` is the distance from `x_a` to the nearest training AT of a different class.

```
test input      predicted      LSA    DSA
typical                 A     1.41   0.01
boundary                A     5.56   1.04
far away                A    52.65   0.80
misclassified           A    44.39   5.95
```

The *far away* input is far from all the classes: LSA is high, but DSA is less than 1.
The *misclassified* input is in the middle of class B, but the predicted class is A.
For this input, LSA is high, and DSA is much more than 1.
As a result, LSA measures how *rare* an input is for its predicted class.
DSA measures how *near* an input is to the boundary between its predicted class and the other classes.

### Conclusion for the 2D example

SA can:

1. Guide the selection of individual inputs: SA gives each input its own score. Every test input has one LSA value and one DSA value, so you can rank new inputs and pick the most surprising ones.
2. Correlate with the likelihood of revealing a problem: the example contains two inputs that are both "very different", and only one of them is a mistake:
- "Far away" is unlike any training input of class A, but it isn't closer to any other class. Nothing suggests the prediction A is wrong.
- "Misclassified" is just as different from class A, and it sits among class B. The prediction A is very likely wrong.

LSA ("how rare is this for class A?") scores both high, about 52 and 44. 
DSA ("is this closer to another class than class A normally is?") scores only the second one high: 0.80 vs 5.95. 
So DSA's notion of "different" correlates better with mistakes.

## SA on MNIST

Run these commands:

```bash
python main.py --sa lsa
python main.py --sa dsa
```

The script computes the ATs of the layer `fc1` for the training set and for the test set, and the SA of each test input.
With LSA, the script takes approximately 50 seconds.
With DSA, the script takes approximately 3.5 minutes, because DSA compares each test AT with all the training ATs.
SA gives a value to each test input, so it can guide the selection of individual test inputs.

The plot `sa_low_and_high_<sa>_fc1.png` shows the 10 test inputs with the lowest SA and the 10 test inputs with the highest SA.
The inputs with the highest SA are digits with an unusual shape.
Some of them are misclassified: 2 of 10 for LSA, and 6 of 10 for DSA.
LSA gives a high value to all unusual inputs, also when the prediction is correct (as for the *far away* input in the 2D example).
DSA gives a high value mostly to inputs that are near a different class (as for the *misclassified* input in the 2D example).

The plot `sa_misclassified_<sa>_fc1.png` divides the test inputs into 10 groups of 1,000 inputs, from the lowest to the highest SA.
For each group, it shows the number of misclassified inputs.
The test set has 91 misclassified inputs.
The group with the highest SA contains 76 (LSA) and 88 (DSA) of them.
Each of the other groups contains at most 12 (LSA) and 1 (DSA).
As a result, SA correlates with the likelihood that an input reveals a problem: the more surprising the input, the more probable a misclassification.

## Layer selection

SA uses the ATs of one layer.
To use the layer `conv1`, run this command:

```bash
python main.py --sa dsa --layer conv1
```

Compare the plots `sa_misclassified_dsa_fc1.png` and `sa_misclassified_dsa_conv1.png`.
At the layer `fc1`, the 1,000 test inputs with the highest DSA contain 88 of the 91 misclassified inputs.
At the layer `conv1`, they contain only 35, and the other misclassified inputs are in almost all the other groups.

The layer decides what "surprising" means.
An early layer, for example `conv1`, finds simple patterns such as strokes and edges, which all the digits have.
At `conv1`, the classes are not separated, and SA shows unusual strokes (for example, very thick, very thin, or broken strokes).
The last hidden layer, `fc1`, shows which digit the network sees, and the classes are well separated (as in the 2D example).
At `fc1`, SA shows inputs that are unusual for the predicted digit, and these inputs are more frequently misclassified.

But the deepest layer is not always the best layer (there is no evidence of this).
The recommendation is therefore to treat the layer as a parameter and check it empirically.

## Surprise coverage

SC divides the range `(0, U]` into `n` buckets of equal size.
SC is the fraction of buckets that contain the SA of one or more test inputs.
The demo uses the following values for MNIST: `n = 1000`, `U = 2000` for LSA, and `U = 2` for DSA (from the original paper).
With DSA, the SC of the test set is 51.90%.

SC changes with the test set.
The right-hand side of the plot `sc_dsa_fc1.png` compares different test sets with the same `U` and `n`.
With DSA, 100 random test inputs give an SC of 7.70%, 1,000 give 30.80%, and all the 10,000 test inputs give 51.90%.
1,000 test inputs of only the digit `1` give 20.90%, because they are less diverse than 1,000 random test inputs.
As a result, SC can show that a test set is small or not diverse.

The other two subfigures of the plot show the problems of SC.
The value of SC depends on parameters that are not related to the test inputs:

- For different values of `U`, the SC of the same test set changes from 0.00% to 87.30%.
- For different numbers of buckets `n`, the SC of the same test set changes from 80.00% (10 buckets) to 30.88% (10,000 buckets).

As a result, you can compare two SC values only if they use the same SA, layer, `U` and `n`.

In the original paper, the authors choose `U` manually.
The option `--upper-bound-percentile` computes `U` from data:

```bash
python main.py --sa dsa --upper-bound-percentile 98
```

The script removes 10% of the training inputs before it builds SA.
Then, `U` is the 98th percentile of the SA of these held-out inputs.
`U` is 176.10 for LSA and 0.81 for DSA.
With these values, the SC of the test set is 76.80% for LSA and 87.60% for DSA.
With this option, DSA takes approximately 5.5 minutes, because it also computes the SA of the held-out inputs.

For the value `n`, the behavior is more predictable, but it can be chosen similarly to `U`.
