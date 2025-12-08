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

# Fast Gradient Sign Method Attack

First show how the attacks work (both untargeted and targeted), lines 54--62 are commented by default but they show what the gradient and the sign mean. The show two examples:

## Images that are hard to attack

```bash
python fgsm_attack.py --model-path mnist.pt --epsilon 0.01
```

The code selects the first image that is correctly classified by default, which is a 7 with confidence 1. Executing the code will plot:

- `original_image_7.png`
- `adversarial_image_7_untargeted.png`

The attack is an *untargeted attack* and has no effect in this case, epsilon is too small and the model is very confident about the prediction. Here `epsilon=0.25` is needed for the prediction to shift to 2.

```bash
python fgsm_attack.py --model-path mnist.pt --epsilon 0.25
```

Inspecting `adversarial_image_7_untargeted.png` will show an image where there is a high level of noise. However, a *targeted attack* is more effective than untargeted at the same noise level:

```bash
python fgsm_attack.py --model-path mnist.pt --epsilon 0.21 # this will give that the confidence on the correct class (i.e., 7) is 0.66
python fgsm_attack.py --model-path mnist.pt --epsilon 0.21 --target-class 2 # this will give that the confidence on the correct class (i.e., 7) is 0.32; the prediction actually shifts to 2
```

The second command plots `adversarial_image_7_to_2.png`.

## Images that are easy to attack

We now want to select an image in which the model is less confident; maybe it is easier to attack with a lower level of noise.

```bash
python fgsm_attack.py --model-path mnist.pt --epsilon 0.03 --confidence-threshold 0.9
```

The `confidence-threshold` parameter lets us select an image of a 3 with confidence 0.7. The untargeted attack above with `epsilon=0.03` makes the prediction shift from 3 to 8, and the confidence on the correct prediction (i.e., 3) drops to 0.31. The image `adversarial_image_3_untargeted.png` is indistinguishable from `original_image_3.png`.

It is possible to make the network misclassify the image to another class, e.g., class 5 by typing:

```bash
python fgsm_attack.py --model-path mnist.pt --epsilon 0.04 --confidence-threshold 0.9 --target-class 5
```

It requires a bit more noise (i.e., `epsilon=0.04` instead of `epsilon=0.03` above) w.r.t. class 8. The noise is also more visible in this case (inspect `adversarial_image_3_to_5`).


# Plot boundary

TODO: add text

## Undefended model

```bash
python plot_boundary.py --model-path mnist.pt --index 18
```

## Adversarial training

```bash
python plot_boundary.py --model-path mnist_adv_training.pt --index 18
```

## Input pertubation (jpeg compression)

```bash
python plot_boundary.py --model-path mnist.pt --index 18 --jpeg-compression
```

## Input pertubation (thermometer encoding)

```bash
python plot_boundary.py --model-path mnist_thermometer_levels_16.pt --index 18 --thermometer-encoding --num-levels 16
```