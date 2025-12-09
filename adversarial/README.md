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

The script `plot_boundary.py` plots the decision boundary of a pre-trained MNIST model in two and three dimensions. The script considers the boundary in two orthogonal random directions and their linear combination:

```python
dir_x = torch.randn_like(img_flat)
dir_x = dir_x / torch.norm(dir_x)

dir_y = torch.randn_like(img_flat)
# making sure that y is orthogonal to x
dir_y = dir_y - (dir_y @ dir_x) * dir_x
dir_y = dir_y / torch.norm(dir_y)
...

perturbed = img_flat + x * dir_x + y * dir_y
```

where `img_flat` is the image flattend, and `x` and `y` in the last line are values of the noise that is added to the original image `img_flat` in the corresponding directions `dir_x` and `dir_y`.

It then computes the worst-case direction in the x-axis (untargeted Fast Gradient Sign Method attack), while keeping the y-axis unchanged (see function `get_adversarial_direction_untargeted_fgsm`). Finally, it adds the z-axis in both cases, where the z-axis is the confidence of the model in the correct prediction. We know this because we know the label of the image we are sampling. The idea is taken from Nicholas Carlini's video on [youtube](https://www.youtube.com/watch?v=Z7D-jRMJWHI).

## Undefended model

The script below:

```bash
python plot_boundary.py --model-path mnist.pt --index 18
```

selects the image in the test dataset of MNIST with index 18, i.e., an image with label 3 and confidence around 0.7. We are selecting this image, because the model is easier to attack on this image with a small amount of noise in the worst-case direction. The script plots several images:

- `original_image_3.svg`, the original image with label 3;
- `decision_boundary_rand_3_mnist_2d.svg`, the decision boundary of the model in 2d. The colors show the different classes in which the digit can be classified when adding noise that ranges from -30 to +30. The origin, shown with a white star, indicates the original image which is classified as a 3 (color red);
- `misclassified_images_rand_pred_5_x_0.00_y_30.00.svg`, the image resulting from adding a noise of 30 in the y direction to the original image. The image is now classified as a 5 (color brown), but it is no longer recognizable as a 3;
- `decision_boundary_rand_3_mnist_3d.svg`, the 3d boundary plot when adding the third dimension of the confidence of the model in the correct prediction (i.e., in the label 3);
- `decision_boundary_adv_untargeted_3_mnist_2d.svg`, the decision boundary when the x direction is the worst-case direction, computed by the FGSM untargeted attack;
- `misclassified_images_adv_pred_8_x_2.73_y_0.00.svg`, the original image when adding noise of value 2.73 in the worst-case x direction (this is the value at the boundary between class 3, the original, and class 8). Here, we barely see the noise but the model misclassifies this image as an 8.
- `decision_boundary_adv_untargeted_3_mnist_3d.svg`, the boundary when adding the model confidence on the z-axis, keeping x as the worst case direction and y unchanged.

The seed is always `0` for reproducibility, and it is set at the beginning of the script.

## Adversarial training

The script below loads the model trained by mixing original training data of MNIST with adversarial data computed using the FGSM attack with maximum epsilon value of 0.3 (i.e., `mnist_adv_training.pt`).

```bash
python plot_boundary.py --model-path mnist_adv_training.pt --index 18
```

The script plots, in addition to `original_image_3.svg`:

- `decision_boundary_rand_3_mnist_adv_training_2d.svg`, the model is stronger in random directions (the map is red everywhere, i.e., all the images, despite the high level of noise, are classified correctly as 3);
- `decision_boundary_rand_3_mnist_adv_training_3d.svg`;
- `decision_boundary_adv_untargeted_3_mnist_adv_training_2d.svg`;
- `decision_boundary_adv_untargeted_3_mnist_adv_training_3d.svg`;
- `misclassified_images_adv_pred_8_x_6.97_y_0.00.svg`, now we have to add more noise (i.e., 6.97 vs 2.73) in the x direction to make the model misclassify it as an 8, i.e., the model is more robust against adversarial attacks.

## Input pertubation (jpeg compression)

The script below loads the original model and, for each image, it applies jpeg compression with quality 95 before feeding the image to the network.

```bash
python plot_boundary.py --model-path mnist.pt --index 18 --jpeg-compression
```

The script plots, in addition to `original_image_3.svg`:

- `decision_boundary_rand_3_mnist_jpeg_defense_2d.svg`, the model is worse in random directions than the original model, the brown region is bigger;
- `decision_boundary_rand_3_mnist_jpeg_defense_3d.svg`;
- `misclassified_images_rand_pred_5_x_0.00_y_30.00.svg`;
- `decision_boundary_adv_untargeted_3_mnist_jpeg_defense_2d.svg`;
- `decision_boundary_adv_untargeted_3_mnist_jpeg_defense_3d.svg`;
- `misclassified_images_adv_pred_8_x_3.33_y_0.00.svg`, now we have to add *slightly* more noise (i.e., 3.33 vs 2.73) in the x direction to make the model misclassify it as an 8, i.e., the model is *slightly* more robust against adversarial attacks.

## Input pertubation (thermometer encoding)

The script loads the model trained using thermometer encoding where each pixel value is encoded using 16 values. The encoding breaks the linearity and the possibility for the gradients to flow through the encoder. This makes computing FGSM impossible; to make the comparison fair with the other defense techniques, we bypass the encoder when computing gradients to make the attack possible.

```bash
python plot_boundary.py --model-path mnist_thermometer_levels_16.pt --index 18 --thermometer-encoding --num-levels 16
```

The script plots, in addition to `original_image_3.svg`:

- `decision_boundary_rand_3_mnist_thermometer_levels_16_thermometer_encoding_2d.svg`, the model is stronger in random directions (the map is red everywhere, i.e., all the images, despite the high level of noise, are classified correctly as 3);
- `decision_boundary_rand_3_mnist_thermometer_levels_16_thermometer_encoding_3d.svg`, the decision boundary is not smooth, making the gradient *more difficult* to compute;
- `decision_boundary_adv_untargeted_3_mnist_thermometer_levels_16_thermometer_encoding_2d.svg`;
- `decision_boundary_adv_untargeted_3_mnist_thermometer_levels_16_thermometer_encoding_3d.svg`;
- `misclassified_images_adv_pred_8_x_6.97_y_0.00.svg`, now we have to add more noise (i.e., 6.97 vs 2.73) in the x direction to make the model misclassify it as an 8, i.e., the model is more robust against adversarial attacks. This is similar to the noise we have to add when the model is defended with adversarial training.