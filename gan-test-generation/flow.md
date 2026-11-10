1. show the diagram in README.md (method). The generator gets a digit y and a latent vector z. 
The search adds p to the activations of the hidden layers, and it uses the gradient of the loss of the classifier. 
The weights of the generator and of the classifier do not change.

2. show gan_samples.png (generator). The GAN makes realistic digits of each class: the classifier (98.11% on the MNIST test set) labels 97.71% of the generated images with the digit of the generator. Compare this with vae_samples.png, produced by a VAE that is not conditional.

3. show tests_early.png (label preservation). The 9 is still a 9, the 4 is not a 4 anymore.

4. show tests_late.png (early vs late layers). Early layers change the shape of the digit (the 5 becomes similar to a 7, the 0 gets a stroke inside). 
Late layers change only local features (spots, parts of the strokes), and they need larger perturbations: the search reaches the target for only 4 of 10 seeds (norms 11.8 to 17.8), against 10 of 10 with the early layers (median norm 7.4). 
The layers decide the type of failure that the search can find.

5. show the fault (biased training data). In the training set of the faulty classifier, the bold 0s have the label 1.
Only the hold-out set of the bold 0s shows the fault clearly (18.67% against 100.00%); on all the test images the accuracy is 94.00% against 98.11%.

6. show results/fault/tests_clean_0to1_early.png and tests_faulty_0to1_early.png (fault detection). Same seeds (the lightest 0s), same search. With the clean classifier, the 0s get thinner and change shape to become a 1. With the faulty classifier, the tests are still 0s, and some are clearly bolder (7 of 10 tests are bolder): the tests show the spurious feature that the classifier uses. The search changes a feature of the image (the boldness), not single pixels; this is not possible with a search in the pixel space.

7. Excercise: 
