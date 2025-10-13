# Setup

Build the docker container:

```bash
docker build --build-arg USERNAME=$USER --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -t stats:latest .
```

Then start the `devcontainer`:

- Download [VSCode](https://code.visualstudio.com/Download) for your platform;
- Install DevContainer Extension;
- In VSCode, use the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P` on macOS) to run the "Dev Containers: Open Folder in Container..." command;
- Select the `neuron-coverage` folder;

Optionally, start the container without `devcontainer` by typing:

```bash
docker run --rm -v $PWD:/home/ -it stats:latest
```

Once within the container, if `devcontainer` is used select the only python intepreter available, i.e., `3.11.13`. Then type:

```bash
python main.py
```

The script computes two quantities, i.e., Mann-Whitney U test (also known as ranksum) and the Vargha-Delaney effect size, for four different pairs of normal distributions, also plotting the boxplots for each pair. 
The Mann-Whitney U test compares two samples, A and B, from two distributions, with the default (or null) hypothesis that such samples come from the same distribution.
For instance, A and B may be accuracies of two different models or coverage values coming from two different search algorithms, and we want to know which one is "better" (higher or lower values depending on the metric) or if they are the same.
If the null hypothesis is true, then the two samples may come from the same distribution (another test is needed to test if we have enough data to say that the two samples come from the same distribution); if it is false, then we can say the two samples come from different distributions, and that there is a statistically significant effect that influences the data (i.e., a better model or a better search algorithm).
The Mann-Whitney U test computes a number called *p-value*, which tells us how likely is the effect on the samples if the null hypothesis were true.
For instance, if we are comparing the accuracies of two model architectures on a certain dataset and we have a p-value of 0.04, then we can say that we'd obtain the same difference or more in 4% of the experiments due to randomness.
In other words, the two samples are quite unlikely if the null hypothesis is true, hence we are reasonably sure that there is a difference between the two samples.
The threshold that is often used in practice to reject the null hypothesis is p-value $< \alpha = 0.05$; this is just a convention, though.

The p-value of Mann-Whitney U test tells us whether there is a significant difference between two samples A and B.
However, it does not tell us the *direction* nor the *magnitude* of the difference.
In other words, two samples A and B might be statistically different but the difference might be small to be of practical value.
To know these two quantities, we have to compute the effect size; in particular, the code computes the Vargha-Delaney effect size, indicated as $\hat{A}_{12}$, which is the probability that a randomly selected value from A is greater than a randomly selected value from list B. 
It returns a float between 0 and 1, where 0.5 indicates no difference between the samples; 1.0 indicates all values in A are greater than those in B, and 0.0 indicates the opposite.

The code has four cases:
- Case 1: null hypothesis rejected, i.e., significance (p-value < 0.05) and A better than B with a large effect size
- Case 2: null hypothesis rejected, i.e., significance (p-value < 0.05) and B better than A with a large effect size
- Case 3: null hypothesis accepted, i.e., not enough evidence to say that A and B are different; in this case the effect size is not computed
- Case 4: null hypothesis rejected, i.e., significance (p-value < 0.05) but the difference between A and B is small (i.e., the effect size is small)

To know more about the field of statistical tests feel free to have a look at the following paper:

*Arcuri, Andrea, and Lionel Briand. "A hitchhiker's guide to statistical tests for assessing randomized algorithms in software engineering." Software Testing, Verification and Reliability 24.3 (2014): 219-250.*



