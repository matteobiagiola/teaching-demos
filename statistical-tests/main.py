import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple
from scipy.stats import ranksums

def a12_unpaired(a: List[float], b: List[float]) -> Tuple[float, str]:

    """
    Compute the A12 effect size between two lists.
    A12 is the probability that a randomly selected value from list `a` is greater than
    a randomly selected value from list `b`. If the values are equal, it counts as half.
    Returns a float between 0 and 1, where 0.5 indicates no difference between the lists.
    1.0 indicates all values in `a` are greater than those in `b`, and 0.0 indicates the opposite.
    
    Args:
        a (List): First list of values.
        b (List): Second list of values.
        
    Returns:
        Tuple[float, str]: A12 effect size and its magnitude ("negligible", "small", "medium", "large").
    """

    more = 0.0
    same = 0.0
    assert len(a) > 0, "A12 is undefined for empty lists"
    assert len(b) > 0, "A12 is undefined for empty lists"
    
    for x in a:

        for y in b:

            if x == y: 
                same += 1
            elif x > y: 
                more += 1
    
    a12 = (more + 0.5 * same) / (len(a) * len(b))

    if 2 * abs(a12 - 0.5) < 0.147:
        return a12, "negligible"
    elif 2 * abs(a12 - 0.5) < 0.334:
        return a12, "small"
    elif 2 * abs(a12 - 0.5) < 0.474:
        return a12, "medium"
    else:
        return a12, "large"



def compute_p_value(a: List[float], b: List[float]) -> float:
    """
    Compute the p-value using the Wilcoxon rank-sum test (Mann-Whitney U test).
    
    Args:
        a (List): First list of values.
        b (List): Second list of values.

    Returns:
        float: The p-value from the rank-sum test.
    """
    return ranksums(a, b).pvalue


def analyze_and_plot_boxplot(a: List[float], b: List[float], labels: List[str], filename: str) -> None:
    """
    Analyze two lists of values, compute the p-value and effect size, and plot a boxplot.
    
    Args:
        data (List[List]): List of lists containing the data to plot.
        labels (List[str]): Labels for each box in the boxplot.
        title (str): Title of the plot.
        ylabel (str): Label for the y-axis.
        filename (str): Filename to save the plot.

    Returns:
        None
    """

    p_value = compute_p_value(a=a, b=b)
    print(f"P-value: {p_value:.5f}")

    if p_value < 0.05:
        print("The difference is statistically significant (p < 0.05).")
        effect_size, magnitude = a12_unpaired(a=a, b=b)
        print(f"A12 effect size: {effect_size:.5f}, magnitude: {magnitude}")
    else:
        print("Not enough evidence to suggest a statistically significant difference (p >= 0.05).")

    plt.boxplot([a, b], labels=labels)
    plt.title("Boxplot of A and B")
    plt.ylabel("Values")
    plt.savefig(filename, dpi=300)
    plt.close()


def main():

    seed = 0
    print(f"Using seed: {seed}")
    np.random.seed(seed)

    # Case 1: Significance and A better than B
    print("\nCase 1: Significance and A better than B")

    a = np.random.normal(loc=1, scale=1, size=40)
    b = np.random.normal(loc=0, scale=1, size=40)

    analyze_and_plot_boxplot(a=a, b=b, labels=["A", "B"], filename="boxplot_case_1.png")

    # Case 2: Significance B better than A
    print("\nCase 2: Significance B better than A")

    a = np.random.normal(loc=0, scale=1, size=35)
    b = np.random.normal(loc=1, scale=1, size=35)

    analyze_and_plot_boxplot(a=a, b=b, labels=["A", "B"], filename="boxplot_case_2.png")

    # Case 3: No statistical significance (p > 0.05)
    print("\nCase 3: No statistical significance")

    a = np.random.normal(loc=0, scale=1, size=50)
    b = np.random.normal(loc=0.2, scale=1, size=50)

    analyze_and_plot_boxplot(a=a, b=b, labels=["A", "B"], filename="boxplot_case_3.png")

    # Case 4: Statistically significant but negligible effect size (p < 0.05, A near 0.5)
    print("\nCase 4: Statistically significant but negligible effect size (p < 0.05, A near 0.5)")

    a = np.random.normal(loc=0, scale=0.1, size=1000)
    b = a + np.random.normal(loc=0.01, scale=0.01, size=1000)  # slight shift

    analyze_and_plot_boxplot(a=a, b=b, labels=["A", "B"], filename="boxplot_case_4.png")


if __name__ == "__main__":
    main()
