from typing import List


def max_product(lst: List[int]) -> int:
    """
    Find the maximum product of any two integers in a list.
    Args:
        lst (List[int]): A list of integers.
    Returns:
        int: The maximum product of any two integers in the list.
    Raises:
        ValueError: If the list has fewer than two elements.
    """
    if len(lst) < 2:
        raise ValueError("List should have more than two elements")
    n1, n2 = sorted(lst, reverse=True)[:2]
    return n1 * n2
