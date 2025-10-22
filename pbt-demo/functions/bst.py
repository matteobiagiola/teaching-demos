from typing import Optional


class Node:
    def __init__(
        self, value: int, left: Optional["Node"] = None, right: Optional["Node"] = None
    ):
        self.value = value
        self.left = left
        self.right = right


def insert(node: Optional[Node], value: int) -> Node:
    """
    Insert a value into the binary search tree.
    Args:
        node (Optional[Node]): The root of the binary search tree or subtree.
        value (int): The value to insert.
    Returns:
        Node: The root of the binary search tree after insertion.
    """
    if node is None:
        return Node(value=value)
    if value < node.value:
        node.left = insert(node=node.left, value=value)
    elif value > node.value:
        node.right = insert(node=node.right, value=value)
    return node


def is_bst(
    node: Optional[Node], lo: float = float("-inf"), hi: float = float("inf")
) -> bool:
    """
    Check if a binary tree is a binary search tree.
    Args:
        node (Optional[Node]): The root of the binary tree or subtree.
        lo (float): The lower bound for node values.
        hi (float): The upper bound for node values.
    Returns:
        bool: True if the tree is a binary search tree, False otherwise.
    """
    if node is None:
        return True
    if not (lo < node.value < hi):
        return False
    return is_bst(node=node.left, lo=lo, hi=node.value) and is_bst(
        node=node.right, lo=node.value, hi=hi
    )
