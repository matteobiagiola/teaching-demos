from typing import Callable, Optional
from hypothesis import given, strategies as st

from functions.bst import Node, insert, is_bst


# @composite provides a callable draw as the first parameter to the decorated function,
# which can be used to dynamically draw a value from any strategy.
@st.composite
def tree_gen(draw: Callable, depth: int = 0, max_depth: int = 4) -> Optional[Node]:
    if depth >= max_depth or draw(st.booleans()):
        return None
    value = draw(st.integers(min_value=-50, max_value=50))
    left = draw(tree_gen(depth=depth + 1, max_depth=max_depth))
    right = draw(tree_gen(depth=depth + 1, max_depth=max_depth))
    return Node(value=value, left=left, right=right)


valid_bst_trees = tree_gen().filter(is_bst)


@given(x=st.integers(), t=valid_bst_trees)
def test_insert_maintains_bst(x: int, t: Node) -> None:
    new_tree = insert(node=t, value=x)
    assert is_bst(new_tree)
