from typing import Tuple
from hypothesis import given, strategies as st
from functions.triangle import is_triangle

# to prevent the generation too many invalid triangles, we limit side lengths to positive integers
sides = st.integers(min_value=0, max_value=1000)

invalid_triangles = st.builds(lambda a, b, c: (a, b, c), sides, sides, sides).filter(
    lambda k: k[0] >= k[1] + k[2] or k[1] >= k[0] + k[2] or k[2] >= k[0] + k[1]
)

valid_triangles = st.builds(lambda a, b, c: (a, b, c), sides, sides, sides).filter(
    lambda k: k[0] < k[1] + k[2] and k[1] < k[0] + k[2] and k[2] < k[0] + k[1]
)


@given(invalid_triangles)
def test_triangle_invalid(k: Tuple[int, int, int]) -> None:
    assert not is_triangle(a=k[0], b=k[1], c=k[2])


@given(valid_triangles)
def test_triangle_valid(k: Tuple[int, int, int]) -> None:
    assert is_triangle(a=k[0], b=k[1], c=k[2])
