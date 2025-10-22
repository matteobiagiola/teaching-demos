from hypothesis import given, strategies as st
from functions.max_product import max_product


@given(st.lists(st.integers(), min_size=2))
def test_max_product(lst: list[int]) -> None:
    result = max_product(lst=lst)
    assert result >= lst[0] * lst[1]
