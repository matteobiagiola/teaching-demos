from hypothesis import event, given, strategies as st
from functions.triangle import is_triangle


# generate all integers, including negatives and zero
@given(
    a=st.integers(),
    b=st.integers(),
    c=st.integers(),
)
def test_triangle_inequality_property(a, b, c) -> None:
    result = is_triangle(a=a, b=b, c=c)

    if result:
        # Must satisfy triangle inequality
        assert a < b + c
        assert b < a + c
        assert c < a + b
        event("triangle valid")
    else:
        # Must violate at least one inequality or have a nonpositive side
        assert a >= b + c or b >= a + c or c >= a + b or a <= 0 or b <= 0 or c <= 0
        event("triangle invalid")
