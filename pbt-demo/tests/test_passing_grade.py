from hypothesis import settings, Verbosity
from hypothesis import given, strategies as st
from functions.passing_grade import passed
import pytest


@given(st.floats(min_value=1.0, max_value=5.0, exclude_max=True))
@settings(verbosity=Verbosity.verbose)
def test_fail_valid_grade(grade: float) -> None:
    result = passed(grade=grade)
    assert isinstance(result, bool)
    assert not result


# @given(st.floats(min_value=5.0, max_value=10.0, exclude_max=False))
# @settings(verbosity=Verbosity.verbose)
# def test_pass_valid_grade(grade: float) -> None:
#     result = passed(grade=grade)
#     assert isinstance(result, bool)
#     assert result


# @given(st.one_of(st.floats(max_value=0.99), st.floats(min_value=10.01)))
# @settings(verbosity=Verbosity.verbose)
# def test_passed_invalid_grade(grade: float) -> None:
#     with pytest.raises(ValueError):
#         passed(grade=grade)
