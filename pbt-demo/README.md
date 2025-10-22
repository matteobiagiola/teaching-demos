# Setup

Build the docker container:

```bash
docker build --build-arg USERNAME=$USER --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -t hypothesis:6.142.1 .
```

The image will be around 188MB. Then start the `devcontainer`:

- Download [VSCode](https://code.visualstudio.com/Download) for your platform;
- Install DevContainer Extension;
- In VSCode, use the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P` on macOS) to run the "Dev Containers: Open Folder in Container..." command;
- Select the `vae-interpolation-example` folder.

Optionally, start the container without `devcontainer` by typing:

```bash
docker run -v $PWD:/home/ -it hypothesis:6.142.1
```

Once within the container, if `devcontainer` is used select the only python intepreter available, i.e., `3.11.13`. 

# Run

## First example

```bash
pytest --hypothesis-show-statistics tests/test_passing_grade.py
```

Change `return grade >= 5.0` in `functions/passing_grade` with `return grade > 5.0` to show how a failure is triggered.

## Second example

```bash
pytest --hypothesis-show-statistics tests/test_is_triangle_bad.py
```

The result should be the following:

```
tests/test_is_triangle_bad.py::test_triangle_inequality_property:

  - during generate phase (0.06 seconds):
    - Typical runtimes: < 1ms, of which < 1ms in data generation
    - 100 passing examples, 0 failing examples, 0 invalid examples
    - Events:
      * 80.00%, triangle invalid
      * 20.00%, triangle valid

  - Stopped because settings.max_examples=100
```

Many of the tests are invalid, and it is more likely to exercise one property (i.e., invalidity) then the other (i.e., validity). Ideally, we would like our data generation to be equally distributed, or we would like to test our properties in an equal way. One solution, would be to split validity and invalidity properties.


```bash
pytest --hypothesis-show-statistics tests/test_is_triangle_good.py
```

## Third example

```bash
pytest --hypothesis-show-statistics tests/test_bst.py
```

Then increase the `max_depth` parameter in the `tree_gen` function in `tests/test_bst.py` and run the command above again, to show that Hypothesis times out and returns an error. Basically, the generation is too slow because many of the trees that are generated are not valid binary search trees.

## Fourth example

In this case, reason with students about the property to come up with for the `functions/max_product.py` function:

```python
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
```

The function has a bug, as it does not account for negative numbers. If we provide GitHub Copilot with this prefix:

```python
from hypothesis import given, strategies as st
from functions.max_product import max_product


@given(st.lists(st.integers(), min_size=2))
```

Copilot generates the following test:

```python
from hypothesis import given, strategies as st
from functions.max_product import max_product


@given(st.lists(st.integers(), min_size=2))
def test_max_product(lst: list[int]) -> None:
    result = max_product(lst)
    sorted_lst = sorted(lst, reverse=True)
    expected = sorted_lst[0] * sorted_lst[1]
    assert result == expected
```

Which has two problems: the first problem is that it does not catch the bug, as if we run it with `pytest` we get no failure:

```bash
pytest --hypothesis-show-statistics tests/test_max_product.py
```

The second problem, which is related to the first one, is that the test is too tied to the implementation; and if the implementation is wrong, then the logic of the test is wrong too. We can come up with another property:


```python
from hypothesis import given, strategies as st
from functions.max_product import max_product


@given(st.lists(st.integers(), min_size=2))
def test_max_product(lst: list[int]) -> None:
    result = max_product(lst=lst)
    assert result >= lst[0] * lst[1]
```

Here, we are saying that the result of running the `max_product` function on a given list should be `>=` any two values of the list, e.g., the first two. If we run this now:

```bash
pytest --hypothesis-show-statistics tests/test_max_product.py
```

We get the following output:

```
lst = [-1, -1, 0]

    @given(st.lists(st.integers(), min_size=2))
    def test_max_product(lst: list[int]) -> None:
        result = max_product(lst=lst)
>       assert result >= lst[0] * lst[1]
E       assert 0 >= (-1 * -1)
E       Falsifying example: test_max_product(
E           lst=[-1, -1, 0],
E       )

tests/test_max_product.py:8: AssertionError
```

Hypothesis found that our implementation is wrong, because if we provide the input `lst=[-1, -1, 0]`, our implementation returns `0` instead of `1`, which is the highest product between two numbers that we can find in the list i.e., `-1` and `-1`.