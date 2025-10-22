def is_triangle(a: int, b: int, c: int) -> bool:
    """
    Determine if three sides can form a triangle.
    Args:
        a (int): Length of side a.
        b (int): Length of side b.
        c (int): Length of side c.
    Returns:
        bool: True if the sides can form a triangle, False otherwise.
    """

    has_bad_side = a >= (b + c) or b >= (a + c) or c >= (a + b)
    return not has_bad_side
