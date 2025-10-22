def passed(grade: float) -> bool:
    """
    Determine if the given grade is a passing grade.
    A passing grade is defined as a grade of 5.0 or higher.
    Args:
        grade (float): The grade to evaluate (must be between 1.0 and 10.0).
    Returns:
        bool: True if the grade is passing, False otherwise.
    Raises:
        ValueError: If the grade is not between 1.0 and 10.0.
    """

    if grade < 1.0 or grade > 10.0:
        raise ValueError("Grade must be between 1.0 and 10.0")

    return grade >= 5.0
