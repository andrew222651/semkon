from . import bar


def foo_func(n: int) -> int:
    """
    ::: {.theorem #foo_func}
    This function always returns a non-negative number.
    :::
    ::: {.proof}
    The square of any integer is non-negative, and
    bar.bar_func(n) is non-negative for any n.
    So the sum is non-negative.
    :::
    """
    return n**2 + bar.bar_func(n)
