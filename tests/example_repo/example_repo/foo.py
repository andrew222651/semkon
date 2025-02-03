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


def big_multiply(n: int) -> int:
    """
    ::: {.theorem #big_multiply}
    This function always returns `7699497396633402`.
    :::
    ::: {.proof}
    It returns `92832938 * 8293929`, which is `7699497396633402`.
    :::
    """
    return 92832938 * 8293929
