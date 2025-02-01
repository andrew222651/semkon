import re


def extract_theorem_ids(text: str) -> list[str]:
    """Extract theorem IDs from theorem-proof block pairs.

    Looks for patterns like:
    ::: {.theorem #id}
    ... (content can contain colons, just not three in a row)
    :::
    ::: {.proof}
    ... (content can contain colons, just not three in a row)
    :::

    Returns a list of theorem IDs that are followed by proof blocks.
    """
    # Pattern matches a complete theorem block followed by a complete proof block
    pattern = (
        r"::: \{\.theorem #([^\s}]+)\}"  # theorem start with ID capture
        r"(?:(?!:::).)*"  # content until next :::
        r":::"  # theorem end
        r"\s*"  # whitespace between blocks
        r"::: \{\.proof\}"  # proof start
        r"(?:(?!:::).)*"  # content until next :::
        r":::"  # proof end
    )
    return re.findall(pattern, text, re.DOTALL)
