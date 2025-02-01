import unittest

from semkon.properties import extract_theorem_ids


class TestTheoremExtraction(unittest.TestCase):
    def test_basic_theorem_proof_pair(self):
        text = """
    ::: {.theorem #basic}
    Simple theorem
    :::
    ::: {.proof}
    Simple proof
    :::
"""
        self.assertEqual(extract_theorem_ids(text), ["basic"])

    def test_multiple_theorem_proof_pairs(self):
        text = """
::: {.theorem #first}
First theorem
:::
::: {.proof}
First proof
:::

::: {.theorem #second}
Second theorem
:::
::: {.proof}
Second proof
:::
"""
        self.assertEqual(extract_theorem_ids(text), ["first", "second"])

    def test_theorem_without_proof_is_ignored(self):
        text = """
::: {.theorem #with_proof}
This one has a proof
:::
::: {.proof}
Here's the proof
:::

::: {.theorem #no_proof}
This one doesn't have a proof
:::
"""
        self.assertEqual(extract_theorem_ids(text), ["with_proof"])

    def test_colons_in_content(self):
        text = """
::: {.theorem #colon_test}
This theorem: has some: colons in it
:::
::: {.proof}
This proof: also has: some colons: in it
:::
"""
        self.assertEqual(extract_theorem_ids(text), ["colon_test"])

    def test_empty_content(self):
        text = """
::: {.theorem #empty}
:::
::: {.proof}
:::
"""
        self.assertEqual(extract_theorem_ids(text), ["empty"])

    def test_no_theorems(self):
        text = "Just some regular text without any theorem blocks"
        self.assertEqual(extract_theorem_ids(text), [])
