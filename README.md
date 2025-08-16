Semkon: _semantika kontrolilo_ (semantic checker)

Semkon uses LLMs to check the correctness of proofs written as comments in your
codebase.

Features:
* automatically finds property+proof blocks in your code
  * see [example](tests/example_repo/example_repo/foo.py)
  * there's no required syntax since detection is done by an LLM. for markdown one option is <https://mystmd.org/guide/proofs-and-theorems>
* recognizes stated "axioms"
* configurable file exclusion
* can execute Python code (off by default)

Requires OpenAI API key. Costs can be unpredictable so to be fully safe, use a
project API key with a project budget limit.

You can choose which OpenAI model to use. As of Feb 2025, only o1 is useful in
@andrew222651's tests on real code, and no non-OpenAI models are useful.

Run: `semkon --help`

Install for development: `pip install -e ".[dev]"`


Alternatives:
* AI code review tools may be able to check some proofs
  * [CodeRabbit](https://www.coderabbit.ai/) was able to check some simple
    proofs with no customizations
  * you may be able to explicitly add proof checking as a custom "style rule"
    if necessary

