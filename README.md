Semkon: _semantika kontrolilo_ (semantic checker)

Semkon uses LLMs to check the correctness of proofs written as comments in your
codebase.

Features:
* automatically finds property+proof blocks in your code
  * see [example](tests/example_repo/example_repo/foo.py)
* recognizes stated "axioms"
* configurable file exclusion, token limits

Requires OpenAI API key.

Run: `semkon --help`

Install for development: `pip install -e ".[dev]"`


Alternatives:
* AI code review tools may be able to check some proofs
  * [CodeRabbit](https://www.coderabbit.ai/) was able to check some simple
    proofs with no customizations
  * you may be able to explicitly add proof checking as a custom "style rule"
    if necessary

