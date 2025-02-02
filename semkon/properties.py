import re

from loguru import logger
from pydantic import BaseModel

from .clients import openai_client
from .code_quoting import format_file


class Proposition(BaseModel):
    line_num: int
    statement: str
    proof: str


class PropositionsResponse(BaseModel):
    data: list[Proposition]


def extract_propositions(
    content: str, filter: str | None = None
) -> list[Proposition]:
    if not re.search(r"\bproof\b", content, re.IGNORECASE):
        return []

    if filter is not None and filter.strip():
        filter_text = f"* {filter}"
    else:
        filter_text = ""

    initial_message = f"""The following file is taken from a repository of source code.
It may (or may not) contain one or more formal propositions that have something to do with the codebase.
These would be written as developer documentation. They may be called "properties", "theorems", etc.
Extract all such propositions that satisfy the following criteria:
* They are written in natural language, not a programming language.
* They are in a mathematical style, like a computer scientist would write.
* They are explicitly labeled as a "property" or "theorem" or similar,
  and have an associated explicitly-labeled proof.
{filter_text}

For example, there may be propositions about running times,
correctness, or auxiliary facts.

{format_file(content)}"""
    logger.debug(initial_message)

    resp = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": initial_message}],
        response_format=PropositionsResponse,
    )
    if resp.choices[0].message.parsed is None:
        return []
    return resp.choices[0].message.parsed.data
