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


def extract_propositions(content: str) -> list[Proposition]:
    if not re.search(r"\bproof\b", content, re.IGNORECASE):
        return []

    initial_message = f"""The following file is taken from a repository of source code.
If it contains developer documentation, it may contain zero or more
propositions that have something to do with the codebase. They will be written in natural
language, not a programming language. The propositions may be called "properties", "theorems", etc.
They will be in a mathematical style, with a statement and a proof.
Please extract them. Only get propositions that have associated proofs.

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
