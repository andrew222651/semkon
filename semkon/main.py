#!/usr/bin/env python3

import json
import sys
from pathlib import Path
from typing import Annotated, Any, Literal, Sequence

import chromadb
import tiktoken
import typer
from chromadb.api import ClientAPI
from loguru import logger
from openai import LengthFinishReasonError
from pydantic import BaseModel

from .clients import openai_client
from .code_quoting import format_file
from .file_filters import get_rel_paths
from .properties import extract_propositions
from .python_deps import get_deps_rec
from .safe_sympy import execute

MAX_FILES_REQUESTED = 5

# o1-preview and o1 gave good results. everything else was bad (deepseek,
# claude, gpt-4o, gemini, o3-mini).
MODEL: dict[str, Any] = {
    "model": "o1",
    "reasoning_effort": "medium",
}
MAX_CONTEXT_LENGTH = 200_000
MAX_OUTPUT_LENGTH = 100_000

CORRECTNESS_BLURB = """By "correct", we mean very high confidence that each step of the proof is valid,
the proof does in fact prove the proposition, and that the proof is supported by
what the code does. Mark the proof as "incorrect" if you understand it and the
code but the proof is wrong. Use "unknown" if e.g. you don't 100% know how an
external library works, or the proof needs more detail. Skeptically and
rigorously check every claim with references to the code. If the proof
references an explicitly-stated axiom (or "assumption", etc) found in the
codebase, you can assume that the axiom is true. If the proof references another
proposition from the codebase, you can assume that the other proposition is true
if the codebase provides a proof for it (you don't have to check that proof) or
if it's well-known or if a reference to the literature is provided. However, if
the proof we're checking is part of a cycle of dependencies where the proof of
one proposition relies on the truth of the next, report this proof as
"incorrect"."""


# https://github.com/openai/tiktoken/issues/337#issuecomment-2392465999
enc = tiktoken.encoding_for_model("gpt-4o")

logger.remove()
logger.add(sink=sys.stderr, level="DEBUG")


class PropertyLocation(BaseModel):
    rel_path: Path
    line_num: int


class CorrectnessExplanation(BaseModel):
    correctness: Literal["correct", "incorrect", "unknown"]
    explanation: str


class FilesRequested(BaseModel):
    files_requested: list[str]


class ExecutePython(BaseModel):
    code: str


SafeData = FilesRequested | CorrectnessExplanation


class FullFilesExcludedResponse(BaseModel):
    data: SafeData


class FullFilesExcludedExecutePythonResponse(BaseModel):
    data: SafeData | ExecutePython


class FullFilesIncludedResponse(BaseModel):
    data: CorrectnessExplanation


class Failure(BaseModel):
    msg: str


class ProofCheckResult(BaseModel):
    tokens_used: int
    property_location: PropertyLocation
    correctness_explanation: CorrectnessExplanation | Failure


class Linter:
    def __init__(
        self,
        directory: Path,
        max_messages: int,
        max_files: int,
        filter_paths: list[str],
        property_filter: str | None,
        always_exclude_full_files: bool,
        execute_python: bool,
    ):
        self._directory: Path = directory
        self._rel_paths: list[Path] = get_rel_paths(
            directory, filter_paths=filter_paths
        )
        if len(self._rel_paths) > max_files:
            raise ValueError(f"Too many files: {len(self._rel_paths)}")
        for p in self._rel_paths:
            logger.debug(f"Found {p}")

        self._chroma_client: ClientAPI = chromadb.Client()
        self._collection: chromadb.Collection = (
            self._chroma_client.create_collection("codebase")
        )
        documents = [
            (directory / rel_path).read_text() for rel_path in self._rel_paths
        ]
        ids = [str(rel_path) for rel_path in self._rel_paths]
        self._collection.add(documents=documents, ids=ids)

        self._property_locations = []
        for p in self._rel_paths:
            props = extract_propositions(
                (directory / p).read_text(), filter=property_filter, rel_path=p
            )
            for prop in props:
                self._property_locations.append(
                    PropertyLocation(rel_path=p, line_num=prop.line_num)
                )
                logger.debug(f"Found property @ {p}:{prop.line_num}")
                logger.debug(prop.statement)

        self._max_messages = max_messages

        # only an approximate calculation
        self._exclude_full_files = always_exclude_full_files or (
            sum(len(enc.encode(doc)) for doc in documents)
            >= MAX_CONTEXT_LENGTH - MAX_OUTPUT_LENGTH
        )

        python_deps = get_deps_rec(
            self._directory, self._directory, self._rel_paths
        )
        if python_deps:
            self._python_deps_text = f"""Here is the dependency graph of the codebase:
{json.dumps(python_deps, indent=2)}

"""
        else:
            self._python_deps_text = ""

        self._execute_python = execute_python

    def _get_response_options(self) -> str:
        options = []
        request_files = f"""Request files

In this response, you may request to see additional files from the codebase in
order to ultimately determine whether the proof is correct. They will be
provided to you in the next message. You will have the opportunity to request
further files if needed, and we will repeat this process until you are ready to
make a final determination. You can request up to {MAX_FILES_REQUESTED} files
at a time."""
        options.append(request_files)

        execute_python = f"""Execute Python code

You may respond with Python 3 code, in which case it will be executed and the
results will be provided in the next message. The code should be literal, bare
Python code, not in a markdown code block or with any other formatting. You can
assume that `import math` and `import sympy` have already been run, so you have
the ability to do calculations and use symbolic methods. You cannot import any
other modules. Do not perform I/O. The object you wish to see should be assigned
to the variable `result`. The next message will provide you with the output of
calling `repr` on the object. If illegal code is provided, or the code raises an
exception, or times out, the result will be the exception object. Use this
response option if you want to run a computation that would help us ultimately
determine whether the proof is correct."""
        if self._execute_python:
            options.append(execute_python)

        correctness_verdict = f"""Correctness verdict

In this response, state whether the proof is correct.
{CORRECTNESS_BLURB}
(Use this response only if none of the other response types are currently
necessary to make a determination.)"""
        options.append(correctness_verdict)

        numbered_options = [f"{i+1}. {opt}" for i, opt in enumerate(options)]
        return "\n\n".join(numbered_options)

    def _build_initial_message(
        self, property_location: PropertyLocation
    ) -> str:
        if not self._exclude_full_files:
            return f"""The following is a listing of all files in a codebase:
{"\n".join(str(p) for p in self._rel_paths)}

At the end of this message is a listing of all file contents.

The file {property_location.rel_path} contains one or more propositions
about the codebase. The proposition we are interested in is on line
{property_location.line_num}, and is followed by a proof.

In your response, state whether the proof (not the proposition) is correct.
{CORRECTNESS_BLURB}
        

File contents:
{"\n".join(format_file((self._directory / p).read_text(), rel_path=p) for p in self._rel_paths)}
            """
        else:
            return f"""The following is a listing of all files in a codebase:
{"\n".join(str(p) for p in self._rel_paths)}

{self._python_deps_text}

At the end of this message is a listing of the contents of {property_location.rel_path}.
This file contains one or more propositions
about the codebase. The proposition we are interested in is on line
{property_location.line_num}, and is followed by a proof.

The goal of this conversation is to determine whether the proof 
(not the proposition) is correct.

Your responses in this conversation can be one of the following.

{self._get_response_options()}

File contents:
{format_file((self._directory / property_location.rel_path).read_text(), rel_path=property_location.rel_path)}
            """

    def _build_subsequent_message(self, data: str) -> str:
        return f"""The requested information is given below.

{data}
        """

    def check_proofs(self) -> list[ProofCheckResult]:
        ret = []
        for property_location in self._property_locations:
            pcr = self.check_proof(property_location)
            ret.append(pcr)

        return ret

    def check_proof(
        self, property_location: PropertyLocation
    ) -> ProofCheckResult:
        files_shown: set[Path] = set()
        all_files = set(self._rel_paths)

        tokens_used = 0

        initial_message = self._build_initial_message(property_location)
        # logger.debug(initial_message)
        messages = [
            {
                "role": "user",
                "content": initial_message,
            }
        ]

        response_format: type[BaseModel]
        if self._exclude_full_files:
            if self._execute_python:
                response_format = FullFilesExcludedExecutePythonResponse
            else:
                response_format = FullFilesExcludedResponse
        else:
            response_format = FullFilesIncludedResponse
        for _ in range(self._max_messages):
            try:
                resp = openai_client.beta.chat.completions.parse(
                    **MODEL,
                    messages=messages,  # type: ignore
                    response_format=response_format,
                )
            except LengthFinishReasonError as lfre:
                usage = lfre.completion.usage
                if usage is None:
                    raise RuntimeError("No token usage available")
                return ProofCheckResult(
                    property_location=property_location,
                    correctness_explanation=Failure(
                        msg="Model token limit reached"
                    ),
                    tokens_used=usage.total_tokens + tokens_used,
                )

            logger.debug(f"Token usage: {resp.usage}")
            if resp.usage is None:
                raise RuntimeError("No token usage available")
            tokens_used += resp.usage.total_tokens

            resp_message = resp.choices[0].message
            if resp_message.content is None:
                raise RuntimeError("No response from LLM")
            logger.debug(resp_message.content)
            messages.append(
                {"role": "assistant", "content": resp_message.content}
            )
            if resp_message.parsed is None:
                raise RuntimeError("No response from LLM")
            response_data = resp_message.parsed.data

            if isinstance(response_data, CorrectnessExplanation):
                return ProofCheckResult(
                    property_location=property_location,
                    correctness_explanation=response_data,
                    tokens_used=tokens_used,
                )
            else:
                if isinstance(response_data, FilesRequested):
                    files_requested = sorted(
                        (
                            all_files
                            & {Path(p) for p in response_data.files_requested}
                        )
                        - files_shown
                    )[:MAX_FILES_REQUESTED]
                    files_shown.update(files_requested)

                    if not files_requested:
                        raise RuntimeError("Invalid response")

                    subsequent_message = self._build_subsequent_message(
                        "\n".join(
                            format_file(
                                (self._directory / p).read_text(), rel_path=p
                            )
                            for p in files_requested
                        )
                    )
                elif isinstance(response_data, ExecutePython):
                    result = execute(response_data.code)
                    subsequent_message = self._build_subsequent_message(result)
                else:
                    assert False
                messages.append(
                    {
                        "role": "user",
                        "content": subsequent_message,
                    }
                )
        else:
            return ProofCheckResult(
                property_location=property_location,
                correctness_explanation=Failure(
                    msg="No result after max responses"
                ),
                tokens_used=tokens_used,
            )


def main(
    directory: Annotated[
        Path,
        typer.Argument(
            help="Repo to analyze",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path("."),
    max_files: Annotated[
        int, typer.Option(help="Max number of files in the codebase")
    ] = 1_000,
    max_responses_per_property: Annotated[
        int,
        typer.Option(
            help="Max number of responses in the LLM conversation about a property"
        ),
    ] = 10,
    filter_path: Annotated[
        list[str] | None,
        typer.Option(
            help="Path to exclude from the analysis in .gitignore format. Repeat as needed.",
        ),
    ] = None,
    property_filter: Annotated[
        str | None,
        typer.Option(
            help="Natural language instructions on which properties to check."
        ),
    ] = None,
    always_exclude_full_files: Annotated[
        bool,
        typer.Option(hidden=True),
    ] = False,
    execute_python: Annotated[
        bool,
        typer.Option(help="Allow running Python code generated by the LLM."),
    ] = False,
):
    linter = Linter(
        directory=directory,
        max_files=max_files,
        max_messages=max_responses_per_property,
        filter_paths=filter_path or [],
        property_filter=property_filter,
        always_exclude_full_files=always_exclude_full_files,
        execute_python=execute_python,
    )
    results = linter.check_proofs()
    print(
        json.dumps(
            [result.model_dump(mode="json") for result in results],
            indent=2,
        )
    )
    if any(
        isinstance(result.correctness_explanation, Failure)
        or result.correctness_explanation.correctness != "correct"
        for result in results
    ):
        exit(1)


def cli() -> int:
    typer.run(main)
    return 0


if __name__ == "__main__":
    exit(cli())
