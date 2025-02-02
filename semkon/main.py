#!/usr/bin/env python3

import json
import sys
from pathlib import Path
from typing import Annotated, Any, Literal, Sequence

import chromadb
import typer
from chromadb.api import ClientAPI
from loguru import logger
from pydantic import BaseModel

from .clients import openai_client
from .code_quoting import format_file
from .file_filters import get_rel_paths
from .properties import extract_propositions
from .python_deps import get_deps_rec

# o1-preview and o1 gave good results. everything else was bad (deepseek,
# claude, gpt-4o, gemini, o3-mini).
MODEL: dict[str, Any] = {
    "model": "o1",
    "reasoning_effort": "medium",
}

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


class FullFilesExcludedResponse(BaseModel):
    data: FilesRequested | CorrectnessExplanation


class FullFilesIncludedResponse(BaseModel):
    data: CorrectnessExplanation


class ProofCheckResult(BaseModel):
    property_location: PropertyLocation
    correctness_explanation: CorrectnessExplanation

class Linter:
    def __init__(
        self,
        directory: Path,
        max_messages: int,
        min_length_to_exclude_full_files: int,
        max_files: int,
    ):
        self._directory: Path = directory
        self._rel_paths: list[Path] = get_rel_paths(directory)
        if len(self._rel_paths) > max_files:
            raise ValueError(f"Too many files: {len(self._rel_paths)}")
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
            props = extract_propositions((directory / p).read_text())
            for prop in props:
                self._property_locations.append(
                    PropertyLocation(rel_path=p, line_num=prop.line_num)
                )
                logger.debug(f"Found property @ {p}:{prop.line_num}")
                logger.debug(f"Prop: {prop.statement}")

        self._max_messages = max_messages
        self._exclude_full_files = (
            sum(len(doc) for doc in documents)
            >= min_length_to_exclude_full_files
        )

        python_deps = "\n".join(
            get_deps_rec(self._directory, self._directory, self._rel_paths)
        )
        if python_deps:
            self._python_deps_text = f"""Here is the dependency graph of the codebase:
{python_deps}


"""
        else:
            self._python_deps_text = ""

    def _build_initial_message(self, property_location: PropertyLocation) -> str:
        correctness_blurb = """By "correct", we mean very high confidence that each step of the proof is valid,
the proof does in fact prove the proposition, and that the proof is supported by
what the code does. Mark the proof as "incorrect" if you understand it and the
code but the proof is wrong. Use "unknown" if e.g. you don't 100% know how an
external library works, or the proof needs more detail. Skeptically and
rigorously check every claim with references to the code. If the proof
references an explicitly-stated axiom (or "assumption", etc), you can assume
that the axiom is correct."""

        if not self._exclude_full_files:
            return f"""The following is a listing of all files in a codebase:
{"\n".join(str(p) for p in self._rel_paths)}

At the end of this message is a listing of all file contents.

The file {property_location.rel_path} contains one or more propositions
about the codebase. The proposition we are interested in is on line
{property_location.line_num}, and is followed by a proof.

In your response, state whether the proof (not the proposition) is correct.
{correctness_blurb}
        

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

1. Request files

In this response, you may request to see additional files from the codebase in
order to ultimately determine whether the proof is correct. They will be
provided to you in the next message. You will have the opportunity to request
further files if needed, and we will repeat this process until you are ready to
make a final determination.

2. Correctness verdict

In this response, state whether the proof is correct.
{correctness_blurb}
(Use this response only if you have seen enough of
the codebase to make a determination.)

File contents:
{format_file((self._directory / property_location.rel_path).read_text(), rel_path=property_location.rel_path)}
            """

    def _build_subsequent_message(self, files_to_show: Sequence[Path]) -> str:
        return f"""The requested files are given below.

{"\n".join(format_file((self._directory / p).read_text(), rel_path=p) for p in files_to_show)}
        """

    def check_proofs(self) -> list[ProofCheckResult]:
        return [
            self.check_proof(property_location)
            for property_location in self._property_locations
        ]

    def check_proof(
        self, property_location: PropertyLocation
    ) -> ProofCheckResult:
        files_shown = set()
        all_files = set(self._rel_paths)
        initial_message = self._build_initial_message(property_location)
        logger.debug(initial_message)
        messages = [
            {
                "role": "user",
                "content": initial_message,
            }
        ]

        for _ in range(self._max_messages):
            resp = openai_client.beta.chat.completions.parse(
                **MODEL,
                messages=messages,  # type: ignore
                response_format=(
                    FullFilesExcludedResponse
                    if self._exclude_full_files
                    else FullFilesIncludedResponse
                ),
            )
            logger.debug(f"Token usage: {resp.usage}")

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
                )
            else:
                files_requested = (
                    all_files & {Path(p) for p in response_data.files_requested}
                ) - files_shown
                files_shown.update(files_requested)

                if not files_requested:
                    raise RuntimeError("Invalid response")

                subsequent_message = self._build_subsequent_message(
                    list(files_requested)
                )
                logger.debug(subsequent_message)
                messages.append(
                    {
                        "role": "user",
                        "content": subsequent_message,
                    }
                )
        else:
            raise TimeoutError("No result after max messages")


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
    max_messages: Annotated[
        int,
        typer.Option(
            help="Max number of messages in each LLM conversation before aborting"
        ),
    ] = 50,
    min_length_to_exclude_full_files: Annotated[
        int,
        typer.Option(
            help="Min size of codebase (in characters) such that we do not include all file contents in the initial prompt"
        ),
    ] = 100_000,
):
    linter = Linter(
        directory=directory,
        max_files=max_files,
        max_messages=max_messages,
        min_length_to_exclude_full_files=min_length_to_exclude_full_files,
    )
    print(
        json.dumps(
            [
                result.model_dump(mode="json")
                for result in linter.check_proofs()
            ],
            indent=2,
        )
    )


def cli() -> int:
    typer.run(main)
    return 0


if __name__ == "__main__":
    exit(cli())
