#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import Annotated, Literal, Sequence

import chromadb
import openai
import typer
from chromadb.api import ClientAPI
from loguru import logger
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from .file_filters import get_rel_paths
from .properties import extract_theorem_ids
from .python_deps import get_deps_rec


# o1-preview and o1 gave good results. everything else was bad (deepseek,
# claude, gpt-4o, gemini, o3-mini).
MODEL = "o1"

logger.remove()
logger.add(sink=sys.stderr, level="DEBUG")


class Settings(BaseSettings):
    OPENAI_API_KEY: str


settings = Settings()  # type: ignore

openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)


class PointOfInterest(BaseModel):
    pass


class Theorem(PointOfInterest):
    rel_path: Path
    theorem_id: str


class CorrectnessExplanation(BaseModel):
    correctness: Literal["correct", "incorrect", "unknown"]
    explanation: str


class FilesRequested(BaseModel):
    files_requested: list[str]


class FullFilesExcludedResponse(BaseModel):
    data: FilesRequested | CorrectnessExplanation


class FullFilesIncludedResponse(BaseModel):
    data: CorrectnessExplanation


class ProofCheckResult(CorrectnessExplanation):
    theorem_id: str


class Linter:
    def _format_file(self, rel_path: Path) -> str:
        return f"""================
{rel_path}
================
{(self._directory / rel_path).read_text()}"""

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

        self._theorems = []
        for p in self._rel_paths:
            ids = extract_theorem_ids((directory / p).read_text())
            for id_ in ids:
                self._theorems.append(Theorem(rel_path=p, theorem_id=id_))

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

    def _build_initial_message(self, theorem: Theorem) -> str:
        if not self._exclude_full_files:
            return f"""The following is a listing of all files in a codebase:
{"\n".join(str(p) for p in self._rel_paths)}

At the end of this message is a listing of all file contents.

The file {theorem.rel_path} contains one or more propositions
about the codebase. The proposition we are interested in starts
with "::: {{.theorem #{theorem.theorem_id}}}", and is followed by a proof.

In your response, state whether the proof (not the proposition) is correct. By
"correct", we mean very high confidence that each step of the proof is valid,
the proof does in fact prove the proposition, and that the proof is supported by
what the code does. Mark the proof as "incorrect" if you understand it and the
code but the proof is wrong. Use "unknown" if e.g. you don't 100% know how an
external library works, or the proof needs more detail. Skeptically and
rigorously check every claim with references to the code.
        

File contents:
{"\n".join(self._format_file(p) for p in self._rel_paths)}
            """
        else:
            return f"""The following is a listing of all files in a codebase:
{"\n".join(str(p) for p in self._rel_paths)}

{self._python_deps_text}

At the end of this message is a listing of the contents of {theorem.rel_path}.
This file contains one or more propositions
about the codebase. The proposition we are interested in starts
with "::: {{.theorem #{theorem.theorem_id}}}", and is followed by a proof.

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

In this response, state whether the proof is correct. By "correct", we mean very
high confidence that each step of the proof is valid, the proof does in fact
prove the proposition, and that the proof is supported by what the code does.
Mark the proof as "incorrect" if you understand it and the code but the proof is
wrong. Use "unknown" if e.g. you don't 100% know how an external library works,
or the proof needs more detail. Skeptically and rigorously check every claim
with references to the code. (Use this response only if you have seen enough of
the codebase to make a determination.)

File contents:
{self._format_file(theorem.rel_path)}
            """

    def _build_subsequent_message(self, files_to_show: Sequence[Path]) -> str:
        return f"""The requested files are given below.

{"\n".join(self._format_file(p) for p in files_to_show)}
        """

    def check_proofs(self) -> list[ProofCheckResult]:
        return [
            self.check_proof(theorem.theorem_id) for theorem in self._theorems
        ]

    def check_proof(self, theorem_id: str) -> ProofCheckResult:
        theorem = next(t for t in self._theorems if t.theorem_id == theorem_id)
        files_shown = set()
        all_files = set(self._rel_paths)
        initial_message = self._build_initial_message(theorem)
        logger.debug(initial_message)
        messages = [
            {
                "role": "user",
                "content": initial_message,
            }
        ]

        for _ in range(self._max_messages):
            resp_msg = (
                openai_client.beta.chat.completions.parse(
                    model=MODEL,
                    messages=messages,  # type: ignore
                    response_format=(
                        FullFilesExcludedResponse
                        if self._exclude_full_files
                        else FullFilesIncludedResponse
                    ),
                )
                .choices[0]
                .message
            )
            if resp_msg.content is None:
                raise RuntimeError("No response from LLM")
            logger.debug(resp_msg.content)
            messages.append({"role": "assistant", "content": resp_msg.content})
            if resp_msg.parsed is None:
                raise RuntimeError("No response from LLM")
            response = resp_msg.parsed.data

            if isinstance(response, CorrectnessExplanation):
                return ProofCheckResult(
                    theorem_id=theorem_id,
                    correctness=response.correctness,
                    explanation=response.explanation,
                )
            else:
                files_requested = (
                    all_files & {Path(p) for p in response.files_requested}
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
    for result in linter.check_proofs():
        print(result)


def cli() -> int:
    typer.run(main)
    return 0


if __name__ == "__main__":
    exit(cli())
