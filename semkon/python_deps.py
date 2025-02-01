import io
import json
import sys
from pathlib import Path
from typing import Sequence

from loguru import logger
from pydeps import cli
from pydeps.pydeps import pydeps


def get_deps_rec(
    repo_directory: Path, package_directory: Path, rel_paths: Sequence[Path]
) -> list[str]:
    init_file = package_directory / "__init__.py"
    if init_file.exists() and init_file.is_file():
        if init_file.relative_to(repo_directory) in rel_paths:
            deps = get_deps(repo_directory, package_directory, rel_paths)
            if deps:
                return [deps]
            else:
                return []
        else:
            return []
    else:
        ret = []
        for child_dir in package_directory.iterdir():
            if child_dir.is_dir():
                deps = get_deps_rec(repo_directory, child_dir, rel_paths)
                ret.extend(deps)
        return ret


def get_deps(
    repo_directory: Path, package_directory: Path, rel_paths: Sequence[Path]
) -> str | None:
    abs_paths = [repo_directory / p for p in rel_paths]

    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    try:
        sys.stdout = new_stdout

        logger.debug(f"Running pydeps on {package_directory}")
        # https://github.com/thebjorn/pydeps/issues/170#issuecomment-1479217955
        pydeps(
            **cli.parse_args(
                ["--show-deps", "--no-dot", str(package_directory)]
            )
        )
        d = json.loads(new_stdout.getvalue())
    finally:
        sys.stdout = old_stdout
        new_stdout.close()

    filtered = dict()
    for k, v in d.items():
        if not v["path"]:
            continue
        p = Path(v["path"])
        if p in abs_paths:
            v["path"] = str(p.relative_to(repo_directory))
            if "imported_by" in v:
                v["imported_by"] = [
                    ib for ib in v["imported_by"] if ib != "__main__"
                ]
            del v["name"]
            del v["bacon"]
            filtered[k] = v

    if filtered:
        return json.dumps(filtered, indent=2)
    else:
        return None
