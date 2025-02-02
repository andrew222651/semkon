import io
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from loguru import logger
from pydeps import cli
from pydeps.pydeps import pydeps


def get_deps_rec(
    repo_directory: Path, package_directory: Path, rel_paths: Sequence[Path]
) -> dict[str, Any]:
    init_file = package_directory / "__init__.py"
    if init_file.exists() and init_file.is_file():
        if init_file.relative_to(repo_directory) in rel_paths:
            ret = get_deps(repo_directory, package_directory, rel_paths)
        else:
            ret = dict()
    else:
        ret = dict()
        for child_dir in package_directory.iterdir():
            if child_dir.is_dir():
                deps = get_deps_rec(repo_directory, child_dir, rel_paths)
                for k, v in deps.items():
                    if k in ret:
                        if ret[k]["path"] != v["path"]:
                            logger.debug("Cannot understand dependency graph")
                            return dict()
                        ret[k]["imported_by"] = list(
                            set(
                                ret[k].get("imported_by", [])
                                + v.get("imported_by", [])
                            )
                        )
                        ret[k]["imports"] = list(
                            set(
                                ret[k].get("imports", []) + v.get("imports", [])
                            )
                        )
                    else:
                        ret[k] = v

    return dict(sorted(ret.items(), key=lambda item: item[0]))


def get_deps(
    repo_directory: Path, package_directory: Path, rel_paths: Sequence[Path]
) -> dict[str, Any]:
    abs_paths = {repo_directory / p for p in rel_paths}

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
        if p not in abs_paths:
            continue
        v["path"] = str(p.relative_to(repo_directory))
        if "imported_by" in v:
            v["imported_by"] = [
                ib for ib in v["imported_by"] if ib != "__main__"
            ]
        del v["name"]
        del v["bacon"]
        filtered[k] = v

    return filtered
