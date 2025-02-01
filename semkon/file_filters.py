from pathlib import Path

from git import Repo


def is_text_file(abs_path: Path) -> bool:
    try:
        abs_path.read_text()
    except UnicodeDecodeError:
        return False
    return True


def is_small_file(abs_path: Path) -> bool:
    return abs_path.stat().st_size < 100 * 1024  # less than 100 KB


def get_rel_paths(directory: Path) -> list[Path]:
    if (directory / ".git").exists() and (directory / ".git").is_dir():
        repo = Repo(directory)
        raw_abs_paths = [
            directory / Path(s)
            for s in (
                repo.git.ls_files("--recurse-submodules").splitlines()
                + repo.git.ls_files(
                    "--others", "--exclude-standard"
                ).splitlines()
            )
        ]
    else:
        raw_abs_paths = [p for p in directory.glob("**/*") if p.is_file()]

    return sorted(
        [
            p.relative_to(directory)
            for p in raw_abs_paths
            if is_text_file(p) and is_small_file(p)
        ]
    )
