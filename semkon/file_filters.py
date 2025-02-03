from pathlib import Path

import gitignore_parser


MAX_FILE_SIZE = 100 * 1024  # 100 KB


def _is_text_file(abs_path: Path) -> bool:
    try:
        abs_path.read_text()
    except UnicodeDecodeError:
        return False
    return True


def _is_small_file(abs_path: Path) -> bool:
    return abs_path.stat().st_size < MAX_FILE_SIZE


FILTER_ALWAYS = [
    "**/.git/",
]


class FileFilters:
    def __init__(self, directory: Path, filter_paths: list[str]):
        self._directory = directory
        raw_rules = [
            gitignore_parser.rule_from_pattern(filter_path, base_path=directory)
            for filter_path in filter_paths + FILTER_ALWAYS
        ]
        self._filter_rules = [r for r in raw_rules if r is not None]

    def get_abs_paths(
        self,
        directory: Path | None = None,
        gitignore_rules: list[gitignore_parser.IgnoreRule] | None = None,
    ) -> list[Path]:
        if directory is None:
            directory = self._directory
        if gitignore_rules is None:
            gitignore_rules = []

        ret = []

        my_gitignore_rules = list(gitignore_rules)

        gitignore_file = directory / ".gitignore"
        if (
            gitignore_file.exists()
            and gitignore_file.is_file()
            and _is_text_file(gitignore_file)
        ):
            raw_rules = [
                gitignore_parser.rule_from_pattern(line, base_path=directory)
                for line in gitignore_file.read_text().splitlines()
            ]
            my_gitignore_rules.extend(r for r in raw_rules if r is not None)

        for f in directory.iterdir():
            if (
                f.is_file()
                and _is_small_file(f)
                and _is_text_file(f)
                and not any(
                    rule.match(f)
                    for rule in my_gitignore_rules + self._filter_rules
                )
            ):
                ret.append(f)

        for dir in directory.iterdir():
            if dir.is_dir() and not any(
                rule.match(dir)
                for rule in my_gitignore_rules + self._filter_rules
            ):
                ret.extend(
                    self.get_abs_paths(dir, gitignore_rules=my_gitignore_rules)
                )

        return ret


def get_rel_paths(directory: Path, filter_paths: list[str]) -> list[Path]:
    ff = FileFilters(directory, filter_paths=filter_paths)
    abs_paths = ff.get_abs_paths()
    return sorted(p.relative_to(directory) for p in abs_paths)
