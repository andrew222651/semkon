import math
from pathlib import Path


def format_file(content: str, rel_path: Path | None = None) -> str:
    content_lines = content.splitlines()
    lines_chars = math.floor(math.log10(len(content_lines))) + 1
    content_w_lines = "\n".join(
        f"{i + 1:>{lines_chars}} | {line}"
        for i, line in enumerate(content_lines)
    )

    file_name = str(rel_path) if rel_path else "<file>"
    return f"""================
{file_name} (line numbers added)
================

{content_w_lines}


"""
