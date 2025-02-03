import math
from typing import Any

import sympy
from RestrictedPython import compile_restricted, safe_globals
from wrapt_timeout_decorator import timeout


TIMEOUT_SECS = 60


@timeout(TIMEOUT_SECS)
def timed_execute(code: str) -> str:
    augmented_safe_globals = safe_globals | {
        "math": math,
        "sympy": sympy,
    }
    try:
        byte_code = compile_restricted(code, "<string>", "exec")
        safe_locals: dict[str, Any] = dict()
        exec(byte_code, augmented_safe_globals, safe_locals)
        return repr(safe_locals.get("result"))
    except Exception as e:
        return repr(e)


def execute(code: str) -> str:
    try:
        return timed_execute(code)
    except TimeoutError as e:
        return repr(e)
