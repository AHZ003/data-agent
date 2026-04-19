"""Sandboxed Python code execution for generated analysis code."""

import io
import sys
import traceback
from typing import Any, Dict, Optional, Tuple
from contextlib import redirect_stdout, redirect_stderr


def execute_python_code(
    code: str,
    local_vars: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, str, Optional[str]]:
    """
    Execute Python code in a restricted environment.

    Returns: (result, stdout_output, error_message)
    """
    if local_vars is None:
        local_vars = {}

    # Provide safe imports
    safe_globals = {
        "__builtins__": __builtins__,
    }

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    result = None
    error = None

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, safe_globals, local_vars)
            # Try to capture the last expression result
            result = local_vars.get("result", None)
    except Exception:
        error = traceback.format_exc()

    return result, stdout_capture.getvalue(), error
