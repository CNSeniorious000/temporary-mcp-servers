#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "fastmcp~=2.13.0.1",
#     "ipython~=9.6.0",
#     "logfire~=4.14.1",
# ]
# ///

"""
IPython MCP Server using FastMCP
A Model Context Protocol server for programmatic IPython session management.
"""

from contextlib import contextmanager, redirect_stderr, redirect_stdout, suppress
from functools import wraps
from io import StringIO
from os import environ, getenv
from sys import stderr, stdout
from typing import Any, TypedDict
from uuid import uuid4

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from IPython import __version__
from IPython.core.interactiveshell import InteractiveShell
from pydantic import Field


class ExecutionResult(TypedDict):
    """Result of code execution in IPython session"""

    success: bool
    stdout: str
    stderr: str
    error: str | None
    result: Any


class IPythonSession:
    """Manages an isolated IPython session"""

    def __init__(self):
        self.shell: InteractiveShell = InteractiveShell()

        showtraceback = self.shell.showtraceback

        @wraps(showtraceback)
        def wrapper(*args, **kwargs):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                return showtraceback(*args, **kwargs)

        self.shell.showtraceback = wrapper

        self.original_showtraceback = showtraceback

    @contextmanager
    def _capture_output(self):
        """Context manager to capture stdout and stderr"""
        with redirect_stdout(stdout := StringIO()), redirect_stderr(stderr := StringIO()):
            results: list[str] = []
            try:
                yield results
            finally:
                results[:] = [stdout.getvalue(), stderr.getvalue()]

    def format_exc(self, exc: BaseException):
        """Format an exception using IPython's traceback formatter"""

        with self._capture_output() as outputs:
            original_color_settings = self.shell.colors
            try:
                self.shell.colors = "nocolor"
                self.original_showtraceback((type(exc), exc, exc.__traceback__))
            finally:
                self.shell.colors = original_color_settings
        return "\n".join(outputs).strip()

    async def run_cell_async(self, code: str) -> ExecutionResult:
        """Execute code asynchronously in the IPython session"""
        with self._capture_output() as outputs:
            result = await self.shell.run_cell_async(code, transformed_cell=self.shell.transform_cell(code), store_history=True)

        stdout, stderr = outputs

        exc = result.error_before_exec or result.error_in_exec

        return {
            "success": result.success,
            "stdout": stdout,
            "stderr": stderr,
            "error": self.format_exc(exc) if exc else None,
            "result": result.result,
        }


sessions: dict[str, IPythonSession] = {}


def _get_session(session_id: str):
    if session_id not in sessions:
        raise ToolError(f"[[ Session {session_id} not found! ]]")
    return sessions[session_id]


def _shorten(text: str, max_length=2000):
    if len(text) <= max_length + 100:
        return text

    half = max_length // 2
    sep = " [...] " if "\n" not in text else "\n\n[...]\n\n"
    return text[:half] + sep + text[-half:]


def _as_xml(data: dict[str, Any]):
    strings = {k: text if "\n" not in (text := str(v).strip()) else f"\n{text}\n" for k, v in data.items()}
    return "\n".join(f"<{k}>{_shorten(v)}</{k}>" for k, v in strings.items())


mcp = FastMCP("Python (IPython)", include_fastmcp_meta=False, version=__version__)
mcp.instructions = """
When you need to execute Python code programmatically, always prefer this IPython session over creating temporary files or using subprocess calls.
This provides a persistent, interactive Python environment with full access to IPython's features including magic commands.
"""


@mcp.tool(title="Execute Python Code")
async def ipython_execute_code(
    code: str = Field(description="Python code to execute"),
    session_id: str | None = Field(None, description="Session ID to use. If not provided, a new session will be created"),
):
    """
    Execute Python code in an IPython session.

    This provides a persistent Python environment where you can:
    - Execute multi-line code blocks
    - Access variables across multiple calls
    - Use IPython magic commands
    - Import modules once and reuse them
    - Execute async/await code seamlessly

    Returns the execution result with output and any errors.

    Useful IPython magic commands:
    - %timeit: Time code execution multiple times for average (avoids outliers), supports custom run count. Example: %timeit [x**2 for x in range(1000)] outputs: 100000 loops, best of 3: 12.1 µs per loop
    - %cd: Change directory. Example: %cd /path/to/directory
    - %env: View or set environment variables with interpolation. Examples: %env PATH (view system path), %env MODEL_PATH = "./models" (set variable)
    - %who / %whos: List variables in current namespace. %who shows names only, %whos shows types and values. Example: %whos outputs: data: DataFrame (1000 rows x 5 columns), model: LinearRegression
    - ? / ??: View object's docstring (?) or source code (??, for objects with accessible source). Examples: pd.DataFrame? (view DataFrame usage), np.mean?? (view mean function source)
    - %run: Execute external Python script with command-line args, variables imported to current namespace. Example: %run train.py --epoch 10 --lr 0.01
    - %prun: Profile code execution with function-level report showing call counts and time percentages. Example: %prun my_complex_function(data)
    """
    if new_session := session_id is None:
        session = sessions[session_id := str(uuid4())] = IPythonSession()
    else:
        session = _get_session(session_id)

    result = await session.run_cell_async(code)

    if not result["success"]:
        assert result["error"] is not None
        if not result["stdout"].strip() and not result["stderr"].strip():
            raise ToolError(result["error"])
        out = {"traceback": result["error"]}
        if stdout := result["stdout"].strip():
            out["stdout"] = stdout
        if stderr := result["stderr"].strip():
            out["stderr"] = stderr
        raise ToolError(_as_xml(out))

    out = {}

    if not result["stdout"].strip() and not result["stderr"].strip():
        if new_session:
            return f"[[ execution successful, stdout/stderr empty, new IPython session created with ID: {session_id} ]]"
        if result["result"] is None:
            return "[[ execution successful, stdout/stderr empty ]]"
        else:
            return repr(result["result"])
    if new_session:
        out["session_id"] = session_id
    if stdout := result["stdout"].strip():
        out["stdout"] = stdout
    if stderr := result["stderr"].strip():
        out["stderr"] = stderr
    if result["result"] is not None:
        out["return"] = repr(result["result"])

    return _as_xml(out)


@mcp.tool(title="List Variables")
def ipython_list_variables(session_id: str):
    """List all user-defined variables in the current IPython namespace"""
    session = _get_session(session_id)
    return _as_xml({name: _shorten(repr(value)) for name, value in session.shell.user_ns.items() if not name.startswith("_") and name not in ("In", "Out", "exit", "quit", "open")})


@mcp.tool(title="Reset IPython Session")
def ipython_clear_context(
    session_id: str,
    delete: bool = Field(False, description="Delete the session after clearing"),  # noqa: FBT001, FBT003
):
    """Reset the IPython session: clear namespaces, history, and cached modules"""
    session = _get_session(session_id)
    session.shell.reset(aggressive=True)

    if delete:
        del sessions[session_id]
        return f"Session {session_id} deleted"
    else:
        return f"Session {session_id} reset"


if LOGFIRE_TOKEN := getenv("LOGFIRE_TOKEN"):
    from threading import Thread
    from time import sleep

    def worker():
        sleep(0.5)
        import logfire

        logfire.configure(scrubbing=False, token=LOGFIRE_TOKEN)
        logfire.instrument_mcp()
        for tool in (
            ipython_list_variables,
            ipython_clear_context,
            ipython_execute_code,
        ):
            tool.fn = logfire.instrument(span_name=f"<<< {tool.name} >>>", record_return=True)(tool.fn)

        environ["LOGFIRE_TOKEN"] = ""  # Avoid duplicate instrumentation

    Thread(target=worker, daemon=True).start()

if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        mcp.run("stdio", show_banner=False)
