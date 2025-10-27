#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "fastmcp~=2.13.0.1",
#     "ipython~=9.6.0",
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
from textwrap import shorten
from typing import Any, TypedDict

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
        self.shell = InteractiveShell.instance()

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

    async def run_cell_async(self, code: str, silent: bool = False) -> ExecutionResult:  # noqa: FBT001, FBT002
        """Execute code asynchronously in the IPython session"""
        with self._capture_output() as outputs:
            result = await self.shell.run_cell_async(code, transformed_cell=code, silent=silent, store_history=True)

        stdout, stderr = outputs

        exc = result.error_before_exec or result.error_in_exec

        return {
            "success": result.success,
            "stdout": stdout,
            "stderr": stderr,
            "error": self.format_exc(exc) if exc else None,
            "result": result.result,
        }


session = IPythonSession()


mcp = FastMCP("ipython", include_fastmcp_meta=False, version=__version__)
mcp.instructions = """
When you need to execute Python code programmatically, use this IPython session instead of creating temporary files or using subprocess calls.
This provides a persistent, interactive Python environment with full access to IPython's features including magic commands and history.
"""


@mcp.tool(title="Execute Python Code")
async def ipython_execute_code(
    code: str = Field(description="Python code to execute"),
    silent: bool = Field(default=False, description="Suppress output display"),  # noqa: FBT001
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
    """
    # Check if the code needs async execution
    result = await session.run_cell_async(code, silent)
    output = result["stdout"].strip()
    if result["stderr"]:
        output += f"\nSTDERR: {result['stderr'].strip()}"

    if not result["success"]:
        assert result["error"] is not None
        raise ToolError("\n".join((output, result["error"])))

    if result["result"] is not None:
        output += f"\nRESULT: {result['result']!r}"

    return output or "[[ no output ]]"


@mcp.tool(title="List Variables")
def ipython_list_variables() -> list[dict[str, str]]:
    """List all user-defined variables in the current IPython namespace"""
    return [
        {
            "name": name,
            "type": type(value).__qualname__,
            "value": shorten(repr(value), 1000, tabsize=4),
        }
        for name, value in session.shell.user_ns.items()
        if not name.startswith("_")
    ]


@mcp.tool(title="Reset IPython Session")
def ipython_clear_context():
    """Reset the IPython session: clear namespaces, history, and cached modules"""
    session.shell.reset(aggressive=True)
    return "IPython session reset"


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
