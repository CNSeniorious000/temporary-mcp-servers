#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastmcp~=2.13.1.0",
#     "hmr~=0.7.4",
#     "ipython~=9.8.0",
#     "logfire~=4.15.0",
#     "objprint~=0.3.0",
#     "uv~=0.7",
# ]
# ///

"""
IPython MCP Server using FastMCP
A Model Context Protocol server for programmatic IPython session management.
"""

from os import environ, getenv
from pathlib import Path
from site import addsitedir, getsitepackages
from sys import executable, path, platform

if parent := getenv("PARENT"):
    cwd = Path.cwd()
    for i in eval(parent):
        addsitedir(i)
    if project_site_packages := getenv("PROJECT_SITE_PACKAGES"):
        path.insert(0, project_site_packages)

elif (venv_path := getenv("VIRTUAL_ENV")) and not Path(executable).is_relative_to(Path.cwd()):
    from subprocess import run
    from tempfile import TemporaryDirectory

    from uv import find_uv_bin

    if all_sitepackages := sorted(Path.cwd().glob("*/Lib/site-packages" if platform == "win32" else "*/lib/python*/site-packages"), key=lambda p: len(str(p))):
        site_packages = all_sitepackages[0]
        assert site_packages.is_dir(), site_packages

        rel_path = "scripts/python.exe" if platform == "win32" else "bin/python"
        python_exe = (site_packages.parent.parent if platform == "win32" else site_packages.parent.parent.parent) / rel_path
        assert python_exe.is_file(), python_exe

        def get_python_version(exe):
            for line in (Path(exe).parent.parent / "pyvenv.cfg").open():
                if line.startswith("version_info = "):
                    return line

        project_py_version = get_python_version(python_exe)
        current_py_version = get_python_version(executable)

        if project_py_version != current_py_version:
            with TemporaryDirectory("-venv", "ipython-mcp-") as temp_path:
                uv = find_uv_bin()
                run([uv, "venv", "-p", str(python_exe), "--seed", temp_path, "--link-mode", "symlink"], check=True)
                site_dirs = getsitepackages()
                new_env = {**environ, "PARENT": str(site_dirs), "VIRTUAL_ENV": temp_path, "UV_LINK_MODE": "symlink"}
                if str(site_packages) not in site_dirs:
                    new_env["PROJECT_SITE_PACKAGES"] = str(site_packages)
                try:
                    exit(run([uv, "run", "-p", str(Path(temp_path, rel_path)), "--active", __file__], env=new_env).returncode)
                except KeyboardInterrupt:
                    exit(1)
        else:
            path.insert(0, str(site_packages))
            addsitedir(str(site_packages))

from contextlib import contextmanager, redirect_stderr, redirect_stdout, suppress
from functools import wraps
from inspect import isclass
from io import StringIO
from operator import call
from sys import stderr
from typing import Any, TypedDict
from uuid import uuid4

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from IPython import __version__
from IPython.core.interactiveshell import InteractiveShell
from IPython.lib.pretty import pretty
from mcp.types import ToolAnnotations
from objprint import ObjPrint
from pydantic import Field
from reactivity.hmr import cache_across_reloads


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
            with redirect_stdout(stderr), redirect_stderr(stderr):
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


@call
@cache_across_reloads
def sessions() -> dict[str, IPythonSession]:
    return {}


def _get_session(session_id: str):
    if session_id not in sessions:
        raise ToolError(f"[[ Session {session_id} not found! ]]")
    return sessions[session_id]


def _shorten(text: str, max_length=70_000):
    if len(text) <= max_length + max_length // 10:  # 110%
        return text

    half = max_length // 2
    sep = " [...] " if "\n" not in text else "\n\n[...]\n\n"
    return text[:half] + sep + text[-half:]


def _as_xml(data: dict[str, str]):
    strings = {k: text if "\n" not in (text := v.strip()) else f"\n{text}\n" for k, v in data.items()}
    return "\n".join(f"<{k}>{_shorten(v)}</{k}>" for k, v in strings.items())


class CustomObjectPrinter(ObjPrint):
    def _objstr(self, obj, memo, indent_level, cfg):
        cfg.attr_pattern = "(?!^__.*__$).*"
        if isclass(obj):
            return repr(obj)
        cls = type(obj)
        if cls.__repr__ is object.__repr__:
            return self._get_custom_object_str(obj, memo, indent_level, cfg)
        if callable(obj):
            return pretty(obj, verbose=True, max_width=320)
        return super()._objstr(obj, memo, indent_level, cfg)


_repr = CustomObjectPrinter().objstr


mcp = FastMCP("Python (IPython)", include_fastmcp_meta=False, version=__version__)
mcp.instructions = """
When you need to execute Python code programmatically, always prefer this IPython session over creating temporary files or using subprocess calls.
This provides a persistent, interactive Python environment with full access to IPython's features including magic commands.
"""


@mcp.tool(title="Execute Python Code", annotations=ToolAnnotations(destructiveHint=False))
async def ipython_execute_code(
    code: str = Field(description="Python code to execute"),
    session_id: str | None = Field(None, description="Existing session ID to use. If not provided, a new session will be created"),
):
    """
    Execute Python code in an IPython session with persistent state across calls.

    Features:
    - Variables persist between calls for incremental development
    - Import modules once, reuse throughout session
    - Magic commands for introspection, profiling, and environment management

    Session Management:
    - Omit session_id for a new session
    - Use same session_id to access previous variables

    Common Magic Commands:
    - Introspection: %whos (list vars), print? (signature), obj?? (source)
    - Performance: %timeit (benchmark), %prun (profile)
    - Environment: %env VAR=value (set), %cd /path (change dir)
    - Files: %run script.py args (execute), %%writefile file.py (save)

    Example input:
    data = [1, 2, 3]
    sum_data = sum(data)
    %whos  # View variables in session
    %timeit [x**2 for x in range(1000)]  # Benchmark
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
            return _repr(result["result"])
    if new_session:
        out["session_id"] = session_id
    if stdout := result["stdout"].strip():
        out["stdout"] = stdout
    if stderr := result["stderr"].strip():
        out["stderr"] = stderr
    if result["result"] is not None:
        out["return"] = _repr(result["result"])

    return _as_xml(out)


@mcp.tool(title="Reset IPython Session", annotations=ToolAnnotations(destructiveHint=False))
def ipython_clear_context(
    session_id: str,
    delete: bool = Field(False, description="Delete the session after clearing"),  # noqa: FBT001, FBT003
):
    """
    Clear an IPython session's namespace and reset its state.

    This will:
    - Remove all user-defined variables from the session
    - Clear execution history
    - Reset the namespace to a clean state
    - Keep the session alive (unless delete=True)

    Args:
        session_id: ID of the session to reset
        delete: If True, delete the entire session after clearing
    """
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

    FLAG = " --- instrumented --- "  # Avoid duplicate instrumentation

    def worker():
        if environ.get(FLAG):
            import logfire
        else:
            environ[FLAG] = "1"
            sleep(0.5)
            import logfire

            logfire.configure(scrubbing=False, token=LOGFIRE_TOKEN, console=False, service_name="ipython")
            logfire.instrument_mcp()

        for tool in (ipython_clear_context, ipython_execute_code):
            tool.fn = logfire.instrument(span_name=f"<<< {tool.name} >>>", record_return=True)(tool.fn)

    Thread(target=worker, daemon=True).start()

if __name__ == "__main__":
    cwd = Path.cwd()
    if not any(Path(i).is_dir() and cwd.samefile(i) for i in path):
        path.insert(0, str(cwd))
    with suppress(KeyboardInterrupt):
        mcp.run("stdio", show_banner=False)
