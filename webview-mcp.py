#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "mcp~=1.19.0",
#     "python-readability~=1.0.0rc0",
#     "pywebview~=6.1",
# ]
# ///


from asyncio import run, timeout, to_thread
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import suppress
from json import dumps
from os import getenv
from typing import Any, NotRequired, TypedDict

from mcp.server import FastMCP
from readability import Article
from readability import parse as _parse
from readability.impl.common import parse_getter
from readability.utils.cases import to_camel_cases, to_snake_cases
from webview import create_window, start, windows

_window = create_window(__file__, hidden=True, minimized=True, focus=False)
assert _window is not None
base_window = _window  # for the narrowed typing


def _eval_js(source: str, callback: Callable[[Any], Any] | None = None):
    return base_window.evaluate_js(source, callback)


def _borrow_signature[T: Callable](_: T) -> Callable[[Callable], T]:
    return lambda f: f  # type: ignore


@_borrow_signature(_parse)
def parse(html: str, /, **options):
    if not base_window.state.get("ready"):
        source = parse_getter()
        _eval_js(f"window.parse = {source}")
        _eval_js("pywebview.state.ready = true")
    result = _eval_js(f"parse({html!r}, {dumps(to_camel_cases(options))})")
    return Article(**to_snake_cases(result))


class Response(TypedDict):
    url: NotRequired[str]
    body: str
    status: int | None


def _fetch(url: str):
    fut = Future[Response]()

    window = create_window(url, url=url, on_top=True)
    assert window is not None

    status: int | None = None

    @window.expose
    def finish():
        window.destroy()
        result: Response = {"status": status, "body": window.state["html"]}
        if window.state["url"] != url:
            result["url"] = window.state["url"]
        fut.set_result(result)

    def on_loaded():
        window.run_js("""
            pywebview.state.html = document.documentElement.outerHTML;
            pywebview.state.url = window.location.href;
            pywebview.api.finish();
        """)

    def on_response_received(response):
        if response.url == url:
            nonlocal status
            status = response.status_code
            window.events.response_received -= on_response_received

    window.events.loaded += on_loaded
    window.events.response_received += on_response_received

    return fut.result()


async def fetch(url: str, _retry_remaining=7) -> Response:
    res = await to_thread(_fetch, url)
    if res["status"] is None and _retry_remaining > 0:  # maybe network issues
        return await fetch(url, _retry_remaining - 1)
    return res


def main():
    """Run webview in the main thread and start MCP server in a sub-thread."""

    @start
    def _():
        try:
            run(mcp.run_stdio_async())
        finally:
            for window in windows:
                window.destroy()


mcp = FastMCP("Webview MCP Server")


@mcp.tool()
async def read_url(url: str, request_timeout: float = 17):
    """Fetch and parse a URL, returning the plain text content."""

    async with timeout(request_timeout):
        res = await fetch(url)

    article = parse(res["body"])
    frontmatter = {**res}
    del frontmatter["body"]
    if title := article.title:
        frontmatter["title"] = title
    if excerpt := article.excerpt:
        frontmatter["excerpt"] = excerpt.replace("\n", " ")
    if byline := article.byline:
        frontmatter["byline"] = byline
    if site := article.site_name:
        frontmatter["site"] = site
    if lang := article.lang:
        frontmatter["language"] = lang
    if published := article.published_time:
        frontmatter["published"] = published

    head = "\n".join(f"{k}: {str(v).strip()}" for k, v in frontmatter.items())
    body = article.text_content or "[[ no content ]]"
    return f"---\n{head}\n---\n\n{body.strip()}"


if LOGFIRE_TOKEN := getenv("LOGFIRE_TOKEN"):
    from threading import Thread
    from time import sleep

    def worker():
        sleep(0.5)
        import logfire

        logfire.configure(scrubbing=False, token=LOGFIRE_TOKEN, service_name="webview")
        logfire.instrument_mcp()
        logfire.log_slow_async_callbacks()
        globals()["_eval_js"] = logfire.instrument(record_return=True)(_eval_js)
        globals()["_fetch"] = logfire.instrument(record_return=True)(_fetch)
        globals()["fetch"] = logfire.instrument(record_return=True)(fetch)

    Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        main()
