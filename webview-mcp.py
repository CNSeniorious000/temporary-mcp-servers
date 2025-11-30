#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "hmr~=0.7.4",
#     "logfire~=4.15.0",
#     "mcp~=1.22.0",
#     "mm-read~=0.0.4.0",
#     "python-readability~=1.0.0rc0",
#     "pywebview~=6.1",
# ]
# ///


from asyncio import Semaphore, Task, create_task, gather, run, to_thread
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import suppress
from json import dumps
from os import getenv
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal, TypedDict
from urllib.parse import unquote

from mcp.server import FastMCP
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from mm_read.parse import to_markdown
from reactivity import async_effect, reactive
from webview import create_window, start, windows

if TYPE_CHECKING:
    from readability import parse
else:
    parse = ...


class ConcurrencyLimiter:
    def __init__(self, *, max_burst: int, max_concurrent: int, refill_interval: float):
        self.max_burst = max_burst
        self.active_count = 0
        self.refill_interval = refill_interval

        self.burst_limiter = Semaphore(max_burst)
        self.concurrency_limiter = Semaphore(max_concurrent)

        self._refill_task: Task | None = None

    async def _refill(self):
        from asyncio import sleep

        while True:
            await sleep(self.refill_interval)
            if self.active_count > 0:
                async with self.concurrency_limiter:
                    if self.active_count > 0:
                        self.active_count -= 1
                        self.burst_limiter.release()

    async def __aenter__(self):
        self.active_count += 1
        if self._refill_task is None or self._refill_task.done():
            self._refill_task = create_task(self._refill())
        await self.burst_limiter.acquire()
        await self.concurrency_limiter.acquire()
        return self

    async def __aexit__(self, *_):
        if self.active_count > 0:
            self.active_count -= 1
            self.burst_limiter.release()
        self.concurrency_limiter.release()


_window = create_window(__file__, hidden=True, minimized=True, focus=False)
assert _window is not None
base_window = _window  # for the narrowed typing


def _eval_js(source: str, callback: Callable[[Any], Any] | None = None):
    return base_window.evaluate_js(source, callback)


def _borrow_signature[T: Callable](_: T) -> Callable[[Callable], T]:
    return lambda f: f  # type: ignore


_readability_use_fallback = False


@_borrow_signature(parse)
def readability_parse(html: str, /, **options):
    global _readability_use_fallback

    from readability import Article, parse
    from readability.impl.common import parse_getter
    from readability.utils.cases import to_camel_cases, to_snake_cases

    if TYPE_CHECKING:
        e1 = e2 = Exception()

    # try the builtin backends first
    if not _readability_use_fallback:
        try:
            return parse(html, **options)
        except Exception as e:
            e1 = e

    # try our webview backend then
    if not base_window.state.get("ready"):
        source = parse_getter()
        _eval_js(f"window.parse = {source}")
        _eval_js("pywebview.state.ready = true")
    try:
        result = _eval_js(f"parse({html!r}, {dumps(to_camel_cases(options))})")
        _readability_use_fallback = True
        return Article(**to_snake_cases(result))
    except Exception as e:
        e2 = e

    # both backends failed
    raise e2 if _readability_use_fallback else e1 from None


del parse  # to avoid accidental usage


class Response(TypedDict):
    url: list[str]  # original and redirected URLs
    body: str
    status: int | None | Literal[False]


def _fetch(url: str, timeout: float):
    fut = Future[Response]()

    window = create_window(url, url, hidden=not getenv("WEBVIEW_VISIBLE"))
    assert window is not None

    fut.add_done_callback(lambda _: window.destroy())

    status: int | None | Literal[False] = False

    @window.expose
    def finish():
        window.destroy()
        result: Response = {"status": status, "body": window.state["html"], "url": [url]}
        if window.state["url"] != url:
            result["url"].append(window.state["url"])
        fut.set_result(result)

    def sync():
        window.run_js("""
            pywebview.state.html = document.documentElement.outerHTML;
            pywebview.state.url = window.location.href;
            pywebview.api.finish();
        """)

    def on_response_received(response):
        if unquote(response.url) == unquote(url):
            nonlocal status
            status = response.status_code
            window.events.response_received -= on_response_received
        else:
            status = None

    window.events.loaded += sync
    window.events.response_received += on_response_received

    with suppress(TimeoutError):
        return fut.result(timeout)
    try:
        sync()
        return fut.result(timeout)
    except TimeoutError as e:
        fut.set_exception(e)
        raise


async def fetch(url: str, timeout, _retry_remaining=2) -> Response:  # noqa: ASYNC109
    url = url.split("#", 1)[0]
    res = await to_thread(_fetch, url, timeout)
    if res["status"] is None and _retry_remaining > 0:  # maybe network issues
        return await fetch(url, timeout, _retry_remaining - 1)
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


async def read_url(url: str, timeout: float = 7):  # noqa: ASYNC109
    """Fetch and parse a URL, returning the plain text content."""

    try:
        res = await fetch(url, timeout)
    except TimeoutError:
        return f"---\nurl: {url}\n---\n\n[[ Timeout {timeout}s exceeded. Possible network issue or slow site. Please retry with longer timeout. ]]"

    article = readability_parse(res["body"], base_uri=res["url"][-1])
    frontmatter: dict[str, Any] = {"url": " -> ".join(res["url"])}
    if status := res["status"]:
        frontmatter["status"] = status
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
    body = to_markdown(article.content or res["body"], res["url"][-1]) if res["body"] else "[[ no content ]]"
    return f"---\n{head}\n---\n\n{body.strip()}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
async def read_urls(urls: list[str], timeout_seconds: float = 7):
    """Fetch and parse multiple URLs, returning their plain text content."""

    match len(urls):
        case 0:
            return CallToolResult(content=[])
        case 1:
            return await read_url(urls[0], timeout_seconds)

    done = reactive(list[str]())

    total = len(urls)
    t = perf_counter()

    @async_effect
    async def _():
        await mcp.get_context().report_progress(len(done), total, done[-1] if done else None)
        if len(done) == total:
            await mcp.get_context().report_progress(total, total, f"Read {total} URLs in {perf_counter() - t:.1f}s")

    limiter = ConcurrencyLimiter(max_burst=3, max_concurrent=15, refill_interval=1)

    async def _read_url(url: str):
        async with limiter:
            try:
                return await read_url(url, timeout_seconds)
            finally:
                done.append(url)

    results = await gather(*(_read_url(url) for url in urls))

    return CallToolResult(content=[TextContent(text=i, type="text") for i in results])


if LOGFIRE_TOKEN := getenv("LOGFIRE_TOKEN"):
    from threading import Thread
    from time import sleep

    def worker():
        sleep(0.5)
        import logfire

        logfire.configure(scrubbing=False, token=LOGFIRE_TOKEN, service_name="webview")
        logfire.instrument_mcp()
        logfire.log_slow_async_callbacks()
        logfire.install_auto_tracing(["readability.impl"], min_duration=0.005)
        globals()["readability_parse"] = logfire.instrument(record_return=True)(readability_parse)
        globals()["_eval_js"] = logfire.instrument(record_return=True)(_eval_js)
        globals()["_fetch"] = logfire.instrument(record_return=True)(_fetch)
        globals()["read_url"] = logfire.instrument(record_return=True)(read_url)

    Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        main()
