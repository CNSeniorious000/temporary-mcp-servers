use std::collections::HashMap;
use std::fmt::Write;
use std::sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};

use dom_smoothie::{Config, Readability, TextMode};
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, ContentBlock, Implementation, ProgressNotificationParam, ServerCapabilities, ServerInfo},
    schemars::JsonSchema,
    service::RequestContext,
    tool, tool_handler, tool_router,
    transport::stdio,
    ErrorData as McpError, RoleServer, ServerHandler, ServiceExt,
};
use serde::Deserialize;
use tao::{
    event::Event,
    event_loop::{ControlFlow, EventLoopBuilder, EventLoopProxy},
    window::{Window, WindowBuilder},
};
use tokio::{
    runtime::Builder as RuntimeBuilder,
    sync::{oneshot, Semaphore},
    task::spawn_blocking,
    time::timeout,
};
use wry::{PageLoadEvent, WebView, WebViewBuilder};

#[derive(Deserialize, Debug)]
struct Fetched {
    #[serde(rename = "url")]
    final_url: String,
    html: String,
    #[serde(default)]
    status: Option<u16>,
}

#[derive(Debug)]
enum UserEvent {
    Fetch { id: u64, url: String, tx: oneshot::Sender<Fetched> },
    ForceSync { id: u64 },
    Cleanup { id: u64 },
}

#[derive(Deserialize, JsonSchema, Debug)]
#[schemars(crate = "rmcp::schemars")]
struct ReadUrlsArgs {
    urls: Vec<String>,
    #[serde(default = "default_timeout")]
    timeout_seconds: f64,
}

const fn default_timeout() -> f64 {
    7.0
}

#[derive(Clone)]
struct WebviewServer {
    proxy: EventLoopProxy<UserEvent>,
    burst: Arc<Semaphore>,
    concurrent: Arc<Semaphore>,
    next_id: Arc<AtomicU64>,
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl WebviewServer {
    fn new(proxy: EventLoopProxy<UserEvent>) -> Self {
        let burst = Arc::new(Semaphore::new(3));
        let concurrent = Arc::new(Semaphore::new(15));
        tokio::spawn({
            let burst = burst.clone();
            async move {
                loop {
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    if burst.available_permits() < 3 {
                        burst.add_permits(1);
                    }
                }
            }
        });
        Self { proxy, burst, concurrent, next_id: Arc::new(AtomicU64::new(1)), tool_router: Self::tool_router() }
    }

    #[tracing::instrument(skip(self), fields(url = %url))]
    async fn read_one(&self, url: String, timeout_s: f64) -> String {
        let url = match url.split_once('#') {
            Some((before, _)) => before.to_owned(),
            None => url,
        };
        let _burst = self.burst.acquire().await.expect("burst semaphore is never closed");
        let _concurrent = self.concurrent.acquire().await.expect("concurrent semaphore is never closed");

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, mut rx) = oneshot::channel();
        if self.proxy.send_event(UserEvent::Fetch { id, url: url.clone(), tx }).is_err() {
            return format!("---\nurl: {url}\n---\n\n[[ event loop is gone ]]");
        }

        // 优先等 load 事件触发的自然抓取；超时则主动让 webview 跑一次抓取脚本，再短等一会儿，
        // 救回那种 DOM 已就位但 load 始终不触发（拖尾资源/第三方脚本卡住）的慢站
        let fetched = if let Ok(Ok(f)) = timeout(Duration::from_secs_f64(timeout_s), &mut rx).await {
            Some(f)
        } else {
            self.proxy.send_event(UserEvent::ForceSync { id }).ok();
            timeout(Duration::from_secs(2), &mut rx).await.ok().and_then(Result::ok)
        };
        self.proxy.send_event(UserEvent::Cleanup { id }).ok();

        let Some(fetched) = fetched else {
            return format!("---\nurl: {url}\n---\n\n[[ Timeout {timeout_s}s exceeded. Possible network issue or slow site. Please retry with longer timeout. ]]");
        };

        spawn_blocking(move || format_article(&url, fetched)).await.unwrap_or_else(|e| format!("[[ format_article task panicked: {e} ]]"))
    }

    #[tool(description = "Fetch and parse multiple URLs, returning their plain text content.")]
    async fn read_urls(&self, Parameters(args): Parameters<ReadUrlsArgs>, ctx: RequestContext<RoleServer>) -> Result<CallToolResult, McpError> {
        let total = args.urls.len();
        let token = ctx.meta.get_progress_token();
        let t0 = Instant::now();
        let timeout_s = args.timeout_seconds;
        let done = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = args
            .urls
            .into_iter()
            .map(|url| {
                let me = self.clone();
                let token = token.clone();
                let peer = ctx.peer.clone();
                let done = done.clone();
                tokio::spawn(async move {
                    let result = me.read_one(url.clone(), timeout_s).await;
                    let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if let Some(tok) = token {
                        peer.notify_progress(ProgressNotificationParam::new(tok, n as f64).with_total(total as f64).with_message(url)).await.ok();
                    }
                    result
                })
            })
            .collect();
        let mut contents = Vec::with_capacity(total);
        for h in handles {
            let text = h.await.unwrap_or_else(|e| format!("[[ task panicked: {e} ]]"));
            contents.push(ContentBlock::text(text));
        }

        if let Some(tok) = token {
            ctx.peer
                .notify_progress(ProgressNotificationParam::new(tok, total as f64).with_total(total as f64).with_message(format!("Read {total} URLs in {:.1}s", t0.elapsed().as_secs_f64())))
                .await
                .ok();
        }

        Ok(CallToolResult::success(contents))
    }
}

#[tool_handler]
impl ServerHandler for WebviewServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_server_info(Implementation::new(env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")))
    }
}

#[tracing::instrument(skip(fetched))]
fn format_article(orig_url: &str, fetched: Fetched) -> String {
    let Fetched { final_url, html, status } = fetched;

    let mut head = if final_url.is_empty() || final_url == orig_url { format!("url: {orig_url}") } else { format!("url: {orig_url} -> {final_url}") };
    if let Some(s) = status {
        write!(head, "\nstatus: {s}").expect("writing to String never fails");
    }

    if html.is_empty() {
        return format!("---\n{head}\n---\n\n[[ no content ]]");
    }

    let cfg = Config { text_mode: TextMode::Markdown, ..Default::default() };
    let Ok(article) = Readability::new(html.clone(), Some(&final_url), Some(cfg)).and_then(|mut r| r.parse()) else {
        return format!("---\n{head}\n---\n\n{}", html.trim());
    };

    for (k, v) in [
        ("title", Some(article.title.as_ref())),
        ("excerpt", article.excerpt.as_deref()),
        ("byline", article.byline.as_deref()),
        ("site", article.site_name.as_deref()),
        ("language", article.lang.as_deref()),
        ("published", article.published_time.as_deref()),
    ] {
        if let Some(s) = v.map(str::trim).filter(|s| !s.is_empty()) {
            write!(head, "\n{k}: {}", s.replace('\n', " ")).expect("writing to String never fails");
        }
    }

    let body = article.text_content.trim();
    let body = if body.is_empty() { "[[ no content ]]" } else { body };
    format!("---\n{head}\n---\n\n{body}")
}

const SNAPSHOT_SCRIPT: &str = "[location.href, document.documentElement.outerHTML]";

type TxCell = Arc<Mutex<Option<oneshot::Sender<Fetched>>>>;

#[cfg(target_os = "macos")]
fn snapshot(webview: &WebView, tx_cell: &TxCell) {
    use objc2::{rc::Retained, runtime::AnyObject, MainThreadMarker};
    use objc2_foundation::{NSArray, NSError, NSString};
    use objc2_web_kit::WKContentWorld;
    use wry::WebViewExtMacOS;

    // run in WKContentWorld::defaultClientWorld — WebKit's user-script isolated world,
    // not subject to page CSP (e.g. raw.githubusercontent.com's `sandbox` directive)
    let wk = webview.webview();
    let tx_cell = tx_cell.clone();
    let handler = block2::RcBlock::new(move |val: *mut AnyObject, _err: *mut NSError| {
        let Some(tx) = tx_cell.lock().expect("ipc tx mutex poisoned").take() else { return };
        // SAFETY: WKWebView fires this completion handler on the main thread; val is either null or a retained NSArray<NSString>
        #[allow(unsafe_code)]
        let arr = unsafe { Retained::retain(val) }.and_then(|v| v.downcast::<NSArray>().ok());
        let read = |a: &NSArray, i| a.objectAtIndex(i).downcast::<NSString>().map(|s| s.to_string()).unwrap_or_default();
        let (final_url, html) = arr.filter(|a| a.count() >= 2).map(|a| (read(&a, 0), read(&a, 1))).unwrap_or_default();
        tx.send(Fetched { final_url, html, status: None }).ok();
    });
    #[allow(unsafe_code)]
    unsafe {
        let mtm = MainThreadMarker::new().expect("snapshot must run on the main thread");
        let world = WKContentWorld::defaultClientWorld(mtm);
        let js = NSString::from_str(SNAPSHOT_SCRIPT);
        wk.evaluateJavaScript_inFrame_inContentWorld_completionHandler(&js, None, &world, Some(&handler));
    }
}

#[cfg(not(target_os = "macos"))]
fn snapshot(webview: &WebView, tx_cell: &TxCell) {
    let tx_cell = tx_cell.clone();
    webview
        .evaluate_script_with_callback(SNAPSHOT_SCRIPT, move |raw| {
            let Some(tx) = tx_cell.lock().expect("ipc tx mutex poisoned").take() else { return };
            // on windows/linux wry stringifies the JS return value to JSON text before invoking us
            let (final_url, html) = serde_json::from_str::<(String, String)>(&raw).unwrap_or_default();
            tx.send(Fetched { final_url, html, status: None }).ok();
        })
        .ok();
}

#[cfg(any(target_os = "windows", target_os = "macos", target_os = "ios", target_os = "android"))]
fn build_webview(b: WebViewBuilder<'_>, w: &Window) -> wry::Result<WebView> {
    b.build(w)
}

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "ios", target_os = "android")))]
fn build_webview(b: WebViewBuilder<'_>, w: &Window) -> wry::Result<WebView> {
    use tao::platform::unix::WindowExtUnix;
    use wry::WebViewBuilderExtUnix;
    b.build_gtk(w.default_vbox().unwrap())
}

fn main() {
    #[allow(unused_mut)]
    let mut event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();
    #[cfg(target_os = "macos")]
    {
        use tao::platform::macos::EventLoopExtMacOS;
        event_loop.set_dock_visibility(false);
    }
    let proxy = event_loop.create_proxy();

    std::thread::spawn(move || {
        dotenvy::dotenv().ok();
        let logfire = std::env::var_os("LOGFIRE_TOKEN").and_then(|_| {
            logfire::configure()
                .with_service_name(env!("CARGO_PKG_NAME"))
                .with_default_level_filter(tracing::level_filters::LevelFilter::INFO)
                .finish()
                .ok()
        });
        let rt = RuntimeBuilder::new_multi_thread().enable_all().build().expect("failed to build tokio runtime");
        rt.block_on(async {
            let Ok(service) = WebviewServer::new(proxy).serve(stdio()).await else { return };
            tokio::select! {
                _ = service.waiting() => {}
                _ = tokio::signal::ctrl_c() => {}
            }
        });
        if let Some(lf) = logfire {
            lf.shutdown().ok();
        }
        std::process::exit(0);
    });

    let mut in_flight = HashMap::new();
    let visible = std::env::var_os("WEBVIEW_VISIBLE").is_some();
    let loop_proxy = event_loop.create_proxy();

    event_loop.run(move |event, target, control_flow| {
        *control_flow = ControlFlow::Wait;
        match event {
            Event::UserEvent(UserEvent::Fetch { id, url, tx }) => {
                let window = match WindowBuilder::new().with_title(&url).with_visible(visible).build(target) {
                    Ok(w) => w,
                    Err(e) => {
                        drop(tx.send(Fetched { final_url: url, html: format!("[[ window build failed: {e} ]]"), status: None }));
                        return;
                    }
                };
                let tx_cell: TxCell = Arc::new(Mutex::new(Some(tx)));
                let proxy_for_load = loop_proxy.clone();
                let builder = WebViewBuilder::new().with_url(&url).with_on_page_load_handler(move |evt, _url| {
                    if matches!(evt, PageLoadEvent::Finished) {
                        proxy_for_load.send_event(UserEvent::ForceSync { id }).ok();
                    }
                });
                let webview = match build_webview(builder, &window) {
                    Ok(w) => w,
                    Err(e) => {
                        let taken = tx_cell.lock().expect("ipc tx mutex poisoned").take();
                        if let Some(tx) = taken {
                            drop(tx.send(Fetched { final_url: url, html: format!("[[ webview build failed: {e} ]]"), status: None }));
                        }
                        return;
                    }
                };
                in_flight.insert(id, (window, webview, tx_cell));
            }
            Event::UserEvent(UserEvent::ForceSync { id }) => {
                if let Some((_, webview, tx_cell)) = in_flight.get(&id) {
                    snapshot(webview, tx_cell);
                }
            }
            Event::UserEvent(UserEvent::Cleanup { id }) => {
                in_flight.remove(&id);
            }
            _ => {}
        }
    });
}
