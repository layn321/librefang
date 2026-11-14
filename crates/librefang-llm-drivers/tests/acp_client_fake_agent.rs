//! Integration tests for the ACP client driver against a fake ACP agent.
//!
//! The fake agent (`tests/fixtures/fake_acp_agent.py`) speaks ACP v1 over stdio and
//! never calls a model, so these tests need no CLI, no adapter and no credentials.
//! They need `python3` on PATH.

use librefang_llm_drivers::drivers::acp_client::{AcpAgentKind, AcpClientDriver, PermissionPolicy};
use librefang_llm_drivers::llm_driver::{
    CompletionRequest, LlmDriver, McpBridgeConfig, StreamEvent,
};
use librefang_types::message::{Message, StopReason};
use librefang_types::tool::ToolDefinition;
use serial_test::serial;
use std::path::PathBuf;
use std::sync::Arc;

/// Secrets that must never reach an adapter: one on the old exact list, an unlisted
/// `*_API_KEY`, an AWS secret, a `*_TOKEN`, and the variable that re-enables bypass.
const SECRET_VARS: &[&str] = &[
    "GROQ_API_KEY",
    "FAKE_UNLISTED_API_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "GITHUB_TOKEN",
    "IS_SANDBOX",
];

fn fixture() -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/fake_acp_agent.py")
        .to_string_lossy()
        .into_owned()
}

fn driver(mode: &str, policy: PermissionPolicy) -> AcpClientDriver {
    AcpClientDriver::new(AcpAgentKind::ClaudeCode, Some("python3".to_string()))
        .with_args(vec![fixture(), mode.to_string()])
        .with_permission_policy(policy)
        .with_message_timeout(30)
}

fn request(prompt: &str) -> CompletionRequest {
    CompletionRequest {
        model: "claude-code-acp".to_string(),
        messages: Arc::new(vec![Message::user(prompt)]),
        ..Default::default()
    }
}

async fn run_stream(
    d: &AcpClientDriver,
    req: CompletionRequest,
) -> (
    Vec<StreamEvent>,
    librefang_llm_drivers::llm_driver::CompletionResponse,
) {
    let (tx, mut rx) = tokio::sync::mpsc::channel(256);
    let collector = tokio::spawn(async move {
        let mut events = Vec::new();
        while let Some(ev) = rx.recv().await {
            events.push(ev);
        }
        events
    });
    let response = d.stream(req, tx).await.expect("ACP turn should succeed");
    (collector.await.unwrap(), response)
}

fn text_of(events: &[StreamEvent]) -> String {
    events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::TextDelta { text } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

/// Full flow: initialize -> session/new (with the MCP bridge) -> session/prompt ->
/// streamed chunks -> permission request allowed by policy -> end_turn.
#[tokio::test]
#[serial]
async fn full_turn_streams_and_allows_permission_per_policy() {
    std::env::set_var("ANTHROPIC_BASE_URL", "http://agentgateway.test:8080");
    for k in SECRET_VARS {
        std::env::set_var(k, "must-not-leak");
    }
    let d = driver("normal", PermissionPolicy::AllowAll).with_mcp_bridge(McpBridgeConfig {
        base_url: "http://127.0.0.1:4545".to_string(),
        api_key: None,
    });
    let mut req = request("hello acp");
    req.tools = Arc::new(vec![ToolDefinition {
        name: "noop".to_string(),
        description: "noop".to_string(),
        input_schema: serde_json::json!({"type": "object"}),
    }]);

    let (events, response) = run_stream(&d, req).await;
    std::env::remove_var("ANTHROPIC_BASE_URL");
    for k in SECRET_VARS {
        std::env::remove_var(k);
    }

    let text = text_of(&events);
    assert!(
        text.contains("echo:hello acp"),
        "prompt reached the agent: {text}"
    );
    assert!(
        text.contains("|base_url:http://agentgateway.test:8080"),
        "ANTHROPIC_BASE_URL passed through: {text}"
    );
    assert!(
        text.contains("|mcp:librefang"),
        "MCP bridge sent in session/new: {text}"
    );
    assert!(
        text.contains("|leaked:<none>"),
        "allowlist stripped listed, unlisted and AWS secrets: {text}"
    );
    assert!(
        text.contains("|mode:default"),
        "driver switched the session to `default` mode: {text}"
    );
    assert!(
        text.contains("|bypass_allowed:false"),
        "session/new _meta disabled bypass: {text}"
    );
    assert!(
        text.contains("|cwd_match:yes"),
        "adapter runs in the session cwd: {text}"
    );
    assert!(
        text.contains("|permission:granted"),
        "allow-all policy granted: {text}"
    );
    assert!(events
        .iter()
        .any(|e| matches!(e, StreamEvent::ThinkingDelta { text } if text == "thinking about it")));
    assert!(events.iter().any(
        |e| matches!(e, StreamEvent::PhaseChange { phase, detail: Some(d) } if phase == "tool_use" && d == "Edit README.md")
    ));
    assert!(matches!(
        events.last(),
        Some(StreamEvent::ContentComplete {
            stop_reason: StopReason::EndTurn,
            ..
        })
    ));
    assert_eq!(response.stop_reason, StopReason::EndTurn);
    assert_eq!(response.text(), text);
    assert!(response.tool_calls.is_empty());
    assert!(
        d.active_pids().is_empty(),
        "adapter PID released after the turn"
    );
}

/// The default policy denies; the agent sees the reject option.
#[tokio::test]
#[serial]
async fn default_deny_policy_rejects_permission() {
    let d = driver("normal", PermissionPolicy::default());
    let (events, _) = run_stream(&d, request("hi")).await;
    let text = text_of(&events);
    assert!(
        text.contains("|permission:denied"),
        "deny policy rejected: {text}"
    );
    assert!(!text.contains("granted"));
}

/// An allowlist that does not include `edit` denies the fake agent's edit request.
#[tokio::test]
#[serial]
async fn allowlist_without_edit_denies() {
    let d = driver(
        "normal",
        PermissionPolicy::parse("allow:read,search").unwrap(),
    );
    let (events, _) = run_stream(&d, request("hi")).await;
    assert!(text_of(&events).contains("|permission:denied"));
    let d = driver("normal", PermissionPolicy::parse("allow:edit").unwrap());
    let (events, _) = run_stream(&d, request("hi")).await;
    assert!(text_of(&events).contains("|permission:granted"));
}

/// A turn with no session/update still completes (empty text, end_turn).
#[tokio::test]
#[serial]
async fn turn_without_updates_completes_empty() {
    let d = driver("drop_stream", PermissionPolicy::AllowAll);
    let (events, response) = run_stream(&d, request("hi")).await;
    assert_eq!(text_of(&events), "");
    assert_eq!(response.stop_reason, StopReason::EndTurn);
}

/// Dropping the stream receiver sends session/cancel; the agent acknowledges it
/// (writes the marker) and the driver returns an error instead of a response.
#[tokio::test]
#[serial]
async fn dropped_receiver_sends_session_cancel() {
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("cancelled");
    let d = AcpClientDriver::new(AcpAgentKind::Codex, Some("python3".to_string()))
        .with_args(vec![
            fixture(),
            "hang".to_string(),
            marker.to_string_lossy().into_owned(),
        ])
        .with_message_timeout(30);
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    let reader = tokio::spawn(async move {
        let first = rx.recv().await;
        drop(rx);
        first
    });
    let result = d.stream(request("hi"), tx).await;
    let first = reader.await.unwrap();
    assert!(matches!(first, Some(StreamEvent::TextDelta { ref text }) if text == "working..."));
    let err = result.expect_err("cancelled turn must not return a response");
    assert!(err.to_string().contains("cancelled"), "{err}");
    assert!(marker.exists(), "agent received session/cancel");
}

/// A missing adapter is a spawn error that names the install step.
#[tokio::test]
#[serial]
async fn missing_adapter_reports_install_hint() {
    let d = AcpClientDriver::new(
        AcpAgentKind::ClaudeCode,
        Some("/nonexistent/claude-agent-acp".to_string()),
    );
    let err = d
        .complete(request("hi"))
        .await
        .expect_err("spawn must fail");
    assert!(err.to_string().contains("claude-agent-acp"), "{err}");
}

/// The Codex adapter gets OPENAI_* but not ANTHROPIC_*, and no unlisted secret.
#[tokio::test]
#[serial]
async fn codex_env_allowlist_passes_openai_only() {
    std::env::set_var("OPENAI_BASE_URL", "http://agentgateway.test:8080/v1");
    std::env::set_var("ANTHROPIC_BASE_URL", "http://agentgateway.test:8080");
    for k in SECRET_VARS {
        std::env::set_var(k, "must-not-leak");
    }
    let d = AcpClientDriver::new(AcpAgentKind::Codex, Some("python3".to_string()))
        .with_args(vec![fixture(), "normal".to_string()])
        .with_permission_policy(PermissionPolicy::AllowAll)
        .with_message_timeout(30);
    let (events, _) = run_stream(&d, request("hi")).await;
    std::env::remove_var("OPENAI_BASE_URL");
    std::env::remove_var("ANTHROPIC_BASE_URL");
    for k in SECRET_VARS {
        std::env::remove_var(k);
    }
    let text = text_of(&events);
    assert!(
        text.contains("|openai_base:http://agentgateway.test:8080/v1"),
        "{text}"
    );
    assert!(
        text.contains("|base_url:<unset>"),
        "ANTHROPIC_* withheld from codex: {text}"
    );
    assert!(text.contains("|leaked:<none>"), "{text}");
    // No mode is enforced for Codex (its mode ids are unverified); the agent's stays.
    assert!(text.contains("|mode:acceptEdits"), "{text}");
}

/// A configured cwd is used as-is; the default is a scratch dir removed after the turn.
#[tokio::test]
#[serial]
async fn session_cwd_is_explicit_and_scratch_by_default() {
    let dir = tempfile::tempdir().unwrap();
    let d = driver("normal", PermissionPolicy::AllowAll).with_cwd(dir.path().to_path_buf());
    let (events, _) = run_stream(&d, request("hi")).await;
    assert!(text_of(&events).contains("|cwd_match:yes"));
    assert!(dir.path().exists(), "a configured cwd is never removed");

    let before = scratch_dirs();
    let d = driver("normal", PermissionPolicy::AllowAll);
    let (events, _) = run_stream(&d, request("hi")).await;
    assert!(text_of(&events).contains("|cwd_match:yes"));
    assert_eq!(scratch_dirs(), before, "per-turn scratch cwd removed");

    let missing = dir.path().join("absent");
    let d = driver("normal", PermissionPolicy::AllowAll).with_cwd(missing);
    assert!(
        d.complete(request("hi")).await.is_err(),
        "missing cwd is an error"
    );
}

fn scratch_dirs() -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(std::env::temp_dir())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("librefang-acp-"))
        .collect();
    v.sort();
    v
}

/// A required mode the agent does not offer fails the turn (fail closed).
#[tokio::test]
#[serial]
async fn missing_required_mode_fails_closed() {
    let d = driver("no_default_mode", PermissionPolicy::AllowAll);
    let err = d
        .complete(request("hi"))
        .await
        .expect_err("must not run in bypass mode");
    assert!(err.to_string().contains("session mode"), "{err}");
}

/// An agent that ignores session/cancel: the driver gives up after the grace period,
/// returns an error and kills the adapter.
#[tokio::test]
#[serial]
async fn agent_ignoring_cancel_is_killed_after_grace() {
    let d = driver("ignore_cancel", PermissionPolicy::AllowAll)
        .with_cancel_grace(std::time::Duration::from_millis(300));
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    let reader = tokio::spawn(async move {
        let first = rx.recv().await;
        drop(rx);
        first
    });
    let started = std::time::Instant::now();
    let err = d.stream(request("hi"), tx).await.expect_err("must fail");
    assert!(reader.await.unwrap().is_some());
    assert!(err.to_string().contains("did not acknowledge"), "{err}");
    assert!(started.elapsed() < std::time::Duration::from_secs(10));
    assert!(d.active_pids().is_empty());
}

fn pid_alive(pid: i32) -> bool {
    // SAFETY: signal 0 only checks for existence; no signal is delivered.
    unsafe { libc::kill(pid, 0) == 0 }
}

/// The message timeout kills the adapter's whole process group, grandchildren included.
#[tokio::test]
#[serial]
async fn timeout_kills_process_group_including_grandchild() {
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("grandchild.pid");
    let d = AcpClientDriver::new(AcpAgentKind::ClaudeCode, Some("python3".to_string()))
        .with_args(vec![
            fixture(),
            "grandchild_hang".to_string(),
            marker.to_string_lossy().into_owned(),
        ])
        .with_message_timeout(2);
    let (tx, mut rx) = tokio::sync::mpsc::channel(16);
    let drain = tokio::spawn(async move { while rx.recv().await.is_some() {} });
    let err = d
        .stream(request("hi"), tx)
        .await
        .expect_err("must time out");
    let _ = drain.await;
    assert!(
        matches!(
            err,
            librefang_llm_drivers::llm_driver::LlmError::TimedOut { .. }
        ),
        "{err}"
    );
    let pid: i32 = std::fs::read_to_string(&marker)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while pid_alive(pid) && std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert!(
        !pid_alive(pid),
        "grandchild {pid} survived the timeout kill"
    );
}

/// Fake test values only: the fake agent never sends them anywhere.
const GW_BASE: &str = "http://litellm-gateway.test:4000/anthropic";
const GW_HEADERS: &str = "x-litellm-api-key: sk-fake-test-value\nX-Lane: acp-test";

/// The live failure (layn321/librefang#1 @de93066): an adapter that only takes
/// gateway credentials through ACP `authenticate` refuses `session/new` with
/// "Authentication required". The driver must advertise gateway support, call
/// `authenticate` with the base URL and headers from its env, then open the session.
#[tokio::test]
#[serial]
async fn gateway_env_is_sent_through_authenticate_before_session_new() {
    clear_oauth_env();
    std::env::set_var("ANTHROPIC_BASE_URL", GW_BASE);
    std::env::set_var("ANTHROPIC_CUSTOM_HEADERS", GW_HEADERS);
    let d = driver("gateway", PermissionPolicy::AllowAll);
    let result = d.complete(request("hello gateway")).await;
    std::env::remove_var("ANTHROPIC_BASE_URL");
    std::env::remove_var("ANTHROPIC_CUSTOM_HEADERS");

    let response = result.expect("turn completes once authenticate has run");
    let text = response.text();
    assert!(text.contains("echo:hello gateway"), "{text}");
    assert!(
        text.contains(&format!("|gw_base:{GW_BASE}")),
        "base URL sent: {text}"
    );
    assert!(
        text.contains("|gw_headers:X-Lane,x-litellm-api-key"),
        "header names sent: {text}"
    );
    assert!(
        text.contains("|gw_match:yes"),
        "header values arrived intact: {text}"
    );
    assert!(
        !text.contains("sk-fake-test-value"),
        "value never echoed: {text}"
    );
    assert_eq!(response.stop_reason, StopReason::EndTurn);
}

/// Without gateway env there is nothing to authenticate with, so the gateway-only
/// adapter refuses the session and the driver reports it, not a silent empty turn.
#[tokio::test]
#[serial]
async fn gateway_only_agent_without_gateway_env_reports_auth_required() {
    clear_oauth_env();
    std::env::remove_var("ANTHROPIC_BASE_URL");
    std::env::remove_var("ANTHROPIC_CUSTOM_HEADERS");
    let d = driver("gateway", PermissionPolicy::AllowAll);
    let err = d
        .complete(request("hello"))
        .await
        .expect_err("session/new is refused");
    assert!(
        err.to_string().contains("Authentication required"),
        "auth error surfaced: {err}"
    );
}

/// A `MakeWriter` that appends everything into one shared buffer.
#[derive(Clone, Default)]
struct Capture(Arc<std::sync::Mutex<Vec<u8>>>);

impl std::io::Write for Capture {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Capture {
    type Writer = Capture;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

/// F9: the ACP crate logs whole outgoing JSON-RPC messages at debug/trace. With a
/// subscriber at TRACE for every target, neither the gateway header value sent in
/// `authenticate` nor the MCP bridge key sent in `session/new` may reach the log.
#[tokio::test]
#[serial]
async fn trace_logging_never_contains_gateway_or_mcp_secrets() {
    const MCP_KEY: &str = "mcp-fake-bridge-key";
    clear_oauth_env();
    let token_dir = tempfile::tempdir().unwrap();
    let token_file = token_dir.path().join("oauth_token");
    std::fs::write(&token_file, format!("{OAUTH_FAKE}\n")).unwrap();
    std::env::set_var("LIBREFANG_CLAUDE_OAUTH_TOKEN_FILE", &token_file);
    std::env::set_var("LIBREFANG_ACP_GATEWAY_AUTH", "authenticate");
    let capture = Capture::default();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_ansi(false)
        .with_writer(capture.clone())
        .finish();
    let _guard = tracing::subscriber::set_default(subscriber);

    std::env::set_var("ANTHROPIC_BASE_URL", GW_BASE);
    std::env::set_var("ANTHROPIC_CUSTOM_HEADERS", GW_HEADERS);
    let d = driver("gateway", PermissionPolicy::AllowAll).with_mcp_bridge(McpBridgeConfig {
        base_url: "http://127.0.0.1:4545".to_string(),
        api_key: Some(MCP_KEY.to_string()),
    });
    let mut req = request("hello trace");
    req.tools = Arc::new(vec![ToolDefinition {
        name: "noop".to_string(),
        description: "noop".to_string(),
        input_schema: serde_json::json!({"type": "object"}),
    }]);
    let result = d.complete(req).await;
    std::env::remove_var("ANTHROPIC_BASE_URL");
    std::env::remove_var("ANTHROPIC_CUSTOM_HEADERS");
    clear_oauth_env();
    drop(_guard);

    let response = result.expect("turn completes");
    assert!(
        response.text().contains("|gw_match:yes"),
        "gateway call ran"
    );
    let log = String::from_utf8(capture.0.lock().unwrap().clone()).unwrap();
    // The capture is live at TRACE: the driver's own debug line is there.
    assert!(
        log.contains("ACP agent initialized"),
        "capture saw nothing: {log}"
    );
    assert!(
        log.contains("authenticating ACP agent against the model gateway"),
        "gateway info line present"
    );
    assert!(
        !log.contains("sk-fake-test-value"),
        "gateway header value reached the log"
    );
    assert!(!log.contains(MCP_KEY), "MCP bridge key reached the log");
    assert!(!log.contains(OAUTH_FAKE), "OAuth token reached the log");
}

/// LOW from the r3 review: an adapter that does not offer `gateway` gets no
/// `authenticate` call; the driver warns and the turn still runs on env passthrough.
#[tokio::test]
#[serial]
async fn agent_without_gateway_method_falls_back_to_env() {
    clear_oauth_env();
    std::env::set_var("ANTHROPIC_BASE_URL", GW_BASE);
    std::env::set_var("ANTHROPIC_CUSTOM_HEADERS", GW_HEADERS);
    let d = driver("normal", PermissionPolicy::AllowAll);
    let result = d.complete(request("hello fallback")).await;
    std::env::remove_var("ANTHROPIC_BASE_URL");
    std::env::remove_var("ANTHROPIC_CUSTOM_HEADERS");

    let text = result.expect("turn completes without authenticate").text();
    assert!(text.contains("echo:hello fallback"), "{text}");
    assert!(
        text.contains(&format!("|base_url:{GW_BASE}")),
        "env passthrough still carries the base URL: {text}"
    );
    assert!(
        !text.contains("|gw_base:"),
        "no authenticate was sent to an agent that did not offer it: {text}"
    );
}

/// Fake subscription token; the fake agent only compares it, never prints it.
const OAUTH_FAKE: &str = "oauth-fake-test-token-0001";

fn clear_oauth_env() {
    for k in [
        "CLAUDE_CODE_OAUTH_TOKEN",
        "LIBREFANG_CLAUDE_OAUTH_TOKEN_FILE",
        "LIBREFANG_ACP_GATEWAY_AUTH",
    ] {
        std::env::remove_var(k);
    }
}

fn passthrough_driver(token_file: &std::path::Path) -> AcpClientDriver {
    AcpClientDriver::new(AcpAgentKind::ClaudeCode, Some("python3".to_string()))
        .with_args(vec![
            fixture(),
            "passthrough".to_string(),
            token_file.to_string_lossy().into_owned(),
        ])
        .with_permission_policy(PermissionPolicy::AllowAll)
        .with_message_timeout(30)
}

/// LiteLLM subscription passthrough (Tank's lab rounds 3-4): with a subscription
/// token present the default is `passthrough`. The adapter must get the real
/// `CLAUDE_CODE_OAUTH_TOKEN` (here loaded from the token file), the base URL and the
/// custom headers through env, and no `authenticate` call, which would make the
/// adapter swap the token for the `acp-proxy` placeholder. Also captured at TRACE.
#[tokio::test]
#[serial]
async fn passthrough_keeps_oauth_token_and_never_authenticates() {
    clear_oauth_env();
    let token_dir = tempfile::tempdir().unwrap();
    let token_file = token_dir.path().join("oauth_token");
    std::fs::write(&token_file, format!("{OAUTH_FAKE}\n")).unwrap();
    std::env::set_var("LIBREFANG_CLAUDE_OAUTH_TOKEN_FILE", &token_file);
    std::env::set_var("ANTHROPIC_BASE_URL", GW_BASE);
    std::env::set_var("ANTHROPIC_CUSTOM_HEADERS", GW_HEADERS);
    let capture = Capture::default();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_ansi(false)
        .with_writer(capture.clone())
        .finish();
    let guard = tracing::subscriber::set_default(subscriber);
    let result = passthrough_driver(&token_file)
        .complete(request("hello passthrough"))
        .await;
    drop(guard);
    std::env::remove_var("ANTHROPIC_BASE_URL");
    std::env::remove_var("ANTHROPIC_CUSTOM_HEADERS");
    clear_oauth_env();

    let text = result
        .expect("passthrough turn completes with no authenticate")
        .text();
    assert!(text.contains("echo:hello passthrough"), "{text}");
    assert!(
        text.contains("|oauth_match:yes"),
        "OAuth token reached the adapter: {text}"
    );
    assert!(
        text.contains("|placeholder:no"),
        "no acp-proxy placeholder: {text}"
    );
    assert!(text.contains(&format!("|base_url:{GW_BASE}")), "{text}");
    assert!(text.contains("|custom_headers_env:yes"), "{text}");
    assert!(
        !text.contains("|gw_base:"),
        "authenticate was not sent: {text}"
    );
    assert!(!text.contains(OAUTH_FAKE), "token never echoed: {text}");
    let log = String::from_utf8(capture.0.lock().unwrap().clone()).unwrap();
    assert!(log.contains("ACP gateway auth mode"), "capture saw nothing");
    assert!(!log.contains(OAUTH_FAKE), "OAuth token reached the log");
    assert!(
        !log.contains("sk-fake-test-value"),
        "gateway header reached the log"
    );
}

/// A token set directly in `CLAUDE_CODE_OAUTH_TOKEN` also selects passthrough and
/// passes the allowlist unchanged.
#[tokio::test]
#[serial]
async fn env_oauth_token_also_selects_passthrough() {
    clear_oauth_env();
    let token_dir = tempfile::tempdir().unwrap();
    let token_file = token_dir.path().join("expected");
    std::fs::write(&token_file, OAUTH_FAKE).unwrap();
    std::env::set_var("CLAUDE_CODE_OAUTH_TOKEN", OAUTH_FAKE);
    std::env::set_var("ANTHROPIC_BASE_URL", GW_BASE);
    std::env::set_var("ANTHROPIC_CUSTOM_HEADERS", GW_HEADERS);
    let result = passthrough_driver(&token_file)
        .complete(request("hello env token"))
        .await;
    std::env::remove_var("ANTHROPIC_BASE_URL");
    std::env::remove_var("ANTHROPIC_CUSTOM_HEADERS");
    clear_oauth_env();

    let text = result.expect("turn completes").text();
    assert!(text.contains("|oauth_match:yes"), "{text}");
    assert!(text.contains("|placeholder:no"), "{text}");
    assert!(!text.contains("|gw_base:"), "{text}");
}

/// An unreadable token file fails the turn before anything is spawned, naming the
/// path but no content.
#[tokio::test]
#[serial]
async fn missing_oauth_token_file_fails_closed() {
    clear_oauth_env();
    std::env::set_var(
        "LIBREFANG_CLAUDE_OAUTH_TOKEN_FILE",
        "/nonexistent/librefang-oauth-token",
    );
    let err = driver("normal", PermissionPolicy::AllowAll)
        .complete(request("hi"))
        .await
        .expect_err("must fail");
    clear_oauth_env();
    assert!(
        err.to_string()
            .contains("cannot read LIBREFANG_CLAUDE_OAUTH_TOKEN_FILE"),
        "{err}"
    );
}
