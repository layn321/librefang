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
    std::env::set_var("GROQ_API_KEY", "must-not-leak");
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
    std::env::remove_var("GROQ_API_KEY");

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
        "env filter stripped GROQ_API_KEY: {text}"
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
