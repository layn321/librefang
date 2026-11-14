//! ACP client driver: drives Claude Code and Codex as ACP agents.
//!
//! LibreFang acts as the **client** of the Agent Client Protocol and spawns an
//! adapter that speaks ACP over stdio and drives the real CLI:
//!
//! * `claude-code-acp` provider -> `claude-agent-acp` (`@agentclientprotocol/claude-agent-acp`,
//!   formerly `@zed-industries/claude-code-acp`), which runs the Claude Code binary through
//!   the Claude Agent SDK. `CLAUDE_CODE_EXECUTABLE` selects that binary.
//! * `codex-acp` provider -> `codex-acp` (`@agentclientprotocol/codex-acp`), which drives
//!   `codex app-server`.
//!
//! Neither path uses a CLI print / one-shot mode: the adapters run the real CLI as a
//! long-lived agent and LibreFang talks to it over ACP.
//!
//! Per request the driver runs `initialize` -> `session/new` (cwd + the LibreFang MCP
//! bridge, when tools are requested) -> `session/prompt`, streams `session/update`
//! notifications into [`StreamEvent`]s, answers `session/request_permission` from a
//! configurable [`PermissionPolicy`] (deny unless configured), and sends `session/cancel`
//! when the caller stops listening.
//!
//! The adapter is spawned with `tokio::process::Command` (not the crate's `AcpAgent`) so
//! the same env filter, process-group kill and PID tracking as the other CLI drivers
//! apply. Its pipes are handed to [`agent_client_protocol::ByteStreams`].

use crate::llm_driver::{CompletionRequest, CompletionResponse, LlmDriver, LlmError, StreamEvent};
use agent_client_protocol::schema::v1::{
    CancelNotification, ClientCapabilities, ContentBlock as AcpContentBlock,
    FileSystemCapabilities, HttpHeader, InitializeRequest, McpServer, McpServerHttp,
    NewSessionRequest, PermissionOption, PermissionOptionKind, PromptRequest,
    RequestPermissionOutcome, RequestPermissionRequest, RequestPermissionResponse,
    SelectedPermissionOutcome, SessionNotification, SessionUpdate, StopReason as AcpStopReason,
    TextContent, ToolKind, Usage as AcpUsage,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{Agent, ByteStreams, ConnectionTo};
use async_trait::async_trait;
use dashmap::DashMap;
use librefang_types::message::{ContentBlock, MessageContent, Role, StopReason, TokenUsage};
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};
use tracing::{debug, info, warn};

pub use crate::llm_driver::McpBridgeConfig;

/// Env var selecting the permission policy (see [`PermissionPolicy::parse`]).
pub const PERMISSION_POLICY_ENV: &str = "LIBREFANG_ACP_PERMISSION_POLICY";
/// Env var overriding the session working directory sent in `session/new`.
pub const CWD_ENV: &str = "LIBREFANG_ACP_CWD";

/// How long to wait for the agent to acknowledge `session/cancel` before the
/// process group is killed.
const CANCEL_GRACE: Duration = Duration::from_secs(5);

/// Which adapter this driver spawns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcpAgentKind {
    /// Claude Code through `claude-agent-acp`.
    ClaudeCode,
    /// Codex through `codex-acp`.
    Codex,
}

impl AcpAgentKind {
    /// Default adapter executable, looked up on PATH.
    pub fn default_command(self) -> &'static str {
        match self {
            Self::ClaudeCode => "claude-agent-acp",
            Self::Codex => "codex-acp",
        }
    }

    /// Env var overriding the adapter executable.
    pub fn path_env(self) -> &'static str {
        match self {
            Self::ClaudeCode => "LIBREFANG_CLAUDE_ACP_PATH",
            Self::Codex => "LIBREFANG_CODEX_ACP_PATH",
        }
    }

    /// Env var with extra whitespace-separated adapter arguments.
    pub fn args_env(self) -> &'static str {
        match self {
            Self::ClaudeCode => "LIBREFANG_CLAUDE_ACP_ARGS",
            Self::Codex => "LIBREFANG_CODEX_ACP_ARGS",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::ClaudeCode => "claude_code_acp",
            Self::Codex => "codex_acp",
        }
    }
}

// ── Permission policy ────────────────────────────────────────────

/// How `session/request_permission` is answered. There is no human in the loop,
/// so the default is [`PermissionPolicy::DenyAll`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum PermissionPolicy {
    /// Allow every request.
    AllowAll,
    /// Reject every request (the default).
    #[default]
    DenyAll,
    /// Allow only tool calls whose ACP `kind` is in the set (`read`, `edit`, `delete`,
    /// `move`, `search`, `execute`, `think`, `fetch`, `switch_mode`, `other`).
    AllowKinds(BTreeSet<String>),
}

/// The decision taken for one permission request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermissionDecision {
    /// Whether the policy allowed the tool call.
    pub allowed: bool,
    /// The option id sent back, or `None` when the outcome is `cancelled`
    /// (no option of the required kind was offered).
    pub option_id: Option<String>,
}

impl PermissionPolicy {
    /// Parse `allow-all`, `deny-all`, or `allow:<kind>[,<kind>...]`.
    pub fn parse(spec: &str) -> Result<Self, String> {
        let spec = spec.trim();
        match spec.to_ascii_lowercase().as_str() {
            "" | "deny" | "deny-all" | "deny_all" => return Ok(Self::DenyAll),
            "allow" | "allow-all" | "allow_all" => return Ok(Self::AllowAll),
            _ => {}
        }
        let Some(list) = spec.strip_prefix("allow:") else {
            return Err(format!(
                "unknown ACP permission policy {spec:?}; expected allow-all, deny-all or allow:<kind>,..."
            ));
        };
        let kinds: BTreeSet<String> = list
            .split(',')
            .map(|k| k.trim().to_ascii_lowercase())
            .filter(|k| !k.is_empty())
            .collect();
        if kinds.is_empty() {
            return Err("ACP permission allowlist is empty".to_string());
        }
        Ok(Self::AllowKinds(kinds))
    }

    /// Read [`PERMISSION_POLICY_ENV`]; unset or invalid falls back to deny-all.
    pub fn from_env() -> Self {
        match std::env::var(PERMISSION_POLICY_ENV) {
            Ok(spec) => Self::parse(&spec).unwrap_or_else(|e| {
                warn!(error = %e, "invalid {PERMISSION_POLICY_ENV}; denying all ACP permission requests");
                Self::DenyAll
            }),
            Err(_) => Self::DenyAll,
        }
    }

    fn allows(&self, kind: Option<&ToolKind>) -> bool {
        match self {
            Self::AllowAll => true,
            Self::DenyAll => false,
            Self::AllowKinds(kinds) => kind.is_some_and(|k| kinds.contains(tool_kind_name(k))),
        }
    }

    /// Decide one request: pick an allow option (once preferred) when allowed,
    /// otherwise a reject option (once preferred).
    pub fn decide(
        &self,
        kind: Option<&ToolKind>,
        options: &[PermissionOption],
    ) -> PermissionDecision {
        let allowed = self.allows(kind);
        let preference: [PermissionOptionKind; 2] = if allowed {
            [
                PermissionOptionKind::AllowOnce,
                PermissionOptionKind::AllowAlways,
            ]
        } else {
            [
                PermissionOptionKind::RejectOnce,
                PermissionOptionKind::RejectAlways,
            ]
        };
        let option_id = preference.iter().find_map(|want| {
            options
                .iter()
                .find(|o| &o.kind == want)
                .map(|o| o.option_id.to_string())
        });
        PermissionDecision { allowed, option_id }
    }
}

/// The wire name of an ACP tool kind.
pub fn tool_kind_name(kind: &ToolKind) -> &'static str {
    match kind {
        ToolKind::Read => "read",
        ToolKind::Edit => "edit",
        ToolKind::Delete => "delete",
        ToolKind::Move => "move",
        ToolKind::Search => "search",
        ToolKind::Execute => "execute",
        ToolKind::Think => "think",
        ToolKind::Fetch => "fetch",
        ToolKind::SwitchMode => "switch_mode",
        _ => "other",
    }
}

fn answer_permission(
    policy: &PermissionPolicy,
    agent: AcpAgentKind,
    request: &RequestPermissionRequest,
) -> RequestPermissionResponse {
    let kind = request.tool_call.fields.kind.as_ref();
    let decision = policy.decide(kind, &request.options);
    // Every decision is logged at info so an operator can audit what the agent was allowed to do.
    info!(
        agent = agent.label(),
        session_id = %request.session_id,
        tool_call_id = %request.tool_call.tool_call_id,
        tool_kind = kind.map(tool_kind_name).unwrap_or("unspecified"),
        title = request.tool_call.fields.title.as_deref().unwrap_or(""),
        allowed = decision.allowed,
        option_id = decision.option_id.as_deref().unwrap_or("<cancelled>"),
        "ACP permission request decided by policy"
    );
    let outcome = match decision.option_id {
        Some(id) => RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(id)),
        None => RequestPermissionOutcome::Cancelled,
    };
    RequestPermissionResponse::new(outcome)
}

// ── Message mapping ──────────────────────────────────────────────

/// Map one `session/update` to the LibreFang stream events it produces.
///
/// Tool calls run inside the agent, so they surface as `PhaseChange`, never as
/// `ToolUseStart` (which would make the LibreFang agent loop try to run them).
pub fn map_session_update(update: &SessionUpdate) -> Vec<StreamEvent> {
    match update {
        SessionUpdate::AgentMessageChunk(chunk) => content_text(&chunk.content)
            .map(|text| vec![StreamEvent::TextDelta { text }])
            .unwrap_or_default(),
        SessionUpdate::AgentThoughtChunk(chunk) => content_text(&chunk.content)
            .map(|text| vec![StreamEvent::ThinkingDelta { text }])
            .unwrap_or_default(),
        SessionUpdate::ToolCall(call) => vec![StreamEvent::PhaseChange {
            phase: "tool_use".to_string(),
            detail: Some(call.title.clone()),
        }],
        _ => Vec::new(),
    }
}

fn content_text(block: &AcpContentBlock) -> Option<String> {
    match block {
        AcpContentBlock::Text(t) if !t.text.is_empty() => Some(t.text.clone()),
        _ => None,
    }
}

/// Map an ACP stop reason. `Cancelled` maps to `EndTurn`; the driver itself
/// turns a cancelled turn into an error.
pub fn map_stop_reason(reason: &AcpStopReason) -> StopReason {
    match reason {
        AcpStopReason::MaxTokens | AcpStopReason::MaxTurnRequests => StopReason::MaxTokens,
        AcpStopReason::Refusal => StopReason::ContentFiltered,
        _ => StopReason::EndTurn,
    }
}

fn map_usage(usage: Option<&AcpUsage>) -> TokenUsage {
    let Some(u) = usage else {
        return TokenUsage::default();
    };
    TokenUsage {
        input_tokens: u.input_tokens,
        output_tokens: u.output_tokens,
        cache_read_input_tokens: u.cached_read_tokens.unwrap_or(0),
        cache_creation_input_tokens: u.cached_write_tokens.unwrap_or(0),
    }
}

/// Flatten the conversation into one text prompt. Each LibreFang call opens a fresh ACP
/// session, so history travels in the prompt, as it does for the claude_code driver.
pub fn build_prompt_text(request: &CompletionRequest) -> String {
    let mut parts = Vec::new();
    if let Some(ref sys) = request.system {
        parts.push(format!("[System]\n{sys}"));
    }
    for msg in request.messages.iter() {
        let role = match msg.role {
            Role::User => "User",
            Role::Assistant => "Assistant",
            Role::System => "System",
        };
        let text = match &msg.content {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text, .. } if !text.is_empty() => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        };
        if !text.is_empty() {
            parts.push(format!("[{role}]\n{text}"));
        }
    }
    parts.join("\n\n")
}

/// The LibreFang MCP bridge as an ACP `http` MCP server, with the same
/// per-connection identity headers the claude_code driver writes into `--mcp-config`.
fn bridge_mcp_server(bridge: &McpBridgeConfig, request: &CompletionRequest) -> McpServer {
    let url = format!("{}/mcp", bridge.base_url.trim_end_matches('/'));
    let mut headers = Vec::new();
    let mut push = |name: &str, value: Option<&str>| {
        if let Some(v) = value.filter(|v| !v.trim().is_empty()) {
            headers.push(HttpHeader::new(name, v));
        }
    };
    push("X-API-Key", bridge.api_key.as_deref());
    push("X-LibreFang-Agent-Id", request.agent_id.as_deref());
    push(
        "X-LibreFang-Current-Peer-Jid",
        request.sender_user_id.as_deref(),
    );
    push(
        "X-LibreFang-Current-Channel",
        request.sender_channel.as_deref(),
    );
    push(
        "X-LibreFang-Current-Chat-Id",
        request.sender_chat_id.as_deref(),
    );
    push(
        "X-LibreFang-Current-Account-Id",
        request.sender_account_id.as_deref(),
    );
    McpServer::Http(McpServerHttp::new("librefang", url).headers(headers))
}

// ── Driver ───────────────────────────────────────────────────────

/// Kills the adapter's process group when dropped (cancel, timeout, error).
struct ProcessGroupGuard {
    pid: Option<u32>,
    label: String,
    pids: Arc<DashMap<String, u32>>,
}

impl Drop for ProcessGroupGuard {
    fn drop(&mut self) {
        self.pids.remove(&self.label);
        crate::cli_process::kill_process_group(self.pid);
    }
}

/// What one prompt turn sends to the agent.
struct TurnInput {
    cwd: PathBuf,
    mcp_servers: Vec<McpServer>,
    prompt: String,
}

/// Outcome of one prompt turn.
struct TurnResult {
    stop_reason: AcpStopReason,
    usage: Option<AcpUsage>,
    cancel_sent: bool,
}

/// LLM driver that drives an ACP adapter (Claude Code or Codex) as an ACP client.
pub struct AcpClientDriver {
    kind: AcpAgentKind,
    command: String,
    args: Vec<String>,
    policy: PermissionPolicy,
    cwd: Option<PathBuf>,
    message_timeout_secs: u64,
    mcp_bridge: Option<McpBridgeConfig>,
    emit_caller_trace_headers: bool,
    active_pids: Arc<DashMap<String, u32>>,
}

impl AcpClientDriver {
    /// Build a driver. `command` (from the provider's `base_url`, as for the other CLI
    /// drivers) wins over the `LIBREFANG_*_ACP_PATH` env var, which wins over the default
    /// adapter name. The permission policy comes from [`PERMISSION_POLICY_ENV`].
    pub fn new(kind: AcpAgentKind, command: Option<String>) -> Self {
        let command = command
            .filter(|s| !s.trim().is_empty())
            .or_else(|| {
                std::env::var(kind.path_env())
                    .ok()
                    .filter(|s| !s.trim().is_empty())
            })
            .unwrap_or_else(|| kind.default_command().to_string());
        let args = std::env::var(kind.args_env())
            .map(|s| s.split_whitespace().map(str::to_string).collect())
            .unwrap_or_default();
        Self {
            kind,
            command,
            args,
            policy: PermissionPolicy::from_env(),
            cwd: std::env::var_os(CWD_ENV).map(PathBuf::from),
            message_timeout_secs: crate::cli_process::DEFAULT_MESSAGE_TIMEOUT_SECS,
            mcp_bridge: None,
            emit_caller_trace_headers: true,
            active_pids: Arc::new(DashMap::new()),
        }
    }

    /// Replace the adapter arguments.
    pub fn with_args(mut self, args: Vec<String>) -> Self {
        self.args = args;
        self
    }

    /// Set the permission policy explicitly.
    pub fn with_permission_policy(mut self, policy: PermissionPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Set the session working directory.
    pub fn with_cwd(mut self, cwd: PathBuf) -> Self {
        self.cwd = Some(cwd);
        self
    }

    /// Set the per-request timeout.
    pub fn with_message_timeout(mut self, secs: u64) -> Self {
        self.message_timeout_secs = secs;
        self
    }

    /// Expose LibreFang tools to the agent through the daemon's `/mcp` endpoint.
    pub fn with_mcp_bridge(mut self, bridge: McpBridgeConfig) -> Self {
        self.mcp_bridge = Some(bridge);
        self
    }

    /// Control the `LIBREFANG_AGENT_ID` / `SESSION_ID` / `STEP_ID` env vars.
    pub fn with_emit_caller_trace_headers(mut self, emit: bool) -> Self {
        self.emit_caller_trace_headers = emit;
        self
    }

    /// The adapter executable this driver spawns.
    pub fn command(&self) -> &str {
        &self.command
    }

    /// The configured permission policy.
    pub fn permission_policy(&self) -> &PermissionPolicy {
        &self.policy
    }

    /// Snapshot of running adapter PIDs.
    pub fn active_pids(&self) -> Vec<(String, u32)> {
        self.active_pids
            .iter()
            .map(|e| (e.key().clone(), *e.value()))
            .collect()
    }

    fn build_command(&self, request: &CompletionRequest) -> tokio::process::Command {
        let mut cmd = tokio::process::Command::new(&self.command);
        cmd.args(&self.args);
        match self.kind {
            AcpAgentKind::ClaudeCode => {
                // Same filter as the claude_code driver: strips other providers' keys and
                // secret-suffixed vars, keeps CLAUDE_* / ANTHROPIC_* (incl. ANTHROPIC_BASE_URL,
                // CLAUDE_CODE_EXECUTABLE) so model calls can go to the gateway.
                let filter = super::claude_code::ClaudeCodeDriver::new(None, false);
                filter.apply_env_filter(&mut cmd);
                super::claude_code::ClaudeCodeDriver::ensure_home_env(&mut cmd);
                if let Some(model) = model_override(&request.model, "claude-code-acp/") {
                    cmd.env("ANTHROPIC_MODEL", model);
                }
            }
            AcpAgentKind::Codex => {
                // Same filter as the codex_cli driver: keeps OPENAI_* / CODEX_*.
                super::codex_cli::CodexCliDriver::apply_env_filter(&mut cmd);
            }
        }
        if self.emit_caller_trace_headers {
            for (key, value) in [
                ("LIBREFANG_AGENT_ID", request.agent_id.as_deref()),
                ("LIBREFANG_SESSION_ID", request.session_id.as_deref()),
                ("LIBREFANG_STEP_ID", request.step_id.as_deref()),
            ] {
                if let Some(v) = value.filter(|v| !v.is_empty()) {
                    cmd.env(key, v);
                }
            }
        }
        cmd.stdin(std::process::Stdio::piped());
        cmd.stdout(std::process::Stdio::piped());
        // Adapter diagnostics go to the daemon's stderr; ACP itself is on stdout only.
        cmd.stderr(std::process::Stdio::inherit());
        cmd.kill_on_drop(true);
        crate::cli_process::set_process_group(&mut cmd);
        cmd
    }

    fn session_cwd(&self) -> PathBuf {
        self.cwd
            .clone()
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| PathBuf::from("/"))
    }
}

fn model_override(model: &str, prefix: &str) -> Option<String> {
    let stripped = model.strip_prefix(prefix).unwrap_or(model).trim();
    if stripped.is_empty() || stripped == prefix.trim_end_matches('/') {
        None
    } else {
        Some(stripped.to_string())
    }
}

/// Run one ACP turn over `transport`. Updates go to `events`; `cancel` fires
/// `session/cancel`.
async fn run_turn(
    transport: impl agent_client_protocol::ConnectTo<agent_client_protocol::Client> + 'static,
    kind: AcpAgentKind,
    policy: PermissionPolicy,
    input: TurnInput,
    events: tokio::sync::mpsc::UnboundedSender<SessionUpdate>,
    cancel: Arc<tokio::sync::Notify>,
) -> Result<TurnResult, agent_client_protocol::Error> {
    let TurnInput {
        cwd,
        mcp_servers,
        prompt,
    } = input;
    agent_client_protocol::Client
        .builder()
        .on_receive_notification(
            async move |notification: SessionNotification, _cx| {
                // Unbounded and non-blocking so the dispatch loop never stalls; the
                // driver applies caller backpressure on the other side.
                let _ = events.send(notification.update);
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .on_receive_request(
            async move |request: RequestPermissionRequest, responder, _cx| {
                responder.respond(answer_permission(&policy, kind, &request))
            },
            agent_client_protocol::on_receive_request!(),
        )
        .connect_with(transport, async move |connection: ConnectionTo<Agent>| {
            let init = connection
                .send_request(
                    InitializeRequest::new(ProtocolVersion::V1).client_capabilities(
                        // No fs / terminal delegation: the agent uses its own tools.
                        ClientCapabilities::new()
                            .fs(FileSystemCapabilities::new())
                            .terminal(false),
                    ),
                )
                .block_task()
                .await?;
            debug!(agent = kind.label(), info = ?init.agent_info, "ACP agent initialized");
            let mcp_servers = if !mcp_servers.is_empty() && !init.agent_capabilities.mcp_capabilities.http {
                warn!(
                    agent = kind.label(),
                    "ACP agent does not advertise HTTP MCP support; LibreFang tools not passed"
                );
                Vec::new()
            } else {
                mcp_servers
            };
            let session = connection
                .send_request(NewSessionRequest::new(cwd).mcp_servers(mcp_servers))
                .block_task()
                .await?;
            let session_id = session.session_id;
            let prompt_fut = connection
                .send_request(PromptRequest::new(
                    session_id.clone(),
                    vec![AcpContentBlock::Text(TextContent::new(prompt))],
                ))
                .block_task();
            tokio::pin!(prompt_fut);
            let mut cancel_sent = false;
            let response = tokio::select! {
                r = &mut prompt_fut => r?,
                _ = cancel.notified() => {
                    info!(agent = kind.label(), session_id = %session_id, "sending ACP session/cancel");
                    connection.send_notification(CancelNotification::new(session_id.clone()))?;
                    cancel_sent = true;
                    match tokio::time::timeout(CANCEL_GRACE, &mut prompt_fut).await {
                        Ok(r) => r?,
                        Err(_) => return Err(agent_client_protocol::util::internal_error(
                            "agent did not acknowledge session/cancel",
                        )),
                    }
                }
            };
            Ok(TurnResult {
                stop_reason: response.stop_reason,
                usage: response.usage,
                cancel_sent,
            })
        })
        .await
}

#[async_trait]
impl LlmDriver for AcpClientDriver {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let (tx, mut rx) = tokio::sync::mpsc::channel(64);
        let drain = tokio::spawn(async move { while rx.recv().await.is_some() {} });
        let result = self.stream(request, tx).await;
        let _ = drain.await;
        result
    }

    async fn stream(
        &self,
        request: CompletionRequest,
        tx: tokio::sync::mpsc::Sender<StreamEvent>,
    ) -> Result<CompletionResponse, LlmError> {
        let label = self.kind.label();
        let mcp_servers = match (&self.mcp_bridge, request.tools.is_empty()) {
            (Some(bridge), false) => vec![bridge_mcp_server(bridge, &request)],
            (None, false) => {
                warn!(
                    agent = label,
                    tool_count = request.tools.len(),
                    "ACP driver received tools but no MCP bridge is configured"
                );
                Vec::new()
            }
            _ => Vec::new(),
        };
        let prompt = build_prompt_text(&request);

        let mut cmd = self.build_command(&request);
        let mut child = cmd.spawn().map_err(|e| {
            LlmError::Http(format!(
                "ACP adapter '{}' not found or failed to start ({e}). Install it: {}",
                self.command,
                match self.kind {
                    AcpAgentKind::ClaudeCode =>
                        "npm install -g @agentclientprotocol/claude-agent-acp",
                    AcpAgentKind::Codex => "npm install -g @agentclientprotocol/codex-acp",
                }
            ))
        })?;
        let pid_label = format!("{}:{}", request.model, uuid::Uuid::new_v4());
        if let Some(pid) = child.id() {
            self.active_pids.insert(pid_label.clone(), pid);
        }
        let _guard = ProcessGroupGuard {
            pid: child.id(),
            label: pid_label,
            pids: Arc::clone(&self.active_pids),
        };
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| LlmError::Http("ACP adapter stdin unavailable".to_string()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| LlmError::Http("ACP adapter stdout unavailable".to_string()))?;
        let transport = ByteStreams::new(stdin.compat_write(), stdout.compat());

        let (ev_tx, mut ev_rx) = tokio::sync::mpsc::unbounded_channel();
        let cancel = Arc::new(tokio::sync::Notify::new());
        let turn = run_turn(
            transport,
            self.kind,
            self.policy.clone(),
            TurnInput {
                cwd: self.session_cwd(),
                mcp_servers,
                prompt,
            },
            ev_tx,
            Arc::clone(&cancel),
        );
        tokio::pin!(turn);
        let deadline = tokio::time::sleep(Duration::from_secs(self.message_timeout_secs));
        tokio::pin!(deadline);

        let mut text = String::new();
        let mut thinking = String::new();
        let mut receiver_gone = false;
        let mut forward = async |update: SessionUpdate, receiver_gone: &mut bool| {
            for event in map_session_update(&update) {
                match &event {
                    StreamEvent::TextDelta { text: t } => text.push_str(t),
                    StreamEvent::ThinkingDelta { text: t } => thinking.push_str(t),
                    _ => {}
                }
                if !*receiver_gone && tx.send(event).await.is_err() {
                    *receiver_gone = true;
                    cancel.notify_one();
                }
            }
        };

        let turn_result = loop {
            tokio::select! {
                biased;
                Some(update) = ev_rx.recv() => forward(update, &mut receiver_gone).await,
                // The caller stopped listening while the agent is idle: cancel now rather
                // than waiting for the next update to fail to send.
                _ = tx.closed(), if !receiver_gone => {
                    receiver_gone = true;
                    cancel.notify_one();
                }
                result = &mut turn => break result,
                _ = &mut deadline => {
                    warn!(agent = label, timeout_secs = self.message_timeout_secs, "ACP turn timed out; killing adapter");
                    return Err(crate::cli_process::timeout_error_with_partial(
                        self.message_timeout_secs,
                        label,
                        text,
                    ));
                }
            }
        };
        // Updates that arrived before the prompt response.
        while let Ok(update) = ev_rx.try_recv() {
            forward(update, &mut receiver_gone).await;
        }

        let turn = turn_result.map_err(|e| LlmError::Http(format!("ACP {label} error: {e}")))?;
        if receiver_gone || turn.cancel_sent || matches!(turn.stop_reason, AcpStopReason::Cancelled)
        {
            return Err(LlmError::Http(format!(
                "ACP {label} turn cancelled (stream receiver dropped)"
            )));
        }
        let stop_reason = map_stop_reason(&turn.stop_reason);
        let usage = map_usage(turn.usage.as_ref());
        if tx
            .send(StreamEvent::ContentComplete { stop_reason, usage })
            .await
            .is_err()
        {
            return Err(LlmError::Http("stream receiver dropped".to_string()));
        }
        let mut content = Vec::new();
        if !thinking.is_empty() {
            debug!(
                agent = label,
                thinking_chars = thinking.len(),
                "ACP agent thought chunks received"
            );
        }
        content.push(ContentBlock::Text {
            text,
            provider_metadata: None,
        });
        Ok(CompletionResponse {
            text_synthesized_from_thinking: false,
            content,
            stop_reason,
            tool_calls: Vec::new(),
            usage,
            actual_provider: None,
            actual_model: None,
        })
    }

    fn family(&self) -> crate::llm_driver::LlmFamily {
        match self.kind {
            AcpAgentKind::ClaudeCode => crate::llm_driver::LlmFamily::Anthropic,
            AcpAgentKind::Codex => crate::llm_driver::LlmFamily::OpenAi,
        }
    }

    fn is_coding_agent(&self) -> bool {
        true
    }
}

/// Whether the adapter for `kind` resolves on PATH (or via its env override).
pub fn acp_adapter_available(kind: AcpAgentKind) -> bool {
    let command = std::env::var(kind.path_env())
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| kind.default_command().to_string());
    let path = std::path::Path::new(&command);
    if path.components().count() > 1 {
        return path.is_file();
    }
    std::env::var_os("PATH")
        .is_some_and(|paths| std::env::split_paths(&paths).any(|dir| dir.join(&command).is_file()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_client_protocol::schema::v1::{ContentChunk, ToolCall};

    fn option(id: &str, kind: PermissionOptionKind) -> PermissionOption {
        PermissionOption::new(id.to_string(), id.to_string(), kind)
    }

    fn standard_options() -> Vec<PermissionOption> {
        vec![
            option("allow", PermissionOptionKind::AllowOnce),
            option("always", PermissionOptionKind::AllowAlways),
            option("reject", PermissionOptionKind::RejectOnce),
        ]
    }

    #[test]
    fn policy_parse_accepts_known_forms() {
        assert_eq!(
            PermissionPolicy::parse("allow-all").unwrap(),
            PermissionPolicy::AllowAll
        );
        assert_eq!(
            PermissionPolicy::parse("deny-all").unwrap(),
            PermissionPolicy::DenyAll
        );
        assert_eq!(
            PermissionPolicy::parse("").unwrap(),
            PermissionPolicy::DenyAll
        );
        assert_eq!(
            PermissionPolicy::parse("allow: Read, search").unwrap(),
            PermissionPolicy::AllowKinds(["read".to_string(), "search".to_string()].into())
        );
        assert!(PermissionPolicy::parse("allow:").is_err());
        assert!(PermissionPolicy::parse("yolo").is_err());
    }

    #[test]
    fn default_policy_is_deny() {
        assert_eq!(PermissionPolicy::default(), PermissionPolicy::DenyAll);
    }

    #[test]
    fn deny_all_picks_reject_option() {
        let d = PermissionPolicy::DenyAll.decide(Some(&ToolKind::Read), &standard_options());
        assert!(!d.allowed);
        assert_eq!(d.option_id.as_deref(), Some("reject"));
    }

    #[test]
    fn allow_all_prefers_allow_once() {
        let d = PermissionPolicy::AllowAll.decide(Some(&ToolKind::Execute), &standard_options());
        assert!(d.allowed);
        assert_eq!(d.option_id.as_deref(), Some("allow"));
    }

    #[test]
    fn allowlist_matches_kind_only() {
        let policy = PermissionPolicy::parse("allow:read,search").unwrap();
        assert!(
            policy
                .decide(Some(&ToolKind::Read), &standard_options())
                .allowed
        );
        assert!(
            !policy
                .decide(Some(&ToolKind::Edit), &standard_options())
                .allowed
        );
        // A request with no kind is never on an allowlist.
        assert!(!policy.decide(None, &standard_options()).allowed);
    }

    #[test]
    fn deny_without_reject_option_cancels() {
        let only_allow = vec![option("allow", PermissionOptionKind::AllowOnce)];
        let d = PermissionPolicy::DenyAll.decide(Some(&ToolKind::Edit), &only_allow);
        assert_eq!(
            d.option_id, None,
            "must never pick an allow option when denying"
        );
    }

    #[test]
    fn maps_message_and_thought_chunks() {
        let text = SessionUpdate::AgentMessageChunk(ContentChunk::new(AcpContentBlock::Text(
            TextContent::new("hi"),
        )));
        assert!(matches!(
            map_session_update(&text).as_slice(),
            [StreamEvent::TextDelta { text }] if text == "hi"
        ));
        let thought = SessionUpdate::AgentThoughtChunk(ContentChunk::new(AcpContentBlock::Text(
            TextContent::new("hmm"),
        )));
        assert!(matches!(
            map_session_update(&thought).as_slice(),
            [StreamEvent::ThinkingDelta { text }] if text == "hmm"
        ));
    }

    #[test]
    fn tool_call_maps_to_phase_change_not_tool_use() {
        let call = SessionUpdate::ToolCall(ToolCall::new("t1", "Edit file"));
        assert!(matches!(
            map_session_update(&call).as_slice(),
            [StreamEvent::PhaseChange { phase, detail: Some(d) }] if phase == "tool_use" && d == "Edit file"
        ));
    }

    #[test]
    fn maps_stop_reasons() {
        assert_eq!(
            map_stop_reason(&AcpStopReason::EndTurn),
            StopReason::EndTurn
        );
        assert_eq!(
            map_stop_reason(&AcpStopReason::MaxTokens),
            StopReason::MaxTokens
        );
        assert_eq!(
            map_stop_reason(&AcpStopReason::MaxTurnRequests),
            StopReason::MaxTokens
        );
        assert_eq!(
            map_stop_reason(&AcpStopReason::Refusal),
            StopReason::ContentFiltered
        );
    }

    #[test]
    fn model_override_strips_prefix() {
        assert_eq!(
            model_override("claude-code-acp/opus", "claude-code-acp/"),
            Some("opus".into())
        );
        assert_eq!(model_override("claude-code-acp", "claude-code-acp/"), None);
        assert_eq!(model_override("", "claude-code-acp/"), None);
    }
}
