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
    AuthCapabilities, AuthMethod, AuthenticateRequest, CancelNotification, ClientCapabilities,
    ContentBlock as AcpContentBlock, FileSystemCapabilities, HttpHeader, InitializeRequest,
    McpServer, McpServerHttp, Meta, NewSessionRequest, PermissionOption, PermissionOptionKind,
    PromptRequest, RequestPermissionOutcome, RequestPermissionRequest, RequestPermissionResponse,
    SelectedPermissionOutcome, SessionModeState, SessionNotification, SessionUpdate,
    SetSessionModeRequest, StopReason as AcpStopReason, TextContent, ToolKind, Usage as AcpUsage,
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
/// Env var setting the session working directory sent in `session/new`. Unset means a
/// fresh, empty scratch directory per turn (removed afterwards), never the daemon's cwd.
pub const CWD_ENV: &str = "LIBREFANG_ACP_CWD";
/// Env var setting the ACP session mode the driver enforces after `session/new`.
/// Unset uses [`AcpAgentKind::default_session_mode`]; an empty value keeps the agent's own mode.
pub const SESSION_MODE_ENV: &str = "LIBREFANG_ACP_SESSION_MODE";

/// How a configured Anthropic gateway is reached: `authenticate` or `passthrough`.
pub const GATEWAY_AUTH_ENV: &str = "LIBREFANG_ACP_GATEWAY_AUTH";

/// File holding a Claude subscription OAuth token, read at spawn time into the
/// adapter's `CLAUDE_CODE_OAUTH_TOKEN` (the way lab workers mount it).
pub const OAUTH_TOKEN_FILE_ENV: &str = "LIBREFANG_CLAUDE_OAUTH_TOKEN_FILE";

/// The CLI's own subscription-token variable.
const OAUTH_TOKEN_VAR: &str = "CLAUDE_CODE_OAUTH_TOKEN";

/// How long to wait for the agent to acknowledge `session/cancel` before the
/// process group is killed.
const DEFAULT_CANCEL_GRACE: Duration = Duration::from_secs(5);

/// Non-secret variables every adapter needs to start (Node, TLS, proxies, locale).
const BASE_ENV_ALLOWLIST: &[&str] = &[
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "TMPDIR",
    "TERM",
    "LANG",
    "LANGUAGE",
    "TZ",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
    "XDG_RUNTIME_DIR",
    "NODE_EXTRA_CA_CERTS",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "all_proxy",
    // Windows process basics.
    "SYSTEMROOT",
    "SystemRoot",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "PATHEXT",
    "COMSPEC",
    "TEMP",
    "TMP",
];

/// Substrings that keep a variable out even when its prefix is allowed: nothing in the
/// adapter's environment may switch a CLI into a permission-bypass mode.
const BYPASS_ENV_MARKERS: &[&str] = &["DANGEROUSLY", "SKIP_PERMISSIONS", "BYPASS_PERMISSIONS"];

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

    /// Provider-prefixed variables this adapter receives (credentials, base URLs, the
    /// CLI's own settings such as `CLAUDE_CODE_EXECUTABLE`).
    pub fn env_prefixes(self) -> &'static [&'static str] {
        match self {
            Self::ClaudeCode => &["CLAUDE_", "ANTHROPIC_"],
            Self::Codex => &["OPENAI_", "CODEX_"],
        }
    }

    /// The ACP session mode enforced after `session/new` unless configured otherwise.
    ///
    /// Claude: `default`, the mode in which Claude Code asks before acting
    /// (`claude-agent-acp` 0.84.0 `dist/permissions/modes.js` maps `default`/`manual` to it).
    /// Codex: none, because `codex-acp`'s mode ids have not been verified here.
    pub fn default_session_mode(self) -> Option<&'static str> {
        match self {
            Self::ClaudeCode => Some("default"),
            Self::Codex => None,
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

// ── Adapter environment ──────────────────────────────────────────

/// Whether `key` may be passed to the adapter for `kind`.
pub fn adapter_env_key_allowed(kind: AcpAgentKind, key: &str) -> bool {
    let upper = key.to_ascii_uppercase();
    if BYPASS_ENV_MARKERS.iter().any(|m| upper.contains(m)) {
        return false;
    }
    BASE_ENV_ALLOWLIST.contains(&key)
        || key.starts_with("LC_")
        || kind.env_prefixes().iter().any(|p| key.starts_with(p))
}

/// The adapter environment: an allowlist, not a denylist. Only the base variables in
/// [`BASE_ENV_ALLOWLIST`], `LC_*`, and the kind's own prefixes
/// ([`AcpAgentKind::env_prefixes`]) pass; everything else (unlisted `*_API_KEY`,
/// `AWS_SECRET_ACCESS_KEY`, other providers' keys, tokens) is dropped.
pub fn allowed_adapter_env(
    kind: AcpAgentKind,
    vars: impl IntoIterator<Item = (String, String)>,
) -> Vec<(String, String)> {
    vars.into_iter()
        .filter(|(k, _)| adapter_env_key_allowed(kind, k))
        .collect()
}

/// The adapter executable: configured value (provider `base_url`) first, then the
/// `LIBREFANG_*_ACP_PATH` env var, then the default name on PATH.
pub fn resolve_adapter_command(kind: AcpAgentKind, configured: Option<&str>) -> String {
    configured
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .or_else(|| {
            std::env::var(kind.path_env())
                .ok()
                .filter(|s| !s.trim().is_empty())
        })
        .unwrap_or_else(|| kind.default_command().to_string())
}

/// `_meta` for `session/new`. For Claude it sets
/// `claudeCode.options.allowDangerouslySkipPermissions = false`, which `claude-agent-acp`
/// 0.84.0 reads to turn bypass off for the session (`dist/acp-agent.js:6486-6488`); a
/// `bypassPermissions` default mode then clamps to `default` (`dist/permissions/modes.js`
/// `resolvePermissionMode`) and the SDK is spawned with `allowDangerouslySkipPermissions:
/// false` (`dist/acp-agent.js:6616`).
fn session_meta(kind: AcpAgentKind) -> Option<Meta> {
    match kind {
        AcpAgentKind::ClaudeCode => {
            let value = serde_json::json!({
                "claudeCode": { "options": { "allowDangerouslySkipPermissions": false } }
            });
            value.as_object().cloned()
        }
        AcpAgentKind::Codex => None,
    }
}

/// ACP auth method id `claude-agent-acp` offers for an Anthropic-protocol gateway
/// (`dist/acp-agent.js:1151-1160`), advertised only when the client sets
/// `clientCapabilities.auth._meta.gateway = true` (`:1150`, `:1288-1291`).
const GATEWAY_AUTH_METHOD: &str = "gateway";

/// A model gateway the Claude adapter is told about through ACP `authenticate`.
///
/// Built from the same `ANTHROPIC_*` variables an operator already sets for the
/// CLI. Header values are credentials: `Debug` prints header names only, and
/// nothing in this module logs a value.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct GatewayAuth {
    base_url: String,
    headers: std::collections::BTreeMap<String, String>,
}

impl std::fmt::Debug for GatewayAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GatewayAuth")
            .field("base_url", &self.base_url)
            .field("header_names", &self.headers.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl GatewayAuth {
    /// Gateway config from an environment, or `None` when no gateway is set.
    ///
    /// Needs `ANTHROPIC_BASE_URL` plus at least one credential header. Headers come
    /// from `ANTHROPIC_CUSTOM_HEADERS` (newline-separated `Name: Value`, the CLI's own
    /// format). `ANTHROPIC_AUTH_TOKEN` and `ANTHROPIC_API_KEY` become `Authorization`
    /// and `x-api-key` unless a custom header already sets them, because the adapter's
    /// gateway route blanks both variables for the CLI (`dist/acp-agent.js:7299-7311`,
    /// `:7343-7347`) and would otherwise drop them.
    pub(crate) fn from_env(
        kind: AcpAgentKind,
        vars: impl IntoIterator<Item = (String, String)>,
    ) -> Option<Self> {
        if kind != AcpAgentKind::ClaudeCode {
            return None;
        }
        let vars: std::collections::HashMap<String, String> = vars.into_iter().collect();
        let base_url = vars.get("ANTHROPIC_BASE_URL")?.trim().to_string();
        if base_url.is_empty() {
            return None;
        }
        let mut headers = std::collections::BTreeMap::new();
        if let Some(raw) = vars.get("ANTHROPIC_CUSTOM_HEADERS") {
            for line in raw.lines() {
                let Some((name, value)) = line.split_once(':') else {
                    continue;
                };
                let name = name.trim();
                if !name.is_empty() {
                    headers.insert(name.to_string(), value.trim().to_string());
                }
            }
        }
        let has = |headers: &std::collections::BTreeMap<String, String>, wanted: &str| {
            headers.keys().any(|k| k.eq_ignore_ascii_case(wanted))
        };
        if let Some(token) = vars.get("ANTHROPIC_AUTH_TOKEN").filter(|v| !v.is_empty()) {
            if !has(&headers, "authorization") {
                headers.insert("Authorization".to_string(), format!("Bearer {token}"));
            }
        }
        if let Some(key) = vars.get("ANTHROPIC_API_KEY").filter(|v| !v.is_empty()) {
            if !has(&headers, "x-api-key") {
                headers.insert("x-api-key".to_string(), key.clone());
            }
        }
        if headers.is_empty() {
            return None;
        }
        Some(Self { base_url, headers })
    }

    /// `authenticate` params `claude-agent-acp` reads (`dist/acp-agent.js:1401-1413`).
    fn authenticate_meta(&self) -> Option<Meta> {
        serde_json::json!({ "gateway": { "baseUrl": self.base_url, "headers": self.headers } })
            .as_object()
            .cloned()
    }
}

/// How the driver gets gateway credentials to the Claude CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatewayAuthMode {
    /// Send the gateway URL and headers through ACP `authenticate`. The adapter
    /// then sets a placeholder `ANTHROPIC_AUTH_TOKEN` and blanks any OAuth token
    /// (`dist/acp-agent.js:7299-7311`, `:7343-7347`): fits a key-based gateway.
    Authenticate,
    /// Never advertise or call `authenticate`; the CLI reads `ANTHROPIC_BASE_URL`,
    /// `ANTHROPIC_CUSTOM_HEADERS` and `CLAUDE_CODE_OAUTH_TOKEN` from its env and
    /// sends the subscription token itself: fits a subscription passthrough gateway.
    Passthrough,
}

impl GatewayAuthMode {
    /// Explicit setting wins; unset defaults to `passthrough` when a subscription
    /// OAuth token is present (so it is never replaced by a placeholder), else
    /// `authenticate`.
    pub fn resolve(raw: Option<&str>, oauth_token_present: bool) -> Result<Self, String> {
        match raw.map(str::trim).filter(|v| !v.is_empty()) {
            None if oauth_token_present => Ok(Self::Passthrough),
            None => Ok(Self::Authenticate),
            Some("authenticate") => Ok(Self::Authenticate),
            Some("passthrough") => Ok(Self::Passthrough),
            // The value is not echoed: someone may have pasted a secret into it.
            Some(_) => Err(format!(
                "{GATEWAY_AUTH_ENV} must be `authenticate` or `passthrough`"
            )),
        }
    }
}

/// Largest token file accepted; real OAuth tokens are about 100 bytes.
const OAUTH_TOKEN_FILE_MAX: u64 = 16 * 1024;

/// Read a subscription token file. Errors name the path, never the content.
///
/// Symlinks are followed (Kubernetes secret mounts are symlinks), but the target
/// must be a regular file of at most [`OAUTH_TOKEN_FILE_MAX`] bytes, so a FIFO,
/// device or endless file cannot block or exhaust the daemon. Reads are async.
async fn read_oauth_token_file(path: &str) -> Result<String, LlmError> {
    match tokio::time::timeout(OAUTH_TOKEN_READ_TIMEOUT, read_oauth_token_file_inner(path)).await {
        Ok(result) => result,
        Err(_) => Err(LlmError::Http(format!(
            "{OAUTH_TOKEN_FILE_ENV} file '{path}': read timed out after {}s",
            OAUTH_TOKEN_READ_TIMEOUT.as_secs()
        ))),
    }
}

/// Upper bound on reading the token file (a hung network mount, a swapped FIFO).
const OAUTH_TOKEN_READ_TIMEOUT: Duration = Duration::from_secs(2);

async fn read_oauth_token_file_inner(path: &str) -> Result<String, LlmError> {
    use tokio::io::AsyncReadExt;
    let fail = |why: String| LlmError::Http(format!("{OAUTH_TOKEN_FILE_ENV} file '{path}': {why}"));
    // Checked before open: opening a FIFO for reading blocks until a writer appears.
    let meta = tokio::fs::metadata(path)
        .await
        .map_err(|e| fail(format!("cannot read ({e})")))?;
    if !meta.is_file() {
        return Err(fail("not a regular file".to_string()));
    }
    if meta.len() > OAUTH_TOKEN_FILE_MAX {
        return Err(fail(format!("larger than {OAUTH_TOKEN_FILE_MAX} bytes")));
    }
    let mut options = tokio::fs::OpenOptions::new();
    options.read(true);
    // Non-blocking open: if the path became a FIFO after the stat, open returns
    // at once instead of waiting for a writer; the re-check below then refuses it.
    #[cfg(unix)]
    options.custom_flags(libc::O_NONBLOCK);
    let file = options
        .open(path)
        .await
        .map_err(|e| fail(format!("cannot read ({e})")))?;
    // Re-check the opened file, in case the path was swapped after the stat.
    let opened = file
        .metadata()
        .await
        .map_err(|e| fail(format!("cannot read ({e})")))?;
    if !opened.is_file() {
        return Err(fail("not a regular file".to_string()));
    }
    let mut raw = Vec::new();
    file.take(OAUTH_TOKEN_FILE_MAX + 1)
        .read_to_end(&mut raw)
        .await
        .map_err(|e| fail(format!("cannot read ({e})")))?;
    if raw.len() as u64 > OAUTH_TOKEN_FILE_MAX {
        return Err(fail(format!("larger than {OAUTH_TOKEN_FILE_MAX} bytes")));
    }
    let raw = String::from_utf8(raw).map_err(|_| fail("not valid UTF-8".to_string()))?;
    let token = raw.trim();
    if token.is_empty() {
        return Err(fail("is empty".to_string()));
    }
    Ok(token.to_string())
}

/// Subscription passthrough with a token: make the token the CLI's only credential
/// and keep it out of the agent's tool subprocesses.
///
/// - Claude Code picks `ANTHROPIC_AUTH_TOKEN` before `CLAUDE_CODE_OAUTH_TOKEN` when it
///   resolves its auth source, so a leftover one would replace the subscription token;
///   `ANTHROPIC_API_KEY` goes through a separate key path. Both are removed.
/// - Claude Code deletes `CLAUDE_CODE_OAUTH_TOKEN`, `ANTHROPIC_API_KEY`,
///   `ANTHROPIC_AUTH_TOKEN` and similar from tool subprocess env only when
///   `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB` is on (off by default outside GitHub
///   Actions). It is turned on here unless the operator set it. When on, Claude
///   Code also forces the `default` permission mode.
///
/// Returns whether scrubbing ends up on.
fn harden_passthrough_env(cmd: &mut tokio::process::Command) -> bool {
    cmd.env_remove("ANTHROPIC_API_KEY");
    cmd.env_remove("ANTHROPIC_AUTH_TOKEN");
    match std::env::var(SUBPROCESS_ENV_SCRUB_VAR) {
        // Claude Code's own truthy set (`1`, `true`, `yes`, `on`).
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => {
            cmd.env(SUBPROCESS_ENV_SCRUB_VAR, "1");
            true
        }
    }
}

/// Whether a configured session mode will be overridden by the subprocess scrub,
/// which makes Claude Code force its `default` permission mode.
fn scrub_overrides_session_mode(scrub_on: bool, session_mode: Option<&str>) -> bool {
    scrub_on && session_mode.is_some_and(|m| !m.is_empty() && m != "default")
}

/// Claude Code's switch for scrubbing credentials from tool subprocesses.
const SUBPROCESS_ENV_SCRUB_VAR: &str = "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB";

/// Client capabilities for `initialize`. A gateway turns on
/// `auth._meta.gateway` so the Claude adapter offers its gateway auth method.
fn client_capabilities(gateway: Option<&GatewayAuth>) -> ClientCapabilities {
    // No fs / terminal delegation: the agent uses its own tools.
    let caps = ClientCapabilities::new()
        .fs(FileSystemCapabilities::new())
        .terminal(false);
    match gateway {
        Some(_) => caps.auth(
            AuthCapabilities::new()
                .meta(serde_json::json!({ "gateway": true }).as_object().cloned()),
        ),
        None => caps,
    }
}

/// Whether the agent advertised `id` as an `authenticate`-able method.
fn advertises_auth_method(methods: &[AuthMethod], id: &str) -> bool {
    methods
        .iter()
        .any(|m| matches!(m, AuthMethod::Agent(a) if &*a.id.0 == id))
}

fn auth_method_ids(methods: &[AuthMethod]) -> Vec<String> {
    methods
        .iter()
        .map(|m| match m {
            AuthMethod::Agent(a) => a.id.0.to_string(),
            AuthMethod::Terminal(t) => format!("terminal:{}", t.id.0),
            _ => "unknown".to_string(),
        })
        .collect()
}

/// Tracing target prefix of `agent-client-protocol` (and its schema crate).
const ACP_CRATE_TARGET: &str = "agent_client_protocol";

/// Whether a callsite belongs to the ACP crate and is more verbose than `info`.
///
/// `agent-client-protocol` 3.3.0 logs whole outgoing messages:
/// `jsonrpc/outgoing_actor.rs:15` and `:37` (`debug!(?message)`) and
/// `jsonrpc/transport_actor.rs:135`/`:145` (`trace!` of the JSON line). Those
/// messages carry the gateway headers (`authenticate`) and the MCP bridge
/// `X-API-Key` (`session/new`).
fn acp_crate_capped(meta: &tracing::Metadata<'_>) -> bool {
    meta.target().starts_with(ACP_CRATE_TARGET) && *meta.level() > tracing::Level::INFO
}

/// Forwards everything to the subscriber that was current when it was built,
/// except ACP-crate callsites above `info`, which it reports as disabled.
struct CapAcpCrateLogs {
    inner: tracing::Dispatch,
    max_level: tracing::level_filters::LevelFilter,
}

impl tracing::Subscriber for CapAcpCrateLogs {
    fn register_callsite(
        &self,
        meta: &'static tracing::Metadata<'static>,
    ) -> tracing::subscriber::Interest {
        if acp_crate_capped(meta) {
            // `sometimes`, not `never`: interest is cached per callsite across all
            // dispatchers, so `never` would also silence it outside this turn.
            tracing::subscriber::Interest::sometimes()
        } else {
            self.inner.register_callsite(meta)
        }
    }
    fn enabled(&self, meta: &tracing::Metadata<'_>) -> bool {
        !acp_crate_capped(meta) && self.inner.enabled(meta)
    }
    fn max_level_hint(&self) -> Option<tracing::level_filters::LevelFilter> {
        Some(self.max_level)
    }
    fn new_span(&self, span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        self.inner.new_span(span)
    }
    fn record(&self, span: &tracing::span::Id, values: &tracing::span::Record<'_>) {
        self.inner.record(span, values)
    }
    fn record_follows_from(&self, span: &tracing::span::Id, follows: &tracing::span::Id) {
        self.inner.record_follows_from(span, follows)
    }
    fn event(&self, event: &tracing::Event<'_>) {
        if !acp_crate_capped(event.metadata()) {
            self.inner.event(event)
        }
    }
    fn enter(&self, span: &tracing::span::Id) {
        self.inner.enter(span)
    }
    fn exit(&self, span: &tracing::span::Id) {
        self.inner.exit(span)
    }
    fn clone_span(&self, id: &tracing::span::Id) -> tracing::span::Id {
        self.inner.clone_span(id)
    }
    fn try_close(&self, id: tracing::span::Id) -> bool {
        self.inner.try_close(id)
    }
    fn current_span(&self) -> tracing_core::span::Current {
        self.inner.current_span()
    }
}

/// A dispatcher for one ACP turn: the current one, with the ACP crate capped at `info`.
fn acp_capped_dispatch() -> tracing::Dispatch {
    tracing::Dispatch::new(CapAcpCrateLogs {
        inner: tracing::dispatcher::get_default(|d| d.clone()),
        // Keep the process-wide static max level as it is rather than raising it.
        max_level: tracing::level_filters::LevelFilter::current(),
    })
}

/// What to do about the session mode after `session/new`.
#[derive(Debug, PartialEq, Eq)]
enum ModeAction {
    /// Already in the required mode, or no mode is required.
    Keep,
    /// Switch with `session/set_mode`.
    Set,
    /// The agent reports no modes; nothing to enforce (logged).
    AgentHasNoModes,
}

/// Decide how to reach `required` from the agent's reported mode state. Fails closed:
/// a required mode the agent does not offer is an error, not a silent pass.
fn plan_session_mode(
    required: Option<&str>,
    modes: Option<&SessionModeState>,
) -> Result<ModeAction, String> {
    let Some(required) = required else {
        return Ok(ModeAction::Keep);
    };
    let Some(state) = modes else {
        return Ok(ModeAction::AgentHasNoModes);
    };
    if state.current_mode_id.to_string() == required {
        return Ok(ModeAction::Keep);
    }
    if state
        .available_modes
        .iter()
        .any(|m| m.id.to_string() == required)
    {
        return Ok(ModeAction::Set);
    }
    Err(format!(
        "ACP agent does not offer required session mode {required:?} (current {:?})",
        state.current_mode_id.to_string()
    ))
}

/// Removes the per-turn scratch working directory.
struct ScratchDir(PathBuf);

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
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
    session_mode: Option<String>,
    cancel_grace: Duration,
    gateway: Option<GatewayAuth>,
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
    session_mode: Option<String>,
    cancel_grace: Duration,
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
        let command = resolve_adapter_command(kind, command.as_deref());
        let session_mode = match std::env::var(SESSION_MODE_ENV) {
            Ok(v) if v.trim().is_empty() => None,
            Ok(v) => Some(v.trim().to_string()),
            Err(_) => kind.default_session_mode().map(str::to_string),
        };
        let args = std::env::var(kind.args_env())
            .map(|s| s.split_whitespace().map(str::to_string).collect())
            .unwrap_or_default();
        Self {
            kind,
            command,
            args,
            policy: PermissionPolicy::from_env(),
            cwd: std::env::var_os(CWD_ENV)
                .filter(|v| !v.is_empty())
                .map(PathBuf::from),
            session_mode,
            cancel_grace: DEFAULT_CANCEL_GRACE,
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

    /// Set the session mode enforced after `session/new` (`None` keeps the agent's own).
    pub fn with_session_mode(mut self, mode: Option<String>) -> Self {
        self.session_mode = mode;
        self
    }

    /// How long to wait for the agent to acknowledge `session/cancel`.
    pub fn with_cancel_grace(mut self, grace: Duration) -> Self {
        self.cancel_grace = grace;
        self
    }

    /// The session mode this driver enforces.
    pub fn session_mode(&self) -> Option<&str> {
        self.session_mode.as_deref()
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

    fn build_command(
        &self,
        request: &CompletionRequest,
        cwd: &std::path::Path,
    ) -> tokio::process::Command {
        let mut cmd = tokio::process::Command::new(&self.command);
        cmd.args(&self.args);
        cmd.current_dir(cwd);
        // Allowlist, not the inherited denylist: start from an empty environment.
        cmd.env_clear();
        cmd.envs(allowed_adapter_env(self.kind, std::env::vars()));
        match self.kind {
            AcpAgentKind::ClaudeCode => {
                // On top of the allowlist, the claude_code driver's filter keeps its OAuth
                // precedence (drops ANTHROPIC_API_KEY when a subscription login exists).
                let filter = super::claude_code::ClaudeCodeDriver::new(None, false);
                filter.apply_env_filter(&mut cmd);
                super::claude_code::ClaudeCodeDriver::ensure_home_env(&mut cmd);
                if let Some(model) = model_override(&request.model, "claude-code-acp/") {
                    cmd.env("ANTHROPIC_MODEL", model);
                }
            }
            AcpAgentKind::Codex => {
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

    /// The session working directory: the configured one (must exist), else a fresh empty
    /// scratch directory that is removed when the returned guard drops.
    fn session_cwd(&self) -> Result<(PathBuf, Option<ScratchDir>), LlmError> {
        if let Some(dir) = &self.cwd {
            if !dir.is_dir() {
                return Err(LlmError::Http(format!(
                    "ACP session cwd {} does not exist or is not a directory",
                    dir.display()
                )));
            }
            return Ok((dir.clone(), None));
        }
        let dir = std::env::temp_dir().join(format!(
            "librefang-acp-{}-{}",
            self.kind.label(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir)
            .map_err(|e| LlmError::Http(format!("failed to create ACP scratch cwd: {e}")))?;
        Ok((dir.clone(), Some(ScratchDir(dir))))
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
        session_mode,
        cancel_grace,
        gateway,
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
                    InitializeRequest::new(ProtocolVersion::V1)
                        .client_capabilities(client_capabilities(gateway.as_ref())),
                )
                .block_task()
                .await?;
            debug!(
                agent = kind.label(),
                info = ?init.agent_info,
                auth_methods = ?auth_method_ids(&init.auth_methods),
                "ACP agent initialized"
            );
            if let Some(gateway) = gateway.as_ref() {
                if advertises_auth_method(&init.auth_methods, GATEWAY_AUTH_METHOD) {
                    // Logs the URL and header names; never a header value.
                    info!(
                        agent = kind.label(),
                        gateway = ?gateway,
                        "authenticating ACP agent against the model gateway"
                    );
                    connection
                        .send_request(
                            AuthenticateRequest::new(GATEWAY_AUTH_METHOD)
                                .meta(gateway.authenticate_meta()),
                        )
                        .block_task()
                        .await?;
                } else {
                    warn!(
                        agent = kind.label(),
                        auth_methods = ?auth_method_ids(&init.auth_methods),
                        "ACP agent did not offer gateway auth; relying on ANTHROPIC_* env passthrough"
                    );
                }
            }
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
                .send_request(
                    NewSessionRequest::new(cwd)
                        .mcp_servers(mcp_servers)
                        .meta(session_meta(kind)),
                )
                .block_task()
                .await?;
            let session_id = session.session_id;
            match plan_session_mode(session_mode.as_deref(), session.modes.as_ref())
                .map_err(agent_client_protocol::util::internal_error)?
            {
                ModeAction::Keep => {}
                ModeAction::AgentHasNoModes => warn!(
                    agent = kind.label(),
                    required = session_mode.as_deref().unwrap_or(""),
                    "ACP agent reports no session modes; mode not enforced"
                ),
                ModeAction::Set => {
                    let mode = session_mode.clone().unwrap_or_default();
                    info!(agent = kind.label(), session_id = %session_id, mode = %mode, "setting ACP session mode");
                    connection
                        .send_request(SetSessionModeRequest::new(session_id.clone(), mode))
                        .block_task()
                        .await?;
                }
            }
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
                    match tokio::time::timeout(cancel_grace, &mut prompt_fut).await {
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

        let (cwd, _scratch) = self.session_cwd()?;
        let mut cmd = self.build_command(&request, &cwd);
        let env_token_set = std::env::var(OAUTH_TOKEN_VAR).is_ok_and(|v| !v.is_empty());
        let mut oauth_token_present = env_token_set;
        if self.kind == AcpAgentKind::ClaudeCode && !env_token_set {
            if let Some(path) = std::env::var(OAUTH_TOKEN_FILE_ENV)
                .ok()
                .filter(|p| !p.is_empty())
            {
                // Value goes straight onto the child env; it is never logged.
                cmd.env(OAUTH_TOKEN_VAR, read_oauth_token_file(&path).await?);
                oauth_token_present = true;
            }
        }
        let gateway = match self.kind {
            AcpAgentKind::ClaudeCode => {
                let mode = GatewayAuthMode::resolve(
                    std::env::var(GATEWAY_AUTH_ENV).ok().as_deref(),
                    oauth_token_present,
                )
                .map_err(LlmError::Http)?;
                debug!(
                    agent = label,
                    ?mode,
                    oauth_token_present,
                    "ACP gateway auth mode"
                );
                match mode {
                    GatewayAuthMode::Authenticate => GatewayAuth::from_env(
                        self.kind,
                        allowed_adapter_env(self.kind, std::env::vars()),
                    ),
                    GatewayAuthMode::Passthrough => {
                        if oauth_token_present {
                            let scrub_on = harden_passthrough_env(&mut cmd);
                            if scrub_overrides_session_mode(scrub_on, self.session_mode.as_deref())
                            {
                                warn!(
                                    agent = label,
                                    session_mode = self.session_mode.as_deref().unwrap_or(""),
                                    "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB is on, so Claude Code forces its \
                                     `default` permission mode; the configured ACP session mode may not hold"
                                );
                            }
                        }
                        None
                    }
                }
            }
            AcpAgentKind::Codex => None,
        };
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
        // The ACP crate logs whole JSON-RPC messages (gateway headers, the MCP
        // bridge key) at debug/trace; cap it at info for this turn, whatever
        // RUST_LOG says.
        let turn = tracing::instrument::WithSubscriber::with_subscriber(
            run_turn(
                transport,
                self.kind,
                self.policy.clone(),
                TurnInput {
                    cwd,
                    mcp_servers,
                    prompt,
                    session_mode: self.session_mode.clone(),
                    cancel_grace: self.cancel_grace,
                    gateway,
                },
                ev_tx,
                Arc::clone(&cancel),
            ),
            acp_capped_dispatch(),
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

/// Whether the adapter for `kind` resolves, using the same precedence as the driver:
/// `configured` (the provider's `base_url`), then the env override, then PATH.
pub fn acp_adapter_available(kind: AcpAgentKind, configured: Option<&str>) -> bool {
    let command = resolve_adapter_command(kind, configured);
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

    fn vars(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn keys(env: &[(String, String)]) -> Vec<&str> {
        env.iter().map(|(k, _)| k.as_str()).collect()
    }

    const PARENT_ENV: &[(&str, &str)] = &[
        ("PATH", "/usr/bin"),
        ("HOME", "/home/u"),
        ("LC_ALL", "C.UTF-8"),
        ("ANTHROPIC_BASE_URL", "http://gw:8080"),
        ("ANTHROPIC_AUTH_TOKEN", "gw-key"),
        ("CLAUDE_CODE_EXECUTABLE", "/opt/claude"),
        ("OPENAI_BASE_URL", "http://gw:8080/v1"),
        ("CODEX_HOME", "/home/u/.codex"),
        ("AWS_SECRET_ACCESS_KEY", "aws"),
        ("STRIPE_API_KEY", "stripe"),
        ("LIBREFANG_API_KEY", "lf"),
        ("GITHUB_TOKEN", "gh"),
        ("IS_SANDBOX", "1"),
        ("CLAUDE_DANGEROUSLY_SKIP_PERMISSIONS", "1"),
    ];

    #[test]
    fn claude_env_is_an_allowlist() {
        let env = allowed_adapter_env(AcpAgentKind::ClaudeCode, vars(PARENT_ENV));
        assert_eq!(
            keys(&env),
            [
                "PATH",
                "HOME",
                "LC_ALL",
                "ANTHROPIC_BASE_URL",
                "ANTHROPIC_AUTH_TOKEN",
                "CLAUDE_CODE_EXECUTABLE"
            ]
        );
    }

    #[test]
    fn codex_env_is_an_allowlist() {
        let env = allowed_adapter_env(AcpAgentKind::Codex, vars(PARENT_ENV));
        assert_eq!(
            keys(&env),
            ["PATH", "HOME", "LC_ALL", "OPENAI_BASE_URL", "CODEX_HOME"]
        );
    }

    #[test]
    fn unlisted_secrets_and_bypass_switches_never_pass() {
        for kind in [AcpAgentKind::ClaudeCode, AcpAgentKind::Codex] {
            for key in [
                "AWS_SECRET_ACCESS_KEY",
                "STRIPE_API_KEY",
                "LIBREFANG_API_KEY",
                "FOO_API_KEY",
                "GITHUB_TOKEN",
                "IS_SANDBOX",
                "CLAUDE_DANGEROUSLY_SKIP_PERMISSIONS",
                "CODEX_BYPASS_PERMISSIONS",
            ] {
                assert!(
                    !adapter_env_key_allowed(kind, key),
                    "{key} must not reach the {kind:?} adapter"
                );
            }
        }
    }

    #[test]
    fn resolve_adapter_command_precedence() {
        assert_eq!(
            resolve_adapter_command(AcpAgentKind::Codex, Some(" /opt/codex-acp ")),
            "/opt/codex-acp"
        );
        // Blank configured value falls through (env var unset in tests by default).
        if std::env::var(AcpAgentKind::Codex.path_env()).is_err() {
            assert_eq!(
                resolve_adapter_command(AcpAgentKind::Codex, Some("  ")),
                "codex-acp"
            );
            assert_eq!(
                resolve_adapter_command(AcpAgentKind::Codex, None),
                "codex-acp"
            );
        }
    }

    #[test]
    fn availability_honours_configured_path() {
        let dir = tempfile::tempdir().unwrap();
        let adapter = dir.path().join("my-claude-acp");
        std::fs::write(&adapter, "").unwrap();
        let configured = adapter.to_string_lossy().into_owned();
        assert!(acp_adapter_available(
            AcpAgentKind::ClaudeCode,
            Some(&configured)
        ));
        let missing = dir.path().join("absent").to_string_lossy().into_owned();
        assert!(!acp_adapter_available(
            AcpAgentKind::ClaudeCode,
            Some(&missing)
        ));
    }

    #[test]
    fn claude_session_meta_disables_bypass() {
        let meta = session_meta(AcpAgentKind::ClaudeCode).expect("claude meta");
        assert_eq!(
            meta["claudeCode"]["options"]["allowDangerouslySkipPermissions"],
            serde_json::json!(false)
        );
        assert!(session_meta(AcpAgentKind::Codex).is_none());
    }

    fn mode_state(current: &str, available: &[&str]) -> SessionModeState {
        use agent_client_protocol::schema::v1::SessionMode;
        SessionModeState::new(
            current.to_string(),
            available
                .iter()
                .map(|m| SessionMode::new(m.to_string(), m.to_string()))
                .collect(),
        )
    }

    #[test]
    fn session_mode_plan_fails_closed() {
        let st = mode_state("acceptEdits", &["default", "acceptEdits"]);
        assert_eq!(
            plan_session_mode(Some("default"), Some(&st)),
            Ok(ModeAction::Set)
        );
        let st = mode_state("default", &["default"]);
        assert_eq!(
            plan_session_mode(Some("default"), Some(&st)),
            Ok(ModeAction::Keep)
        );
        let st = mode_state("bypassPermissions", &["bypassPermissions"]);
        assert!(plan_session_mode(Some("default"), Some(&st)).is_err());
        assert_eq!(
            plan_session_mode(Some("default"), None),
            Ok(ModeAction::AgentHasNoModes)
        );
        assert_eq!(plan_session_mode(None, Some(&st)), Ok(ModeAction::Keep));
        assert_eq!(
            AcpAgentKind::ClaudeCode.default_session_mode(),
            Some("default")
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

    fn env(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn gateway_needs_base_url_and_a_credential_header() {
        let kind = AcpAgentKind::ClaudeCode;
        assert_eq!(GatewayAuth::from_env(kind, env(&[])), None);
        assert_eq!(
            GatewayAuth::from_env(kind, env(&[("ANTHROPIC_BASE_URL", "http://gw/anthropic")])),
            None,
            "a base URL alone keeps the env passthrough path"
        );
        assert_eq!(
            GatewayAuth::from_env(kind, env(&[("ANTHROPIC_CUSTOM_HEADERS", "x-k: v")])),
            None
        );
        assert_eq!(
            GatewayAuth::from_env(
                AcpAgentKind::Codex,
                env(&[
                    ("ANTHROPIC_BASE_URL", "http://gw"),
                    ("ANTHROPIC_CUSTOM_HEADERS", "x-k: v")
                ])
            ),
            None,
            "codex has no Anthropic gateway"
        );
    }

    #[test]
    fn gateway_parses_custom_headers_and_maps_token_and_key() {
        let gw = GatewayAuth::from_env(
            AcpAgentKind::ClaudeCode,
            env(&[
                ("ANTHROPIC_BASE_URL", " http://gw/anthropic "),
                (
                    "ANTHROPIC_CUSTOM_HEADERS",
                    "x-litellm-api-key: sk-one\r\nX-Route:lane\nbogus\n",
                ),
                ("ANTHROPIC_AUTH_TOKEN", "tok"),
                ("ANTHROPIC_API_KEY", "key"),
            ]),
        )
        .expect("gateway");
        assert_eq!(gw.base_url, "http://gw/anthropic");
        let names: Vec<_> = gw.headers.keys().cloned().collect();
        assert_eq!(
            names,
            ["Authorization", "X-Route", "x-api-key", "x-litellm-api-key"]
        );
        assert_eq!(gw.headers["x-litellm-api-key"], "sk-one");
        assert_eq!(gw.headers["X-Route"], "lane");
        assert_eq!(gw.headers["Authorization"], "Bearer tok");
        assert_eq!(gw.headers["x-api-key"], "key");

        // A custom header already carrying the credential wins over the variables.
        let gw = GatewayAuth::from_env(
            AcpAgentKind::ClaudeCode,
            env(&[
                ("ANTHROPIC_BASE_URL", "http://gw"),
                (
                    "ANTHROPIC_CUSTOM_HEADERS",
                    "authorization: Bearer hdr\nX-Api-Key: hk",
                ),
                ("ANTHROPIC_AUTH_TOKEN", "tok"),
                ("ANTHROPIC_API_KEY", "key"),
            ]),
        )
        .expect("gateway");
        assert_eq!(gw.headers.len(), 2);
        assert_eq!(gw.headers["authorization"], "Bearer hdr");
        assert_eq!(gw.headers["X-Api-Key"], "hk");
    }

    #[test]
    fn gateway_debug_never_prints_header_values() {
        let gw = GatewayAuth::from_env(
            AcpAgentKind::ClaudeCode,
            env(&[
                ("ANTHROPIC_BASE_URL", "http://gw"),
                (
                    "ANTHROPIC_CUSTOM_HEADERS",
                    "x-litellm-api-key: sk-SECRET-VALUE",
                ),
                ("ANTHROPIC_AUTH_TOKEN", "TOKEN-SECRET"),
            ]),
        )
        .expect("gateway");
        let shown = format!("{gw:?}");
        assert!(shown.contains("x-litellm-api-key"), "{shown}");
        assert!(!shown.contains("SECRET"), "header value leaked: {shown}");
    }

    #[test]
    fn gateway_meta_matches_adapter_shape() {
        let gw = GatewayAuth::from_env(
            AcpAgentKind::ClaudeCode,
            env(&[
                ("ANTHROPIC_BASE_URL", "http://gw"),
                ("ANTHROPIC_CUSTOM_HEADERS", "x-k: v"),
            ]),
        )
        .expect("gateway");
        let meta = serde_json::Value::Object(gw.authenticate_meta().expect("meta"));
        assert_eq!(
            meta,
            serde_json::json!({ "gateway": { "baseUrl": "http://gw", "headers": { "x-k": "v" } } })
        );
        let caps = serde_json::to_value(client_capabilities(Some(&gw))).unwrap();
        assert_eq!(caps["auth"]["_meta"]["gateway"], serde_json::json!(true));
        let caps = serde_json::to_value(client_capabilities(None)).unwrap();
        assert!(caps["auth"]["_meta"].is_null(), "{caps}");
    }

    #[test]
    fn gateway_auth_mode_defaults_to_passthrough_with_oauth_token() {
        use GatewayAuthMode::*;
        assert_eq!(GatewayAuthMode::resolve(None, true), Ok(Passthrough));
        assert_eq!(GatewayAuthMode::resolve(None, false), Ok(Authenticate));
        assert_eq!(GatewayAuthMode::resolve(Some(" "), true), Ok(Passthrough));
        assert_eq!(
            GatewayAuthMode::resolve(Some("authenticate"), true),
            Ok(Authenticate)
        );
        assert_eq!(
            GatewayAuthMode::resolve(Some("passthrough"), false),
            Ok(Passthrough)
        );
        assert!(GatewayAuthMode::resolve(Some("bogus"), false).is_err());
    }

    #[tokio::test]
    async fn oauth_token_file_is_capped_regular_and_never_echoed() {
        let dir = tempfile::tempdir().unwrap();
        let p = |n: &str| dir.path().join(n).to_string_lossy().into_owned();
        std::fs::write(p("empty"), "  \n").unwrap();
        let err = read_oauth_token_file(&p("empty")).await.unwrap_err();
        assert!(err.to_string().contains("is empty"), "{err}");

        std::fs::write(p("tok"), "tok-fake-123\n").unwrap();
        assert_eq!(
            read_oauth_token_file(&p("tok")).await.unwrap(),
            "tok-fake-123"
        );

        let err = read_oauth_token_file(&p("missing")).await.unwrap_err();
        assert!(err.to_string().contains("cannot read"), "{err}");

        let big = "x".repeat(OAUTH_TOKEN_FILE_MAX as usize + 1);
        std::fs::write(p("big"), &big).unwrap();
        let err = read_oauth_token_file(&p("big")).await.unwrap_err();
        assert!(err.to_string().contains("larger than"), "{err}");
        assert!(!err.to_string().contains("xxxx"), "content echoed: {err}");

        std::fs::create_dir(p("dir")).unwrap();
        let err = read_oauth_token_file(&p("dir")).await.unwrap_err();
        assert!(err.to_string().contains("not a regular file"), "{err}");

        #[cfg(unix)]
        {
            // Symlinked secret mounts are accepted; a symlink to a FIFO is not,
            // and must be refused without blocking on open.
            std::os::unix::fs::symlink(p("tok"), p("link")).unwrap();
            assert_eq!(
                read_oauth_token_file(&p("link")).await.unwrap(),
                "tok-fake-123"
            );
            let status = std::process::Command::new("mkfifo")
                .arg(p("fifo"))
                .status()
                .unwrap();
            assert!(status.success());
            std::os::unix::fs::symlink(p("fifo"), p("fifolink")).unwrap();
            let err = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                read_oauth_token_file(&p("fifolink")),
            )
            .await
            .expect("FIFO refused without blocking")
            .unwrap_err();
            assert!(err.to_string().contains("not a regular file"), "{err}");
        }
    }

    #[test]
    fn invalid_gateway_auth_mode_is_not_echoed() {
        let err = GatewayAuthMode::resolve(Some("sk-pasted-by-mistake"), false).unwrap_err();
        assert!(!err.contains("sk-pasted"), "{err}");
    }

    #[test]
    fn scrub_conflict_only_for_non_default_modes() {
        assert!(scrub_overrides_session_mode(true, Some("acceptEdits")));
        assert!(!scrub_overrides_session_mode(true, Some("default")));
        assert!(!scrub_overrides_session_mode(true, Some("")));
        assert!(!scrub_overrides_session_mode(true, None));
        assert!(!scrub_overrides_session_mode(false, Some("acceptEdits")));
    }
}
