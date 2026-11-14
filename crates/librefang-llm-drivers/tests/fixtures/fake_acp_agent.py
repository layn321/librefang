#!/usr/bin/env python3
"""Fake ACP agent for the acp_client driver integration tests.

Speaks ACP v1 (newline-delimited JSON-RPC 2.0) over stdio. It never calls a
model and needs no credentials. Behaviour is selected by argv[1] (mode) and
argv[2] (cancel marker path, `hang` mode only):

  normal       stream chunks, ask for one `edit` permission, report the outcome
  drop_stream  answer the prompt with end_turn but send no session/update
  hang         send one chunk, then wait for session/cancel and answer
               stopReason=cancelled (writes the marker file if given)
  ignore_cancel     send one chunk, then never answer (ignores session/cancel)
  grandchild_hang   start a `sleep` grandchild, write its pid to the marker file,
                    send one chunk, then never answer
  no_default_mode   session/new reports only `bypassPermissions` (no `default`)
"""

import subprocess

import json
import os
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "normal"
CANCEL_MARKER = sys.argv[2] if len(sys.argv) > 2 else None
_next_id = 1000


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def read():
    line = sys.stdin.readline()
    if not line:
        sys.exit(0)
    return json.loads(line)


def chunk(session_id, text, kind="agent_message_chunk"):
    send({
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": session_id,
            "update": {"sessionUpdate": kind, "content": {"type": "text", "text": text}},
        },
    })


def request(method, params):
    """Send a request to the client and wait for its response."""
    global _next_id
    _next_id += 1
    rid = _next_id
    send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
    while True:
        msg = read()
        if msg.get("id") == rid and "method" not in msg:
            return msg


def handle_prompt(rid, params, state):
    sid = params["sessionId"]
    prompt_text = "".join(b.get("text", "") for b in params.get("prompt", []))
    if MODE == "drop_stream":
        send({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": "end_turn"}})
        return
    if MODE in ("ignore_cancel", "grandchild_hang"):
        if MODE == "grandchild_hang":
            child = subprocess.Popen(["sleep", "300"])
            with open(CANCEL_MARKER, "w") as f:
                f.write(str(child.pid))
        chunk(sid, "working...")
        while True:
            read()
    if MODE == "hang":
        chunk(sid, "working...")
        while True:
            msg = read()
            if msg.get("method") == "session/cancel":
                if CANCEL_MARKER:
                    with open(CANCEL_MARKER, "w") as f:
                        f.write("cancelled")
                send({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": "cancelled"}})
                return
    chunk(sid, "thinking about it", kind="agent_thought_chunk")
    chunk(sid, "echo:" + prompt_text.splitlines()[-1] if prompt_text else "echo:")
    chunk(sid, "|base_url:" + os.environ.get("ANTHROPIC_BASE_URL", "<unset>"))
    chunk(sid, "|mcp:" + ",".join(state.get("mcp", [])))
    watched = ["GROQ_API_KEY", "AWS_SECRET_ACCESS_KEY", "FAKE_UNLISTED_API_KEY",
               "GITHUB_TOKEN", "IS_SANDBOX"]
    leaked = [k for k in watched if k in os.environ]
    chunk(sid, "|leaked:" + (",".join(leaked) or "<none>"))
    chunk(sid, "|openai_base:" + os.environ.get("OPENAI_BASE_URL", "<unset>"))
    chunk(sid, "|mode:" + state.get("mode", "<none>"))
    chunk(sid, "|bypass_allowed:" + state.get("bypass", "<unset>"))
    chunk(sid, "|cwd_match:" + ("yes" if os.getcwd() == os.path.realpath(state.get("cwd", "")) else "no"))
    send({
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": sid,
            "update": {
                "sessionUpdate": "tool_call",
                "toolCallId": "t1",
                "title": "Edit README.md",
                "kind": "edit",
                "status": "pending",
            },
        },
    })
    resp = request("session/request_permission", {
        "sessionId": sid,
        "toolCall": {"toolCallId": "t1", "kind": "edit", "title": "Edit README.md"},
        "options": [
            {"optionId": "allow", "name": "Allow", "kind": "allow_once"},
            {"optionId": "reject", "name": "Reject", "kind": "reject_once"},
        ],
    })
    outcome = resp.get("result", {}).get("outcome", {})
    if outcome.get("outcome") == "selected" and outcome.get("optionId") == "allow":
        chunk(sid, "|permission:granted")
    else:
        chunk(sid, "|permission:denied")
    send({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": "end_turn"}})


def main():
    state = {}
    while True:
        msg = read()
        method = msg.get("method")
        rid = msg.get("id")
        params = msg.get("params", {})
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": rid, "result": {
                "protocolVersion": 1,
                "agentCapabilities": {"mcpCapabilities": {"http": True}},
                "agentInfo": {"name": "fake-acp-agent", "version": "0.0.1"},
                "authMethods": [],
            }})
        elif method == "session/new":
            state["mcp"] = [s.get("name", "") for s in params.get("mcpServers", [])]
            state["cwd"] = params.get("cwd", "")
            opts = params.get("_meta", {}).get("claudeCode", {}).get("options", {})
            if "allowDangerouslySkipPermissions" in opts:
                state["bypass"] = str(opts["allowDangerouslySkipPermissions"]).lower()
            if MODE == "no_default_mode":
                modes = {"currentModeId": "bypassPermissions",
                         "availableModes": [{"id": "bypassPermissions", "name": "Bypass"}]}
            else:
                modes = {"currentModeId": "acceptEdits",
                         "availableModes": [{"id": "default", "name": "Default"},
                                            {"id": "acceptEdits", "name": "Accept edits"}]}
            state["mode"] = modes["currentModeId"]
            send({"jsonrpc": "2.0", "id": rid,
                  "result": {"sessionId": "sess-1", "modes": modes}})
        elif method == "session/set_mode":
            state["mode"] = params.get("modeId", "")
            send({"jsonrpc": "2.0", "id": rid, "result": {}})
        elif method == "session/prompt":
            handle_prompt(rid, params, state)
        elif rid is not None and method is not None:
            send({"jsonrpc": "2.0", "id": rid,
                  "error": {"code": -32601, "message": "method not found"}})


if __name__ == "__main__":
    main()
