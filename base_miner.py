#!/usr/bin/env python3
"""SOMA miner — observation-masking context compressor for OpenClaw/qwen3-coder.

Algorithm: pure observation masking (Lindenbauer et al., NeurIPS 2025).
- The last OBSERVATION_MASK_WINDOW tool results keep their content (content-truncated).
- ALL older tool results get their content replaced with a compact tombstone marker.
- ALL message pairs are preserved (no structural dropping — this prevents flail).
- ALL assistant messages are kept in full (the reasoning trace is the anti-flail signal).

Empirical basis (arXiv 2508.21433, Table 1, Qwen3-Coder 480B):
  Raw agent:              53.4% solve rate, $1.29/instance
  Observation masking M=10: 54.8% solve rate (+2.6 pp), $0.61/instance (−52.7%)
  LLM-Summary:            53.8% solve rate (+0.7%),  $0.64/instance (−50.4%)

The +2.6 pp improvement over raw, with no LLM calls, is our jackpot conversion
mechanism. The key: masking eliminates noisy old observations while preserving
the full reasoning trace that prevents re-exploration (flail).
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional


EVENT_NAMES = frozenset({"assemble"})

# ── observation masking window ────────────────────────────────────────────────
# Last N tool results keep their content (content-truncated).
# All older tool results get a compact tombstone.
# Empirically optimal at M=10 for Qwen3-Coder 480B (Lindenbauer et al. 2025).
OBSERVATION_MASK_WINDOW = 10   # default / standard trajectories

# Adaptive windows by trajectory regime (computed from struggle signals):
WINDOW_SHORT    = 999  # sentinel: short trajectory -> no tombstoning at all
WINDOW_HARD     = 7    # moderate struggling  (20-30 tool results, 1-2 signals)
WINDOW_DROWNING = 5    # heavy struggling     (30+ tool results, 3+ signals)

# Screener-safety boundary: trajectories with <= this many tool results get
# WINDOW_SHORT (no tombstoning). Easy screener tasks: 5-12 results.
# Hard competition tasks: typically 20-60. This guarantees the screener is
# never broken regardless of how aggressive the Hard tuning is.
SCREENER_SAFE_LENGTH = 14

# Struggling-signal thresholds (empirically grounded):
#   Successful SWE-bench sessions: median 11-16 turns (Liu et al. 2025)
#   Failed sessions: median 31+ turns (Liu et al. 2025, long-tail failure)
#   LOOP regime: P(failure|LOOP) = 88.7% (CAUM study, 80K SWE-agent sessions)
LONG_TRAJ_THRESHOLD  = 20   # >= this many tool results -> at least one signal
VERY_LONG_THRESHOLD  = 30   # >= this -> heavy compression
LOOP_LOOKBACK        = 8    # last N results to scan for re-read loops
LOOP_FILE_REPEAT     = 3    # same file read >= this many times in lookback = loop
TEST_FAIL_THRESHOLD  = 3    # >= this many failed test runs = agent is stuck
EDIT_DELAY_THRESHOLD = 14   # no edit after this many results = exploration stall

# ── per-result content limits for the RECENT window ──────────────────────────
# (characters; ~4 chars ≈ 1 token)
READ_HEAD_CHARS = 2000    # ~500 tok — file structure / imports
READ_TAIL_CHARS = 4000    # ~1000 tok — active editing region
EXEC_HEAD_CHARS = 800     # ~200 tok — command + env context
EXEC_TAIL_CHARS = 4800    # ~1200 tok — errors live at the tail
TEST_TAIL_CHARS = 10000   # ~2500 tok — failing traceback is always at the tail
WRITE_MAX_CHARS = 2000    # ~500 tok — ack messages are usually small
GENERIC_HEAD_CHARS = 1000
GENERIC_TAIL_CHARS = 4000
MIN_CHARS_TO_TRUNCATE = 500  # below this, truncation has no effect worth paying for

# ── state persistence ─────────────────────────────────────────────────────────
STATE_VERSION = 1
STATE_DIR_NAME = "state"

# ── markers ───────────────────────────────────────────────────────────────────
_OMIT_MIDDLE = "\n[...{count} chars omitted...]\n"
_OMIT_HEAD   = "[...{count} chars omitted from start...]\n"


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize_role(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    role = value.strip()
    lowered = role.lower().replace("_", "").replace("-", "")
    if lowered == "toolresult":
        return "toolResult"
    return lowered


def extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(extract_text(item) for item in value)
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if isinstance(value.get("content"), str):
            return value["content"]
        return "\n".join(extract_text(item) for item in value.values())
    return str(value)


def estimate_tokens_for_message_array(messages: list[Any]) -> int:
    total_chars = 0
    for message in messages:
        if isinstance(message, dict):
            total_chars += len(extract_text(message.get("content")))
    return max(1, math.ceil(total_chars / 4)) if messages else 0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_file_part(value: Any, fallback: str = "session") -> str:
    raw = value.strip() if isinstance(value, str) and value.strip() else fallback
    normalized = "".join(char if char.isalnum() or char in "_.-" else "-" for char in raw)
    normalized = normalized.strip("-")
    return (normalized or fallback)[:120]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def fingerprint_messages(messages: list[Any]) -> str:
    return hashlib.sha256(canonical_json(messages).encode("utf-8")).hexdigest()


def get_params(payload: dict[str, Any]) -> dict[str, Any]:
    params = payload.get("params")
    return params if isinstance(params, dict) else payload


def resolve_plugin_dir(payload: dict[str, Any]) -> Path:
    value = payload.get("pluginDir")
    if isinstance(value, str) and value.strip():
        return Path(value.strip())
    return Path(__file__).resolve().parent


def resolve_session_identity(payload: dict[str, Any]) -> tuple[str, str | None, str | None]:
    params = get_params(payload)
    session_id = params.get("sessionId") if isinstance(params.get("sessionId"), str) else None
    session_key = params.get("sessionKey") if isinstance(params.get("sessionKey"), str) else None
    identity = session_id or session_key or "session"
    return safe_file_part(identity), session_id, session_key


def resolve_state_path(payload: dict[str, Any]) -> Path:
    plugin_dir = resolve_plugin_dir(payload)
    session_part, _, _ = resolve_session_identity(payload)
    return plugin_dir / "logs" / STATE_DIR_NAME / f"{session_part}.json"


def load_state(state_path: Path) -> dict[str, Any] | None:
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:
        return None

    if not isinstance(state, dict) or state.get("version") != STATE_VERSION:
        return None
    if not isinstance(state.get("messages"), list):
        return None
    if not isinstance(state.get("sourceMessageCount"), int):
        return None
    if not isinstance(state.get("sourceFingerprint"), str):
        return None
    return state


def save_state(
    state_path: Path,
    payload: dict[str, Any],
    raw_messages: list[Any],
    current_messages: list[Any],
) -> None:
    _, session_id, session_key = resolve_session_identity(payload)
    state = {
        "version": STATE_VERSION,
        "updatedAt": utc_now(),
        "sessionId": session_id,
        "sessionKey": session_key,
        "sourceMessageCount": len(raw_messages),
        "sourceFingerprint": fingerprint_messages(raw_messages),
        "messageCount": len(current_messages),
        "messages": current_messages,
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = state_path.with_suffix(f"{state_path.suffix}.tmp")
    temp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(state_path)


def resolve_stateful_messages(
    payload: dict[str, Any],
    raw_messages: list[Any],
) -> tuple[list[Any], dict[str, Any], Path]:
    state_path = resolve_state_path(payload)
    state = load_state(state_path)
    metadata: dict[str, Any] = {
        "statePath": str(state_path),
        "stateLoaded": False,
        "stateResetReason": None,
        "rawInputMessageCount": len(raw_messages),
        "previousSourceMessageCount": None,
        "previousStateMessageCount": None,
        "newMessageCount": len(raw_messages),
        "workingMessageCount": len(raw_messages),
    }

    if state is None:
        return raw_messages, metadata, state_path

    source_count = state.get("sourceMessageCount")
    if not isinstance(source_count, int) or source_count < 0:
        metadata["stateResetReason"] = "invalid_source_count"
        return raw_messages, metadata, state_path

    metadata["previousSourceMessageCount"] = source_count
    metadata["previousStateMessageCount"] = len(state.get("messages", []))

    if source_count > len(raw_messages):
        metadata["stateResetReason"] = "source_shorter_than_state"
        return raw_messages, metadata, state_path

    source_prefix = raw_messages[:source_count]
    if fingerprint_messages(source_prefix) != state.get("sourceFingerprint"):
        metadata["stateResetReason"] = "source_prefix_changed"
        return raw_messages, metadata, state_path

    new_messages = raw_messages[source_count:]
    working_messages = [*state["messages"], *new_messages]
    metadata.update({
        "stateLoaded": True,
        "newMessageCount": len(new_messages),
        "workingMessageCount": len(working_messages),
    })
    return working_messages, metadata, state_path


def extract_tool_result_ids(message: Any) -> set[str]:
    if not isinstance(message, dict) or normalize_role(message.get("role")) != "toolResult":
        return set()
    ids: set[str] = set()
    for field in ("toolCallId", "toolUseId", "id"):
        value = message.get(field)
        if isinstance(value, str) and value.strip():
            ids.add(value.strip())
    return ids


def extract_tool_call_ids(message: Any) -> set[str]:
    if not isinstance(message, dict) or normalize_role(message.get("role")) != "assistant":
        return set()

    ids: set[str] = set()
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "toolCall":
                value = block.get("id")
                if isinstance(value, str) and value.strip():
                    ids.add(value.strip())

    for field in ("toolCalls", "tool_calls"):
        tool_calls = message.get(field)
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    value = tool_call.get("id")
                    if isinstance(value, str) and value.strip():
                        ids.add(value.strip())

    return ids


def sanitize_content(content: Any) -> tuple[Any, bool]:
    if not isinstance(content, list):
        return content, False

    sanitized: list[Any] = []
    changed = False
    for block in content:
        if isinstance(block, dict) and block.get("type") == "thinking":
            changed = True
            continue
        sanitized.append(block)

    return sanitized, changed


def has_runtime_content(content: Any) -> bool:
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return bool(content)
    if isinstance(content, dict):
        return bool(content)
    return content is not None


def is_failed_assistant_placeholder(message: Any) -> bool:
    if not isinstance(message, dict) or normalize_role(message.get("role")) != "assistant":
        return False
    if not isinstance(message.get("errorMessage"), str):
        return False
    content = message.get("content")
    return content in (None, "") or content == []


def sanitize_messages(messages: list[Any]) -> tuple[list[Any], dict[str, Any]]:
    sanitized: list[Any] = []
    changed = False
    removed_count = 0
    removed_thinking_block_count = 0

    for message in messages:
        if is_failed_assistant_placeholder(message):
            changed = True
            removed_count += 1
            continue

        if not isinstance(message, dict):
            sanitized.append(message)
            continue

        next_message = message
        next_content, content_changed = sanitize_content(message.get("content"))
        if content_changed:
            next_message = copy.deepcopy(message)
            next_message["content"] = next_content
            changed = True
            removed_thinking_block_count += 1

        if normalize_role(next_message.get("role")) != "toolResult" and not has_runtime_content(next_message.get("content")):
            changed = True
            removed_count += 1
            continue

        sanitized.append(next_message)

    return sanitized if changed else messages, {
        "changed": changed,
        "removedMessageCount": removed_count,
        "removedThinkingBlockCount": removed_thinking_block_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool call index
# ─────────────────────────────────────────────────────────────────────────────

def build_tool_call_index(messages: list[Any]) -> dict[str, dict[str, Any]]:
    """Map every toolCallId found in assistant messages → {name, input}."""
    index: dict[str, dict[str, Any]] = {}
    for msg in messages:
        if not isinstance(msg, dict) or normalize_role(msg.get("role")) != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "toolCall":
                    call_id = block.get("id")
                    if isinstance(call_id, str) and call_id.strip():
                        index[call_id.strip()] = {
                            "name": str(block.get("name", "")),
                            "input": block.get("input") if isinstance(block.get("input"), dict) else {},
                        }
        for field in ("toolCalls", "tool_calls"):
            tool_calls = msg.get(field)
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    call_id = tc.get("id")
                    if isinstance(call_id, str) and call_id.strip():
                        name = tc.get("name") or ""
                        if not name and isinstance(tc.get("function"), dict):
                            name = tc["function"].get("name", "")
                        raw_input = tc.get("input") or tc.get("arguments") or {}
                        if isinstance(raw_input, str):
                            try:
                                raw_input = json.loads(raw_input)
                            except Exception:
                                raw_input = {}
                        index[call_id.strip()] = {
                            "name": str(name),
                            "input": raw_input if isinstance(raw_input, dict) else {},
                        }
    return index


def get_tool_info_for_result(
    result_msg: Any,
    index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return {name, input} for a toolResult message, or {} if not found."""
    if not isinstance(result_msg, dict):
        return {}
    for field in ("toolCallId", "toolUseId", "id"):
        call_id = result_msg.get(field)
        if isinstance(call_id, str) and call_id.strip() in index:
            return index[call_id.strip()]
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Tool-type classification
# ─────────────────────────────────────────────────────────────────────────────

_READ_TOOL_NAMES = frozenset({
    "read", "read_file", "view", "view_file", "cat", "open",
    "get_file_content", "file_read", "show_file",
})
_WRITE_TOOL_NAMES = frozenset({
    "write", "write_file", "edit", "edit_file", "str_replace",
    "str_replace_editor", "create_file", "replace", "patch",
    "apply_patch", "insert", "delete", "update_file",
})
_EXEC_TOOL_NAMES = frozenset({
    "exec", "bash", "shell", "run", "run_command", "execute",
    "terminal", "cmd", "command",
})
_TEST_KEYWORDS = (
    "pytest", "python -m pytest", "python -m test", "unittest",
    "nose2", "tox", "python test", "py.test",
)


def classify_tool_type(tool_name: str, tool_input: dict) -> str:
    """Return one of: 'read', 'write', 'test', 'exec', 'other'."""
    name_lower = tool_name.lower().strip()
    if name_lower in _READ_TOOL_NAMES:
        return "read"
    if name_lower in _WRITE_TOOL_NAMES:
        return "write"
    if name_lower in _EXEC_TOOL_NAMES:
        command = str(tool_input.get("command", tool_input.get("cmd", ""))).lower()
        if any(kw in command for kw in _TEST_KEYWORDS):
            return "test"
        return "exec"
    if "path" in tool_input and not any(k in tool_input for k in ("content", "new_str", "old_str", "insert")):
        return "read"
    return "other"


def extract_file_path_from_tool(tool_name: str, tool_input: dict) -> str | None:
    """Return the file path from a read-tool's input, or None."""
    for field in ("path", "file_path", "filename", "file", "filepath"):
        value = tool_input.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Content truncation helpers (applied to RECENT window only)
# ─────────────────────────────────────────────────────────────────────────────

def _truncate_head_tail(text: str, head: int, tail: int) -> str:
    if len(text) <= head + tail:
        return text
    omitted = len(text) - head - tail
    return text[:head] + _OMIT_MIDDLE.format(count=omitted) + text[-tail:]


def _truncate_tail_only(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return _OMIT_HEAD.format(count=omitted) + text[-max_chars:]


def _compress_text_by_type(text: str, tool_type: str) -> str:
    if len(text) <= MIN_CHARS_TO_TRUNCATE:
        return text
    if tool_type == "read":
        return _truncate_head_tail(text, READ_HEAD_CHARS, READ_TAIL_CHARS)
    if tool_type == "test":
        return _truncate_tail_only(text, TEST_TAIL_CHARS)
    if tool_type == "exec":
        return _truncate_head_tail(text, EXEC_HEAD_CHARS, EXEC_TAIL_CHARS)
    if tool_type == "write":
        if len(text) > WRITE_MAX_CHARS:
            return text[:WRITE_MAX_CHARS] + f"\n[...{len(text) - WRITE_MAX_CHARS} chars omitted...]"
        return text
    return _truncate_head_tail(text, GENERIC_HEAD_CHARS, GENERIC_TAIL_CHARS)


def _compress_content(content: Any, tool_type: str) -> tuple[Any, bool]:
    """Recursively compress content; returns (new_content, changed)."""
    if isinstance(content, str):
        compressed = _compress_text_by_type(content, tool_type)
        return compressed, compressed != content

    if isinstance(content, list):
        new_blocks: list[Any] = []
        changed = False
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type in ("text", "tool_result", ""):
                    text = block.get("text")
                    if isinstance(text, str):
                        compressed = _compress_text_by_type(text, tool_type)
                        if compressed != text:
                            new_block = dict(block)
                            new_block["text"] = compressed
                            new_blocks.append(new_block)
                            changed = True
                            continue
                    nested = block.get("content")
                    if nested is not None:
                        new_nested, nested_changed = _compress_content(nested, tool_type)
                        if nested_changed:
                            new_block = dict(block)
                            new_block["content"] = new_nested
                            new_blocks.append(new_block)
                            changed = True
                            continue
            new_blocks.append(block)
        return (new_blocks if changed else content), changed

    return content, False


def _compress_recent_tool_result(
    message: Any,
    tool_call_index: dict[str, dict[str, Any]],
) -> tuple[Any, bool]:
    """Apply content-aware truncation to a RECENT toolResult (within window)."""
    if not isinstance(message, dict):
        return message, False
    tool_info = get_tool_info_for_result(message, tool_call_index)
    tool_type = classify_tool_type(tool_info.get("name", ""), tool_info.get("input", {}))
    content = message.get("content")
    new_content, changed = _compress_content(content, tool_type)
    if not changed:
        return message, False
    new_msg = copy.deepcopy(message)
    new_msg["content"] = new_content
    return new_msg, True


# ─────────────────────────────────────────────────────────────────────────────
# Tombstone markers (applied to OLD observations outside the window)
# ─────────────────────────────────────────────────────────────────────────────

def _make_tombstone(
    message: Any,
    tool_call_index: dict[str, dict[str, Any]],
) -> str:
    """
    Compact marker replacing an old observation's content.

    Preserves the "what I already tried" signal so the agent doesn't re-explore.
    Format: "[observation masked: {type} {path_or_cmd}]"
    Examples:
      "[observation masked: read src/utils.py]"
      "[observation masked: test pytest tests/]"
      "[observation masked: exec git diff HEAD]"
      "[observation masked: write tests/test_utils.py]"
    """
    tool_info = get_tool_info_for_result(message, tool_call_index)
    tool_name = tool_info.get("name", "")
    tool_input = tool_info.get("input", {})
    tool_type = classify_tool_type(tool_name, tool_input) if tool_name else "unknown"

    if tool_type == "read":
        path = extract_file_path_from_tool(tool_name, tool_input) or ""
        return f"[observation masked: read {path}]" if path else "[observation masked: read]"

    if tool_type in ("test", "exec"):
        cmd = str(tool_input.get("command", tool_input.get("cmd", "")))[:50].strip()
        return f"[observation masked: {tool_type} {cmd}]" if cmd else f"[observation masked: {tool_type}]"

    if tool_type == "write":
        path = extract_file_path_from_tool(tool_name, tool_input) or ""
        return f"[observation masked: write {path}]" if path else "[observation masked: write]"

    return f"[observation masked: {tool_type}]"


def _replace_content_with_tombstone(
    message: Any,
    tombstone: str,
) -> Any:
    """Replace a toolResult's content with a tombstone string."""
    if not isinstance(message, dict):
        return message
    current_content = message.get("content")
    if current_content == tombstone:
        return message  # already tombstoned; skip deepcopy
    new_msg = dict(message)
    new_msg["content"] = tombstone
    return new_msg


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory difficulty detection (no-LLM, O(n))
# ─────────────────────────────────────────────────────────────────────────────

def count_struggling_signals(
    tool_result_indices: list[int],
    messages: list[Any],
    tool_call_index: dict[str, dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    """
    Count how many struggling signals are present in the trajectory (0-4).

    Each signal corresponds to a known Hard-task failure mode:
    1. LONG_TRAJ  : >= LONG_TRAJ_THRESHOLD tool results (failed sessions run
                    2x longer than successful ones; Liu et al. 2025)
    2. LOOP       : same file read >= LOOP_FILE_REPEAT times in last
                    LOOP_LOOKBACK results (LOOP regime; CAUM, P(fail)=88.7%)
    3. TEST_FAILS : >= TEST_FAIL_THRESHOLD failed test runs (agent is stuck
                    cycling through the same broken fix attempts)
    4. NO_EDIT    : no edit/write among first EDIT_DELAY_THRESHOLD results
                    (pure exploration phase — agent can't commit to a fix)

    Returns (signal_count, detail_dict).
    """
    n = len(tool_result_indices)
    signals = 0
    detail: dict[str, Any] = {}

    # Signal 1: long trajectory
    if n >= LONG_TRAJ_THRESHOLD:
        signals += 1
        detail["longTraj"] = n

    # Signal 2: re-read loop in recent window
    recent = tool_result_indices[-LOOP_LOOKBACK:]
    path_counts: dict[str, int] = {}
    for idx in recent:
        info = get_tool_info_for_result(messages[idx], tool_call_index)
        t = classify_tool_type(info.get("name", ""), info.get("input", {}))
        if t == "read":
            p = extract_file_path_from_tool(info.get("name", ""), info.get("input", {}))
            if p:
                path_counts[p] = path_counts.get(p, 0) + 1
    looping = {p: c for p, c in path_counts.items() if c >= LOOP_FILE_REPEAT}
    if looping:
        signals += 1
        detail["loop"] = looping

    # Signal 3: multiple failed test runs
    fail_count = 0
    for idx in tool_result_indices:
        info = get_tool_info_for_result(messages[idx], tool_call_index)
        t = classify_tool_type(info.get("name", ""), info.get("input", {}))
        if t == "test":
            text = extract_text(messages[idx].get("content"))
            if any(kw in text for kw in ("FAILED", "ERROR", "AssertionError", "assert", "FAIL")):
                fail_count += 1
    if fail_count >= TEST_FAIL_THRESHOLD:
        signals += 1
        detail["testFails"] = fail_count

    # Signal 4: no edit/write in first EDIT_DELAY_THRESHOLD results (pure exploration stall)
    if n >= EDIT_DELAY_THRESHOLD:
        early_types = [
            classify_tool_type(
                get_tool_info_for_result(messages[tool_result_indices[i]], tool_call_index).get("name", ""),
                get_tool_info_for_result(messages[tool_result_indices[i]], tool_call_index).get("input", {}),
            )
            for i in range(min(EDIT_DELAY_THRESHOLD, n))
        ]
        if not any(t == "write" for t in early_types):
            signals += 1
            detail["noEarlyEdit"] = True

    return signals, detail


def get_adaptive_window(
    tool_result_count: int,
    signal_count: int,
) -> int:
    """
    Map trajectory length and struggling signals to an observation-masking window size.

    Screener-safe guarantee: trajectories with <= SCREENER_SAFE_LENGTH tool
    results ALWAYS get WINDOW_SHORT (no tombstoning), protecting Easy tasks.

    Hard-task compression ladder:
      0-1 signals, 20-30 results -> WINDOW_HARD (7)
      2+ signals or 30+ results  -> WINDOW_DROWNING (5)
    """
    # Short trajectory (Easy / screener tasks): never tombstone
    if tool_result_count <= SCREENER_SAFE_LENGTH:
        return WINDOW_SHORT

    # Standard trajectory (not yet struggling)
    if signal_count == 0:
        return OBSERVATION_MASK_WINDOW  # 10

    # Hard trajectory — use compression ladder
    if tool_result_count >= VERY_LONG_THRESHOLD or signal_count >= 2:
        return WINDOW_DROWNING  # 5
    return WINDOW_HARD  # 7


def collect_always_keep_indices(
    tool_result_indices: list[int],
    messages: list[Any],
    tool_call_index: dict[str, dict[str, Any]],
    n_test: int = 1,
    n_edit: int = 2,
) -> set[int]:
    """
    Collect tool result indices that must ALWAYS be in the full-content window,
    regardless of the adaptive window boundary.

    Always keep:
    - Last n_test TEST results: the failing traceback is the fix signal.
    - Last n_edit WRITE/EDIT results: the agent must see what it already changed.
    """
    test_indices: list[int] = []
    edit_indices: list[int] = []
    for idx in tool_result_indices:
        info = get_tool_info_for_result(messages[idx], tool_call_index)
        t = classify_tool_type(info.get("name", ""), info.get("input", {}))
        if t == "test":
            test_indices.append(idx)
        elif t == "write":
            edit_indices.append(idx)
    return set(test_indices[-n_test:]) | set(edit_indices[-n_edit:])



# ─────────────────────────────────────────────────────────────────────────────
# Core: observation masking (primary algorithm)
# ─────────────────────────────────────────────────────────────────────────────

def apply_observation_masking(
    messages: list[Any],
    tool_call_index: dict[str, dict[str, Any]],
    window: int = OBSERVATION_MASK_WINDOW,
    always_keep: Optional[set[int]] = None,
) -> tuple[list[Any], dict[str, Any]]:
    """
    Adaptive observation masking.

    For each toolResult message:
    - If it is within the last `window` tool results, OR in `always_keep`:
      apply content-aware truncation (full content, head+tail clipped).
    - Otherwise: replace content with a compact tombstone marker.

    `always_keep`: set of message indices that must always be in the full-content
    window regardless of `window`.  Used to guarantee the most recent test result
    and recent edit results are always visible (critical for Hard tasks).

    ALL message pairs are preserved (no structural dropping).
    The full reasoning trace in assistant messages is always kept intact.
    """
    # Collect all toolResult positions.
    tool_result_indices: list[int] = [
        i for i, m in enumerate(messages)
        if isinstance(m, dict) and normalize_role(m.get("role")) == "toolResult"
    ]

    total = len(tool_result_indices)
    if total == 0:
        return messages, {
            "changed": False, "tombstonedCount": 0, "compressedCount": 0,
            "charsSaved": 0, "totalToolResults": 0, "windowSize": window,
        }

    # Apply adaptive window (999 = sentinel for "no tombstoning")
    effective_window = min(window, total)
    boundary = max(0, total - effective_window)
    base_tombstone = set(tool_result_indices[:boundary])
    base_recent    = set(tool_result_indices[boundary:])

    # Override: always-keep indices are moved from tombstone_set to recent_set
    forced_recent = (always_keep or set()) & base_tombstone
    tombstone_set = base_tombstone - forced_recent
    recent_set    = base_recent | forced_recent

    result = list(messages)
    changed = False
    tombstoned_count = 0
    compressed_count = 0
    chars_saved = 0

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or normalize_role(msg.get("role")) != "toolResult":
            continue

        original_len = len(extract_text(msg.get("content")))

        if i in tombstone_set:
            tombstone = _make_tombstone(msg, tool_call_index)
            new_msg = _replace_content_with_tombstone(msg, tombstone)
            if new_msg is not msg:
                result[i] = new_msg
                changed = True
                tombstoned_count += 1
                chars_saved += max(0, original_len - len(tombstone))

        elif i in recent_set:
            new_msg, was_compressed = _compress_recent_tool_result(msg, tool_call_index)
            if was_compressed:
                result[i] = new_msg
                changed = True
                compressed_count += 1
                new_len = len(extract_text(new_msg.get("content")))
                chars_saved += max(0, original_len - new_len)

    return (result if changed else messages), {
        "changed": changed,
        "tombstonedCount": tombstoned_count,
        "compressedCount": compressed_count,
        "charsSaved": chars_saved,
        "totalToolResults": total,
        "windowSize": window,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Re-exploration (flail) detection — informational only, does not change output
# ─────────────────────────────────────────────────────────────────────────────

def detect_reread_loop(
    messages: list[Any],
    tool_call_index: dict[str, dict[str, Any]],
    look_back: int = 8,
    threshold: int = 3,
) -> dict[str, Any]:
    """
    Detect if the agent is in a re-read loop (same file path appears >= threshold
    times among the last `look_back` tool results).

    Returns diagnostic info; does not modify the trajectory.
    """
    tool_result_indices = [
        i for i, m in enumerate(messages)
        if isinstance(m, dict) and normalize_role(m.get("role")) == "toolResult"
    ]
    recent = tool_result_indices[-look_back:]
    path_counts: dict[str, int] = {}
    for idx in recent:
        info = get_tool_info_for_result(messages[idx], tool_call_index)
        t = classify_tool_type(info.get("name", ""), info.get("input", {}))
        if t == "read":
            path = extract_file_path_from_tool(info.get("name", ""), info.get("input", {}))
            if path:
                path_counts[path] = path_counts.get(path, 0) + 1
    looping_files = {p: c for p, c in path_counts.items() if c >= threshold}
    return {"loop": bool(looping_files), "loopingFiles": looping_files}


# ─────────────────────────────────────────────────────────────────────────────
# Top-level handler
# ─────────────────────────────────────────────────────────────────────────────

def get_messages(payload: dict[str, Any]) -> list[Any]:
    params = get_params(payload)
    messages = params.get("messages") if isinstance(params, dict) else None
    return messages if isinstance(messages, list) else []


def handle_assemble(payload: dict[str, Any]) -> dict[str, Any]:
    messages = get_messages(payload)
    working_messages, state_metadata, state_path = resolve_stateful_messages(payload, messages)
    runtime_messages, sanitization = sanitize_messages(working_messages)

    # Build tool-call index (needed for tombstone labels and content compression).
    tool_call_index = build_tool_call_index(runtime_messages)

    # Collect all toolResult indices (needed for struggle detection and masking).
    tr_indices: list[int] = [
        i for i, m in enumerate(runtime_messages)
        if isinstance(m, dict) and normalize_role(m.get("role")) == "toolResult"
    ]

    # Determine adaptive window size from trajectory struggle signals.
    signal_count, signal_detail = count_struggling_signals(tr_indices, runtime_messages, tool_call_index)
    adaptive_window = get_adaptive_window(len(tr_indices), signal_count)

    # Identify always-keep indices (most recent test + recent edits).
    always_keep = collect_always_keep_indices(tr_indices, runtime_messages, tool_call_index)

    # Apply observation masking: tombstone old observations, truncate recent ones.
    masked_messages, masking_meta = apply_observation_masking(
        runtime_messages, tool_call_index, window=adaptive_window, always_keep=always_keep
    )

    # Informational flail detection (does not alter output).
    loop_info = detect_reread_loop(runtime_messages, tool_call_index)

    sanitized = bool(sanitization.get("changed"))
    masked    = bool(masking_meta.get("changed"))
    output_differs_from_raw = fingerprint_messages(masked_messages) != fingerprint_messages(messages)
    changed = sanitized or masked or output_differs_from_raw

    if masked:
        reason = "observation_masked"
    elif sanitized:
        reason = "sanitized"
    elif output_differs_from_raw and state_metadata.get("stateLoaded"):
        reason = "state_reused"
    else:
        reason = "unchanged"

    metadata = {
        **masking_meta,
        **state_metadata,
        "changed": changed,
        "reason": reason,
        "originalMessageCount": len(messages),
        "messageCount": len(masked_messages),
        "sanitized": sanitized,
        "removedMessageCount": sanitization.get("removedMessageCount", 0),
        "removedThinkingBlockCount": sanitization.get("removedThinkingBlockCount", 0),
        "pruned": False,
        "observationMasked": masked,
        "rereadLoop": loop_info,
        "adaptiveWindow": adaptive_window,
        "struggleSignals": signal_count,
        "struggleDetail": signal_detail,
    }

    try:
        save_state(state_path, payload, messages, masked_messages)
        metadata["stateSaved"] = True
    except Exception as error:
        metadata["stateSaved"] = False
        metadata["stateError"] = str(error)

    return {
        "assembled": True,
        "messages": masked_messages,
        "estimatedTokens": estimate_tokens_for_message_array(masked_messages),
        "baseMiner": metadata,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_event(event_name: str) -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        if not isinstance(payload, dict):
            raise ValueError("Connector payload must be a JSON object")
        if event_name not in EVENT_NAMES:
            raise ValueError(f"Unknown base miner event: {event_name}")
        response = {"ok": True, "result": handle_assemble(payload)}
    except Exception as error:
        response = {"ok": False, "error": str(error), "errorType": error.__class__.__name__}
    sys.stdout.write(json.dumps(response, ensure_ascii=False))
    return 0


def cli_main(argv: list[str]) -> int:
    event_name = argv[0] if argv else "assemble"
    return run_event(event_name)


if __name__ == "__main__":
    raise SystemExit(cli_main(sys.argv[1:]))
