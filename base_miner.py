#!/usr/bin/env python3
"""SOMA miner for assemble-time trajectory pruning.

Algorithmic (no-LLM) context compressor for OpenClaw/qwen3-coder SWE-bench agents.
Key strategy: content-aware truncation + stale-read deduplication + smart result
selection keeps the agent focused without losing solve-critical signals.
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

# ── structural window ─────────────────────────────────────────────────────────
# How many tool results to keep in the final compressed trajectory.
# Raised from 4 → 6 so the agent keeps a broader recent horizon; content
# truncation (below) compensates for the extra messages.
KEEP_TOOL_RESULT_COUNT = 6

# ── per-result content limits (characters; ~4 chars ≈ 1 token) ───────────────
# File-read results: keep structure (imports / class defs) from the head,
# and the active editing region from the tail.
READ_HEAD_CHARS = 2000   # ~500 tokens
READ_TAIL_CHARS = 4000   # ~1000 tokens  (most-recent code is load-bearing)

# Shell / exec results: environment line from head, errors/output from tail.
EXEC_HEAD_CHARS = 800    # ~200 tokens
EXEC_TAIL_CHARS = 4800   # ~1200 tokens  (errors live at the end)

# Test-run results: critical – failing traceback is always at the tail.
TEST_TAIL_CHARS = 10000  # ~2500 tokens  (keep full failure output)

# Write / edit / patch results: usually small acknowledgements.
WRITE_MAX_CHARS = 2000   # ~500 tokens

# Generic fallback for unrecognised tool types.
GENERIC_HEAD_CHARS = 1000
GENERIC_TAIL_CHARS = 4000

# Minimum content length before truncation is applied (avoid truncating tiny results).
MIN_CHARS_TO_TRUNCATE = 500

# ── state persistence ─────────────────────────────────────────────────────────
STATE_VERSION = 1
STATE_DIR_NAME = "state"

# ── truncation marker ─────────────────────────────────────────────────────────
_OMIT_MIDDLE = "\n[...{count} chars omitted...]\n"
_OMIT_HEAD   = "[...{count} chars omitted from start...]\n"


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers (unchanged from baseline)
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


def find_first_user_index(messages: list[Any]) -> Optional[int]:
    for index, message in enumerate(messages):
        if isinstance(message, dict) and normalize_role(message.get("role")) == "user":
            return index
    return None


def find_tool_call_indices(messages: list[Any], tool_result_indices: list[int]) -> list[int]:
    wanted_ids: set[str] = set()
    for index in tool_result_indices:
        wanted_ids.update(extract_tool_result_ids(messages[index]))

    matched: list[int] = []
    if wanted_ids:
        for index, message in enumerate(messages):
            if extract_tool_call_ids(message) & wanted_ids:
                matched.append(index)

    if matched:
        return matched

    first_tool_result_index = min(tool_result_indices)
    for index in range(first_tool_result_index - 1, -1, -1):
        if extract_tool_call_ids(messages[index]):
            return [index]

    return []


def filter_tool_call_message(message: Any, wanted_ids: set[str]) -> Any:
    if not isinstance(message, dict) or not wanted_ids:
        return message

    filtered = copy.deepcopy(message)
    content = filtered.get("content")
    if isinstance(content, list):
        filtered["content"] = [
            block for block in content
            if not (
                isinstance(block, dict)
                and block.get("type") == "toolCall"
                and isinstance(block.get("id"), str)
                and block["id"].strip() not in wanted_ids
            )
        ]

    for field in ("toolCalls", "tool_calls"):
        tool_calls = filtered.get(field)
        if isinstance(tool_calls, list):
            filtered[field] = [
                tool_call for tool_call in tool_calls
                if not (
                    isinstance(tool_call, dict)
                    and isinstance(tool_call.get("id"), str)
                    and tool_call["id"].strip() not in wanted_ids
                )
            ]

    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Tool call index  (new)
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
# Tool-type classification  (new)
# ─────────────────────────────────────────────────────────────────────────────

# Names that indicate a file-read operation.
_READ_TOOL_NAMES = frozenset({
    "read", "read_file", "view", "view_file", "cat", "open",
    "get_file_content", "file_read", "show_file",
})

# Names that indicate a file-write / edit / patch operation.
_WRITE_TOOL_NAMES = frozenset({
    "write", "write_file", "edit", "edit_file", "str_replace",
    "str_replace_editor", "create_file", "replace", "patch",
    "apply_patch", "insert", "delete", "update_file",
})

# Shell-execution names.
_EXEC_TOOL_NAMES = frozenset({
    "exec", "bash", "shell", "run", "run_command", "execute",
    "terminal", "cmd", "command",
})

# Keywords that identify a test run inside a shell command.
_TEST_KEYWORDS = ("pytest", "python -m pytest", "python -m test", "unittest",
                  "nose2", "tox", "python test", "py.test")


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
    # Heuristic fallback: if input has a "path" key it's likely a read.
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
# Content truncation helpers  (new)
# ─────────────────────────────────────────────────────────────────────────────

def _truncate_head_tail(text: str, head: int, tail: int) -> str:
    """Keep the first `head` and last `tail` characters, with an omission marker."""
    if len(text) <= head + tail:
        return text
    omitted = len(text) - head - tail
    marker = _OMIT_MIDDLE.format(count=omitted)
    return text[:head] + marker + text[-tail:]


def _truncate_tail_only(text: str, max_chars: int) -> str:
    """Keep only the last `max_chars` characters (errors live at the end)."""
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    marker = _OMIT_HEAD.format(count=omitted)
    return marker + text[-max_chars:]


def _compress_text_by_type(text: str, tool_type: str) -> str:
    """Apply type-specific truncation to a raw text string."""
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
            omitted = len(text) - WRITE_MAX_CHARS
            return text[:WRITE_MAX_CHARS] + f"\n[...{omitted} chars omitted...]"
        return text
    # Generic fallback.
    return _truncate_head_tail(text, GENERIC_HEAD_CHARS, GENERIC_TAIL_CHARS)


def _compress_content(content: Any, tool_type: str) -> tuple[Any, bool]:
    """Recursively compress content, returning (new_content, changed)."""
    if isinstance(content, str):
        compressed = _compress_text_by_type(content, tool_type)
        return compressed, compressed != content

    if isinstance(content, list):
        new_blocks: list[Any] = []
        changed = False
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type", "")
                # Text blocks inside tool results.
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
                    # Recurse into nested content.
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


def compress_tool_result_message(
    message: Any,
    tool_call_index: dict[str, dict[str, Any]],
) -> tuple[Any, bool]:
    """Return a (possibly content-truncated) copy of a toolResult message."""
    if not isinstance(message, dict):
        return message, False
    tool_info = get_tool_info_for_result(message, tool_call_index)
    tool_name = tool_info.get("name", "")
    tool_input = tool_info.get("input", {})
    tool_type = classify_tool_type(tool_name, tool_input)

    content = message.get("content")
    new_content, changed = _compress_content(content, tool_type)
    if not changed:
        return message, False
    new_msg = copy.deepcopy(message)
    new_msg["content"] = new_content
    return new_msg, True


def compress_all_tool_results(
    messages: list[Any],
    tool_call_index: dict[str, dict[str, Any]],
) -> tuple[list[Any], dict[str, Any]]:
    """Apply content-aware truncation to every toolResult in the list."""
    result = list(messages)
    compressed_count = 0
    chars_saved = 0

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or normalize_role(msg.get("role")) != "toolResult":
            continue
        original_len = len(extract_text(msg.get("content")))
        new_msg, changed = compress_tool_result_message(msg, tool_call_index)
        if changed:
            result[i] = new_msg
            compressed_count += 1
            chars_saved += original_len - len(extract_text(new_msg.get("content")))

    any_changed = compressed_count > 0
    return (result if any_changed else messages), {
        "changed": any_changed,
        "compressedCount": compressed_count,
        "charsSaved": chars_saved,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Smart tool-result selection  (new)
# ─────────────────────────────────────────────────────────────────────────────

def select_tool_results_smart(
    all_result_indices: list[int],
    messages: list[Any],
    tool_call_index: dict[str, dict[str, Any]],
    keep_count: int,
) -> list[int]:
    """
    Select which `keep_count` tool result indices to keep, using these rules:

    1. Deduplicate reads: for the same file path, keep only the most-recent
       read result (earlier reads of that file are superseded).
    2. Prioritise test-run outputs — they carry failing-traceback signal.
    3. Fill remaining slots with the most-recent results.

    Chronological ordering is preserved in the returned list.
    """
    if len(all_result_indices) <= keep_count:
        return all_result_indices

    # Build per-index metadata.
    meta: list[dict[str, Any]] = []
    for idx in all_result_indices:
        msg = messages[idx]
        tool_info = get_tool_info_for_result(msg, tool_call_index)
        tool_name = tool_info.get("name", "")
        tool_input = tool_info.get("input", {})
        tool_type = classify_tool_type(tool_name, tool_input)
        file_path = (
            extract_file_path_from_tool(tool_name, tool_input)
            if tool_type == "read" else None
        )
        meta.append({"idx": idx, "type": tool_type, "file_path": file_path})

    # Deduplicate reads: scan newest-first, keep only first occurrence per path.
    seen_paths: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for m in reversed(meta):
        if m["type"] == "read" and m["file_path"]:
            if m["file_path"] in seen_paths:
                continue  # stale read – superseded by a later read
            seen_paths.add(m["file_path"])
        deduped.append(m)
    deduped.reverse()  # restore chronological order

    if len(deduped) <= keep_count:
        return [m["idx"] for m in deduped]

    # Among the deduplicated set, prefer to keep test outputs.
    test_indices = [m["idx"] for m in deduped if m["type"] == "test"]
    non_test = [m["idx"] for m in deduped if m["type"] != "test"]

    # Always keep up to 2 most-recent test outputs.
    kept_tests = test_indices[-2:] if len(test_indices) > 2 else test_indices
    remaining_slots = keep_count - len(kept_tests)
    kept_non_test = non_test[-remaining_slots:] if remaining_slots > 0 else []

    # Merge and sort by original index to restore chronological order.
    selected = sorted(set(kept_tests) | set(kept_non_test))
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Structural pruning  (enhanced)
# ─────────────────────────────────────────────────────────────────────────────

def prune_messages(
    messages: list[Any],
    tool_call_index: Optional[dict[str, dict[str, Any]]] = None,
) -> tuple[list[Any], dict[str, Any]]:
    first_user_index = find_first_user_index(messages)
    if first_user_index is None:
        return messages, {"changed": False, "reason": "missing_first_user_message"}

    tool_result_indices = [
        index for index, message in enumerate(messages)
        if isinstance(message, dict) and normalize_role(message.get("role")) == "toolResult"
    ]
    if len(tool_result_indices) < KEEP_TOOL_RESULT_COUNT:
        return messages, {
            "changed": False,
            "reason": "fewer_than_threshold_tool_results",
            "toolResultCount": len(tool_result_indices),
        }

    # Smart selection (dedup + priority) when index is available.
    if tool_call_index is not None:
        kept_tool_result_indices = select_tool_results_smart(
            tool_result_indices, messages, tool_call_index, KEEP_TOOL_RESULT_COUNT
        )
    else:
        kept_tool_result_indices = tool_result_indices[-KEEP_TOOL_RESULT_COUNT:]

    kept_tool_result_ids: set[str] = set()
    for index in kept_tool_result_indices:
        kept_tool_result_ids.update(extract_tool_result_ids(messages[index]))

    tool_call_indices = find_tool_call_indices(messages, kept_tool_result_indices)
    if not tool_call_indices:
        return messages, {"changed": False, "reason": "missing_invoking_tool_call"}

    keep_indices = {first_user_index, *kept_tool_result_indices, *tool_call_indices}
    pruned = [
        filter_tool_call_message(message, kept_tool_result_ids) if index in tool_call_indices else message
        for index, message in enumerate(messages)
        if index in keep_indices
    ]
    changed = len(pruned) != len(messages)
    return pruned if changed else messages, {
        "changed": changed,
        "reason": "pruned" if changed else "nothing_to_remove",
        "originalMessageCount": len(messages),
        "messageCount": len(pruned) if changed else len(messages),
        "keptToolResultCount": KEEP_TOOL_RESULT_COUNT,
        "keptToolCallMessageCount": len(tool_call_indices),
    }


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

    # Build tool-call index before any compression/pruning.
    tool_call_index = build_tool_call_index(runtime_messages)

    # Content-aware truncation of every toolResult (in-place, no structural changes).
    content_compressed, compression_meta = compress_all_tool_results(runtime_messages, tool_call_index)

    # Structural pruning with smart tool-result selection.
    pruned_messages, prune_meta = prune_messages(content_compressed, tool_call_index)

    pruned   = bool(prune_meta.get("changed"))
    sanitized = bool(sanitization.get("changed"))
    content_changed = bool(compression_meta.get("changed"))
    output_differs_from_raw = fingerprint_messages(pruned_messages) != fingerprint_messages(messages)
    changed = pruned or sanitized or content_changed or output_differs_from_raw

    reason = prune_meta.get("reason")
    if pruned:
        reason = "pruned"
    elif content_changed:
        reason = "content_compressed"
    elif sanitized:
        reason = "sanitized"
    elif output_differs_from_raw and state_metadata.get("stateLoaded"):
        reason = "state_reused"

    metadata = {
        **prune_meta,
        **state_metadata,
        "changed": changed,
        "reason": reason,
        "originalMessageCount": len(messages),
        "messageCount": len(pruned_messages),
        "sanitized": sanitized,
        "removedMessageCount": sanitization.get("removedMessageCount", 0),
        "removedThinkingBlockCount": sanitization.get("removedThinkingBlockCount", 0),
        "pruned": pruned,
        "contentCompressed": content_changed,
        "compressedToolResultCount": compression_meta.get("compressedCount", 0),
        "charsSaved": compression_meta.get("charsSaved", 0),
    }

    try:
        save_state(state_path, payload, messages, pruned_messages)
        metadata["stateSaved"] = True
    except Exception as error:
        metadata["stateSaved"] = False
        metadata["stateError"] = str(error)

    return {
        "assembled": True,
        "messages": pruned_messages,
        "estimatedTokens": estimate_tokens_for_message_array(pruned_messages),
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
