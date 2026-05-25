#!/usr/bin/env python3
"""Minimal SOMA miner baseline for assemble-time trajectory pruning."""

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
KEEP_TOOL_RESULT_COUNT = 4
STATE_VERSION = 1
STATE_DIR_NAME = "state"


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


def save_state(state_path: Path, payload: dict[str, Any], raw_messages: list[Any], current_messages: list[Any]) -> None:
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


def resolve_stateful_messages(payload: dict[str, Any], raw_messages: list[Any]) -> tuple[list[Any], dict[str, Any], Path]:
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


def prune_messages(messages: list[Any]) -> tuple[list[Any], dict[str, Any]]:
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
            "reason": "fewer_than_four_tool_results",
            "toolResultCount": len(tool_result_indices),
        }

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


def get_messages(payload: dict[str, Any]) -> list[Any]:
    params = get_params(payload)
    messages = params.get("messages") if isinstance(params, dict) else None
    return messages if isinstance(messages, list) else []


def handle_assemble(payload: dict[str, Any]) -> dict[str, Any]:
    messages = get_messages(payload)
    working_messages, state_metadata, state_path = resolve_stateful_messages(payload, messages)
    runtime_messages, sanitization = sanitize_messages(working_messages)
    pruned_messages, metadata = prune_messages(runtime_messages)
    pruned = bool(metadata.get("changed"))
    sanitized = bool(sanitization.get("changed"))
    output_differs_from_raw = fingerprint_messages(pruned_messages) != fingerprint_messages(messages)
    changed = pruned or sanitized or output_differs_from_raw
    reason = metadata.get("reason")
    if pruned:
        reason = "pruned"
    elif sanitized:
        reason = "sanitized"
    elif output_differs_from_raw and state_metadata.get("stateLoaded"):
        reason = "state_reused"

    metadata = {
        **metadata,
        **state_metadata,
        "changed": changed,
        "reason": reason,
        "originalMessageCount": len(messages),
        "messageCount": len(pruned_messages),
        "sanitized": sanitized,
        "removedMessageCount": sanitization.get("removedMessageCount", 0),
        "removedThinkingBlockCount": sanitization.get("removedThinkingBlockCount", 0),
        "pruned": pruned,
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