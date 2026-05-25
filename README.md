# SOMA Miner Context Engine for OpenClaw

SOMA Miner is an OpenClaw context-engine plugin with a connector-only JavaScript layer and an active baseline Python miner.

## Layout

- `index.js` - OpenClaw connector. It registers the plugin, owns OpenClaw lifecycle hooks, exposes the small CLI surface, and calls Python from `assemble` before OpenClaw sends context to the LLM.
- `base_miner.py` - It exposes one connector event, `assemble`, stores the current mined trajectory per session, and prunes by keeping the first user message, the last 4 tool results, and the assistant tool-call message that invoked them.
- `openclaw.plugin.json` - plugin manifest without compression or mining config.
- `requirements.txt` - Python dependency list.

## Install

```bash
cd /path/to/SOMA-plugin
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
openclaw plugins install --link /path/to/SOMA-plugin --dangerously-force-unsafe-install
openclaw soma-miner on
```

The connector runs `./.venv/bin/python` when present and falls back to `python3`.

## CLI

```bash
openclaw soma-miner on
openclaw soma-miner off
openclaw soma-miner trajectory on
openclaw soma-miner trajectory off
```

`on` enables SOMA Miner as the active OpenClaw context engine. `off` clears it from the context-engine slot.

`trajectory` is connector-only:

- `on` makes the connector return the trajectory produced by `base_miner.py` during `assemble`.
- `off` leaves OpenClaw messages unchanged for trajectory-changing hooks.

The miner still runs during `assemble` in both modes; the flag only decides whether the connector returns the Python trajectory or the original OpenClaw messages.

The `trajectory` setting is stored in `connector-state.json` next to the plugin. It is not read from OpenClaw config.

The connector does not accept compression ratios, thresholds, artifact paths, or mining settings. The active baseline behavior lives in `base_miner.py`.

`base_miner.py` stores the canonical mined trajectory under `logs/state/`. On each `assemble`, it loads the previous state for the session and appends only messages that are new in OpenClaw's full runtime history before sanitizing and pruning again. This keeps the next LLM input based on the last mined trajectory rather than repeatedly starting from the entire OpenClaw session log.

When OpenClaw exposes or can infer the native session file, the connector rewrites `agents/<agent>/sessions/<session>.jsonl` so its message rows are the current mined message state returned by Python. The raw pre-compaction history remains available in `logs/io/` samples for debugging, while the OpenClaw session trajectory reflects the state that will be used on the next `assemble`.

The Python connector surface is intentionally just:

```bash
python3 base_miner.py assemble
```

OpenClaw lifecycle hooks such as register, ingest, compact, maintain, and dispose stay in `index.js`. The connector calls the Python `assemble` event from OpenClaw's `assemble` hook, so the miner can react before the next LLM request is sent.

## Configuration Boundary

OpenClaw config is used only when `soma-miner on/off` switches the active context-engine slot. The connector does not read compression configuration from `openclaw.json`, and it does not pass OpenClaw config snapshots into Python.

The baseline miner has no external dependencies. Runtime IO samples are written by the connector under `logs/io/` when pruning happens. The per-session current trajectory is written by the Python miner under `logs/state/`, and the connector mirrors that state into the native OpenClaw session JSONL when that path is available.
