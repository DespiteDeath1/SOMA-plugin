import { appendFileSync, existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import path from "node:path";
import { createHash, randomUUID } from "node:crypto";
import { fileURLToPath } from "node:url";

const PLUGIN_ID = "soma-miner";
const PLUGIN_NAME = "SOMA Miner";
const DEFAULT_REPLACE_TRAJECTORY = true;
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const CONNECTOR_STATE_FILE = path.join(__dirname, "connector-state.json");
const IO_LOG_DIR = path.join(__dirname, "logs", "io");
const RUNTIME_LOG_FILE = path.join(__dirname, "logs", "runtime-hooks.jsonl");
const contextEngineSessionFiles = new Map();

function cloneJson(value, fallback = null) {
  try {
    return JSON.parse(JSON.stringify(value ?? fallback));
  } catch {
    return fallback;
  }
}

function getRuntimeConfig(api) {
  return api.runtime?.config ?? {};
}

function loadConfigForConnectorWrite(api) {
  const runtimeConfig = getRuntimeConfig(api);
  try {
    const current = typeof runtimeConfig.current === "function"
      ? runtimeConfig.current()
      : runtimeConfig.loadConfig?.();
    if (current) {
      return cloneJson(current.cfg ?? current, {});
    }
  } catch {
    // Fall through to api.config.
  }
  return cloneJson(api.config ?? {}, {});
}

async function writeOpenClawConfig(api, cfg) {
  const writer = getRuntimeConfig(api).writeConfigFile;
  if (typeof writer !== "function") {
    throw new Error("Config write API is not available in this OpenClaw runtime");
  }
  await writer(cfg);
}

function ensureObjectProperty(target, key) {
  if (!target[key] || typeof target[key] !== "object" || Array.isArray(target[key])) {
    target[key] = {};
  }
  return target[key];
}

function readConnectorState() {
  let raw = {};
  try {
    raw = JSON.parse(readFileSync(CONNECTOR_STATE_FILE, "utf-8"));
  } catch {
    raw = {};
  }
  return {
    replaceTrajectory: typeof raw.replaceTrajectory === "boolean"
      ? raw.replaceTrajectory
      : DEFAULT_REPLACE_TRAJECTORY,
  };
}

function writeConnectorState(nextState) {
  const merged = {
    ...readConnectorState(),
    ...nextState,
    updatedAt: new Date().toISOString(),
  };
  writeFileSync(CONNECTOR_STATE_FILE, `${JSON.stringify(merged, null, 2)}\n`, "utf-8");
  return merged;
}

function buildPayload(params = null, extra = {}) {
  return {
    pluginId: PLUGIN_ID,
    pluginName: PLUGIN_NAME,
    pluginDir: __dirname,
    params: cloneJson(params, null),
    ...extra,
  };
}

function cacheContextEngineSessionFile(params = {}) {
  const directSessionFile = typeof params?.sessionFile === "string" && params.sessionFile.trim()
    ? params.sessionFile.trim()
    : null;
  if (!directSessionFile) {
    return null;
  }

  const sessionId = typeof params?.sessionId === "string" ? params.sessionId.trim() : "";
  const sessionKey = typeof params?.sessionKey === "string" ? params.sessionKey.trim() : "";
  if (sessionId) {
    contextEngineSessionFiles.set(`id:${sessionId}`, directSessionFile);
  }
  if (sessionKey) {
    contextEngineSessionFiles.set(`key:${sessionKey}`, directSessionFile);
  }
  return directSessionFile;
}

function parseAgentIdFromSessionKey(sessionKey) {
  if (typeof sessionKey !== "string") {
    return null;
  }
  const match = sessionKey.trim().match(/^agent:([^:]+):/);
  return match?.[1] ?? null;
}

function inferContextEngineSessionFile(params = {}) {
  const sessionId = typeof params?.sessionId === "string" && params.sessionId.trim()
    ? params.sessionId.trim()
    : null;
  if (!sessionId) {
    return null;
  }

  const agentId = typeof params?.agentId === "string" && params.agentId.trim()
    ? params.agentId.trim()
    : parseAgentIdFromSessionKey(params?.sessionKey);
  if (!agentId) {
    return null;
  }

  const stateDir = typeof process.env.OPENCLAW_STATE_DIR === "string" && process.env.OPENCLAW_STATE_DIR.trim()
    ? process.env.OPENCLAW_STATE_DIR.trim()
    : path.join(process.env.HOME || "/home/node", ".openclaw");
  const sessionFile = path.join(stateDir, "agents", agentId, "sessions", `${sessionId}.jsonl`);
  return existsSync(sessionFile) ? sessionFile : null;
}

function resolveContextEngineSessionFile(params = {}) {
  const directSessionFile = cacheContextEngineSessionFile(params);
  if (directSessionFile) {
    return directSessionFile;
  }

  const sessionKey = typeof params?.sessionKey === "string" ? params.sessionKey.trim() : "";
  if (sessionKey) {
    const cachedSessionFile = contextEngineSessionFiles.get(`key:${sessionKey}`);
    if (typeof cachedSessionFile === "string" && cachedSessionFile.trim()) {
      return cachedSessionFile;
    }
  }

  const sessionId = typeof params?.sessionId === "string" ? params.sessionId.trim() : "";
  if (sessionId) {
    const cachedSessionFile = contextEngineSessionFiles.get(`id:${sessionId}`);
    if (typeof cachedSessionFile === "string" && cachedSessionFile.trim()) {
      return cachedSessionFile;
    }
  }

  const inferredSessionFile = inferContextEngineSessionFile(params);
  if (inferredSessionFile) {
    cacheContextEngineSessionFile({
      sessionId: params?.sessionId,
      sessionKey: params?.sessionKey,
      sessionFile: inferredSessionFile,
    });
    return inferredSessionFile;
  }

  return null;
}

function resolveCompressionServiceUrl() {
  const raw = typeof process.env.SOMA_COMPRESSION_SERVICE_URL === "string"
    ? process.env.SOMA_COMPRESSION_SERVICE_URL.trim()
    : "";
  return raw || null;
}

async function callCompressionService(params) {
  const serviceUrl = resolveCompressionServiceUrl();
  if (!serviceUrl) {
    throw new Error("SOMA_COMPRESSION_SERVICE_URL is not set — compression service URL is required");
  }

  const messages = Array.isArray(params?.messages) ? params.messages : [];
  const body = {
    messages,
    session_id: params?.sessionId ?? null,
    session_key: params?.sessionKey ?? null,
    current_token_count: Number.isFinite(params?.currentTokenCount) ? params.currentTokenCount : null,
  };

  let response;
  try {
    response = await fetch(`${serviceUrl}/compress`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Compression service unreachable at ${serviceUrl}: ${message}`);
  }

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`Compression service returned HTTP ${response.status}: ${text.slice(0, 200)}`);
  }

  const result = await response.json();

  if (!result.compress) {
    return null;
  }

  return result.trajectory ?? null;
}

async function setPluginEnabled(api, enabled) {
  const cfg = loadConfigForConnectorWrite(api);
  const plugins = ensureObjectProperty(cfg, "plugins");
  const entries = ensureObjectProperty(plugins, "entries");
  const pluginEntry = ensureObjectProperty(entries, PLUGIN_ID);
  const slots = ensureObjectProperty(plugins, "slots");

  pluginEntry.enabled = true;
  if (enabled) {
    slots.contextEngine = PLUGIN_ID;
  } else if (slots.contextEngine === PLUGIN_ID) {
    delete slots.contextEngine;
  }

  await writeOpenClawConfig(api, cfg);
  console.log(JSON.stringify({ ok: true, enabled, contextEngine: slots.contextEngine ?? null }, null, 2));
}

async function setConnectorTrajectoryMode(enabled) {
  const state = writeConnectorState({ replaceTrajectory: enabled });
  console.log(JSON.stringify({ ok: true, replaceTrajectory: state.replaceTrajectory }, null, 2));
}

function safeFilePart(value, fallback = "session") {
  const raw = typeof value === "string" && value.trim() ? value.trim() : fallback;
  const normalized = raw.replace(/[^A-Za-z0-9_.-]+/g, "-").replace(/^-+|-+$/g, "");
  return (normalized || fallback).slice(0, 80);
}

function hasCompressionResult(trajectory) {
  return trajectory?.compaction?.compacted === true
    || trajectory?.compacted === true
    || trajectory?.baseMiner?.pruned === true
    || trajectory?.baseMiner?.reason === "pruned";
}

function shouldUseMinerTrajectory(trajectory) {
  return hasCompressionResult(trajectory)
    || trajectory?.baseMiner?.changed === true;
}

function writeJsonFile(filePath, payload) {
  writeFileSync(filePath, `${JSON.stringify(payload, null, 2)}\n`, "utf-8");
}

function writeRuntimeHookMarker(eventType, payload = {}) {
  try {
    mkdirSync(path.dirname(RUNTIME_LOG_FILE), { recursive: true });
    const entry = {
      timestamp: new Date().toISOString(),
      pluginId: PLUGIN_ID,
      eventType,
      ...payload,
    };
    appendFileSync(RUNTIME_LOG_FILE, `${JSON.stringify(entry)}\n`, "utf-8");
    console.info(`[${PLUGIN_ID}] ${eventType} ${JSON.stringify(payload)}`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.warn(`[${PLUGIN_ID}] failed to write runtime hook marker: ${message}`);
  }
}

function normalizeRole(value) {
  if (typeof value !== "string") {
    return "";
  }
  const lowered = value.trim().toLowerCase().replace(/[_-]+/g, "");
  return lowered === "toolresult" ? "toolResult" : lowered;
}

function sortForJson(value) {
  if (Array.isArray(value)) {
    return value.map((item) => sortForJson(item));
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, sortForJson(value[key])]),
    );
  }
  return value;
}

function fingerprintValue(value) {
  return createHash("sha256")
    .update(JSON.stringify(sortForJson(value)))
    .digest("hex");
}

function extractToolResultIds(message) {
  if (!message || typeof message !== "object" || normalizeRole(message.role) !== "toolResult") {
    return [];
  }
  return [message.toolCallId, message.toolUseId, message.id]
    .filter((value) => typeof value === "string" && value.trim())
    .map((value) => value.trim());
}

function extractToolCallIds(message) {
  if (!message || typeof message !== "object" || normalizeRole(message.role) !== "assistant") {
    return [];
  }

  const ids = [];
  if (Array.isArray(message.content)) {
    for (const block of message.content) {
      if (block && typeof block === "object" && block.type === "toolCall" && typeof block.id === "string" && block.id.trim()) {
        ids.push(block.id.trim());
      }
    }
  }

  for (const field of ["toolCalls", "tool_calls"]) {
    if (!Array.isArray(message[field])) {
      continue;
    }
    for (const toolCall of message[field]) {
      if (toolCall && typeof toolCall === "object" && typeof toolCall.id === "string" && toolCall.id.trim()) {
        ids.push(toolCall.id.trim());
      }
    }
  }

  return [...new Set(ids)];
}

function messageIdentityKeys(message) {
  if (!message || typeof message !== "object") {
    return [];
  }

  const role = normalizeRole(message.role);
  const keys = [];
  if (["string", "number"].includes(typeof message.timestamp) && String(message.timestamp).trim()) {
    keys.push(`${role}:timestamp:${String(message.timestamp).trim()}`);
  }

  if (role === "toolResult") {
    for (const toolCallId of extractToolResultIds(message).sort()) {
      keys.push(`${role}:toolResult:${toolCallId}`);
    }
  } else if (role === "assistant") {
    const toolCallIds = extractToolCallIds(message).sort();
    if (toolCallIds.length > 0) {
      keys.push(`${role}:toolCallSet:${toolCallIds.join(",")}`);
    }
    for (const toolCallId of toolCallIds) {
      keys.push(`${role}:toolCall:${toolCallId}`);
    }
  }

  keys.push(`${role}:fingerprint:${fingerprintValue(message)}`);
  return keys;
}

function isMessageEntry(entry) {
  return entry?.type === "message" && entry?.message && typeof entry.message === "object";
}

function readTranscriptEntries(sessionFile) {
  const raw = readFileSync(sessionFile, "utf-8");
  const entries = [];
  for (const line of raw.split(/\r?\n/u)) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }
    try {
      const parsed = JSON.parse(trimmed);
      if (parsed && typeof parsed === "object") {
        entries.push(parsed);
      }
    } catch {
      // Ignore malformed diagnostic rows instead of blocking the runtime hook.
    }
  }
  return entries;
}

function getLeadingStructuralEntries(entries) {
  const leading = [];
  for (const entry of entries) {
    if (isMessageEntry(entry) || entry?.type === "compaction" || entry?.customType === "soma-miner-state") {
      break;
    }
    leading.push(cloneJson(entry, entry));
  }
  return leading;
}

function buildExistingMessageEntryMap(entries) {
  const entryMap = new Map();
  for (const entry of entries) {
    if (!isMessageEntry(entry)) {
      continue;
    }
    for (const key of messageIdentityKeys(entry.message)) {
      if (!entryMap.has(key)) {
        entryMap.set(key, []);
      }
      entryMap.get(key).push(entry);
    }
  }
  return entryMap;
}

function findExistingMessageEntry(message, entryMap, usedEntryIds) {
  for (const key of messageIdentityKeys(message)) {
    for (const entry of entryMap.get(key) ?? []) {
      const marker = typeof entry.id === "string" && entry.id.trim() ? entry.id.trim() : fingerprintValue(entry);
      if (usedEntryIds.has(marker)) {
        continue;
      }
      usedEntryIds.add(marker);
      return entry;
    }
  }
  return null;
}

function buildMessageEntry(message, existingEntry, parentId, index) {
  const entry = existingEntry && typeof existingEntry === "object" ? cloneJson(existingEntry, {}) : {};
  entry.type = "message";
  if (typeof entry.id !== "string" || !entry.id.trim()) {
    entry.id = `soma-message-${index}-${randomUUID()}`;
  }
  entry.parentId = parentId;
  if (typeof entry.timestamp !== "string" || !entry.timestamp.trim()) {
    entry.timestamp = new Date().toISOString();
  }
  entry.message = message;
  return entry;
}

function updateSessionIndexAfterTrajectoryState(sessionFile, messageCount) {
  const indexFile = path.join(path.dirname(sessionFile), "sessions.json");
  if (!existsSync(indexFile)) {
    return;
  }

  try {
    const parsed = JSON.parse(readFileSync(indexFile, "utf-8"));
    const sessionId = path.basename(sessionFile, ".jsonl");
    const matchedKey = Object.keys(parsed ?? {}).find((key) => {
      const entry = parsed?.[key];
      return entry && typeof entry === "object" && (entry.sessionId === sessionId || entry.sessionFile === sessionFile);
    });
    if (!matchedKey || !parsed[matchedKey] || typeof parsed[matchedKey] !== "object") {
      return;
    }
    parsed[matchedKey].updatedAt = Date.now();
    parsed[matchedKey].messageCount = messageCount;
    writeJsonFile(indexFile, parsed);
  } catch {
    // Session index is diagnostic; failing to update it must not block assemble.
  }
}

function persistTrajectoryState(params, minerTrajectory, { sessionFile, useMinerTrajectory }) {
  const outputMessages = Array.isArray(minerTrajectory?.messages) ? minerTrajectory.messages : [];
  const result = {
    sessionFile: sessionFile ?? null,
    trajectoryStateSaved: false,
    trajectoryStateMessageCount: null,
    trajectoryStateError: null,
  };

  if (!useMinerTrajectory) {
    return result;
  }
  if (!sessionFile) {
    return { ...result, trajectoryStateError: "missing_session_file" };
  }
  if (!existsSync(sessionFile)) {
    return { ...result, trajectoryStateError: "session_file_not_found" };
  }
  if (outputMessages.length === 0) {
    return { ...result, trajectoryStateError: "missing_output_messages" };
  }

  try {
    const entries = readTranscriptEntries(sessionFile);
    const leadingEntries = getLeadingStructuralEntries(entries);
    const entryMap = buildExistingMessageEntryMap(entries);
    let previousId = leadingEntries.length > 0 ? leadingEntries[leadingEntries.length - 1]?.id : null;
    previousId = typeof previousId === "string" && previousId.trim() ? previousId : null;

    const usedEntryIds = new Set();
    const rewrittenEntries = [...leadingEntries];
    outputMessages.forEach((message, index) => {
      const existingEntry = findExistingMessageEntry(message, entryMap, usedEntryIds);
      const messageEntry = buildMessageEntry(message, existingEntry, previousId, index);
      rewrittenEntries.push(messageEntry);
      previousId = messageEntry.id;
    });

    const tempPath = `${sessionFile}.${process.pid}.${randomUUID()}.tmp`;
    writeFileSync(tempPath, `${rewrittenEntries.map((entry) => JSON.stringify(entry)).join("\n")}\n`, "utf-8");
    renameSync(tempPath, sessionFile);
    updateSessionIndexAfterTrajectoryState(sessionFile, outputMessages.length);

    return {
      ...result,
      trajectoryStateSaved: true,
      trajectoryStateMessageCount: outputMessages.length,
    };
  } catch (error) {
    return {
      ...result,
      trajectoryStateError: error instanceof Error ? error.message : String(error),
    };
  }
}

function attachTrajectoryStateMetadata(minerTrajectory, stateMetadata) {
  if (!minerTrajectory || typeof minerTrajectory !== "object") {
    return minerTrajectory;
  }
  if (!minerTrajectory.baseMiner || typeof minerTrajectory.baseMiner !== "object") {
    minerTrajectory.baseMiner = {};
  }
  Object.assign(minerTrajectory.baseMiner, stateMetadata);
  return minerTrajectory;
}

function writeTrajectoryIoLogs(params, minerTrajectory, { replaceTrajectory }) {
  if (!hasCompressionResult(minerTrajectory)) {
    return;
  }

  try {
    mkdirSync(IO_LOG_DIR, { recursive: true });
    const timestamp = new Date().toISOString();
    const fileTimestamp = timestamp.replace(/[:.]/g, "-");
    const sessionPart = safeFilePart(params?.sessionId ?? params?.sessionKey);
    const prefix = `${fileTimestamp}-${sessionPart}-${randomUUID()}`;
    const inputMessages = Array.isArray(params?.messages) ? params.messages : [];
    const outputMessages = Array.isArray(minerTrajectory?.messages) ? minerTrajectory.messages : [];
    const metadata = {
      timestamp,
      pluginId: PLUGIN_ID,
      sourceHook: "assemble",
      sessionId: params?.sessionId ?? null,
      sessionKey: params?.sessionKey ?? null,
      sessionFile: minerTrajectory?.baseMiner?.sessionFile ?? null,
      replaceTrajectory,
      inputMessageCount: inputMessages.length,
      outputMessageCount: outputMessages.length,
      estimatedTokens: minerTrajectory?.estimatedTokens ?? null,
      state: {
        loaded: minerTrajectory?.baseMiner?.stateLoaded ?? null,
        saved: minerTrajectory?.baseMiner?.stateSaved ?? null,
        rawInputMessageCount: minerTrajectory?.baseMiner?.rawInputMessageCount ?? inputMessages.length,
        workingMessageCount: minerTrajectory?.baseMiner?.workingMessageCount ?? null,
        previousStateMessageCount: minerTrajectory?.baseMiner?.previousStateMessageCount ?? null,
        previousSourceMessageCount: minerTrajectory?.baseMiner?.previousSourceMessageCount ?? null,
        newMessageCount: minerTrajectory?.baseMiner?.newMessageCount ?? null,
        resetReason: minerTrajectory?.baseMiner?.stateResetReason ?? null,
      },
      trajectoryState: {
        saved: minerTrajectory?.baseMiner?.trajectoryStateSaved ?? null,
        messageCount: minerTrajectory?.baseMiner?.trajectoryStateMessageCount ?? null,
        error: minerTrajectory?.baseMiner?.trajectoryStateError ?? null,
      },
      compression: {
        compacted: minerTrajectory?.compaction?.compacted ?? minerTrajectory?.compacted ?? null,
        baseMiner: minerTrajectory?.baseMiner ?? null,
      },
    };

    writeJsonFile(path.join(IO_LOG_DIR, `${prefix}.input-trajectory.json`), {
      ...metadata,
      trajectory: inputMessages,
    });
    writeJsonFile(path.join(IO_LOG_DIR, `${prefix}.output-trajectory.json`), {
      ...metadata,
      trajectory: outputMessages,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.warn(`[${PLUGIN_ID}] failed to write trajectory io logs: ${message}`);
  }
}

function buildPassThroughResult(command, params = {}) {
  if (command === "assemble") {
    const messages = Array.isArray(params?.messages) ? params.messages : [];
    return {
      messages,
      estimatedTokens: estimateTokensForMessageArray(messages),
    };
  }

  if (command === "compact") {
    const currentTokenCount = Number.isFinite(params?.currentTokenCount) ? params.currentTokenCount : 0;
    return {
      ok: true,
      compacted: false,
      reason: "replaceTrajectory disabled",
      result: {
        summary: "",
        tokensBefore: currentTokenCount,
        tokensAfter: currentTokenCount,
      },
    };
  }

  if (command === "maintain") {
    return {
      changed: false,
      bytesFreed: 0,
      rewrittenEntries: 0,
    };
  }

  return null;
}

function estimateTokensForMessageArray(messages) {
  if (!Array.isArray(messages)) {
    return 0;
  }

  const totalChars = messages.reduce((sum, message) => sum + extractText(message?.content).length, 0);
  return Math.ceil(totalChars / 4);
}

function extractText(content) {
  if (typeof content === "string") {
    return content.trim();
  }
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (!part || typeof part !== "object") {
          return "";
        }
        if (typeof part.text === "string") {
          return part.text.trim();
        }
        if (typeof part.content === "string") {
          return part.content.trim();
        }
        if (part.type === "toolCall") {
          return JSON.stringify(part);
        }
        return "";
      })
      .filter(Boolean)
      .join("\n")
      .trim();
  }
  if (content && typeof content === "object") {
    if (typeof content.text === "string") {
      return content.text.trim();
    }
    if (typeof content.content === "string") {
      return content.content.trim();
    }
  }
  return "";
}

function parseTrajectoryMode(mode) {
  const normalized = String(mode ?? "").trim().toLowerCase();
  if (["on", "true", "yes", "1"].includes(normalized)) {
    return true;
  }
  if (["off", "false", "no", "0"].includes(normalized)) {
    return false;
  }
  throw new Error("Trajectory mode must be on/off");
}

const somaMinerPlugin = {
  id: PLUGIN_ID,
  kind: "context-engine",
  name: PLUGIN_NAME,
  description: "OpenClaw context engine connector for the local SOMA Miner Python script.",
  register(api) {
    if (typeof api.registerContextEngine !== "function") {
      throw new Error(
        "This OpenClaw build does not expose public context-engine plugin registration. soma-miner requires a newer/runtime-compatible OpenClaw version.",
      );
    }

    if (typeof api.registerCli === "function") {
      api.registerCli(({ program }) => {
        const cmd = program.command("soma-miner").description("Control the SOMA Miner connector");

        cmd
          .command("on")
          .description("Enable SOMA Miner as the active context engine")
          .action(async () => {
            await setPluginEnabled(api, true);
          });

        cmd
          .command("off")
          .description("Switch back to the native OpenClaw context engine")
          .action(async () => {
            await setPluginEnabled(api, false);
          });

        cmd
          .command("trajectory")
          .description("Choose whether the connector replaces the runtime trajectory")
          .argument("<mode>", "on/off")
          .action(async (mode) => {
            await setConnectorTrajectoryMode(parseTrajectoryMode(mode));
          });
      }, { commands: ["soma-miner"] });
    }

    api.registerContextEngine(PLUGIN_ID, () => ({
      info: {
        id: PLUGIN_ID,
        name: PLUGIN_NAME,
        version: "0.0.1",
        ownsCompaction: true,
      },

      async ingest(params) {
        return { ingested: true };
      },

      async assemble(params) {
        const currentParams = params ?? {};
        const sessionFile = resolveContextEngineSessionFile(currentParams);
        writeRuntimeHookMarker("assemble:start", {
          sessionId: currentParams.sessionId ?? null,
          sessionKey: currentParams.sessionKey ?? null,
          sessionFile,
          inputMessageCount: Array.isArray(currentParams.messages) ? currentParams.messages.length : 0,
        });
        try {
          const serviceResult = await callCompressionService(currentParams);
          if (serviceResult === null) {
            // Service decided not to compress.
            writeRuntimeHookMarker("assemble:finish", {
              sessionId: currentParams.sessionId ?? null,
              sessionKey: currentParams.sessionKey ?? null,
              sessionFile,
              compress: false,
              inputMessageCount: Array.isArray(currentParams.messages) ? currentParams.messages.length : 0,
              outputMessageCount: Array.isArray(currentParams.messages) ? currentParams.messages.length : 0,
            });
            return buildPassThroughResult("assemble", currentParams);
          }

          const minerTrajectory = serviceResult;
          const replaceTrajectory = readConnectorState().replaceTrajectory;
          const useMinerTrajectory = replaceTrajectory && shouldUseMinerTrajectory(minerTrajectory);
          const trajectoryState = persistTrajectoryState(currentParams, minerTrajectory, {
            sessionFile,
            useMinerTrajectory,
          });
          attachTrajectoryStateMetadata(minerTrajectory, trajectoryState);
          writeTrajectoryIoLogs(currentParams, minerTrajectory, { replaceTrajectory });
          writeRuntimeHookMarker("assemble:finish", {
            sessionId: currentParams.sessionId ?? null,
            sessionKey: currentParams.sessionKey ?? null,
            sessionFile,
            compress: true,
            replaceTrajectory,
            useMinerTrajectory,
            inputMessageCount: Array.isArray(currentParams.messages) ? currentParams.messages.length : 0,
            outputMessageCount: Array.isArray(minerTrajectory?.messages) ? minerTrajectory.messages.length : 0,
            changed: minerTrajectory?.baseMiner?.changed ?? null,
            pruned: minerTrajectory?.baseMiner?.pruned ?? null,
            reason: minerTrajectory?.baseMiner?.reason ?? null,
            trajectoryStateSaved: trajectoryState.trajectoryStateSaved,
            trajectoryStateError: trajectoryState.trajectoryStateError,
          });
          if (useMinerTrajectory) {
            return minerTrajectory;
          }
          return buildPassThroughResult("assemble", currentParams);
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          writeRuntimeHookMarker("assemble:error", {
            sessionId: currentParams.sessionId ?? null,
            sessionKey: currentParams.sessionKey ?? null,
            sessionFile,
            inputMessageCount: Array.isArray(currentParams.messages) ? currentParams.messages.length : 0,
            error: message,
          });
          throw error;
        }
      },

      async compact(params) {
        const currentParams = params ?? {};
        writeRuntimeHookMarker("compact", {
          sessionId: currentParams.sessionId ?? null,
          sessionKey: currentParams.sessionKey ?? null,
          currentTokenCount: Number.isFinite(currentParams.currentTokenCount) ? currentParams.currentTokenCount : null,
          targetTokenCount: Number.isFinite(currentParams.targetTokenCount) ? currentParams.targetTokenCount : null,
        });
        return buildPassThroughResult("compact", currentParams);
      },

      async afterTurn(params) {
        return null;
      },

      async maintain(params) {
        return buildPassThroughResult("maintain", params ?? {});
      },

      async dispose(params) {
        return null;
      },
    }));
  },
};

export default somaMinerPlugin;
