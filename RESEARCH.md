# SOMA SN114 CoT-Compression Research Brief
## Maximizing the Algorithmic Context Miner to Decisively Win

**Empirical baseline:** We are #2 (0.544), leader is #1 (0.644). The entire gap is on 45 competition tasks. Leader solves MORE tasks (jackpots 10.0 vs 6.2, solid passes 22.0 vs 17.6) while using ~1.4× fewer tokens (0.82M vs 1.20M median). Our -4 protection is already better than the leader; our weakness is compressing too lightly, causing the agent to drift and solve fewer tasks.

---

## Q1 — Context Management for Code Agents: Load-Bearing vs. Droppable

### Findings

**The lost-in-the-middle phenomenon is real and severe for code agents.** Liu et al. (TACL 2024, "Lost in the Middle") established the U-shaped attention curve: models attend strongly to the *beginning* and *end* of context, and 30%+ less to information in the middle. This applies even to models explicitly designed for long contexts. A 2025 EMNLP finding ("Context Length Alone Hurts LLM Performance Despite Perfect Retrieval") reinforced this: even when a model can perfectly retrieve evidence, performance still degrades 24% as irrelevant context length grows.

For coding agents, the practical consequence (Morph/WarpGrep analysis, 2025): when an agent reads 8 files, the relevant code in file #4 sits in the model's blind spot. The agent has the right information in context but cannot effectively attend to it — it may hallucinate edits to the wrong file, repeat work already done, or produce code that contradicts what it just read.

**What is load-bearing:**
1. **Initial task description / issue text** — must be at position 0 (highest attention). Loss of the issue statement is the single most common cause of goal drift.
2. **Most recent failing test output** — the traceback is the ground truth for what is broken. Must be close to the current position (high attention). Failing tests at the *end* of context are the jackpot signal.
3. **Most recent state of files being actively edited** — the agent needs to know what the file currently looks like to produce a correct patch.
4. **Most recent edit/write results** — confirms that a change was actually applied; prevents re-applying the same patch.
5. **The last 1-3 shell exec results** — recent environment feedback (git status, ls output, import errors) orients the agent.

**What is safely droppable:**
- Stale file reads: if a file was read at step 5 and then edited at step 12, the step-5 read is superseded — it shows a version of the file that no longer exists.
- Duplicate reads: the agent often reads the same file 3-5 times. All but the most recent copy are noise.
- Old directory listings (ls, find): these reflect the repo state at exploration time, which may have changed.
- Verbose middle-trajectory grep/search dumps: the agent had already located the relevant lines; keeping the result forever adds noise.
- Passing test output from many steps ago: only the *most recent* test run matters.
- Assistant reasoning text from solved sub-problems: if the agent fixed one bug and moved on, earlier reasoning chains add distraction.

**Techniques ranked by expected score impact:**
1. **Deduplicate repeated reads of the same file** — direct token reduction, no information loss.
2. **Drop old exploration output (grep, ls) beyond a sliding window** — large token reduction with minimal loss.
3. **Keep failing test output at the end of context** — ensures high attention on the ground truth signal.
4. **Truncate file read content to head+tail** — preserves structure and active region.

**Risk (what could break solves / trigger -4):**
- Dropping the ONLY copy of a critical file read before the agent has used it to write a patch.
- Discarding a test run that showed a *passing* test (the agent needs to know what still works).
- Mitigations: keep the most recent read for each active file; always keep the last test result regardless of pass/fail.

**Citations:** Liu et al. TACL 2024; EMNLP 2025 findings-1264; Morph/WarpGrep context-rot analysis 2025; Anthropic "Effective context engineering" 2025.

---

## Q2 — Algorithmic (No-LLM) Compression Techniques

### Findings & Techniques

**A. Content-aware truncation of tool results (highest impact)**

Observation masking — simply limiting how many tokens a single tool result can inject — is surprisingly powerful. The paper "Observation Masking vs. LLM-Summary" (arXiv 2508.21433) showed observation masking **halves cost without reducing performance**, and a hybrid approach reduces cost by a further 7–11% while *improving* solve rate by 2.6 pp. The key insight: most tool outputs contain far more tokens than the agent needs to act, especially for large file reads.

**Type-specific strategies (ranked by token savings):**

| Tool type | Strategy | Justification |
|-----------|----------|---------------|
| `read` (file content) | Head 2K + Tail 4K chars | Head: imports/class structure. Tail: most-recently-edited region. Error line usually in tail. |
| `exec` (shell command) | Head 800 + Tail 4.8K chars | Head: command + environment. Tail: errors live at the end of output. |
| `bash` with pytest/unittest | Tail 10K chars only | Test failures are always at the end; setup noise at the top is droppable. |
| `write`/`edit`/`patch` | Keep up to 2K chars | Acknowledgement messages are short; truncation rarely needed. |

**Pseudocode:**
```python
def compress_text(text, tool_type):
    if len(text) <= 500:
        return text  # Too small to bother
    if tool_type == "read":
        return head_tail(text, head=2000, tail=4000)
    if tool_type == "test":
        return tail_only(text, max=10000)
    if tool_type == "exec":
        return head_tail(text, head=800, tail=4800)
    if tool_type == "write":
        return text[:2000] if len(text) > 2000 else text
    return head_tail(text, head=1000, tail=4000)  # generic

def head_tail(text, head, tail):
    if len(text) <= head + tail:
        return text
    omitted = len(text) - head - tail
    return text[:head] + f"\n[...{omitted} chars omitted...]\n" + text[-tail:]
```

**Expected impact:** ~3–5× reduction in per-result token count for large file reads. Across 4–8 kept tool results, this translates to a 50–70% per-turn token reduction.

**B. Deduplication of stale reads (high impact, zero risk)**

When an agent reads the same file path at step 3, step 7, and step 14, only the step-14 copy is the current file state (or after edits, the most recently edited version). Steps 3 and 7 are pure noise.

**Algorithm:**
```python
def deduplicate_reads(tool_result_indices, messages, tool_call_index):
    seen_paths = set()
    kept = []
    for idx in reversed(tool_result_indices):  # newest-first
        tool_info = get_tool_info(messages[idx], tool_call_index)
        if tool_info["type"] == "read":
            path = tool_info["input"].get("path")
            if path and path in seen_paths:
                continue  # stale read — drop it
            if path:
                seen_paths.add(path)
        kept.append(idx)
    return list(reversed(kept))  # restore chronological order
```

**Expected impact:** In a typical SWE-bench trajectory where the agent reads the same file 3–5 times, this removes 2–4 full (large) tool results from the kept set, freeing those slots for more useful recent context.

**Risk:** Near-zero if paired with keeping the most-recent read. The only risk is if an edit *introduces a regression* and the agent needs to see the pre-edit version — but pre-edit file states are never used by the model to write patches.

**C. Smart structural selection (medium impact)**

Instead of blindly keeping the last N tool results by index, use a priority queue:
1. Always keep the most recent N_test (up to 2) test-run results.
2. Deduplicate reads, keeping only the latest per file path.
3. Fill remaining slots with the most recent results.
4. Sort the kept set chronologically so the agent sees an ordered narrative.

**D. Edit/write ledger awareness (medium impact)**

When a file is written or edited, any prior read of that file is no longer the current state. A smarter version of dedup checks whether there is an intervening write between two reads, and if so, marks the pre-write read as superseded even if the file paths differ slightly (e.g., normalized paths).

**E. Assistant message content truncation (low impact, worth considering)**

After removing thinking blocks (already done), assistant messages contain the tool-call setup reasoning. This is usually < 500 tokens and safe to keep. However, very long chains of natural language reasoning between tool calls (if the agent is verbose) could be truncated to the last 2K chars. This is a lower-priority improvement.

**Prior art (no-LLM methods used in production):**
- **Claude Code**: "tool result clearing" — once a tool result is deep in history, it clears the raw content and keeps only metadata. The agent continues with the 5 most recently accessed files.
- **OpenHands rolling condenser**: head (first 4 events = system + task) + tail (most recent N events). LLM-free variant simply drops the middle.
- **ECA editor**: per-tool line limits (default 2000 lines) + `[OUTPUT TRUNCATED]` marker.
- **Agentless**: provides file *skeletons* (structure without full implementation) to reduce read context.

---

## Q3 — Solve-Rate Maximization on Hard Tasks (Jackpots)

### Findings

**Why the baseline fails on hard tasks:**
1. Context grows too large → the failing test + relevant error line sink into the middle blind spot → agent makes patches that don't address the actual failure.
2. The agent enters read/re-read loops (re-reads the same file 5+ times) instead of writing a patch — a classic sign of lost focus.
3. Dead-end exploration accumulates in context: grep dumps, ls outputs, incorrect patches already reversed — these compete with the current error for attention.

**How context shaping can raise P(correct patch) on tasks the baseline fails:**

1. **Ensure the failing test traceback is in the last-position slot** — this is the most attended position (tail of U-curve). If the most recent test run shows a failure, it should be the LAST message in the compressed context. Our `select_tool_results_smart` prioritizes test-type results so they stay in the window even when other results are dropped.

2. **Deduplicate re-reads to break the loop signal** — when the same file read appears 5 times, the LLM's attention is diluted across identical content. After dedup, each slot is a *different* piece of information, reducing the probability of the model spinning on redundant context.

3. **Keep the edit/write results visible** — if the agent wrote a patch at step N, that patch acknowledgement should stay in context. This prevents the agent from re-applying the same patch (we've seen this failure mode in SWE-bench trajectories).

4. **Issue description salience** — the original issue text at position 0 is always highly attended. Our implementation always keeps the first user message at position 0. This is free jackpot insurance.

5. **Avoid over-compression on short/dense trajectories** — on hard tasks, the agent may make fewer but more targeted tool calls. A trajectory with only 4–8 tool results should NOT be pruned (< KEEP_TOOL_RESULT_COUNT threshold). The current implementation correctly does not prune when tool results are below threshold.

**Anti-loop heuristic (algorithmic):**
```python
def detect_loop(tool_result_indices, messages, tool_call_index):
    """Returns True if the agent is in a read-loop (same file read 3+ times recently)."""
    recent = tool_result_indices[-6:]  # last 6 tool results
    path_counts = {}
    for idx in recent:
        info = get_tool_info(messages[idx], tool_call_index)
        if info.get("type") == "read":
            path = info.get("input", {}).get("path")
            if path:
                path_counts[path] = path_counts.get(path, 0) + 1
    return any(count >= 3 for count in path_counts.values())
```
If a loop is detected, be MORE aggressive: drop older loop iterations, keep only the latest read + the failing test result. This may help the agent escape the loop by removing the redundant confirmation it has been seeking.

---

## Q4 — Optimal Token Budget / Knee of the Curve

### Findings

**The scoring formula's maximum token bonus saturates at ~13.5% of baseline tokens:**
```
max bonus = 0.5 * clamp(ln(Tok_B/Tok_A), -2, 2) = 0.5 * 2.0 = +1.0
Clamp at +2: ln(Tok_B/Tok_A) = 2 → Tok_A = Tok_B * e^-2 ≈ 0.135 * Tok_B
```
Getting to this ceiling (13.5% of baseline) on all 45 tasks would add +1.0 per passing task vs the minimum +0.0. The expected value of aggressive compression is high, but only if solve rates are maintained.

**Empirical knee of the curve for coding agents:**
- SWE-Pruner (arXiv 2601.16746): 23–54% token reduction **without degrading solve rate**. This suggests the "safe zone" for non-degrading compression is at least 46–77% of raw token count.
- OpenHands condensation: ~50% cost reduction with **no quality loss** on SWE-bench benchmarks.
- Active Context Compression (arXiv 2601.07190): 22.7% savings with matched accuracy when compressing every 10–15 tool calls.
- Observation masking study (arXiv 2508.21433): masking halved costs with **+2.6 pp improvement** in solve rate.

**The key insight from observation masking:** removing noise (long file reads, redundant output) actually *helps* the model focus. The benefit of compression at moderate levels is not just cost savings — it is higher attention density on the remaining tokens.

**Proposed adaptive policy:**

```python
def adaptive_keep_count(messages, tool_call_index):
    """
    Adjust KEEP_TOOL_RESULT_COUNT based on trajectory characteristics.
    - Long/redundant trajectories: compress harder (fewer slots, more dedup)  
    - Short/dense trajectories: compress lighter (more slots, preserve context)
    """
    total_tokens = estimate_tokens_for_message_array(messages)
    tool_results = [m for m in messages if normalize_role(m.get("role")) == "toolResult"]
    
    # Detect redundancy: ratio of reads to unique paths
    unique_paths = set()
    total_reads = 0
    for msg in messages:
        info = get_tool_info(msg, tool_call_index)
        if info.get("type") == "read":
            total_reads += 1
            path = info.get("input", {}).get("path")
            if path:
                unique_paths.add(path)
    redundancy = total_reads / max(len(unique_paths), 1)
    
    if total_tokens > 100_000 or redundancy > 2.5:
        return 4  # aggressive compression for long/loopy trajectories
    elif total_tokens > 40_000:
        return 6  # moderate compression (current default)
    else:
        return 8  # light compression for short/dense trajectories
```

**Current implementation calibration:**
- `KEEP_TOOL_RESULT_COUNT = 6` (content-truncated)
- This targets approximately 15–25% of raw trajectory tokens
- Content truncation ensures each of the 6 results is ≤ 6,000 chars (~1,500 tokens)
- 6 × 1,500 + overhead = ~10,000–12,000 tokens per turn
- At 40 turns: ~0.45M total tokens = well within the bonus zone vs ~2M baseline

---

## Q5 — Reliability Under 3× Majority Vote

### Findings

**The majority vote rule (≥2/3 to "solve") means variance kills us in three ways:**
1. Task solves inconsistently → counts as unsolved (missed jackpot or missed pass)
2. Task fails inconsistently → counts as solved (avoided a -4)
3. Token counts vary → affects the bonus even when all 3 runs pass

**Determinism and idempotency are free reliability improvements:**

Our compression is already stateless-by-design from the scorer's perspective (each run gets a fresh session). But within a run, the compression must be deterministic: same input → same output every time.

**Checklist for deterministic compression:**
1. No randomness in truncation (head+tail cuts are fixed by char position — ✓)
2. Set iteration order is stable (we use lists, not sets, for ordering — ✓ in implementation)
3. Deduplication order is by original message index (not hash-based) — ✓
4. File path comparison is exact string match (not normalized) — acceptable, same trajectory = same paths

**Context stability reduces variance:**
- A stable, focused context (fewer tokens, no noise) gives the model less "room to wander" between identical runs. If the context is identical across 3 runs (which it is, since compression is deterministic), the only source of variance is LLM sampling temperature.
- Shorter, focused context → fewer candidate wrong actions → lower variance in model output.

**Test priority in selection reduces jackpot variance:**
- By always keeping the most recent test output in the compressed context, we ensure that across all 3 runs, the agent sees the same failing test signal. This increases consistency in the "try to fix this specific test" behavior.

---

## Q6 — Legitimate Scoring / Mechanism Edges

### Findings

**A. Multiple hotkeys for difficulty-slice specialization**

The scoring incentive is layered:
> "Incentive is layered: winners are picked per category-subset (overall + difficulty pairs + single difficulties Easy/Medium/Hard), each subset won by the single highest scorer."

This means:
- A hotkey optimized for **Hard tasks** (more aggressive compression to maximize jackpots, tolerate higher -4 risk) can win the Hard category.
- A hotkey optimized for **Easy tasks** (conservative compression to avoid -4s, maximize +1.0 pass bonuses) can win the Easy category.
- A third hotkey optimized for **overall** (balanced) can win the overall category.

**Concrete specialization strategy:**
- **Easy hotkey:** `KEEP_TOOL_RESULT_COUNT = 8`, lighter truncation (READ_TAIL_CHARS = 6000). Goal: never break an easy task, maximize token bonus on reliable solves.
- **Hard hotkey:** `KEEP_TOOL_RESULT_COUNT = 4`, more aggressive truncation (READ_TAIL_CHARS = 3000), stronger dedup, loop detection. Goal: jackpots from tasks baseline fails; accept slightly higher -4 risk.
- **Overall hotkey:** Current implementation (`KEEP_TOOL_RESULT_COUNT = 6`). Balanced.

**B. Screener gate: pass with margin**

The screener gate (top ~20% by screener-only score) must be passed first. With 5 screener tasks:
- If baseline solves 3/5 screener tasks, we need to also solve ≥3/5 (ideally all 5) with token savings.
- Risk: over-compressing screener tasks → breaking all 5 → failing gate, scoring 0.
- Mitigation: for screener tasks, use a more conservative configuration (higher `KEEP_TOOL_RESULT_COUNT`, lower truncation limits). But since we're stateless, we can't distinguish screener from competition tasks without external info.

**C. Both-fail token bonus**

On tasks where neither baseline nor miner solves (score = 0.1 * clamp), aggressive compression gives up to +0.2 per task. With ~30 both-fail tasks and maximum compression:
- Bonus: 30 × 0.2 = +6.0 total — significant!
- But the same aggressive compression might cause -4s on 2 tasks = -8.0.
- Net: risky for overall, but if the Hard hotkey is dedicated to jackpot hunting, this is acceptable.

**D. Avoid over-engineering the -4 protection**

Our current -4 protection is *already better* than the leader. This is a strength to maintain, not a direction to improve further. The marginal value of avoiding one more -4 is high (saves 4 points), but if we're already best-in-class at this, more protection would require reducing compression, which costs us token bonus and solve rate.

**CRITICAL: Do NOT attempt:**
- Injecting the hidden gold test patch (bannable)
- Leaking instance IDs to fetch known solutions (bannable)
- Any cheat that exploits knowledge of hidden test cases
- Guessing specific test names or expected outputs

Top miners are manually reviewed. Legitimate algorithmic strategy only.

---

## Q7 — Qwen3-Coder Specifics

### Findings

**Architecture: Hybrid DeltaNet (linear attention) + MoE**

Qwen3-Coder (both 480B and Coder-Next variants) uses a **3:1 hybrid attention layout**: three Gated DeltaNet layers (linear attention, O(1) per token) followed by one full Gated Attention layer (O(n²)), repeated 12 times. The full-attention layers happen every 4th layer.

**Critical implication for context compression:** Gated DeltaNet layers have a "recall tax" — they compress long context into a fixed-size recurrent state. Long-range, precise needle-in-a-haystack retrieval suffers compared to standard attention. The full-attention layers (1/4 of layers) periodically "rebuild the global picture," but this is a less accurate reconstruction than pure softmax attention.

**Exploitation: position matters more for Qwen3-Coder than for pure-attention models**

Because the recurrent state degrades precision for long-range recall:
- Information at the **very end** of context is processed by the most recent recurrent state update AND gets a full-attention scan. Highest recall.
- Information at the **beginning** of context was incorporated into recurrent state early and may have been partially overwritten by subsequent updates. Moderate recall.
- Information in the **long middle** has the worst recall — the recurrent state has been updated many times since, partially washing it out.

**Context shaping exploit:** Place the most critical information at the END of the compressed trajectory:
1. Most recent test failure → last toolResult
2. Issue description → first user message (already done)
3. The file being edited → second-to-last toolResult

Our `select_tool_results_smart` already achieves this by keeping test outputs in the selection and placing them at their natural chronological position (which is typically recent = near the end).

**Qwen3-Coder tool-calling format:**
- Uses custom XML-style tool calling (`qwen3_coder` parser), NOT JSON, specifically to avoid JSON-escaping issues with large code blocks.
- The format is critical: malformed tool calls cause the agent to retry, wasting turns.
- Our compression MUST NOT corrupt the `toolCall` block structure (IDs, names, inputs). The current implementation only modifies `toolResult` content (not toolCall blocks), so this is safe. ✓

**Qwen3-Coder non-thinking mode:**
- Qwen3-Coder does NOT generate `<thinking>` blocks. Our `sanitize_messages` function strips thinking blocks just in case, but this overhead is minimal.
- The removal of thinking blocks is a no-op for Qwen3-Coder (no benefit, no risk).

**Failure modes to exploit:**
- Long-range recall degradation: remove stale reads that force the model to maintain old file state in its recurrent memory. Fresher, compressed reads are more reliably recalled.
- MoE routing: different experts likely activate for Python code vs shell commands vs test output. Compressed, well-typed content may route more cleanly to the correct experts vs mixed/noisy content.

---

## Q8 — Prior Art: How Top SWE-bench Scaffolds Manage Context

### Findings

**Claude Code (Anthropic)**
- **Keeps:** Architectural decisions, unresolved bugs, implementation details, 5 most recently accessed files.
- **Drops:** Redundant tool outputs, old exploration messages.
- **Method:** Tool result clearing (removes raw content of old tool results, keeps metadata).
- **Trigger:** Proactive, when context approaches window limit.
- **Reference:** "Effective Context Engineering" (Anthropic engineering blog, Sep 2025).

**OpenHands (All Hands AI)**
- **Keeps:** First 4 events (system prompt + initial task), last N events (recent work).
- **Drops:** Middle events (old tool results, old reasoning).
- **Method:** LLMSummarizingCondenser — LLM summary of dropped middle, or RollingWindowCondenser (pure heuristic head+tail).
- **Trigger:** When event count exceeds `max_size` (default 80-120).
- **Claimed result:** ~50% cost reduction, higher solve rate than baseline.
- **Reference:** OpenHands context condensation blog post (Apr 2025) and docs.

**SWE-agent (Princeton)**
- **Keeps:** Full trajectory without compression by default.
- **Recent work:** History truncation and state-based context (maintains "open file + working directory" state as a compact header before each LLM call).
- **The ACI (Agent-Computer Interface):** special-purpose commands that return compact, task-relevant output rather than raw terminal output. E.g., `search_file` returns formatted matching lines rather than raw grep.
- **Reference:** Yang et al. 2024 (SWE-agent paper).

**Agentless (UIUC)**
- **Does not use agents:** Instead uses a structured 3-phase pipeline (localization → repair → validation).
- **Context management:** File *skeletons* (class/function signatures without bodies) for initial localization. Full file content only for the specific function to patch.
- **Implication for us:** In agent mode (our constraint), the principle is to give minimal but precisely-targeted file context — our content truncation (head+tail) approximates this.
- **Reference:** Xia et al. 2024 ("Agentless").

**SWE-AGILE (KDE Group)**
- **Keeps:** Full environmental observations (tool results), last-N-steps of detailed CoT reasoning.
- **Drops:** Older CoT reasoning (replaced with compressed "Reasoning Digests").
- **Method:** Dynamic Reasoning Context — sliding window of detailed reasoning + digests of older steps.
- **Result:** State-of-the-art for 7B-8B models with only 2.2K trajectories.
- **Reference:** arXiv 2604.11716.

**CaT / SWE-Compressor (ShanghaiAI)**
- **Keeps:** Fixed task anchors (issue description, constraints) + structured long-term memory + short-term working memory (last K steps verbatim).
- **Method:** Stage-wise summarization at task boundaries ("segmented compression").
- **Result:** Context stabilizes at <32K tokens (vs ReAct exhausting window by step 60).
- **Reference:** arXiv 2512.22087.

**SWE-Pruner (observation masking)**
- **Method:** Goal-hint-guided line-level pruning using a 0.6B neural skimmer. Task-aware: preserves lines relevant to the current goal.
- **Result:** 23–54% token reduction on SWE-bench while *improving* solve rate.
- **NOT directly applicable** (requires a 0.6B model = no-LLM constraint violated). But the principle — task-aware preservation of relevant lines — can be approximated algorithmically.
- **Approximation:** Keep first N lines (imports/structure = always task-relevant) + last N lines (current edit region = always task-relevant). This is exactly our head+tail strategy.
- **Reference:** arXiv 2601.16746.

**Key takeaway from prior art:**
All high-performing scaffolds converge on the same strategy: **issue description first, recent tool results (truncated) last, stale exploration dropped.** Our implementation now matches this pattern.

---

## Prioritized Roadmap: Top 5 Changes to Move from 0.544 to Decisive #1

### Change 1: Content-Aware Tool Result Truncation
**What:** Apply type-specific truncation (read: head 2K + tail 4K; test: tail 10K; exec: head 800 + tail 4.8K) to every kept toolResult, instead of keeping full content.  
**Expected delta:** From 1.20M → ~0.55M median tokens per task = additional +0.3 bonus per passing task (17–22 tasks passing = +5–7 total points).  
**Risk:** Low — the key signal (error lines, active code region) is preserved in head+tail positions.  
**A/B test:** Run with and without truncation on the 5 screener tasks. If screener pass rate is maintained, deploy.  
**Status:** ✅ Implemented in `base_miner.py`.

### Change 2: Stale-Read Deduplication
**What:** For repeated reads of the same file path, keep only the most-recent read result and drop older ones from the structural window.  
**Expected delta:** Removes 2–4 large tool results from typical trajectories, freeing slots for more recent information. Estimated additional -0.2M tokens per task = +0.1 bonus per passing task.  
**Risk:** Near-zero — older reads of the same file are superseded by newer reads.  
**A/B test:** Compare solve rates on tasks with known re-read loops vs without.  
**Status:** ✅ Implemented in `select_tool_results_smart()`.

### Change 3: Test-Output Prioritization
**What:** When selecting which tool results to keep within the structural window, always include the most recent test run results (up to 2). This ensures the failing test traceback is visible in the compressed context.  
**Expected delta:** Direct contribution to jackpots: tasks where the baseline fails because the agent loses track of the failing test will now have a higher P(solve). Estimated +0.5–1.0 jackpot points (1–2 additional jackpots).  
**Risk:** Low — test outputs are always relevant; the only risk is keeping a "all tests pass" result at the expense of a more recent relevant read.  
**A/B test:** Track how many jackpot attempts (baseline fail, miner succeeds) occur with vs without test prioritization.  
**Status:** ✅ Implemented in `select_tool_results_smart()`.

### Change 4: Increase KEEP_TOOL_RESULT_COUNT from 4 to 6
**What:** Raise the structural window size from 4 to 6 tool results. Combined with content truncation, total tokens per turn actually *decrease* vs current (6 × truncated_result < 4 × full_result), while the agent gets broader recent context.  
**Expected delta:** Agent can "remember" more recent steps without re-reading. Reduces re-read loops → higher solve rate on medium tasks. Estimated +1–2 additional pass points.  
**Risk:** Low if content truncation is applied simultaneously; slightly higher if not.  
**A/B test:** Compare solve rate at KEEP=4 (no truncation) vs KEEP=6 (with truncation). Token count should be roughly equal; solve rate should improve.  
**Status:** ✅ Implemented (`KEEP_TOOL_RESULT_COUNT = 6`).

### Change 5: Multiple Hotkeys for Difficulty-Slice Specialization
**What:** Submit 2–3 hotkeys with different tuning:
- **Easy hotkey:** `KEEP=8`, lighter truncation (READ_TAIL=6000). Conservative, never breaks easy tasks, maximizes bonus.
- **Hard hotkey:** `KEEP=4`, aggressive truncation (READ_TAIL=3000), strong dedup, loop detection. Jackpot hunting.
- **Overall hotkey:** Current implementation. Wins overall category.

**Expected delta:** Wins additional category payout per difficulty slice. If each category wins are worth similar incentive, winning 3 categories vs 1 could multiply rewards 3×.  
**Risk:** Hard hotkey has higher -4 risk on hard tasks (but baseline already fails many hard tasks, so net risk is low).  
**A/B test:** Run hard hotkey only on hard-category tasks if possible; otherwise A/B on full competition.  
**Status:** ⬜ Not yet implemented. Requires creating tunable configurations for `base_miner.py`.

---

## Implementation Notes

### Current Implementation (this PR)

The `base_miner.py` now includes:

1. **`build_tool_call_index(messages)`** — maps every `toolCallId` found in assistant messages to `{name, input}` metadata. Required for content-aware operations.

2. **`classify_tool_type(name, input)`** — returns one of `read`, `write`, `test`, `exec`, `other` based on tool name and input structure.

3. **`compress_all_tool_results(messages, index)`** — applies type-specific truncation to every `toolResult` message content. Operates on content only; does not touch message structure.

4. **`select_tool_results_smart(indices, messages, index, keep)`** — selects which tool results to keep using deduplication + test prioritization, vs the previous naive "last N".

5. **`prune_messages(messages, index)`** — enhanced version of the original structural pruning, now accepts the tool call index for smart selection.

6. **`handle_assemble(payload)`** — orchestrates: sanitize → build index → content compress → smart prune → save state.

### Token Count Calibration

| Scenario | Tokens/turn | 40-turn task | Scoring impact |
|----------|------------|--------------|---------------|
| Baseline (no miner) | ~50K growing | ~2M total | Tok_B reference |
| Current miner (v1) | ~30K | ~1.2M | +0.35 bonus/pass |
| This PR (v2) | ~12K | ~0.48M | +0.73 bonus/pass |
| Max compression | ~6K | ~0.24M | +1.00 bonus/pass |

The current implementation targets the "This PR (v2)" tier — significant compression, minimal solve-rate risk.

### Maintenance and A/B Testing

To A/B test different compression levels, modify the constants at the top of `base_miner.py`:

```python
# Conservative (Easy hotkey):
KEEP_TOOL_RESULT_COUNT = 8
READ_TAIL_CHARS = 6000
TEST_TAIL_CHARS = 15000

# Aggressive (Hard hotkey):  
KEEP_TOOL_RESULT_COUNT = 4
READ_HEAD_CHARS = 1000
READ_TAIL_CHARS = 2500
EXEC_TAIL_CHARS = 3000
TEST_TAIL_CHARS = 8000
```
