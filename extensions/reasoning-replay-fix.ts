/**
 * Reasoning-replay poison fix (reasoning models behind validating gateways)
 *
 * Root cause (verified by direct API experiments against NanoGPT): with pi's
 * stateless replay (`store: false` in pi-ai's openai-responses path), every
 * tool-call turn in the replayed input must carry a non-empty reasoning item
 * before its function_call items. Item ids are irrelevant (unpaired fc_/rs_
 * ids are accepted); the item just has to exist with non-empty
 * `content[].reasoning_text` — a summary-only or encrypted-only item does not
 * satisfy the check. Reasoning models occasionally emit a tool call with zero
 * reasoning tokens, so that turn has no reasoning item at all, and the
 * request is rejected with 400 `missing_tool_call_reasoning`. The history
 * stays poisoned: every retry, resume, or steer re-sends the same items and
 * fails identically.
 *
 * Fix: when a reasoning-model turn carries tool call(s) but no replayable
 * reasoning item, synthesize a minimal valid reasoning item as a synthetic
 * `thinking` block placed before the tool call. pi-ai replays thinking blocks
 * via their `thinkingSignature` JSON (openai-responses-shared.js pushes the
 * parsed item verbatim). The placeholder text is honest about what happened;
 * existing item ids are left untouched.
 *
 * Applies to all reasoning targets: the synthesized item is shaped exactly like
 * a real replayed reasoning item, so gateways that never validate simply
 * ignore it. Whether a turn is injected is decided by the request target, not
 * the turn's own model: replaying any tool-call turn to a reasoning model
 * requires the item, while a non-reasoning target is left alone (injecting
 * reasoning into a non-reasoning replay would be wrong rather than harmless).
 *
 *  - `context` (per LLM request, runs on a pi-managed clone; stored history
 *    untouched): heals already-poisoned sessions and their resumes. When the
 *    request target is a reasoning model, every tool-call turn is also stamped
 *    with the target identity, because pi-ai drops a `thinking` signature while
 *    replaying a turn from a different model — which would otherwise leave the
 *    gateway-facing function_call without its reasoning item.
 *  - `message_end` (replaces the finalized assistant message in place, before
 *    tool execution and persistence): heals agent state and stored history so
 *    new turns never poison anything.
 *  - `session_start` on `resume`/`reload`/`fork`: retroactively scans the
 *    stored branch and appends branch-relative `context_edit` entries that
 *    rewrite each poisoned assistant entry's content in place, so an old
 *    session is permanently healed on disk, not just per request. The edit is
 *    derived from the resolved projection, so it preserves earlier edits and
 *    is idempotent; the append-only session file is never rewritten.
 *
 * Idempotent.
 */

import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";

const PLACEHOLDER_TEXT = "(reasoning not captured for this turn)";

interface ContentBlock {
	type: string;
	id?: string;
	thinking?: string;
	thinkingSignature?: string;
}

interface AnyMessage {
	role: string;
	model?: string;
	provider?: string;
	api?: string;
	content?: unknown;
}

/** Provider/api/model identity used to replay a turn without pi-ai stripping it. */
export interface TargetModel {
	provider: string;
	api?: string;
	id: string;
	reasoning?: boolean;
}

function isAssistantMessage(msg: AnyMessage): msg is AnyMessage & { content: ContentBlock[] } {
	return msg.role === "assistant" && Array.isArray(msg.content);
}

/**
 * Whether this turn was produced by a reasoning-capable model. Resolved from
 * the message's own provider/model via the model registry; falls back to a
 * conservative id heuristic when the registry cannot resolve the entry, and
 * resolves to false for unknown or non-reasoning models.
 */
function isReasoningModelTurn(ctx: ExtensionContext, msg: AnyMessage): boolean {
	if (!isAssistantMessage(msg)) return false;
	const provider = msg.provider;
	const model = msg.model;
	if (typeof provider !== "string" || typeof model !== "string") return false;
	const registryModel = (ctx.modelRegistry as { find?: (p: string, m: string) => { reasoning?: boolean } | undefined })
		.find?.(provider, model);
	if (registryModel) return registryModel.reasoning === true;
	return /(?:^|[:/])(?:thinking|reason)/i.test(model);
}

/**
 * Whether the turn carries a reasoning item the gateway accepts. A signed
 * `thinking` block is only replayable when its `thinkingSignature` parses to a
 * Responses reasoning item with non-empty `content[].reasoning_text` — the
 * gateway rejects summary-only and encrypted-only items. A non-JSON signature
 * (for example an Anthropic signature) is outside this fix's scope and is left
 * alone rather than misclassified.
 */
export function hasReplayableReasoning(msg: AnyMessage): boolean {
	if (!isAssistantMessage(msg)) return false;
	return msg.content.some((block) => {
		if (block.type !== "thinking" || !block.thinkingSignature) return false;
		let parsed: unknown;
		try {
			parsed = JSON.parse(block.thinkingSignature);
		} catch {
			return true;
		}
		if (typeof parsed !== "object" || parsed === null) return true;
		const item = parsed as { type?: unknown; content?: unknown };
		if (item.type !== "reasoning") return true;
		const content = Array.isArray(item.content) ? item.content : [];
		return content.some((entry) => {
			const entryBlock = entry as { type?: unknown; text?: unknown };
			return entryBlock?.type === "reasoning_text" && typeof entryBlock.text === "string" && entryBlock.text.trim().length > 0;
		});
	});
}

/**
 * A turn poisons stateless replay when it carries tool call(s) but no
 * reasoning item to satisfy the gateway's replay check.
 */
export function isUnpairedToolCallTurn(msg: AnyMessage, isReasoningModel: boolean): boolean {
	if (!isReasoningModel || !isAssistantMessage(msg)) return false;
	if (hasReplayableReasoning(msg)) return false;
	return msg.content.some((block) => block.type === "toolCall");
}

/** Deterministic per-turn synthetic reasoning item (stable across retries). */
export function syntheticReasoningItem(msg: AnyMessage): ContentBlock {
	const callIds = (msg.content as ContentBlock[])
		.filter((block) => block.type === "toolCall" && typeof block.id === "string")
		.map((block) => block.id)
		.join(",");
	let hash = 0;
	for (let i = 0; i < callIds.length; i++) hash = (hash * 31 + callIds.charCodeAt(i)) >>> 0;
	return {
		type: "thinking",
		thinking: PLACEHOLDER_TEXT,
		thinkingSignature: JSON.stringify({
			type: "reasoning",
			id: `rs_recovered_${hash.toString(36)}`,
			content: [{ type: "reasoning_text", text: PLACEHOLDER_TEXT }],
		}),
	};
}

/**
 * Predicate form so the transforms stay pure and testable; `isReasoningModel`
 * decides per message (handlers derive it from the model registry).
 */
export function synthesizeMissingReasoning<T extends AnyMessage>(
	msg: T,
	isReasoningModel: (msg: AnyMessage) => boolean,
): T {
	if (!isUnpairedToolCallTurn(msg, isReasoningModel(msg))) return msg;
	// Drop existing thinking blocks: none of them is replayable (otherwise this
	// turn would not be poisoned), and leaving a summary-only or encrypted-only
	// item between the synthetic one and the tool call would keep the gateway
	// from finding reasoning_text on the paired item.
	const content = (msg.content as ContentBlock[]).filter((block) => block.type !== "thinking");
	return { ...msg, content: [syntheticReasoningItem(msg), ...content] } as T;
}

/** Whole-history pass used by `context`. Idempotent. */
export function synthesizeMissingReasoningInHistory<T extends AnyMessage>(
	messages: T[],
	isReasoningModel: (msg: AnyMessage) => boolean,
): T[] {
	let changed = false;
	const out = messages.map((msg) => {
		const rewritten = synthesizeMissingReasoning(msg, isReasoningModel);
		changed ||= rewritten !== msg;
		return rewritten;
	});
	return changed ? out : messages;
}

/** One persisted repair: an `context_edit` replacing a stored entry's content. */
export interface HistoryRepair {
	targetId: string;
	content: ContentBlock[];
}

/**
 * Request-time replay hardening for cross-model histories. pi-ai's
 * `transformMessages` preserves a `thinking` signature only while the message's
 * own `provider`/`api`/`model` match the request target; otherwise it rewrites
 * the block to plain text and drops the signature, so every tool-call turn
 * replayed to a different reasoning model reaches the gateway stripped of its
 * reasoning item. When the target is a reasoning model, ensure every assistant
 * turn with a tool call carries acceptable reasoning (synthesizing one if the
 * real item is missing, summary-only, or encrypted-only) and stamp the turn
 * with the target identity so pi-ai keeps that item. Runs on pi's request
 * clone, so stored history is untouched.
 *
 * Only same-API turns are rewritten: the signature format is owned by the API,
 * and an `anthropic-messages` thinking signature would make pi-ai's Responses
 * serializer throw if it were relabelled as an `openai-responses` message.
 * Cross-API tool-call turns cannot carry a reasoning item into a Responses
 * request and are left untouched.
 */
export function replaySafeHistory<T extends AnyMessage>(
	messages: T[],
	target: TargetModel,
): T[] {
	if (target.reasoning !== true) return messages;
	let changed = false;
	const out = messages.map((msg) => {
		if (!isAssistantMessage(msg) || !msg.content.some((block) => block.type === "toolCall")) return msg;
		if (msg.api !== target.api) return msg;
		const next: AnyMessage = hasReplayableReasoning(msg)
			? msg
			: synthesizeMissingReasoning(msg, () => true);
		// pi-ai JSON-parses every retained signature once the turn looks same-model.
		if (!isAssistantMessage(next) || !thinkingSignaturesAreJson(next)) return msg;
		if (
			next === msg &&
			next.provider === target.provider &&
			next.api === target.api &&
			next.model === target.id
		) return msg;
		changed = true;
		return { ...next, provider: target.provider, api: target.api, model: target.id } as T;
	});
	return changed ? out : messages;
}

/** Whether every signed thinking block parses as JSON (safe for Responses replay). */
function thinkingSignaturesAreJson(msg: AnyMessage & { content: ContentBlock[] }): boolean {
	return msg.content.every((block) => {
		if (block.type !== "thinking" || !block.thinkingSignature) return true;
		try {
			const parsed = JSON.parse(block.thinkingSignature);
			return typeof parsed === "object" && parsed !== null;
		} catch {
			return false;
		}
	});
}

/**
 * Plan the persisted `context_edit` entries that heal a stored branch. Input is
 * a session projection: already-compacted, already-edited messages together
 * with the id of the entry that owns each message. Using the effective
 * (projected) content preserves earlier replacements, and a target that already
 * carries a replayable reasoning item yields no repair, so replanning is
 * idempotent and never appends a duplicate edit.
 */
export function planHistoryRepairs(
	entries: ReadonlyArray<{ sourceEntry: { id: string }; messages: readonly AnyMessage[] }>,
	isReasoningModel: (msg: AnyMessage) => boolean,
): HistoryRepair[] {
	const repairs: HistoryRepair[] = [];
	for (const entry of entries) {
		const message = entry.messages.find((m) => isAssistantMessage(m));
		if (!message) continue;
		const healed = synthesizeMissingReasoning(message, isReasoningModel);
		if (healed === message) continue;
		repairs.push({ targetId: entry.sourceEntry.id, content: healed.content });
	}
	return repairs;
}

/** Minimal append surface pi exposes beyond the readonly session-manager type. */
interface MutableSessionManager {
	buildSessionProjection(): {
		entries: ReadonlyArray<{ sourceEntry: { id: string }; messages: readonly AnyMessage[] }>;
	};
	appendContextEdit(targetId: string, replacement: { content: ContentBlock[] }): string;
}

/**
 * Resolve the internal session-edit surface at runtime. Both members are
 * internal pi APIs, so a pi release can remove or rename them. The result is
 * undefined when the running build no longer provides them.
 */
export function sessionEditApi(manager: unknown): MutableSessionManager | undefined {
	if (!manager || typeof manager !== "object") return undefined;
	const candidate = manager as Partial<MutableSessionManager>;
	if (typeof candidate.buildSessionProjection !== "function") return undefined;
	if (typeof candidate.appendContextEdit !== "function") return undefined;
	return candidate as MutableSessionManager;
}

export default function reasoningReplayFix(pi: ExtensionAPI) {
	const isReasoningModel = (ctx: ExtensionContext) => (msg: AnyMessage) =>
		isReasoningModelTurn(ctx, msg);

	// Request-time repair: makes already-poisoned sessions (and resumes of them)
	// replay cleanly. Runs on a pi-managed clone, so stored history is untouched.
	// A reasoning target gets the cross-model-safe pass; otherwise fall back to
	// the same-model synthesis.
	pi.on("context", async (event, ctx) => {
		const target = ctx.model;
		const messages = target?.reasoning
			? replaySafeHistory(event.messages, target)
			: synthesizeMissingReasoningInHistory(event.messages, isReasoningModel(ctx));
		if (messages === event.messages) return;
		return { messages };
	});

	// History repair: rewrite the finalized assistant message in place (pi
	// applies the replacement before tool execution and persistence), so the
	// poison never enters agent state or the session file. Tool calls are left
	// untouched, so execution is unaffected.
	pi.on("message_end", async (event, ctx) => {
		const message = synthesizeMissingReasoning(event.message, isReasoningModel(ctx));
		if (message === event.message) return;
		return { message };
	});

	// Resume-time history repair: retroactively scan the stored branch and
	// append `context_edit` entries that heal already-poisoned turns on disk.
	// The session stays append-only; pi projects the latest edit per target.
	// The runtime hooks above heal every request, so a missing or failing
	// internal API degrades to per-request repair only.
	pi.on("session_start", async (event, ctx) => {
		if (event.reason !== "resume" && event.reason !== "reload" && event.reason !== "fork") return;
		const manager = sessionEditApi(ctx.sessionManager);
		if (!manager) return;
		try {
			const projection = manager.buildSessionProjection();
			for (const repair of planHistoryRepairs(projection.entries, isReasoningModel(ctx))) {
				manager.appendContextEdit(repair.targetId, { content: repair.content });
			}
		} catch {
			// Stored history stays unhealed for this session. Nothing to repair at
			// runtime, because the request-time hooks read the same entries.
		}
	});
}
