# pi-reasoning-replay-fix

Repair for replayed reasoning-model turns that carry a tool call but no reasoning item.

## Problem

pi replays session history without server state. A gateway that validates reasoning items rejects a replayed tool-call turn when that turn has no non-empty `reasoning_text` item. The gateway returns `400 missing_tool_call_reasoning`.

A reasoning model sometimes emits a tool call without reasoning tokens. That turn has no reasoning item. Every retry, resume, or steer then sends the same items and fails again.

## Fix

The extension adds a `thinking` block before the tool call of an affected turn. pi replays a thinking block through its `thinkingSignature` JSON, so the gateway receives a complete reasoning item.

The placeholder text is `(reasoning not captured for this turn)`. Existing item ids stay unchanged.

## Hooks

| Hook | Action |
| --- | --- |
| `context` | Runs on each request, on a pi-managed clone. Heals stored history and resumes. Stamps each tool-call turn with the request target identity when the target is a reasoning model. |
| `message_end` | Replaces the finalized assistant message before tool execution and before persistence. A new turn does not add a defect. |
| `session_start` | Runs on `resume`, `reload`, and `fork`. Scans the stored branch and appends branch-relative `context_edit` entries. The session file on disk becomes correct. |

The `context_edit` entry is idempotent. It preserves earlier edits. The append-only session file is never rewritten.

## Scope

The extension is provider-agnostic. It acts on the request target, not on the model of the turn. A non-reasoning target is left alone.

## Install

```sh
pi install git:github.com/DecibillyJoel/pi-reasoning-replay-fix
```

pi loads `extensions/*.ts` from a package.

## Caveat

`session_start` needs internal pi APIs: `context.modelRegistry.find(...)`, `sessionManager.buildSessionProjection()`, and `appendContextEdit(...)`. A pi release can change them without a changelog entry.

## Development

No build step. The file is a plain TypeScript source that pi loads directly.

Typecheck against the pi types that the dev dependencies install:

```sh
npm install
npm run typecheck
```

`npm install` fetches the pi package that `tsconfig.json` needs to resolve the types.
