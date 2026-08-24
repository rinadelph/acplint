# Changelog

All notable changes to acplint are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Findings no longer describe acplint's own category selection as an agent
  defect.** Running a subset such as `--categories initialization
  session_lifecycle` always emitted findings claiming the agent sent no
  `agent_thought_chunk`, `available_commands_update`, or `usage_update`. Two
  causes: coverage was recorded only inside `_test_streaming` and `_test_plans`,
  so notifications delivered during `session/new` were invisible even though the
  transport had queued them; and `_assemble_findings` never consulted the
  selected categories, so it reported prompt-only update types as missing when
  nothing had prompted the agent. The transport now records `sessionUpdate`
  types as notifications arrive, and findings for optional update types are
  raised only when a selected category could have elicited them. Fixes #3.
- **No more `RuntimeError: Event loop is closed` traceback at interpreter
  shutdown.** `AcpTransport.__aexit__` cancelled the reader tasks without
  awaiting them, did not await the process after `kill()`, and never closed the
  asyncio subprocess transport. The unclosed transport outlived the loop that
  `asyncio.run()` closed, and its `__del__` scheduled a callback on the dead
  loop. Teardown now awaits the cancelled tasks, reaps a killed process, and
  closes the subprocess transport while the loop is still open. Fixes #4.

### Added

- Test suite under `tests/`, using the `pytest` and `pytest-asyncio` dev
  dependencies and the `asyncio_mode = "auto"` setting already declared in
  `pyproject.toml`. Covers both fixes above by linting throwaway ACP agents
  built for each case.

## [0.2.0] - 2026-07-10

ACP v1 spec-conformance alignment for the runner and schema. These are general
ACP v1 fixes applicable to any ACP v1 agent, verified end-to-end against a live
ACP server bridge reaching Full Conformance (all 14 categories pass).

### Fixed

- **`protocolVersion` is an integer, not a string.** ACP v1 defines
  `ProtocolVersion` as `integer (uint16)` (`PROTOCOL_VERSION = 1`). acplint sent
  and compared the string `"1"`, which SDK-based agents reject with
  `-32602 "expected number, received string"`, failing the mandatory
  `initialization` category. The `ProtocolVersion` enum, all `initialize` send
  sites, the `protocol_version_returned` comparison, and `schema_validation`
  params now use integer `1`.
- **Permission request/response shape aligned to ACP v1.**
  - `RequestPermissionRequest` is now `{ sessionId, toolCall, options }` — the
    stale required `description` field was dropped and `toolCall` is required.
  - `PermissionOptionChoice` is now `{ optionId, name, kind }` with a
    `PermissionOptionKind` enum (`allow_once` / `allow_always` / `reject_once` /
    `reject_always`).
  - Auto-allow handlers now emit the v1 `RequestPermissionOutcome`
    discriminated union: `{ outcome: "selected", optionId }` /
    `{ outcome: "cancelled" }`.
  - `select_allow_option()` prefers an allow-kind option so a reject-first
    ordering cannot produce a false negative; no options / no allow option now
    responds `cancelled` (ACP v1 has no `minItems` on `options`) instead of a
    JSON-RPC error.
- **`PromptRequest` field, `sessionUpdate` casing, and streaming timeout.**
  - `PromptRequest` uses `prompt` (not the non-v1 `content`); renamed at the
    schema and all runner send sites.
  - `sessionUpdate` discriminator values are snake_case per the ACP SDK
    (`agent_message_chunk`, `tool_call`, `tool_call_update`, `usage_update`,
    etc.); previously camelCase constants were compared against snake_case wire
    values, causing the required `agent_message_chunk_received` check to
    false-fail.
  - `_test_streaming` now honors `--timeout` instead of a hardcoded 20s wait.

### Added

- `StopReason` gains `max_tokens`, `max_turn_requests`, and `cancelled` for ACP
  v1 alignment.

## [0.1.0] - 2026-07-10

### Added

- Initial release of acplint — the linter for the Agent Client Protocol (ACP).

[0.2.0]: https://github.com/rinadelph/acplint/releases/tag/v0.2.0
[0.1.0]: https://github.com/rinadelph/acplint/releases/tag/v0.1.0
