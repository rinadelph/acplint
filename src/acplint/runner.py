"""Conformance test runner — orchestrates all ACP protocol tests."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import pydantic

from acplint.report import ConformanceReport, TestResult, TestStatus
from acplint.schema import (
    METHOD_AUTHENTICATE,
    METHOD_CANCEL,
    METHOD_CLOSE_SESSION,
    METHOD_CREATE_TERMINAL,
    METHOD_DELETE_SESSION,
    METHOD_FORK_SESSION,
    METHOD_INITIALIZE,
    METHOD_LIST_SESSIONS,
    METHOD_LOAD_SESSION,
    METHOD_LOGOUT,
    METHOD_NEW_SESSION,
    METHOD_PROMPT,
    METHOD_READ_TEXT_FILE,
    METHOD_REQUEST_PERMISSION,
    METHOD_RESUME_SESSION,
    METHOD_SESSION_UPDATE,
    METHOD_SET_SESSION_CONFIG_OPTION,
    METHOD_SET_SESSION_MODE,
    METHOD_WRITE_TEXT_FILE,
    METHOD_RESPONSE_MODELS,
    AgentCapabilities,
    AvailableCommandsUpdate,
    InitializeResponse,
    NewSessionResponse,
    Plan,
    RequestPermissionRequest,
    SessionModeState,
    SessionNotification,
    ToolCall,
    ToolCallUpdate,
    UsageUpdate,
    select_allow_option,
)
from acplint.transport import AcpTransport

ALL_CATEGORIES = [
    "initialization",
    "authentication",
    "session_lifecycle",
    "streaming",
    "tool_calls",
    "permissions",
    "file_operations",
    "terminals",
    "plans",
    "session_modes",
    "config_options",
    "cancel",
    "stress",
    "schema_validation",
]

# All known session update types that we check for during streaming
ALL_SESSION_UPDATE_TYPES = [
    "agent_message_chunk",
    "agent_thought_chunk",
    "user_message_chunk",
    "tool_call",
    "tool_call_update",
    "available_commands_update",
    "plan",
    "current_mode_update",
    "config_option_update",
    "session_info_update",
    "usage_update",
]

# Updates that are REQUIRED for conformance — missing these is a FAIL
REQUIRED_UPDATE_TYPES = {"agent_message_chunk"}

# Categories whose tests can actually elicit each optional update type. When none
# of the listed categories were selected, the agent was never given a chance to
# send the update, so "never received" describes acplint's own configuration
# rather than the agent, and no finding is raised.
UPDATE_TYPE_SOURCE_CATEGORIES = {
    "agent_thought_chunk": {"streaming"},
    "available_commands_update": {"session_lifecycle", "streaming"},
    "usage_update": {"streaming"},
}

# Updates that are recommended — missing these generates a finding but not a FAIL
RECOMMENDED_UPDATE_TYPES = {
    "agent_thought_chunk",
    "available_commands_update",
    "usage_update",
    "tool_call_update",
}

# Agent->client request methods we track for coverage
ALL_AGENT_REQUEST_METHODS = [
    METHOD_REQUEST_PERMISSION,
    METHOD_READ_TEXT_FILE,
    METHOD_WRITE_TEXT_FILE,
    METHOD_CREATE_TERMINAL,
]


class ConformanceRunner:
    """Runs the full ACP conformance test suite against an agent."""

    def __init__(
        self,
        agent_command: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        categories: list[str] | None = None,
        timeout: float = 30.0,
        auto_allow_permissions: bool = True,
    ) -> None:
        self._command = agent_command
        self._cwd = cwd
        self._env = env
        self._categories = categories or ALL_CATEGORIES
        self._timeout = timeout
        self._auto_allow = auto_allow_permissions
        self._report = ConformanceReport()
        self._transport: AcpTransport | None = None
        self._session_id: str | None = None
        self._init_result: dict[str, Any] = {}
        self._agent_capabilities_raw: dict[str, Any] = {}
        self._auth_methods: list[dict[str, Any]] = []
        self._agent_info: dict[str, Any] = {}

        # Coverage tracking: what we actually saw
        self._coverage: dict[str, Any] = {
            "methods_called": set(),
            "update_types_seen": set(),
            "agent_request_methods_seen": set(),
        }

        # Human-readable findings accumulated across all tests
        self._findings: list[str] = []

        # Collected notifications from the most recent streaming prompt
        self._stream_notifications: list[dict[str, Any]] = []

        # Collected agent requests during streaming prompts
        self._stream_agent_requests: list[dict[str, Any]] = []

    def run_all(self) -> ConformanceReport:
        """Run all configured test categories and return the report."""
        asyncio.run(self._run_all_async())
        self._assemble_findings()
        self._report.findings = list(self._findings)
        return self._report

    async def _run_all_async(self) -> None:
        permission_handler = self._default_permission_handler if self._auto_allow else None
        transport = AcpTransport(
            command=self._command,
            cwd=self._cwd,
            env=self._env,
            permission_handler=permission_handler,
        )
        self._transport = transport

        try:
            async with transport:
                # Initialization is always required
                await self._test_initialization()

                # Run remaining categories
                for category in self._categories:
                    if category == "initialization":
                        continue
                    try:
                        await self._run_category(category)
                    except Exception as e:
                        self._record("category_setup", category, TestStatus.ERROR, str(e))
        except Exception as e:
            self._record("connection", "initialization", TestStatus.ERROR, f"Failed to connect to agent: {e}")

        self._report.agent_info = self._agent_info

    async def _run_category(self, category: str) -> None:
        handler = {
            "session_lifecycle": self._test_session_lifecycle,
            "authentication": self._test_authentication,
            "streaming": self._test_streaming,
            "tool_calls": self._test_tool_calls,
            "permissions": self._test_permissions,
            "file_operations": self._test_file_operations,
            "terminals": self._test_terminals,
            "plans": self._test_plans,
            "session_modes": self._test_session_modes,
            "config_options": self._test_config_options,
            "cancel": self._test_cancel,
            "stress": self._test_stress,
            "schema_validation": self._test_schema_validation,
        }.get(category)
        if handler:
            await handler()

    # -----------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------

    async def _test_initialization(self) -> None:
        testing = "initialization" in self._categories

        start = time.monotonic()
        try:
            response = await self._transport.send_request(
                METHOD_INITIALIZE,
                {
                    "protocolVersion": 1,
                    "clientCapabilities": {},
                    "clientInfo": {"name": "acp-conformance", "version": "0.1.0"},
                },
            )
            duration = (time.monotonic() - start) * 1000
            self._coverage["methods_called"].add(METHOD_INITIALIZE)
            result = response.get("result", {})
            self._init_result = result

            # Validate response with Pydantic
            try:
                init_resp = InitializeResponse.model_validate(result)
                self._agent_capabilities_raw = result.get("agentCapabilities", {})
                self._auth_methods = [
                    {"id": m.id, "label": m.label}
                    for m in (init_resp.auth_methods or [])
                ]
                if init_resp.agent_info:
                    self._agent_info = {
                        "name": init_resp.agent_info.name,
                        "version": init_resp.agent_info.version,
                    }
                if testing:
                    self._record("initialize_v1", "initialization", TestStatus.PASS, duration_ms=duration)
            except pydantic.ValidationError as e:
                error_detail = self._format_validation_error(e)
                if testing:
                    self._record(
                        "initialize_v1", "initialization", TestStatus.FAIL,
                        f"Initialize response schema invalid: {error_detail}",
                        duration_ms=duration,
                        details={"validation_error": error_detail, "raw_result": result},
                    )

            # Test: protocol version must be "1"
            protocol_version = result.get("protocolVersion")
            if testing:
                if protocol_version == 1:
                    self._record("protocol_version_returned", "initialization", TestStatus.PASS)
                else:
                    self._record(
                        "protocol_version_returned", "initialization", TestStatus.FAIL,
                        f"Expected protocolVersion=1, got {protocol_version!r}",
                        details={"expected": 1, "actual": protocol_version},
                    )

            # Test: agentCapabilities must be present
            if testing:
                if "agentCapabilities" in result:
                    self._record("agent_capabilities_present", "initialization", TestStatus.PASS)
                else:
                    self._record(
                        "agent_capabilities_present", "initialization", TestStatus.FAIL,
                        "No agentCapabilities in initialize response",
                        details={"keys_present": list(result.keys())},
                    )

            # Test: agentInfo should be present (WARN-level finding if missing)
            if testing:
                if "agentInfo" in result and result["agentInfo"]:
                    self._record("agent_info_present", "initialization", TestStatus.PASS,
                                 details={"agent_info": self._agent_info})
                else:
                    self._record(
                        "agent_info_present", "initialization", TestStatus.PASS,
                        "agentInfo not provided — agents SHOULD include agentInfo for identification",
                    )
                    self._findings.append("⚠ No agentInfo in initialize response — agents should identify themselves")

            # Test: validate agentCapabilities structure if present
            if testing and "agentCapabilities" in result:
                try:
                    AgentCapabilities.model_validate(result["agentCapabilities"])
                    self._record("agent_capabilities_schema_valid", "initialization", TestStatus.PASS)
                except pydantic.ValidationError as e:
                    self._record(
                        "agent_capabilities_schema_valid", "initialization", TestStatus.FAIL,
                        f"agentCapabilities schema invalid: {self._format_validation_error(e)}",
                        details={"validation_error": self._format_validation_error(e)},
                    )

        except Exception as e:
            if testing:
                self._record("initialize_v1", "initialization", TestStatus.ERROR, str(e))
            raise

        # Ensure we have a session for subsequent tests
        await self._ensure_session()

    async def _initialize(self) -> None:
        """Initialize without recording test results (used when initialization category is not being tested)."""
        response = await self._transport.send_request(
            METHOD_INITIALIZE,
            {
                "protocolVersion": 1,
                "clientCapabilities": {},
                "clientInfo": {"name": "acp-conformance", "version": "0.1.0"},
            },
        )
        self._init_result = response.get("result", {})
        self._agent_capabilities_raw = response.get("result", {}).get("agentCapabilities", {})
        try:
            init_resp = InitializeResponse.model_validate(response.get("result", {}))
            self._auth_methods = [
                {"id": m.id, "label": m.label}
                for m in (init_resp.auth_methods or [])
            ]
            if init_resp.agent_info:
                self._agent_info = {
                    "name": init_resp.agent_info.name,
                    "version": init_resp.agent_info.version,
                }
        except pydantic.ValidationError:
            pass

    async def _ensure_session(self) -> None:
        if self._session_id:
            return
        cwd = self._cwd or "/tmp"
        response = await self._transport.send_request(
            METHOD_NEW_SESSION,
            {"cwd": cwd, "mcpServers": []},
        )
        self._session_id = response.get("result", {}).get("sessionId")
        if not self._session_id:
            self._session_id = response.get("result", {}).get("session_id")

    # -----------------------------------------------------------------------
    # Session lifecycle
    # -----------------------------------------------------------------------

    async def _test_session_lifecycle(self) -> None:
        cwd = self._cwd or "/tmp"

        # New session
        start = time.monotonic()
        try:
            response = await self._transport.send_request(
                METHOD_NEW_SESSION,
                {"cwd": cwd, "mcpServers": []},
            )
            duration = (time.monotonic() - start) * 1000
            self._coverage["methods_called"].add(METHOD_NEW_SESSION)
            session_id = response.get("result", {}).get("sessionId") or response.get("result", {}).get("session_id")
            if session_id:
                # Validate the response schema
                try:
                    NewSessionResponse.model_validate(response.get("result", {}))
                    self._record("new_session", "session_lifecycle", TestStatus.PASS, duration_ms=duration,
                                 details={"sessionId": session_id})
                except pydantic.ValidationError as e:
                    self._record(
                        "new_session", "session_lifecycle", TestStatus.FAIL,
                        f"newSession response schema invalid: {self._format_validation_error(e)}",
                        duration_ms=duration,
                        details={"validation_error": self._format_validation_error(e), "raw_result": response.get("result", {})},
                    )
            else:
                self._record(
                    "new_session", "session_lifecycle", TestStatus.FAIL,
                    "No sessionId in response — newSession MUST return a sessionId",
                    duration_ms=duration,
                    details={"raw_result": response.get("result", {})},
                )
        except Exception as e:
            self._record("new_session", "session_lifecycle", TestStatus.ERROR, str(e))
            return

        # List sessions
        try:
            response = await self._transport.send_request(METHOD_LIST_SESSIONS, {"cwd": cwd})
            self._coverage["methods_called"].add(METHOD_LIST_SESSIONS)
            sessions = response.get("result", {}).get("sessions", [])
            if isinstance(sessions, list):
                self._record("list_sessions", "session_lifecycle", TestStatus.PASS,
                             details={"session_count": len(sessions)})
            else:
                self._record(
                    "list_sessions", "session_lifecycle", TestStatus.FAIL,
                    f"Expected sessions array, got {type(sessions).__name__}",
                    details={"raw_result": response.get("result", {})},
                )
        except Exception as e:
            self._record("list_sessions", "session_lifecycle", TestStatus.ERROR, str(e))

        # Load session — test if agent claims support
        load_supported = self._agent_capabilities_raw.get("loadSession", False)
        if load_supported:
            try:
                response = await self._transport.send_request(
                    METHOD_LOAD_SESSION,
                    {"sessionId": session_id, "cwd": cwd, "mcpServers": []},
                )
                self._coverage["methods_called"].add(METHOD_LOAD_SESSION)
                if "error" in response:
                    self._record(
                        "load_session", "session_lifecycle", TestStatus.FAIL,
                        f"Agent advertises loadSession but returned error: {response['error']}",
                        details={"error": response["error"]},
                    )
                else:
                    self._record("load_session", "session_lifecycle", TestStatus.PASS)
            except Exception as e:
                self._record(
                    "load_session", "session_lifecycle", TestStatus.FAIL,
                    f"Agent advertises loadSession but call failed: {e}",
                )
        else:
            # Try it anyway — if it returns method not found, that's fine
            try:
                response = await self._transport.send_request(
                    METHOD_LOAD_SESSION,
                    {"sessionId": session_id, "cwd": cwd, "mcpServers": []},
                )
                self._coverage["methods_called"].add(METHOD_LOAD_SESSION)
                if "error" in response and response["error"].get("code") == -32601:
                    self._record("load_session", "session_lifecycle", TestStatus.PASS,
                                 "Method not found (not advertised, acceptable)")
                elif "error" in response:
                    self._record("load_session", "session_lifecycle", TestStatus.PASS,
                                 f"Not advertised, returned error: {response['error']}")
                else:
                    self._record("load_session", "session_lifecycle", TestStatus.PASS,
                                 "Works despite not being advertised")
            except Exception as e:
                self._record("load_session", "session_lifecycle", TestStatus.PASS,
                             f"Not advertised, call failed: {e}")

        # Resume session
        try:
            response = await self._transport.send_request(
                METHOD_RESUME_SESSION,
                {"sessionId": session_id, "cwd": cwd, "mcpServers": []},
            )
            self._coverage["methods_called"].add(METHOD_RESUME_SESSION)
            if "error" in response and response["error"].get("code") == -32601:
                self._record("resume_session", "session_lifecycle", TestStatus.PASS,
                             "Method not found — not supported")
            elif "error" in response:
                self._record(
                    "resume_session", "session_lifecycle", TestStatus.FAIL,
                    f"Resume session returned error: {response['error']}",
                    details={"error": response["error"]},
                )
            else:
                self._record("resume_session", "session_lifecycle", TestStatus.PASS)
        except Exception as e:
            self._record("resume_session", "session_lifecycle", TestStatus.ERROR, str(e))

        # Close session — check if capabilities say it's supported
        # close_caps not needed.get("sessionCapabilities", {}).get("close", None)
        try:
            response = await self._transport.send_request(
                METHOD_CLOSE_SESSION,
                {"sessionId": session_id},
            )
            self._coverage["methods_called"].add(METHOD_CLOSE_SESSION)
            if "error" in response and response["error"].get("code") == -32601:
                self._record("close_session", "session_lifecycle", TestStatus.PASS,
                             "Method not found — not supported")
            elif "error" in response:
                self._record(
                    "close_session", "session_lifecycle", TestStatus.FAIL,
                    f"Close session returned error: {response['error']}",
                    details={"error": response["error"]},
                )
            else:
                self._record("close_session", "session_lifecycle", TestStatus.PASS)
        except Exception as e:
            self._record("close_session", "session_lifecycle", TestStatus.ERROR, str(e))

        # Delete session
        try:
            response = await self._transport.send_request(
                METHOD_DELETE_SESSION,
                {"sessionId": session_id},
            )
            self._coverage["methods_called"].add(METHOD_DELETE_SESSION)
            if "error" in response and response["error"].get("code") == -32601:
                self._record("delete_session", "session_lifecycle", TestStatus.PASS,
                             "Method not found — not supported")
            elif "error" in response:
                self._record(
                    "delete_session", "session_lifecycle", TestStatus.FAIL,
                    f"Delete session returned error: {response['error']}",
                    details={"error": response["error"]},
                )
            else:
                self._record("delete_session", "session_lifecycle", TestStatus.PASS)
        except Exception as e:
            self._record("delete_session", "session_lifecycle", TestStatus.ERROR, str(e))

        # Fork session
        try:
            response = await self._transport.send_request(
                METHOD_FORK_SESSION,
                {"sessionId": self._session_id or session_id, "cwd": cwd, "mcpServers": []},
            )
            self._coverage["methods_called"].add(METHOD_FORK_SESSION)
            if "error" in response and response["error"].get("code") == -32601:
                self._record("fork_session", "session_lifecycle", TestStatus.PASS,
                             "Method not found — not supported")
            elif "error" in response:
                self._record(
                    "fork_session", "session_lifecycle", TestStatus.FAIL,
                    f"Fork session returned error: {response['error']}",
                    details={"error": response["error"]},
                )
            else:
                self._record("fork_session", "session_lifecycle", TestStatus.PASS)
        except Exception as e:
            self._record("fork_session", "session_lifecycle", TestStatus.ERROR, str(e))

    # -----------------------------------------------------------------------
    # Authentication
    # -----------------------------------------------------------------------

    async def _test_authentication(self) -> None:
        if self._auth_methods:
            self._record("auth_methods_advertised", "authentication", TestStatus.PASS,
                         details={"count": len(self._auth_methods), "methods": self._auth_methods})
        else:
            self._record(
                "auth_methods_advertised", "authentication", TestStatus.PASS,
                "No auth methods advertised — agent may not require authentication",
            )
            self._findings.append("⚠ No auth methods advertised — agent may not require authentication")

        for method_info in self._auth_methods:
            method_id = method_info.get("id", "")
            try:
                response = await self._transport.send_request(
                    METHOD_AUTHENTICATE,
                    {"methodId": method_id},
                )
                self._coverage["methods_called"].add(METHOD_AUTHENTICATE)
                if "error" in response:
                    self._record(
                        f"authenticate_{method_id}", "authentication", TestStatus.FAIL,
                        f"Authentication with method '{method_id}' returned error: {response['error']}",
                        details={"error": response["error"]},
                    )
                else:
                    self._record(f"authenticate_{method_id}", "authentication", TestStatus.PASS)
            except Exception as e:
                self._record(f"authenticate_{method_id}", "authentication", TestStatus.ERROR, str(e))

        # Logout
        try:
            response = await self._transport.send_request(METHOD_LOGOUT, {})
            self._coverage["methods_called"].add(METHOD_LOGOUT)
            if "error" in response and response["error"].get("code") == -32601:
                self._record("logout", "authentication", TestStatus.PASS,
                             "Method not found — not supported")
            elif "error" in response:
                self._record(
                    "logout", "authentication", TestStatus.FAIL,
                    f"Logout returned error: {response['error']}",
                    details={"error": response["error"]},
                )
            else:
                self._record("logout", "authentication", TestStatus.PASS)
        except Exception as e:
            self._record("logout", "authentication", TestStatus.ERROR, str(e))

    # -----------------------------------------------------------------------
    # Streaming — the big one
    # -----------------------------------------------------------------------

    async def _test_streaming(self) -> None:
        await self._ensure_session()

        # Send a prompt and collect session/update notifications
        try:
            prompt_task = asyncio.create_task(
                self._transport.send_request(
                    METHOD_PROMPT,
                    {
                        "sessionId": self._session_id,
                        "prompt": [{"type": "text", "text": "Say 'hello' and nothing else."}],
                    },
                )
            )

            # Collect notifications while the prompt is processing
            notifications = await self._transport.collect_notifications(
                method=METHOD_SESSION_UPDATE,
                duration=5.0,
            )

            try:
                response = await asyncio.wait_for(prompt_task, timeout=self._timeout)
            except asyncio.TimeoutError:
                self._record("prompt_response_received", "streaming", TestStatus.FAIL,
                             f"Prompt response timed out after {self._timeout}s")
                return

            self._coverage["methods_called"].add(METHOD_PROMPT)
            self._stream_notifications = notifications

            # Check we got at least one notification
            if notifications:
                self._record("session_update_received", "streaming", TestStatus.PASS,
                             details={"count": len(notifications)})
            else:
                self._record(
                    "session_update_received", "streaming", TestStatus.FAIL,
                    "No session/update notifications received during prompt — "
                    "agents MUST send session/update notifications to stream their response",
                )

            # Validate each notification against the schema with full error details
            update_types: dict[str, int] = {}
            validation_errors: list[str] = []
            for idx, notif in enumerate(notifications):
                params = notif.get("params", {})
                try:
                    session_notif = SessionNotification.model_validate(params)
                    update_type = session_notif.update.session_update
                    update_types[update_type] = update_types.get(update_type, 0) + 1
                    self._coverage["update_types_seen"].add(update_type)
                except pydantic.ValidationError as e:
                    error_msg = self._format_validation_error(e)
                    # Try to extract the update type even if validation fails
                    raw_update_type = (
                        params.get("update", {})
                        .get("sessionUpdate", "unknown")
                    )
                    validation_errors.append(
                        f"Notification #{idx} (type={raw_update_type}): {error_msg}"
                    )
                    update_types[raw_update_type] = update_types.get(raw_update_type, 0) + 1

            # Record overall schema validation
            if not notifications:
                pass  # already recorded FAIL above
            elif not validation_errors:
                self._record("session_update_schema_valid", "streaming", TestStatus.PASS,
                             details=update_types)
            else:
                self._record(
                    "session_update_schema_valid", "streaming", TestStatus.FAIL,
                    f"{len(notifications) - len(validation_errors)}/{len(notifications)} notifications valid. "
                    f"Errors: {'; '.join(validation_errors[:5])}",
                    details={"update_types": update_types, "validation_errors": validation_errors},
                )

            # ---- Check each specific update type ----

            # agentMessageChunk — REQUIRED
            if update_types.get("agent_message_chunk", 0) > 0:
                self._record("agent_message_chunk_received", "streaming", TestStatus.PASS,
                             details={"count": update_types["agent_message_chunk"]})
            else:
                self._record(
                    "agent_message_chunk_received", "streaming", TestStatus.FAIL,
                    "No agentMessageChunk updates received — "
                    "agents MUST stream agent messages via agentMessageChunk",
                )

            # agentThoughtChunk — RECOMMENDED
            if update_types.get("agent_thought_chunk", 0) > 0:
                self._record("agent_thought_chunk_received", "streaming", TestStatus.PASS,
                             details={"count": update_types["agent_thought_chunk"]})
            else:
                self._record(
                    "agent_thought_chunk_received", "streaming", TestStatus.PASS,
                    "No agentThoughtChunk updates received — "
                    "agents SHOULD stream thinking via agentThoughtChunk for transparency",
                )
                self._findings.append("⚠ Agent doesn't stream thinking (agentThoughtChunk) — agents should stream thoughts for transparency")

            # userMessageChunk
            if update_types.get("user_message_chunk", 0) > 0:
                self._record("user_message_chunk_received", "streaming", TestStatus.PASS,
                             details={"count": update_types["user_message_chunk"]})
            else:
                self._record(
                    "user_message_chunk_received", "streaming", TestStatus.PASS,
                    "No userMessageChunk updates — agent doesn't echo user messages via userMessageChunk",
                )

            # toolCall notifications
            if update_types.get("tool_call", 0) > 0:
                self._record("tool_call_notifications_received", "streaming", TestStatus.PASS,
                             details={"count": update_types["tool_call"]})
            else:
                self._record(
                    "tool_call_notifications_received", "streaming", TestStatus.PASS,
                    "No toolCall notifications during test prompt — "
                    "agent may not have used tools for this simple prompt",
                )

            # toolCallUpdate notifications
            if update_types.get("tool_call_update", 0) > 0:
                self._record("tool_call_update_received", "streaming", TestStatus.PASS,
                             details={"count": update_types["tool_call_update"]})
            else:
                self._record(
                    "tool_call_update_received", "streaming", TestStatus.PASS,
                    "No toolCallUpdate notifications during test prompt",
                )
                if update_types.get("tool_call", 0) > 0:
                    self._findings.append("⚠ Tool calls received but no toolCallUpdate notifications — agents should provide status updates for tool calls")

            # availableCommandsUpdate — RECOMMENDED (hooks/commands)
            if update_types.get("available_commands_update", 0) > 0:
                self._record("available_commands_update_received", "streaming", TestStatus.PASS,
                             details={"count": update_types["available_commands_update"]})
                # Validate schema of the first one
                for notif in notifications:
                    params = notif.get("params", {})
                    update = params.get("update", {})
                    if update.get("sessionUpdate") == "available_commands_update":
                        try:
                            AvailableCommandsUpdate.model_validate(update)
                            self._record("available_commands_schema_valid", "streaming", TestStatus.PASS)
                        except pydantic.ValidationError as e:
                            self._record(
                                "available_commands_schema_valid", "streaming", TestStatus.FAIL,
                                f"availableCommandsUpdate schema invalid: {self._format_validation_error(e)}",
                                details={"validation_error": self._format_validation_error(e)},
                            )
                        break
            else:
                self._record(
                    "available_commands_update_received", "streaming", TestStatus.PASS,
                    "No availableCommandsUpdate received — agent doesn't advertise hooks/commands",
                )
                self._findings.append("⚠ No hooks/available commands advertised (availableCommandsUpdate)")

            # plan
            if update_types.get("plan", 0) > 0:
                self._record("plan_notification_received_streaming", "streaming", TestStatus.PASS,
                             details={"count": update_types["plan"]})
            else:
                self._record(
                    "plan_notification_received_streaming", "streaming", TestStatus.PASS,
                    "No plan notifications during test prompt",
                )

            # currentModeUpdate
            if update_types.get("current_mode_update", 0) > 0:
                self._record("current_mode_update_received", "streaming", TestStatus.PASS,
                             details={"count": update_types["current_mode_update"]})
            else:
                self._record(
                    "current_mode_update_received", "streaming", TestStatus.PASS,
                    "No currentModeUpdate notifications — agent doesn't send mode updates",
                )

            # configOptionUpdate
            if update_types.get("config_option_update", 0) > 0:
                self._record("config_option_update_received", "streaming", TestStatus.PASS,
                             details={"count": update_types["config_option_update"]})
            else:
                self._record(
                    "config_option_update_received", "streaming", TestStatus.PASS,
                    "No configOptionUpdate notifications — agent doesn't send config option updates",
                )

            # sessionInfoUpdate
            if update_types.get("session_info_update", 0) > 0:
                self._record("session_info_update_received", "streaming", TestStatus.PASS,
                             details={"count": update_types["session_info_update"]})
            else:
                self._record(
                    "session_info_update_received", "streaming", TestStatus.PASS,
                    "No sessionInfoUpdate notifications — agent doesn't update session info (title, timestamps)",
                )

            # usageUpdate — RECOMMENDED
            if update_types.get("usage_update", 0) > 0:
                self._record("usage_update_received", "streaming", TestStatus.PASS,
                             details={"count": update_types["usage_update"]})
                # Validate first usage update
                for notif in notifications:
                    params = notif.get("params", {})
                    update = params.get("update", {})
                    if update.get("sessionUpdate") == "usage_update":
                        try:
                            UsageUpdate.model_validate(update)
                            self._record("usage_update_schema_valid", "streaming", TestStatus.PASS)
                        except pydantic.ValidationError as e:
                            self._record(
                                "usage_update_schema_valid", "streaming", TestStatus.FAIL,
                                f"usageUpdate schema invalid: {self._format_validation_error(e)}",
                                details={"validation_error": self._format_validation_error(e)},
                            )
                        break
            else:
                self._record(
                    "usage_update_received", "streaming", TestStatus.PASS,
                    "No usageUpdate notifications — agent doesn't report token usage",
                )
                self._findings.append("⚠ Agent doesn't report token usage (usageUpdate)")

            # Stop reason in prompt response
            stop_reason = response.get("result", {}).get("stopReason") or response.get("result", {}).get("stop_reason")
            if stop_reason:
                self._record("prompt_stop_reason", "streaming", TestStatus.PASS,
                             details={"stop_reason": stop_reason})
            else:
                self._record(
                    "prompt_stop_reason", "streaming", TestStatus.FAIL,
                    "No stopReason in prompt response — agents MUST return stopReason to indicate why generation stopped",
                    details={"raw_result": response.get("result", {})},
                )

        except Exception as e:
            self._record("streaming_prompt", "streaming", TestStatus.ERROR, str(e))

    # -----------------------------------------------------------------------
    # Tool calls — thorough lifecycle check
    # -----------------------------------------------------------------------

    async def _test_tool_calls(self) -> None:
        await self._ensure_session()

        # If we already have streaming notifications, extract tool calls from those
        # Otherwise, send a tool-use prompt
        tool_call_data: list[dict[str, Any]] = []
        tool_update_data: list[dict[str, Any]] = []

        if self._stream_notifications:
            for notif in self._stream_notifications:
                params = notif.get("params", {})
                update = params.get("update", {})
                update_type = update.get("sessionUpdate", "")
                if update_type == "tool_call":
                    tool_call_data.append(update.get("tool_call", update))
                elif update_type == "tool_call_update":
                    tool_update_data.append(update.get("tool_call_update", update))

        # Also try a tool-use prompt
        try:
            prompt_task = asyncio.create_task(
                self._transport.send_request(
                    METHOD_PROMPT,
                    {
                        "sessionId": self._session_id,
                        "prompt": [{"type": "text", "text": "Read the file /tmp/acp_test.txt and tell me its contents"}],
                    },
                )
            )

            notifications = await self._transport.collect_notifications(
                method=METHOD_SESSION_UPDATE,
                duration=5.0,
            )

            try:
                await asyncio.wait_for(prompt_task, timeout=15.0)
            except asyncio.TimeoutError:
                pass

            for notif in notifications:
                params = notif.get("params", {})
                update = params.get("update", {})
                update_type = update.get("sessionUpdate", "")
                if update_type == "tool_call":
                    tool_call_data.append(update.get("tool_call", update))
                elif update_type == "tool_call_update":
                    tool_update_data.append(update.get("tool_call_update", update))

        except Exception as e:
            self._record("tool_calls_prompt", "tool_calls", TestStatus.ERROR, str(e))

        # Now analyze all collected tool call data
        if not tool_call_data:
            self._record(
                "tool_call_notification_received", "tool_calls", TestStatus.PASS,
                "No toolCall notifications received from either prompt — "
                "agent may not use tools for simple prompts",
            )
            self._findings.append("⚠ No tool calls detected during test prompts — unable to validate tool call lifecycle")
            return

        self._record("tool_call_notification_received", "tool_calls", TestStatus.PASS,
                     details={"count": len(tool_call_data)})

        # Validate each tool call schema
        valid_count = 0
        schema_errors: list[str] = []
        for idx, tc in enumerate(tool_call_data):
            try:
                ToolCall.model_validate(tc)
                valid_count += 1
            except pydantic.ValidationError as e:
                schema_errors.append(f"Tool call #{idx}: {self._format_validation_error(e)}")

        if not schema_errors:
            self._record("tool_call_schema_valid", "tool_calls", TestStatus.PASS)
        else:
            self._record(
                "tool_call_schema_valid", "tool_calls", TestStatus.FAIL,
                f"{valid_count}/{len(tool_call_data)} tool calls valid. Errors: {'; '.join(schema_errors[:5])}",
                details={"validation_errors": schema_errors},
            )

        # Check required fields: toolCallId, title, kind, status
        missing_required: list[str] = []
        for idx, tc in enumerate(tool_call_data):
            for field in ("toolCallId", "title", "kind", "status"):
                if field not in tc:
                    missing_required.append(f"Tool call #{idx} missing '{field}'")
        if missing_required:
            self._record(
                "tool_call_required_fields", "tool_calls", TestStatus.FAIL,
                f"Required fields missing: {'; '.join(missing_required[:5])}",
                details={"missing": missing_required},
            )
        else:
            self._record("tool_call_required_fields", "tool_calls", TestStatus.PASS)

        # Check rawInput — agents SHOULD provide it
        raw_input_count = sum(1 for tc in tool_call_data if "rawInput" in tc and tc["rawInput"] is not None)
        if raw_input_count == len(tool_call_data):
            self._record("tool_call_raw_input", "tool_calls", TestStatus.PASS,
                         details={"count": raw_input_count})
        elif raw_input_count > 0:
            self._record(
                "tool_call_raw_input", "tool_calls", TestStatus.PASS,
                f"rawInput present in {raw_input_count}/{len(tool_call_data)} tool calls — "
                f"agents SHOULD provide rawInput for all tool calls",
            )
            self._findings.append(f"⚠ Tool calls missing rawInput — {len(tool_call_data) - raw_input_count}/{len(tool_call_data)} tool calls lack raw input")
        else:
            self._record(
                "tool_call_raw_input", "tool_calls", TestStatus.PASS,
                "No rawInput in any tool call — agents SHOULD provide rawInput for debugging",
            )
            self._findings.append("⚠ Tool calls missing rawInput — agents should provide raw input for debugging")

        # Check rawOutput on completed tool calls — agents SHOULD provide it
        # A tool call is "completed" if its status is completed/failed OR if we have an update showing completion
        completed_tool_calls: list[dict[str, Any]] = []
        for tc in tool_call_data:
            status = tc.get("status", "")
            if status in ("completed", "failed"):
                completed_tool_calls.append(tc)
        # Also check tool call updates for completions
        for tu in tool_update_data:
            status = tu.get("status", "")
            if status in ("completed", "failed"):
                completed_tool_calls.append(tu)

        if completed_tool_calls:
            raw_output_count = sum(
                1 for tc in completed_tool_calls
                if "rawOutput" in tc and tc["rawOutput"] is not None
            )
            if raw_output_count == len(completed_tool_calls):
                self._record("tool_call_raw_output", "tool_calls", TestStatus.PASS,
                             details={"count": raw_output_count})
            elif raw_output_count > 0:
                self._record(
                    "tool_call_raw_output", "tool_calls", TestStatus.PASS,
                    f"rawOutput present in {raw_output_count}/{len(completed_tool_calls)} completed tool calls",
                )
                self._findings.append(f"⚠ Tool calls missing rawOutput — {len(completed_tool_calls) - raw_output_count}/{len(completed_tool_calls)} completed tool calls lack raw output")
            else:
                self._record(
                    "tool_call_raw_output", "tool_calls", TestStatus.PASS,
                    "No rawOutput in any completed tool call — agents SHOULD provide rawOutput for debugging",
                )
                self._findings.append("⚠ Tool calls missing rawOutput — agents should provide raw output for debugging")

        # Check content on completed tool calls
        if completed_tool_calls:
            content_count = sum(
                1 for tc in completed_tool_calls
                if "content" in tc and tc["content"] is not None and len(tc["content"]) > 0
            )
            if content_count == len(completed_tool_calls):
                self._record("tool_call_content", "tool_calls", TestStatus.PASS,
                             details={"count": content_count})
            elif content_count > 0:
                self._record(
                    "tool_call_content", "tool_calls", TestStatus.PASS,
                    f"Content present in {content_count}/{len(completed_tool_calls)} completed tool calls",
                )
                self._findings.append(f"⚠ Tool calls missing content — {len(completed_tool_calls) - content_count}/{len(completed_tool_calls)} completed tool calls lack content blocks")
            else:
                self._record(
                    "tool_call_content", "tool_calls", TestStatus.PASS,
                    "No content in any completed tool call — agents SHOULD provide content blocks for rendering",
                )
                self._findings.append("⚠ Tool calls missing content — agents should provide content blocks for rendering")

        # Check status transitions: pending → in_progress → completed/failed
        if tool_update_data:
            self._record("tool_call_update_received", "tool_calls", TestStatus.PASS,
                         details={"count": len(tool_update_data)})

            # Validate tool call update schema
            valid_updates = 0
            for tu in tool_update_data:
                try:
                    ToolCallUpdate.model_validate(tu)
                    valid_updates += 1
                except pydantic.ValidationError:
                    pass
            if valid_updates == len(tool_update_data):
                self._record("tool_call_update_schema_valid", "tool_calls", TestStatus.PASS)
            else:
                self._record(
                    "tool_call_update_schema_valid", "tool_calls", TestStatus.FAIL,
                    f"{valid_updates}/{len(tool_update_data)} tool call updates valid",
                )

            # Check status progression makes sense
            tool_statuses: dict[str, list[str]] = {}
            for tc in tool_call_data:
                tc_id = tc.get("toolCallId", "unknown")
                tool_statuses.setdefault(tc_id, []).insert(0, tc.get("status", ""))
            for tu in tool_update_data:
                tc_id = tu.get("toolCallId", "unknown")
                status = tu.get("status", "")
                if status:
                    tool_statuses.setdefault(tc_id, []).append(status)

            bad_transitions: list[str] = []
            for tc_id, statuses in tool_statuses.items():
                if len(statuses) < 2:
                    continue
                # Check no backward transitions (completed → pending, etc.)
                terminal = {"completed", "failed"}
                seen_terminal = False
                for status in statuses:
                    if seen_terminal and status not in terminal:
                        bad_transitions.append(f"Tool call {tc_id}: transition after terminal status ({statuses})")
                        break
                    if status in terminal:
                        seen_terminal = True

            if bad_transitions:
                self._record(
                    "tool_call_status_transitions", "tool_calls", TestStatus.FAIL,
                    f"Invalid status transitions: {'; '.join(bad_transitions[:5])}",
                    details={"bad_transitions": bad_transitions, "all_statuses": tool_statuses},
                )
            else:
                self._record("tool_call_status_transitions", "tool_calls", TestStatus.PASS,
                             details={"tool_statuses": {k: v for k, v in tool_statuses.items()}})
        else:
            self._record(
                "tool_call_update_received", "tool_calls", TestStatus.PASS,
                "No toolCallUpdate notifications — agents SHOULD send status updates for tool calls",
            )

        # Check sub-agent meta: look for tool calls with sub-agent indicators in meta
        sub_agent_count = 0
        for tc in tool_call_data:
            meta = tc.get("_meta", tc.get("meta", {}))
            if meta and any(k in meta for k in ("subAgent", "sub_agent", "childAgent", "child_agent", "spawnedAgent", "spawned_agent")):
                sub_agent_count += 1
            # Also check if kind suggests sub-agent
            kind = tc.get("kind", "")
            if kind in ("sub_agent", "subAgent", "spawn"):
                sub_agent_count += 1

        if sub_agent_count > 0:
            self._record("sub_agent_tool_calls_detected", "tool_calls", TestStatus.PASS,
                         details={"count": sub_agent_count})
        else:
            self._record(
                "sub_agent_tool_calls_detected", "tool_calls", TestStatus.PASS,
                "No sub-agent tool calls detected — agent may not support spawning sub-agents",
            )
            self._findings.append("⚠ No sub-agent support detected in tool call meta — agent may not support spawning sub-agents")

        # Check for skills in tool calls
        skills_detected = False
        for tc in tool_call_data:
            meta = tc.get("_meta", tc.get("meta", {}))
            if meta and any(k in meta for k in ("skill", "skills", "skillId", "skill_id")):
                skills_detected = True
                break
            kind = tc.get("kind", "")
            if kind in ("skill", "skill_use"):
                skills_detected = True
                break
        if skills_detected:
            self._record("skills_tool_calls_detected", "tool_calls", TestStatus.PASS)
        else:
            self._record(
                "skills_tool_calls_detected", "tool_calls", TestStatus.PASS,
                "No skills support detected in tool call meta",
            )
            self._findings.append("⚠ No skills support detected in tool call meta")

    # -----------------------------------------------------------------------
    # Permissions
    # -----------------------------------------------------------------------

    async def _test_permissions(self) -> None:
        await self._ensure_session()

        # Send a prompt likely to trigger a permission request
        try:
            prompt_task = asyncio.create_task(
                self._transport.send_request(
                    METHOD_PROMPT,
                    {
                        "sessionId": self._session_id,
                        "prompt": [{"type": "text", "text": "Write 'test' to /tmp/acplint_test.txt"}],
                    },
                )
            )

            # Wait for a permission request from the agent
            try:
                agent_req = await asyncio.wait_for(
                    self._transport.wait_for_agent_request(
                        method=METHOD_REQUEST_PERMISSION,
                        timeout=10.0,
                    ),
                    timeout=12.0,
                )
                self._coverage["agent_request_methods_seen"].add(METHOD_REQUEST_PERMISSION)
                self._record("permission_request_received", "permissions", TestStatus.PASS,
                             details={"method": METHOD_REQUEST_PERMISSION})

                # Validate schema with detailed error
                params = agent_req.get("params", {})
                try:
                    validated = RequestPermissionRequest.model_validate(params)
                    self._record("permission_request_schema_valid", "permissions", TestStatus.PASS,
                                 details={"session_id": validated.session_id,
                                          "tool_call_id": validated.tool_call.tool_call_id,
                                          "option_count": len(validated.options)})
                except pydantic.ValidationError as e:
                    error_detail = self._format_validation_error(e)
                    self._record(
                        "permission_request_schema_valid", "permissions", TestStatus.FAIL,
                        f"Permission request schema invalid: {error_detail}",
                        details={"validation_error": error_detail, "raw_params": params},
                    )
                    # Check which specific fields are missing
                    missing = self._extract_missing_fields(e)
                    if missing:
                        self._record(
                            "permission_request_missing_fields", "permissions", TestStatus.FAIL,
                            f"Permission request missing required fields: {', '.join(missing)}",
                            details={"missing_fields": missing},
                        )

            except (TimeoutError, asyncio.TimeoutError):
                self._record(
                    "permission_request_received", "permissions", TestStatus.PASS,
                    "No permission request received for a write operation — "
                    "agent may be auto-approving dangerous operations without user consent",
                )
                self._findings.append("⚠ Agent doesn't request permissions — it may be auto-approving dangerous operations")

            try:
                await asyncio.wait_for(prompt_task, timeout=15.0)
            except asyncio.TimeoutError:
                pass

        except Exception as e:
            self._record("permissions_prompt", "permissions", TestStatus.ERROR, str(e))

    # -----------------------------------------------------------------------
    # File operations
    # -----------------------------------------------------------------------

    async def _test_file_operations(self) -> None:
        await self._ensure_session()

        try:
            prompt_task = asyncio.create_task(
                self._transport.send_request(
                    METHOD_PROMPT,
                    {
                        "sessionId": self._session_id,
                        "prompt": [{"type": "text", "text": "Read /tmp/acplint_test.txt and tell me its contents"}],
                    },
                )
            )

            # Wait for a read file request
            got_read = False
            try:
                read_req = await asyncio.wait_for(
                    self._transport.wait_for_agent_request(
                        method=METHOD_READ_TEXT_FILE,
                        timeout=10.0,
                    ),
                    timeout=12.0,
                )
                got_read = True
                self._coverage["agent_request_methods_seen"].add(METHOD_READ_TEXT_FILE)
                self._record("read_text_file_request_received", "file_operations", TestStatus.PASS)

                # Validate schema
                params = read_req.get("params", {})
                try:
                    from acplint.schema import ReadTextFileRequest
                    ReadTextFileRequest.model_validate(params)
                    self._record("read_text_file_schema_valid", "file_operations", TestStatus.PASS)
                except pydantic.ValidationError as e:
                    self._record(
                        "read_text_file_schema_valid", "file_operations", TestStatus.FAIL,
                        f"fs/read_text_file request schema invalid: {self._format_validation_error(e)}",
                        details={"validation_error": self._format_validation_error(e), "raw_params": params},
                    )

                # Respond with file content
                await self._transport.send_response(read_req["id"], {"content": "hello from acp-conformance"})

            except (TimeoutError, asyncio.TimeoutError):
                self._record(
                    "read_text_file_request_received", "file_operations", TestStatus.PASS,
                    "No fs/read_text_file request received — "
                    "agent doesn't request file operations via ACP",
                )
                self._findings.append("⚠ Agent doesn't request file reads via ACP (fs/read_text_file)")

            # Also try a write prompt
            write_prompt_task = asyncio.create_task(
                self._transport.send_request(
                    METHOD_PROMPT,
                    {
                        "sessionId": self._session_id,
                        "prompt": [{"type": "text", "text": "Create a file at /tmp/acplint_write_test.txt with content 'hello'"}],
                    },
                )
            )

            try:
                write_req = await asyncio.wait_for(
                    self._transport.wait_for_agent_request(
                        method=METHOD_WRITE_TEXT_FILE,
                        timeout=10.0,
                    ),
                    timeout=12.0,
                )
                self._coverage["agent_request_methods_seen"].add(METHOD_WRITE_TEXT_FILE)
                self._record("write_text_file_request_received", "file_operations", TestStatus.PASS)

                # Validate schema
                params = write_req.get("params", {})
                try:
                    from acplint.schema import WriteTextFileRequest
                    WriteTextFileRequest.model_validate(params)
                    self._record("write_text_file_schema_valid", "file_operations", TestStatus.PASS)
                except pydantic.ValidationError as e:
                    self._record(
                        "write_text_file_schema_valid", "file_operations", TestStatus.FAIL,
                        f"fs/write_text_file request schema invalid: {self._format_validation_error(e)}",
                        details={"validation_error": self._format_validation_error(e), "raw_params": params},
                    )

                # Respond
                await self._transport.send_response(write_req["id"], {})

            except (TimeoutError, asyncio.TimeoutError):
                self._record(
                    "write_text_file_request_received", "file_operations", TestStatus.PASS,
                    "No fs/write_text_file request received — "
                    "agent doesn't request file writes via ACP",
                )
                self._findings.append("⚠ Agent doesn't request file writes via ACP (fs/write_text_file)")

            if not got_read:
                self._findings.append("⚠ Agent doesn't request file operations via ACP (fs/read_text_file, fs/write_text_file)")

            try:
                await asyncio.wait_for(prompt_task, timeout=15.0)
            except asyncio.TimeoutError:
                pass

            try:
                await asyncio.wait_for(write_prompt_task, timeout=15.0)
            except asyncio.TimeoutError:
                pass

        except Exception as e:
            self._record("file_operations_prompt", "file_operations", TestStatus.ERROR, str(e))

    # -----------------------------------------------------------------------
    # Terminals
    # -----------------------------------------------------------------------

    async def _test_terminals(self) -> None:
        await self._ensure_session()

        try:
            prompt_task = asyncio.create_task(
                self._transport.send_request(
                    METHOD_PROMPT,
                    {
                        "sessionId": self._session_id,
                        "prompt": [{"type": "text", "text": "Run 'echo hello' in a terminal"}],
                    },
                )
            )

            # Wait for terminal/create request
            try:
                create_req = await asyncio.wait_for(
                    self._transport.wait_for_agent_request(
                        method=METHOD_CREATE_TERMINAL,
                        timeout=10.0,
                    ),
                    timeout=12.0,
                )
                self._coverage["agent_request_methods_seen"].add(METHOD_CREATE_TERMINAL)
                self._record("terminal_create_received", "terminals", TestStatus.PASS)

                # Validate schema
                params = create_req.get("params", {})
                try:
                    from acplint.schema import CreateTerminalRequest
                    CreateTerminalRequest.model_validate(params)
                    self._record("terminal_create_schema_valid", "terminals", TestStatus.PASS)
                except pydantic.ValidationError as e:
                    self._record(
                        "terminal_create_schema_valid", "terminals", TestStatus.FAIL,
                        f"terminal/create request schema invalid: {self._format_validation_error(e)}",
                        details={"validation_error": self._format_validation_error(e), "raw_params": params},
                    )

                # Respond with a terminal ID
                terminal_id = str(uuid.uuid4())
                await self._transport.send_response(
                    create_req["id"],
                    {"terminalId": terminal_id},
                )
                self._record("terminal_create_responded", "terminals", TestStatus.PASS,
                             details={"terminalId": terminal_id})

            except (TimeoutError, asyncio.TimeoutError):
                self._record(
                    "terminal_create_received", "terminals", TestStatus.PASS,
                    "No terminal/create request received — "
                    "agent doesn't request terminal creation via ACP",
                )
                self._findings.append("⚠ Agent doesn't request terminal creation via ACP (terminal/create)")

            try:
                await asyncio.wait_for(prompt_task, timeout=15.0)
            except asyncio.TimeoutError:
                pass

        except Exception as e:
            self._record("terminals_prompt", "terminals", TestStatus.ERROR, str(e))

    # -----------------------------------------------------------------------
    # Plans
    # -----------------------------------------------------------------------

    async def _test_plans(self) -> None:
        await self._ensure_session()

        # Check if we already saw plan notifications in streaming
        plan_from_streaming = False
        for notif in self._stream_notifications:
            params = notif.get("params", {})
            update = params.get("update", {})
            if update.get("sessionUpdate") == "plan":
                plan_from_streaming = True
                break

        # Also try a plan-specific prompt
        try:
            prompt_task = asyncio.create_task(
                self._transport.send_request(
                    METHOD_PROMPT,
                    {
                        "sessionId": self._session_id,
                        "prompt": [{"type": "text", "text": "Create a simple plan: 1) read a file 2) summarize it"}],
                    },
                )
            )

            notifications = await self._transport.collect_notifications(
                method=METHOD_SESSION_UPDATE,
                duration=5.0,
            )

            try:
                await asyncio.wait_for(prompt_task, timeout=15.0)
            except asyncio.TimeoutError:
                pass

            # Check for plan notifications
            plan_data_list: list[dict[str, Any]] = []
            for notif in notifications:
                params = notif.get("params", {})
                update = params.get("update", {})
                if update.get("sessionUpdate") == "plan":
                    plan_data = update.get("plan", {})
                    plan_data_list.append(plan_data)
                    self._coverage["update_types_seen"].add("plan")

            if plan_data_list or plan_from_streaming:
                self._record("plan_notification_received", "plans", TestStatus.PASS,
                             details={"count": len(plan_data_list) + (1 if plan_from_streaming else 0)})
                # Validate plan schema
                for plan_data in plan_data_list:
                    try:
                        Plan.model_validate(plan_data)
                        self._record("plan_schema_valid", "plans", TestStatus.PASS)
                    except pydantic.ValidationError as e:
                        self._record(
                            "plan_schema_valid", "plans", TestStatus.FAIL,
                            f"Plan schema invalid: {self._format_validation_error(e)}",
                            details={"validation_error": self._format_validation_error(e), "raw_plan": plan_data},
                        )
                    break
            else:
                self._record(
                    "plan_notification_received", "plans", TestStatus.PASS,
                    "No plan notifications received — agent doesn't share execution plans via session/update plan",
                )
                self._findings.append("⚠ Agent doesn't share execution plans via session/update plan")

        except Exception as e:
            self._record("plans_prompt", "plans", TestStatus.ERROR, str(e))

    # -----------------------------------------------------------------------
    # Session modes
    # -----------------------------------------------------------------------

    async def _test_session_modes(self) -> None:
        await self._ensure_session()

        # Check if session has modes from the new session response
        modes_available = False
        try:
            response = await self._transport.send_request(
                METHOD_NEW_SESSION,
                {"cwd": self._cwd or "/tmp", "mcpServers": []},
            )
            modes_data = response.get("result", {}).get("modes", {})
            if modes_data:
                try:
                    SessionModeState.model_validate(modes_data)
                    modes_available = True
                except pydantic.ValidationError:
                    pass
        except Exception:
            pass

        if not modes_available:
            self._record(
                "session_modes_available", "session_modes", TestStatus.PASS,
                "No session modes advertised in newSession response",
            )
            self._findings.append("⚠ Agent doesn't advertise session modes")

        # Try set_session_mode
        try:
            response = await self._transport.send_request(
                METHOD_SET_SESSION_MODE,
                {"sessionId": self._session_id, "modeId": "code"},
            )
            self._coverage["methods_called"].add(METHOD_SET_SESSION_MODE)
            if "error" in response and response["error"].get("code") == -32601:
                self._record("set_session_mode", "session_modes", TestStatus.PASS,
                             "Method not found — not supported")
            elif "error" in response:
                # Error is expected if mode "code" doesn't exist
                self._record(
                    "set_session_mode", "session_modes", TestStatus.PASS,
                    f"Returned error (may be invalid modeId): {response['error']}",
                    details={"error": response["error"]},
                )
            else:
                self._record("set_session_mode", "session_modes", TestStatus.PASS)
        except Exception as e:
            self._record("set_session_mode", "session_modes", TestStatus.ERROR, str(e))

    # -----------------------------------------------------------------------
    # Config options
    # -----------------------------------------------------------------------

    async def _test_config_options(self) -> None:
        await self._ensure_session()

        try:
            response = await self._transport.send_request(
                METHOD_SET_SESSION_CONFIG_OPTION,
                {"sessionId": self._session_id, "optionId": "test", "value": True},
            )
            self._coverage["methods_called"].add(METHOD_SET_SESSION_CONFIG_OPTION)
            if "error" in response and response["error"].get("code") == -32601:
                self._record("set_config_option", "config_options", TestStatus.PASS,
                             "Method not found — not supported")
            elif "error" in response:
                # Error is expected if option "test" doesn't exist
                self._record(
                    "set_config_option", "config_options", TestStatus.PASS,
                    f"Returned error (may be invalid optionId): {response['error']}",
                    details={"error": response["error"]},
                )
            else:
                self._record("set_config_option", "config_options", TestStatus.PASS)
        except Exception as e:
            self._record("set_config_option", "config_options", TestStatus.ERROR, str(e))

    # -----------------------------------------------------------------------
    # Cancel
    # -----------------------------------------------------------------------

    async def _test_cancel(self) -> None:
        await self._ensure_session()

        # Send a prompt, then cancel it
        try:
            prompt_task = asyncio.create_task(
                self._transport.send_request(
                    METHOD_PROMPT,
                    {
                        "sessionId": self._session_id,
                        "prompt": [{"type": "text", "text": "Write me a very long essay about everything"}],
                    },
                )
            )

            # Give the agent a moment to start processing
            await asyncio.sleep(0.5)

            # Send cancel notification
            await self._transport.send_notification(
                METHOD_CANCEL,
                {"sessionId": self._session_id},
            )
            self._coverage["methods_called"].add(METHOD_CANCEL)
            self._record("cancel_notification_sent", "cancel", TestStatus.PASS)

            # The prompt may or may not complete after cancel
            try:
                await asyncio.wait_for(prompt_task, timeout=10.0)
                self._record("cancel_graceful_handling", "cancel", TestStatus.PASS,
                             "Prompt completed after cancel — agent handled cancellation gracefully")
            except asyncio.TimeoutError:
                self._record("cancel_graceful_handling", "cancel", TestStatus.PASS,
                             "Prompt did not complete after cancel — expected behavior")
            except Exception as e:
                self._record(
                    "cancel_graceful_handling", "cancel", TestStatus.FAIL,
                    f"Agent failed after cancel notification: {e}",
                )

        except Exception as e:
            self._record("cancel_test", "cancel", TestStatus.ERROR, str(e))

    # -----------------------------------------------------------------------
    # Stress tests
    # -----------------------------------------------------------------------

    async def _test_stress(self) -> None:
        await self._ensure_session()

        # Rapid sequential prompts
        try:
            for i in range(5):
                try:
                    await self._transport.send_request(
                        METHOD_PROMPT,
                        {
                            "sessionId": self._session_id,
                            "prompt": [{"type": "text", "text": f"Say 'test {i}'"}],
                        },
                    )
                except TimeoutError:
                    self._record(
                        f"stress_sequential_prompt_{i}", "stress", TestStatus.FAIL,
                        f"Prompt {i} timed out — agent should handle rapid sequential prompts",
                    )
                    break
            else:
                self._record("stress_sequential_prompts", "stress", TestStatus.PASS,
                             details={"count": 5})
        except Exception as e:
            self._record("stress_sequential_prompts", "stress", TestStatus.ERROR, str(e))

        # Concurrent sessions
        try:
            cwd = self._cwd or "/tmp"
            tasks = []
            for _ in range(3):
                task = asyncio.create_task(
                    self._transport.send_request(
                        METHOD_NEW_SESSION,
                        {"cwd": cwd, "mcpServers": []},
                    )
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            successes = sum(
                1 for r in results
                if not isinstance(r, Exception) and "error" not in (r if isinstance(r, dict) else {})
            )
            self._record(
                "stress_concurrent_sessions", "stress",
                TestStatus.PASS if successes >= 2 else TestStatus.FAIL,
                f"{successes}/{len(tasks)} concurrent session creations succeeded",
                details={"successes": successes, "total": len(tasks)},
            )
        except Exception as e:
            self._record("stress_concurrent_sessions", "stress", TestStatus.ERROR, str(e))

        # Large content prompt
        try:
            large_text = "x" * 10_000
            await self._transport.send_request(
                METHOD_PROMPT,
                {
                    "sessionId": self._session_id,
                    "prompt": [{"type": "text", "text": f"Repeat this word: {large_text[:100]}"}],
                },
            )
            self._record("stress_large_prompt", "stress", TestStatus.PASS)
        except Exception as e:
            self._record(
                "stress_large_prompt", "stress", TestStatus.FAIL,
                f"Agent failed on large prompt: {e}",
            )

        # Rapid cancel
        try:
            prompt_task = asyncio.create_task(
                self._transport.send_request(
                    METHOD_PROMPT,
                    {
                        "sessionId": self._session_id,
                        "prompt": [{"type": "text", "text": "Do something complex"}],
                    },
                )
            )
            await asyncio.sleep(0.1)
            await self._transport.send_notification(METHOD_CANCEL, {"sessionId": self._session_id})
            try:
                await asyncio.wait_for(prompt_task, timeout=5.0)
            except asyncio.TimeoutError:
                pass
            self._record("stress_rapid_cancel", "stress", TestStatus.PASS)
        except Exception as e:
            self._record(
                "stress_rapid_cancel", "stress", TestStatus.FAIL,
                f"Agent crashed on rapid cancel: {e}",
            )

    # -----------------------------------------------------------------------
    # Schema validation
    # -----------------------------------------------------------------------

    async def _test_schema_validation(self) -> None:
        await self._ensure_session()

        methods_to_validate = [
            (METHOD_INITIALIZE, {"protocolVersion": 1, "clientCapabilities": {}, "clientInfo": {"name": "acp-conformance", "version": "0.1.0"}}),
            (METHOD_NEW_SESSION, {"cwd": self._cwd or "/tmp", "mcpServers": []}),
            (METHOD_LIST_SESSIONS, {"cwd": self._cwd or "/tmp"}),
        ]

        for method, params in methods_to_validate:
            try:
                response = await self._transport.send_request(method, params)
                self._coverage["methods_called"].add(method)
                result = response.get("result", {})
                model_class = METHOD_RESPONSE_MODELS.get(method)
                if model_class:
                    try:
                        model_class.model_validate(result)
                        self._record(f"schema_{method.replace('/', '_')}", "schema_validation", TestStatus.PASS)
                    except pydantic.ValidationError as e:
                        error_detail = self._format_validation_error(e)
                        self._record(
                            f"schema_{method.replace('/', '_')}", "schema_validation", TestStatus.FAIL,
                            f"Response schema invalid for {method}: {error_detail}",
                            details={"validation_error": error_detail, "method": method, "raw_result": result},
                        )
                else:
                    self._record(
                        f"schema_{method.replace('/', '_')}", "schema_validation", TestStatus.PASS,
                        "No Pydantic model for method",
                    )
            except Exception as e:
                self._record(f"schema_{method.replace('/', '_')}", "schema_validation", TestStatus.ERROR, str(e))

        # Validate every notification we collected against SessionNotification
        all_notifications = self._stream_notifications
        if all_notifications:
            for idx, notif in enumerate(all_notifications[:20]):  # Validate up to 20
                params = notif.get("params", {})
                try:
                    SessionNotification.model_validate(params)
                except pydantic.ValidationError as e:
                    update_type = params.get("update", {}).get("sessionUpdate", "unknown")
                    self._record(
                        f"schema_notification_{idx}", "schema_validation", TestStatus.FAIL,
                        f"Notification #{idx} (type={update_type}) schema invalid: {self._format_validation_error(e)}",
                        details={"validation_error": self._format_validation_error(e), "update_type": update_type},
                    )
                    break
            else:
                self._record("schema_all_notifications_valid", "schema_validation", TestStatus.PASS,
                             details={"count": min(len(all_notifications), 20)})

        # Coverage summary test
        self._record(
            "coverage_methods_exercised", "schema_validation", TestStatus.PASS,
            details={
                "methods_called": sorted(self._coverage["methods_called"]),
                "update_types_seen": sorted(self._observed_update_types()),
                "agent_request_methods_seen": sorted(self._coverage["agent_request_methods_seen"]),
            },
        )

    # -----------------------------------------------------------------------
    # Findings assembly
    # -----------------------------------------------------------------------

    def _observed_update_types(self) -> set[str]:
        """Every session update type seen, from any test or none.

        The transport records update types as notifications arrive, which catches
        the ones an agent sends outside a prompt turn (available_commands_update
        on session/new, for example). Individual tests also record what they saw,
        so the two are unioned.
        """
        observed = set(self._coverage["update_types_seen"])
        if self._transport is not None:
            observed |= self._transport.seen_update_types
        return observed

    def _update_type_was_exercised(self, update_type: str) -> bool:
        """Whether any selected category could have elicited this update type."""
        sources = UPDATE_TYPE_SOURCE_CATEGORIES.get(update_type)
        if sources is None:
            return True
        return bool(sources & set(self._categories))

    def _assemble_findings(self) -> None:
        """Collect findings from coverage gaps that weren't caught by individual tests."""
        seen_update_types = self._observed_update_types()

        # Check for update types we never saw at all
        for update_type in ALL_SESSION_UPDATE_TYPES:
            if update_type not in seen_update_types and self._update_type_was_exercised(update_type):
                if update_type in REQUIRED_UPDATE_TYPES:
                    # Already handled as FAIL in streaming tests
                    pass
                elif update_type == "agent_thought_chunk":
                    if "⚠ Agent doesn't stream thinking" not in " ".join(self._findings):
                        self._findings.append(f"⚠ No {update_type} notifications received at all")
                elif update_type == "available_commands_update":
                    if "⚠ No hooks/available commands" not in " ".join(self._findings):
                        self._findings.append(f"⚠ No {update_type} notifications received — agent doesn't advertise commands/hooks")
                elif update_type == "usage_update":
                    if "⚠ Agent doesn't report token usage" not in " ".join(self._findings):
                        self._findings.append(f"⚠ No {update_type} notifications received — agent doesn't report usage")

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _default_permission_handler(self, params: dict[str, Any]) -> dict[str, Any]:
        chosen = select_allow_option(params.get("options", []))
        if chosen:
            # ACP v1 SelectedPermissionOutcome = {outcome:"selected", optionId}.
            return {"outcome": {"outcome": "selected", "optionId": chosen}}
        # No options to select → cancelled (ACP v1 CancelledPermissionOutcome).
        return {"outcome": {"outcome": "cancelled"}}

    def _record(
        self,
        name: str,
        category: str,
        status: TestStatus,
        message: str | None = None,
        duration_ms: float = 0.0,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._report.add(TestResult(
            name=name,
            category=category,
            status=status,
            message=message,
            duration_ms=duration_ms,
            details=details,
        ))

    @staticmethod
    def _format_validation_error(error: pydantic.ValidationError) -> str:
        """Format a Pydantic validation error into a concise, specific string."""
        parts = []
        for err in error.errors():
            loc = " → ".join(str(part) for part in err["loc"])
            msg = err["msg"]
            input_val = err.get("input", "")
            input_str = str(input_val)[:80] if input_val is not None else "None"
            parts.append(f"{loc}: {msg} (got {input_str})")
        return "; ".join(parts)

    @staticmethod
    def _extract_missing_fields(error: pydantic.ValidationError) -> list[str]:
        """Extract field names that are missing from a validation error."""
        missing = []
        for err in error.errors():
            if err["type"] in ("missing", "value_error.missing"):
                loc = " → ".join(str(part) for part in err["loc"])
                missing.append(loc)
        return missing
