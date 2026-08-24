"""JSON-RPC transport for communicating with ACP agents over stdio."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any, Callable

from acplint.schema import (
    METHOD_CREATE_TERMINAL,
    METHOD_KILL_TERMINAL,
    METHOD_READ_TEXT_FILE,
    METHOD_RELEASE_TERMINAL,
    METHOD_REQUEST_PERMISSION,
    METHOD_SESSION_UPDATE,
    METHOD_TERMINAL_OUTPUT,
    METHOD_WAIT_FOR_TERMINAL_EXIT,
    METHOD_WRITE_TEXT_FILE,
    select_allow_option,
)

logger = logging.getLogger(__name__)


class AcpTransport:
    """Manages JSON-RPC communication with an ACP agent over stdio."""

    def __init__(
        self,
        command: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        permission_handler: Callable | None = None,
        read_file_handler: Callable | None = None,
        write_file_handler: Callable | None = None,
        terminal_handler: Callable | None = None,
    ):
        self._command = command
        self._cwd = cwd
        self._env = env
        self._next_id = 1
        self._pending_requests: dict[int | str, asyncio.Future] = {}
        self._notifications: asyncio.Queue[dict] = asyncio.Queue()
        self._agent_requests: asyncio.Queue[dict] = asyncio.Queue()
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._permission_handler = permission_handler
        self._read_file_handler = read_file_handler
        self._write_file_handler = write_file_handler
        self._terminal_handler = terminal_handler
        self._stderr_lines: list[str] = []
        self._stderr_task: asyncio.Task | None = None
        # Session update types observed on the wire, regardless of which test
        # was running when they arrived.
        self.seen_update_types: set[str] = set()

    async def __aenter__(self) -> AcpTransport:
        self._process = await asyncio.create_subprocess_exec(
            *self._command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._cwd,
            env=self._env,
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        self._stderr_task = asyncio.create_task(self._stderr_loop())
        return self

    async def __aexit__(self, *args: Any) -> None:
        # Await the cancelled tasks so their cancellation is observed while the
        # loop is still running.
        for task in (self._reader_task, self._stderr_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        if self._process is not None:
            if self._process.returncode is None:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self._process.kill()
                    # Reap the killed process too. Without this the pipe teardown
                    # callbacks never run before the loop closes.
                    await self._process.wait()
            self._close_process_transport()
        # Cancel any pending request futures
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

    def _close_process_transport(self) -> None:
        """Close the subprocess transport while the loop is still open.

        asyncio.subprocess.Process exposes no public close(), so an unclosed
        transport is left to the garbage collector. By then asyncio.run() has
        closed the loop, and BaseSubprocessTransport.__del__ calls close(),
        which reaches write_eof() on a still-open stdin pipe and schedules a
        callback on a closed loop. CPython prints that as "Exception ignored in"
        with a RuntimeError traceback, after a run that otherwise succeeded.
        """
        transport = getattr(self._process, "_transport", None)
        if transport is not None:
            transport.close()

    def _allocate_id(self) -> int:
        request_id = self._next_id
        self._next_id += 1
        return request_id

    async def send_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        request_id: int | None = None,
    ) -> dict[str, Any]:
        """Send a JSON-RPC request and wait for the response."""
        if request_id is None:
            request_id = self._allocate_id()

        message: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            message["params"] = params

        future: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future

        payload = json.dumps(message) + "\n"
        assert self._process and self._process.stdin
        self._process.stdin.write(payload.encode())
        await self._process.stdin.drain()

        logger.debug("→ req id=%s method=%s", request_id, method)

        try:
            result = await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise TimeoutError(f"Request {method} (id={request_id}) timed out after 30s")
        return result

    async def send_notification(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        message: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            message["params"] = params

        payload = json.dumps(message) + "\n"
        assert self._process and self._process.stdin
        self._process.stdin.write(payload.encode())
        await self._process.stdin.drain()
        logger.debug("→ notif method=%s", method)

    async def send_response(self, request_id: int | str, result: Any) -> None:
        """Send a JSON-RPC response back to the agent."""
        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
        }
        payload = json.dumps(message) + "\n"
        assert self._process and self._process.stdin
        self._process.stdin.write(payload.encode())
        await self._process.stdin.drain()
        logger.debug("→ resp id=%s", request_id)

    async def send_error_response(
        self, request_id: int | str, code: int, message: str
    ) -> None:
        """Send a JSON-RPC error response."""
        message_dict = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }
        payload = json.dumps(message_dict) + "\n"
        assert self._process and self._process.stdin
        self._process.stdin.write(payload.encode())
        await self._process.stdin.drain()

    async def wait_for_notification(
        self,
        method: str | None = None,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        """Wait for a notification from the agent, optionally filtering by method."""
        deadline = asyncio.get_event_loop().time() + timeout
        checked: list[dict] = []

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                # Put checked items back
                for item in checked:
                    await self._notifications.put(item)
                raise TimeoutError(
                    f"Timed out waiting for notification"
                    f"{' with method=' + method if method else ''}"
                )

            try:
                notification = await asyncio.wait_for(
                    self._notifications.get(), timeout=remaining
                )
            except asyncio.TimeoutError:
                for item in checked:
                    await self._notifications.put(item)
                raise TimeoutError(
                    f"Timed out waiting for notification"
                    f"{' with method=' + method if method else ''}"
                )

            if method is None or notification.get("method") == method:
                # Put checked items back
                for item in checked:
                    await self._notifications.put(item)
                return notification

            checked.append(notification)

    async def collect_notifications(
        self,
        method: str | None = None,
        duration: float = 2.0,
    ) -> list[dict[str, Any]]:
        """Collect all notifications for a given duration, optionally filtering by method."""
        results: list[dict[str, Any]] = []
        deadline = asyncio.get_event_loop().time() + duration

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            try:
                notification = await asyncio.wait_for(
                    self._notifications.get(), timeout=remaining
                )
                if method is None or notification.get("method") == method:
                    results.append(notification)
            except asyncio.TimeoutError:
                break
        return results

    async def wait_for_agent_request(
        self,
        method: str | None = None,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        """Wait for a request from the agent (agent→client direction)."""
        deadline = asyncio.get_event_loop().time() + timeout
        checked: list[dict] = []

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                for item in checked:
                    await self._agent_requests.put(item)
                raise TimeoutError(
                    f"Timed out waiting for agent request"
                    f"{' with method=' + method if method else ''}"
                )

            try:
                request = await asyncio.wait_for(
                    self._agent_requests.get(), timeout=remaining
                )
            except asyncio.TimeoutError:
                for item in checked:
                    await self._agent_requests.put(item)
                raise TimeoutError(
                    f"Timed out waiting for agent request"
                    f"{' with method=' + method if method else ''}"
                )

            if method is None or request.get("method") == method:
                for item in checked:
                    await self._agent_requests.put(item)
                return request

            checked.append(request)

    @property
    def stderr_lines(self) -> list[str]:
        return list(self._stderr_lines)

    async def _read_loop(self) -> None:
        """Read JSON-RPC messages from the agent's stdout."""
        assert self._process and self._process.stdout
        buffer = b""
        try:
            while True:
                chunk = await self._process.stdout.read(4096)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        message = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON from agent: %s", line[:200])
                        continue
                    await self._dispatch(message)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Read loop error: %s", e)

    async def _stderr_loop(self) -> None:
        """Read stderr from the agent process."""
        assert self._process and self._process.stderr
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                decoded = line.decode(errors="replace").rstrip()
                self._stderr_lines.append(decoded)
                logger.debug("stderr: %s", decoded)
        except asyncio.CancelledError:
            pass

    async def _dispatch(self, message: dict[str, Any]) -> None:
        """Route an incoming message to the appropriate handler."""
        if "id" in message:
            if "method" in message:
                # This is a request from the agent (agent→client)
                request_id = message["id"]
                method = message.get("method", "")
                logger.debug("← req id=%s method=%s", request_id, method)
                await self._handle_agent_request(message)
            else:
                # This is a response to our request
                request_id = message["id"]
                future = self._pending_requests.pop(request_id, None)
                if future and not future.done():
                    future.set_result(message)
                    logger.debug("← resp id=%s", request_id)
                else:
                    logger.debug("Received response for unknown id=%s (likely agent request response)", request_id)
        else:
            # This is a notification
            method = message.get("method", "")
            logger.debug("← notif method=%s", method)
            self._record_update_type(message)
            await self._notifications.put(message)

    def _record_update_type(self, message: dict[str, Any]) -> None:
        """Track session/update types as they arrive.

        Coverage is recorded here rather than inside an individual test because
        agents legitimately send notifications outside a prompt turn, for
        example available_commands_update in response to session/new. Recording
        only where a particular test happens to look makes an agent appear not
        to support something it demonstrably sent.
        """
        if message.get("method") != METHOD_SESSION_UPDATE:
            return
        params = message.get("params")
        if not isinstance(params, dict):
            return
        update = params.get("update")
        if not isinstance(update, dict):
            return
        update_type = update.get("sessionUpdate")
        if isinstance(update_type, str) and update_type:
            self.seen_update_types.add(update_type)

    async def _handle_agent_request(self, message: dict[str, Any]) -> None:
        """Handle an incoming request from the agent by auto-responding or queuing it."""
        method = message.get("method", "")
        request_id = message["id"]
        params = message.get("params", {})

        if method == METHOD_REQUEST_PERMISSION:
            if self._permission_handler:
                result = self._permission_handler(params)
                await self.send_response(request_id, result)
            else:
                # Default: allow the first allow-kind option (ACP v1
                # SelectedPermissionOutcome = {outcome:"selected", optionId}).
                # No options / no allow-kind option → CancelledPermissionOutcome
                # ({outcome:"cancelled"}); options has no minItems in v1, so an
                # empty array is a valid request that still requires a valid
                # RequestPermissionResponse, not a JSON-RPC error.
                chosen = select_allow_option(params.get("options", []))
                if chosen:
                    await self.send_response(request_id, {
                        "outcome": {"outcome": "selected", "optionId": chosen}
                    })
                else:
                    await self.send_response(request_id, {
                        "outcome": {"outcome": "cancelled"}
                    })
        elif method == METHOD_READ_TEXT_FILE:
            if self._read_file_handler:
                result = self._read_file_handler(params)
                await self.send_response(request_id, result)
            else:
                await self._agent_requests.put(message)
        elif method == METHOD_WRITE_TEXT_FILE:
            if self._write_file_handler:
                result = self._write_file_handler(params)
                await self.send_response(request_id, result)
            else:
                await self._agent_requests.put(message)
        elif method in (
            METHOD_CREATE_TERMINAL,
            METHOD_TERMINAL_OUTPUT,
            METHOD_RELEASE_TERMINAL,
            METHOD_WAIT_FOR_TERMINAL_EXIT,
            METHOD_KILL_TERMINAL,
        ):
            if self._terminal_handler:
                result = self._terminal_handler(method, params)
                await self.send_response(request_id, result)
            else:
                await self._agent_requests.put(message)
        else:
            # Unknown method — queue for the test to handle
            await self._agent_requests.put(message)
