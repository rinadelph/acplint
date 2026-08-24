"""The transport must be torn down while the event loop is still open.

Regression tests for the "RuntimeError: Event loop is closed" traceback printed
at interpreter shutdown after an otherwise successful run.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

from acplint.transport import AcpTransport

STUBBORN_AGENT = textwrap.dedent(
    '''
    import json, signal, sys
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    def send(obj):
        sys.stdout.write(json.dumps(obj) + "\\n")
        sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        mid = msg.get("id")
        if msg.get("method") == "initialize":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": 1,
                "agentCapabilities": {"loadSession": True},
                "authMethods": [],
                "agentInfo": {"name": "stubborn", "version": "1.0.0"},
            }})
        elif mid is not None:
            send({"jsonrpc": "2.0", "id": mid,
                  "error": {"code": -32601, "message": "Method not found"}})
    '''
)

COOPERATIVE_AGENT = textwrap.dedent(
    '''
    import json, sys

    def send(obj):
        sys.stdout.write(json.dumps(obj) + "\\n")
        sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        mid = msg.get("id")
        if msg.get("method") == "initialize":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": 1,
                "agentCapabilities": {},
                "authMethods": [],
            }})
        elif mid is not None:
            send({"jsonrpc": "2.0", "id": mid,
                  "error": {"code": -32601, "message": "Method not found"}})
    '''
)


async def test_teardown_completes_when_the_agent_ignores_sigterm(tmp_path):
    """The kill path must still reap the process and close the transport.

    A cooperative agent exits on terminate() and the loop tends to run its
    teardown callbacks anyway, so the leak only shows on the kill path. This
    uses an agent that ignores SIGTERM to exercise it deterministically.
    """
    agent = tmp_path / "stubborn.py"
    agent.write_text(STUBBORN_AGENT)

    transport = AcpTransport(command=[sys.executable, str(agent)])
    async with transport:
        await transport.send_request(
            "initialize", {"protocolVersion": 1, "clientCapabilities": {}}
        )

    # The process was reaped rather than left for the GC.
    assert transport._process.returncode is not None
    # No unclosed subprocess transport survives the loop.
    assert transport._process._transport.is_closing()
    # Cancelled tasks were awaited, so nothing is pending at loop close.
    assert transport._reader_task.done()
    assert transport._stderr_task.done()


def test_no_event_loop_traceback_for_a_stubborn_agent(tmp_path):
    """An agent that ignores SIGTERM forces the kill path, which used to leak.

    Runs acplint end to end in a subprocess so interpreter shutdown, where the
    traceback was printed, actually happens.
    """
    agent = tmp_path / "stubborn.py"
    agent.write_text(STUBBORN_AGENT)
    report = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "acplint.cli",
            "--agent",
            sys.executable,
            "--agent-args",
            str(agent),
            "--categories",
            "initialization",
            "--output",
            "json",
            "--output-file",
            str(report),
            "--cwd",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    combined = result.stdout + result.stderr
    assert "Event loop is closed" not in combined, combined
    assert "Exception ignored in" not in combined, combined
