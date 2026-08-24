"""Shared helpers for building throwaway ACP agents to lint against."""
from __future__ import annotations

import stat
import sys

import pytest

AGENT_PREAMBLE = '''\
import json, sys

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\\n")
    sys.stdout.flush()

INIT_RESULT = {
    "protocolVersion": 1,
    "agentCapabilities": {"loadSession": True},
    "authMethods": [],
    "agentInfo": {"name": "test-agent", "version": "1.0.0"},
}

def handle(msg):
    method = msg.get("method")
    mid = msg.get("id")
    if method == "initialize":
        ON_INITIALIZE(mid)
    elif method == "session/new":
        ON_NEW_SESSION(mid)
    elif method == "session/list":
        send({"jsonrpc": "2.0", "id": mid, "result": {"sessions": []}})
    elif method in ("session/load", "session/resume", "session/close", "session/delete"):
        send({"jsonrpc": "2.0", "id": mid, "result": {}})
    elif mid is not None:
        send({"jsonrpc": "2.0", "id": mid,
              "error": {"code": -32601, "message": "Method not found"}})
'''

AGENT_MAIN = '''
for line in sys.stdin:
    line = line.strip()
    if line:
        handle(json.loads(line))
'''


@pytest.fixture
def make_agent(tmp_path):
    """Write an executable Python ACP agent and return its path."""

    def _make(body: str, name: str = "agent.py") -> str:
        path = tmp_path / name
        path.write_text(AGENT_PREAMBLE + body + AGENT_MAIN)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        return str(path)

    return _make


@pytest.fixture
def agent_python() -> str:
    """The interpreter used to run generated agents."""
    return sys.executable


@pytest.fixture
def run_dir(tmp_path) -> str:
    d = tmp_path / "cwd"
    d.mkdir(exist_ok=True)
    return str(d)
