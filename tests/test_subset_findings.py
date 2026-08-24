"""Findings must describe the agent, not acplint's own category selection.

Regression tests for the case where a category subset made acplint report
update types as "never received" when either the agent had already sent one, or
no selected category could have elicited it.
"""
from __future__ import annotations

from acplint.runner import ConformanceRunner

ADVERTISES_COMMANDS = '''
ON_INITIALIZE = lambda mid: send({"jsonrpc": "2.0", "id": mid, "result": INIT_RESULT})

def ON_NEW_SESSION(mid):
    send({"jsonrpc": "2.0", "method": "session/update", "params": {
        "sessionId": "s1",
        "update": {
            "sessionUpdate": "available_commands_update",
            "availableCommands": [{"name": "help", "description": "Show help"}],
        },
    }})
    send({"jsonrpc": "2.0", "id": mid, "result": {"sessionId": "s1"}})
'''

SILENT = '''
ON_INITIALIZE = lambda mid: send({"jsonrpc": "2.0", "id": mid, "result": INIT_RESULT})
ON_NEW_SESSION = lambda mid: send({"jsonrpc": "2.0", "id": mid, "result": {"sessionId": "s1"}})
'''


def run(agent_python, agent_path, run_dir, categories):
    runner = ConformanceRunner(
        agent_command=[agent_python, agent_path],
        cwd=run_dir,
        categories=categories,
    )
    return runner.run_all()


def coverage_update_types(report):
    for result in report.results:
        if result.name == "coverage_methods_exercised":
            return result.details["update_types_seen"]
    raise AssertionError("coverage_methods_exercised was not recorded")


def test_available_commands_on_session_new_counts_as_received(
    make_agent, agent_python, run_dir
):
    """An update sent outside a prompt turn still counts as received."""
    agent = make_agent(ADVERTISES_COMMANDS)
    report = run(
        agent_python, agent, run_dir, ["initialization", "session_lifecycle"]
    )

    assert not any("available_commands_update" in f for f in report.findings)


def test_available_commands_appears_in_coverage(make_agent, agent_python, run_dir):
    agent = make_agent(ADVERTISES_COMMANDS)
    report = run(
        agent_python,
        agent,
        run_dir,
        ["initialization", "session_lifecycle", "schema_validation"],
    )

    assert "available_commands_update" in coverage_update_types(report)


def test_silent_agent_still_reported_when_session_lifecycle_ran(
    make_agent, agent_python, run_dir
):
    """session/new is a real opportunity to advertise, so silence is a finding."""
    agent = make_agent(SILENT)
    report = run(
        agent_python, agent, run_dir, ["initialization", "session_lifecycle"]
    )

    assert any("available_commands_update" in f for f in report.findings)


def test_prompt_only_update_types_not_reported_without_streaming(
    make_agent, agent_python, run_dir
):
    """agent_thought_chunk and usage_update only occur during a prompt turn."""
    agent = make_agent(SILENT)
    report = run(
        agent_python, agent, run_dir, ["initialization", "session_lifecycle"]
    )

    assert not any("agent_thought_chunk" in f for f in report.findings)
    assert not any("usage_update" in f for f in report.findings)


def test_initialization_only_run_reports_no_update_findings(
    make_agent, agent_python, run_dir
):
    """Nothing that produces session updates ran, so nothing is claimed."""
    agent = make_agent(SILENT)
    report = run(agent_python, agent, run_dir, ["initialization"])

    for update_type in (
        "agent_thought_chunk",
        "available_commands_update",
        "usage_update",
    ):
        assert not any(update_type in f for f in report.findings), update_type
