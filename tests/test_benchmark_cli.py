"""Offline tests for benchmark math, splits, and the CLI/server."""

import json
import threading
import urllib.request

import pytest

from agent_fast_decider.benchmark import (
    DEV_IDS,
    HELDOUT_IDS,
    demo_report,
    run_backend,
)
from agent_fast_decider.decider import Decision, FastLocalDecider
from agent_fast_decider.triage import TriageAgent, load_tickets


def test_splits_cover_all_40():
    assert len(DEV_IDS) == 20
    assert len(HELDOUT_IDS) == 20
    assert DEV_IDS.isdisjoint(HELDOUT_IDS)
    assert {f"T{i:02d}" for i in range(1, 41)} == DEV_IDS | HELDOUT_IDS


def test_run_backend_report_shape():
    report = run_backend(FastLocalDecider(), load_tickets())
    d = report.to_dict()
    assert d["backend"] == "local"
    assert d["n"] == 40
    for key in ("route_accuracy", "urgency_accuracy", "frustration_accuracy"):
        assert 0.0 <= d[key] <= 1.0
    assert d["mean_latency_ms_per_decision"] < 50.0
    assert 0.0 <= d["escalation_rate"] <= 1.0
    assert 0 <= d["confident_decisions"] <= 40
    assert 0.0 <= d["confident_accuracy"] <= 1.0


def test_benchmark_local_beats_chance():
    # A keyword scorer must beat the 25% route chance baseline clearly.
    report = run_backend(FastLocalDecider(), load_tickets())
    assert report.route_acc >= 0.6
    assert report.urgent_acc >= 0.6
    assert report.frustration_acc >= 0.5


def test_demo_report_structure():
    agent = TriageAgent(decider=FastLocalDecider(), llm_fn=lambda *a: {})
    # llm_fn returning {} would crash on escalation; use a stub that
    # returns full decisions only if called — instead use a confident
    # stub decider so no escalation happens.
    from agent_fast_decider.decider import Decision as D

    class Confident:
        backend = "local"

        def decide(self, state, questions):
            return {
                "route": D("choice", "sales", 0.99, {"sales": 0.99}, 0.1, "local"),
                "is_urgent": D("noul", False, 0.98, {"yes": 0.02, "no": 0.98}, 0.1, "local"),
                "frustration": D("score", 0.0, 0.97, {"calm": 0.97}, 0.1, "local"),
            }

    agent = TriageAgent(decider=Confident(), llm_fn=lambda *a: (_ for _ in ()).throw(AssertionError("should not escalate")))
    report = demo_report(agent, load_tickets()[:3])
    assert report["n_tickets"] == 3
    assert report["escalated"] == 0
    assert report["escalation_rate"] == 0.0
    assert len(report["results"]) == 3


def test_cli_demo_runs(capsys):
    from agent_fast_decider.cli import main

    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "agent-fast-decider demo" in out
    assert "T01" in out


def test_cli_benchmark_runs(capsys):
    from agent_fast_decider.cli import main

    assert main(["benchmark"]) == 0
    out = capsys.readouterr().out
    assert "benchmark" in out
    assert "heldout_T21_T40" in out


def test_cli_benchmark_json():
    from agent_fast_decider.cli import main
    import io, contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert main(["benchmark", "--json"]) == 0
    payload = json.loads(buf.getvalue())
    assert payload["backend"] == "local"
    assert payload["all"]["n"] == 40


def test_server_decide_endpoint():
    from agent_fast_decider.server import serve
    import http.client

    # Run the server on an ephemeral port in a thread.
    from agent_fast_decider.decider import load_decider
    from agent_fast_decider.server import make_handler
    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(load_decider("local")))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request(
            "POST",
            "/decide",
            body=json.dumps({
                "state": "I was charged twice, please refund the duplicate charge",
                "questions": {
                    "route": {
                        "type": "choice",
                        "instructions": "Which team?",
                        "options": ["billing", "technical", "returns", "sales"],
                    },
                    "is_urgent": {"type": "noul", "instructions": "Urgent?"},
                },
            }),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        assert resp.status == 200
        payload = json.loads(resp.read())
        assert payload["answers"]["route"]["answer"] == "billing"
        assert "latency_ms" in payload
        conn.close()

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("GET", "/health")
        assert conn.getresponse().status == 200
        conn.close()

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("GET", "/nope")
        assert conn.getresponse().status == 404
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
