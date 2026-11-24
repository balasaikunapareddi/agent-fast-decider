"""Offline tests for triage routing, escalation logic, and the agent loop."""

import pytest

from agent_fast_decider.decider import Decision, FastLocalDecider
from agent_fast_decider.triage import (
    CONFIDENCE_THRESHOLD,
    Ticket,
    TriageAgent,
    load_tickets,
)


def _stub_llm(model, ticket, questions):
    return {
        "route": "billing",
        "route_reason": "stub: mentions a charge",
        "urgent": True,
        "urgent_reason": "stub: says urgent",
        "frustration": 2,
        "frustration_reason": "stub: very upset",
        "reply_draft": "Stub draft: sorry about the charge, we are looking into it.",
    }


def _always_confident_decider():
    class Confident:
        backend = "local"
        name = "confident-stub"

        def decide(self, state, questions):
            return {
                "route": Decision("choice", "sales", 0.99, {"sales": 0.99}, 0.1, "local"),
                "is_urgent": Decision("noul", False, 0.98, {"yes": 0.02, "no": 0.98}, 0.1, "local"),
                "frustration": Decision("score", 0.0, 0.97, {"calm": 0.97}, 0.1, "local"),
            }

    return Confident()


def _always_unsure_decider():
    class Unsure:
        backend = "local"
        name = "unsure-stub"

        def decide(self, state, questions):
            return {
                "route": Decision("choice", "billing", 0.4, {"billing": 0.4}, 0.1, "local"),
                "is_urgent": Decision("noul", True, 0.45, {"yes": 0.45, "no": 0.55}, 0.1, "local"),
                "frustration": Decision("score", 1.0, 0.5, {"frustrated": 0.5}, 0.1, "local"),
            }

    return Unsure()


def test_threshold_is_0_9():
    assert CONFIDENCE_THRESHOLD == 0.9


def test_load_tickets_count_and_labels():
    tickets = load_tickets()
    assert len(tickets) == 40
    routes = {t.gold_route for t in tickets}
    assert routes == {"billing", "technical", "returns", "sales"}
    assert all(t.gold_urgent in (True, False) for t in tickets)
    assert all(t.gold_frustration in (0, 1, 2) for t in tickets)
    assert len({t.id for t in tickets}) == 40


def test_no_escalation_when_confident():
    agent = TriageAgent(decider=_always_confident_decider(), llm_fn=_stub_llm)
    result = agent.triage(Ticket("X1", "Hi", "Do you carry this in walnut?"))
    assert result.escalated == []
    assert result.reply_draft is None
    assert result.route == "sales"
    assert result.urgent is False


def test_escalation_calls_llm_once_with_low_questions():
    calls = []

    def counting_llm(model, ticket, questions):
        calls.append((ticket.id, sorted(questions)))
        return _stub_llm(model, ticket, questions)

    agent = TriageAgent(decider=_always_unsure_decider(), llm_fn=counting_llm)
    result = agent.triage(Ticket("X2", "Help", "charged twice, urgent, furious"))
    assert len(calls) == 1
    assert calls[0][0] == "X2"
    assert calls[0][1] == ["frustration", "is_urgent", "route"]
    assert result.escalated == ["route", "is_urgent", "frustration"] or set(
        result.escalated
    ) == {"route", "is_urgent", "frustration"}
    assert result.reply_draft.startswith("Stub draft")
    # LLM decisions override the fast ones.
    assert result.route == "billing"
    assert result.urgent is True
    assert result.frustration == 2


def test_partial_escalation_only_low_confidence():
    class Mixed:
        backend = "local"

        def decide(self, state, questions):
            return {
                "route": Decision("choice", "returns", 0.95, {"returns": 0.95}, 0.1, "local"),
                "is_urgent": Decision("noul", False, 0.6, {"yes": 0.4, "no": 0.6}, 0.1, "local"),
                "frustration": Decision("score", 0.0, 0.99, {"calm": 0.99}, 0.1, "local"),
            }

    seen = []

    def counting_llm(model, ticket, questions):
        seen.append(sorted(questions))
        return _stub_llm(model, ticket, questions)

    agent = TriageAgent(decider=Mixed(), llm_fn=counting_llm)
    result = agent.triage(Ticket("X3", "Hi", "return please, not urgent"))
    assert seen == [["is_urgent"]]
    assert result.route == "returns"  # fast decision kept
    assert result.urgent is True  # LLM overrode the unsure one


def test_real_local_decider_end_to_end_demo():
    agent = TriageAgent(decider=FastLocalDecider(), llm_fn=_stub_llm)
    tickets = load_tickets()[:5]
    results = agent.triage_all(tickets)
    assert len(results) == 5
    for r in results:
        assert r.route in ("billing", "technical", "returns", "sales")
        assert isinstance(r.urgent, bool)
        assert 0.0 <= r.frustration <= 2.0
        assert r.latency_ms >= 0.0


def test_triage_result_to_dict_shape():
    agent = TriageAgent(decider=_always_confident_decider(), llm_fn=_stub_llm)
    d = agent.triage(Ticket("X4", "Hi", "question")).to_dict()
    assert d["ticket"] == "X4"
    assert set(d["fast_decisions"]) == {"route", "is_urgent", "frustration"}
    assert d["backend"] == "local"
    assert "latency_ms" in d


def test_llm_failure_keeps_fast_decisions():
    def failing_llm(model, ticket, questions):
        raise RuntimeError("no API key configured")

    agent = TriageAgent(decider=_always_unsure_decider(), llm_fn=failing_llm)
    result = agent.triage(Ticket("X5", "Help", "charged twice"))
    assert result.escalated != []
    assert result.llm_error is not None
    assert "no API key" in result.llm_error
    assert result.route == "billing"  # fast decision still stands
    assert result.reply_draft is None


def test_agent_backend_label():
    agent = TriageAgent(decider=FastLocalDecider())
    assert agent.backend == "local"
