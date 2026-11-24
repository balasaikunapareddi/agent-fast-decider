"""Offline tests for the local decider: question types, calibration shape."""

import math

import pytest

from agent_fast_decider.decider import FastLocalDecider, load_decider
from agent_fast_decider.triage import TRIAGE_QUESTIONS, load_tickets

DECIDER = FastLocalDecider()


def test_load_decider_local():
    d = load_decider("local")
    assert isinstance(d, FastLocalDecider)
    assert d.backend == "local"


def test_load_decider_unknown_backend():
    with pytest.raises(ValueError):
        load_decider("nope")


def test_choice_shape():
    out = DECIDER.decide(
        "I was charged twice for my order",
        {"route": TRIAGE_QUESTIONS["route"]},
    )
    d = out["route"]
    assert d.kind == "choice"
    assert d.answer in ("billing", "technical", "returns", "sales")
    assert 0.0 <= d.confidence <= 1.0
    assert abs(sum(d.distribution.values()) - 1.0) < 1e-6
    assert d.backend == "local"
    assert d.latency_ms >= 0.0


def test_choice_billing_strong_evidence():
    out = DECIDER.decide(
        "I was charged twice, please refund the duplicate charge to my bank",
        {"route": TRIAGE_QUESTIONS["route"]},
    )
    assert out["route"].answer == "billing"
    assert out["route"].confidence > 0.5


def test_choice_sales_strong_evidence():
    out = DECIDER.decide(
        "Do you carry this in walnut, and is there a bulk discount for 20?",
        {"route": TRIAGE_QUESTIONS["route"]},
    )
    assert out["route"].answer == "sales"


def test_choice_returns_strong_evidence():
    out = DECIDER.decide(
        "The lamp arrived damaged with a cracked base, I want a replacement",
        {"route": TRIAGE_QUESTIONS["route"]},
    )
    assert out["route"].answer == "returns"


def test_choice_technical_strong_evidence():
    out = DECIDER.decide(
        "The app crashes on login every time, the website checkout errors out",
        {"route": TRIAGE_QUESTIONS["route"]},
    )
    assert out["route"].answer == "technical"


def test_noul_urgent_yes():
    out = DECIDER.decide(
        "URGENT: our system is down, we cannot take orders, help immediately",
        {"is_urgent": TRIAGE_QUESTIONS["is_urgent"]},
    )
    d = out["is_urgent"]
    assert d.kind == "noul"
    assert d.answer is True
    assert d.confidence > 0.5
    assert abs(d.distribution["yes"] + d.distribution["no"] - 1.0) < 1e-6


def test_noul_urgent_no():
    out = DECIDER.decide(
        "Do you carry the same bookshelf in walnut? What is the price? Thanks!",
        {"is_urgent": TRIAGE_QUESTIONS["is_urgent"]},
    )
    assert out["is_urgent"].answer is False


def test_score_frustration_calm():
    out = DECIDER.decide(
        "Hello, I love the bookshelf. Do you carry it in walnut? Thanks!",
        {"frustration": TRIAGE_QUESTIONS["frustration"]},
    )
    d = out["frustration"]
    assert d.kind == "score"
    assert 0.0 <= d.answer <= 2.0
    assert d.answer < 1.0  # calm ticket should score low
    assert abs(sum(d.distribution.values()) - 1.0) < 1e-6


def test_score_frustration_very():
    out = DECIDER.decide(
        "This is the second damaged delivery in a row and I have lost all "
        "patience. This is unacceptable, I am very upset.",
        {"frustration": TRIAGE_QUESTIONS["frustration"]},
    )
    assert out["frustration"].answer >= 1.0


def test_all_three_questions_one_call():
    out = DECIDER.decide(
        "Help! My payouts have been failing for 3 days!",
        TRIAGE_QUESTIONS,
    )
    assert set(out) == {"route", "is_urgent", "frustration"}
    assert out["route"].answer == "billing"
    assert out["is_urgent"].answer is True


def test_unknown_question_type_raises():
    with pytest.raises(ValueError):
        DECIDER.decide("hello", {"x": {"type": "telepathy"}})


def test_empty_state_does_not_crash():
    out = DECIDER.decide("", TRIAGE_QUESTIONS)
    assert set(out) == {"route", "is_urgent", "frustration"}
    for d in out.values():
        assert 0.0 <= d.confidence <= 1.0
        assert math.isfinite(d.confidence)


def test_latency_is_milliseconds_fast():
    tickets = load_tickets()
    import time

    t0 = time.perf_counter()
    for t in tickets:
        DECIDER.decide(t.state, TRIAGE_QUESTIONS)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms / len(tickets) < 50.0  # local scorer: ms per ticket


def test_deterministic_same_input_same_output():
    state = "I was charged twice, refund please, this is urgent"
    a = DECIDER.decide(state, TRIAGE_QUESTIONS)
    b = DECIDER.decide(state, TRIAGE_QUESTIONS)
    for k in a:
        assert a[k].answer == b[k].answer
        assert a[k].confidence == b[k].confidence


def test_calibration_constants_sane():
    from agent_fast_decider import calibration as C

    assert 0.1 <= C.ROUTE_TEMPERATURE <= 5.0
    assert 0.1 <= C.FRUSTRATION_TEMPERATURE <= 5.0
    assert C.URGENCY_SLOPE > 0
    assert -5.0 <= C.URGENCY_INTERCEPT <= 5.0
