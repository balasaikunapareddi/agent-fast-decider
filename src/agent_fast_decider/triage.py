"""Triage agent: System-1 fast decisions, System-2 escalation.

Every ticket is first decided by the fast decider (choice route,
noul urgency, score frustration). Any decision whose confidence is
below :data:`CONFIDENCE_THRESHOLD` escalates: the big LLM re-decides
that question and drafts the reply. High-confidence tickets are
acted on immediately with no LLM call at all.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .decider import Decision, Decider, FastLocalDecider
from .prompts import SYSTEM_PROMPT

# The Strands Decider calibration rule, applied to every fast decision:
# answers at confidence >= 0.9 are trusted; below that, confirm or ask.
CONFIDENCE_THRESHOLD = 0.9

ROUTES = ["billing", "technical", "returns", "sales"]
FRUSTRATION_LABELS = ["calm", "frustrated", "very frustrated"]

TRIAGE_QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "options": ROUTES,
    },
    "is_urgent": {
        "type": "noul",
        "instructions": "Does this convey urgency?",
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated is the writer?",
        "options": FRUSTRATION_LABELS,
    },
}


@dataclass
class Ticket:
    id: str
    subject: str
    body: str
    gold_route: str | None = None
    gold_urgent: bool | None = None
    gold_frustration: int | None = None

    @property
    def state(self) -> str:
        return f"{self.subject}\n{self.body}"


@dataclass
class TriageResult:
    ticket: Ticket
    decisions: dict[str, Decision]
    escalated: list[str] = field(default_factory=list)
    llm_decisions: dict[str, dict] = field(default_factory=dict)
    reply_draft: str | None = None
    llm_error: str | None = None
    latency_ms: float = 0.0
    backend: str = "local"

    @property
    def route(self) -> str:
        return self.llm_decisions.get("route", {}).get(
            "value", self.decisions["route"].answer
        )

    @property
    def urgent(self) -> bool:
        return self.llm_decisions.get("is_urgent", {}).get(
            "value", self.decisions["is_urgent"].answer
        )

    @property
    def frustration(self) -> float:
        return self.llm_decisions.get("frustration", {}).get(
            "value", self.decisions["frustration"].answer
        )

    def to_dict(self) -> dict:
        return {
            "ticket": self.ticket.id,
            "subject": self.ticket.subject,
            "route": self.route,
            "urgent": self.urgent,
            "frustration": self.frustration,
            "escalated": self.escalated,
            "fast_decisions": {
                k: v.to_dict() for k, v in self.decisions.items()
            },
            "llm_decisions": self.llm_decisions,
            "reply_draft": self.reply_draft,
            "llm_error": self.llm_error,
            "latency_ms": round(self.latency_ms, 2),
            "backend": self.backend,
        }


def load_tickets(path: Path | None = None) -> list[Ticket]:
    """Load the labeled benchmark set (default: the bundled 40)."""
    path = path or Path(__file__).parent / "data" / "tickets.yaml"
    raw = yaml.safe_load(path.read_text())
    return [Ticket(**row) for row in raw]


def _call_with_retry(fn, *, attempts: int = 3, wait_s: float = 30.0):
    """Retry an LLM call on rate limits (Gemini free tier: 15 req/min).

    Retries only when the error looks like a 429/rate-limit; anything
    else is raised immediately. Sleeps ``wait_s`` between attempts.
    """
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - re-raised below if not 429
            msg = str(exc).lower()
            if "429" not in msg and "rate limit" not in msg and "quota" not in msg:
                raise
            last = exc
            if i < attempts - 1:
                time.sleep(wait_s)
    raise last  # type: ignore[misc]


def _llm_call(model: str, ticket: Ticket, questions: list[str]) -> dict:
    """One System-2 call for the escalated questions. Returns parsed JSON."""
    import litellm

    # Keep LiteLLM's "Give Feedback" banner out of demo/benchmark output.
    litellm.suppress_debug_info = True
    litellm.set_verbose = False

    user = (
        f"Ticket {ticket.id}: {ticket.subject}\n{ticket.body}\n\n"
        f"The fast layer was unsure about: {', '.join(questions)}. "
        "Re-decide those, and draft the reply."
    )

    def _do():
        resp = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            timeout=120,
        )
        return json.loads(resp.choices[0].message.content)

    return _call_with_retry(_do)


class TriageAgent:
    """The triage agent: fast decider first, big LLM only on escalation."""

    def __init__(
        self,
        decider: Decider | None = None,
        model: str | None = None,
        threshold: float = CONFIDENCE_THRESHOLD,
        llm_fn=None,
    ):
        self.decider = decider or FastLocalDecider()
        self.model = model or os.environ.get(
            "LLM_MODEL", "gemini/gemini-3.5-flash-lite"
        )
        self.threshold = threshold
        # Injectable for tests: llm_fn(model, ticket, questions) -> dict.
        self._llm_fn = llm_fn or _llm_call

    @property
    def backend(self) -> str:
        return getattr(self.decider, "backend", "local")

    def triage(self, ticket: Ticket) -> TriageResult:
        started = time.perf_counter()
        decisions = self.decider.decide(ticket.state, TRIAGE_QUESTIONS)
        low = [
            name
            for name, d in decisions.items()
            if d.confidence < self.threshold
        ]
        result = TriageResult(
            ticket=ticket,
            decisions=decisions,
            escalated=low,
            backend=self.backend,
        )
        if low:
            try:
                parsed = self._llm_fn(self.model, ticket, low)
            except Exception as exc:  # LLM unavailable: keep fast decisions
                result.llm_error = f"{type(exc).__name__}: {exc}"
                result.latency_ms = (time.perf_counter() - started) * 1000.0
                return result
            if "route" in low:
                result.llm_decisions["route"] = {
                    "value": parsed["route"],
                    "reason": parsed.get("route_reason", ""),
                }
            if "is_urgent" in low:
                result.llm_decisions["is_urgent"] = {
                    "value": bool(parsed["urgent"]),
                    "reason": parsed.get("urgent_reason", ""),
                }
            if "frustration" in low:
                result.llm_decisions["frustration"] = {
                    "value": int(parsed["frustration"]),
                    "reason": parsed.get("frustration_reason", ""),
                }
            result.reply_draft = parsed.get("reply_draft")
        result.latency_ms = (time.perf_counter() - started) * 1000.0
        return result

    def triage_all(self, tickets: list[Ticket]) -> list[TriageResult]:
        return [self.triage(t) for t in tickets]
