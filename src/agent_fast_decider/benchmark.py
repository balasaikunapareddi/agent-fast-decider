"""Benchmark: measure the System-1 layer against gold labels.

Reports, per backend: route accuracy, urgency accuracy, frustration
accuracy (nearest integer), mean latency per decision, escalation
rate, and calibration — accuracy among decisions the decider was
confident about (>= 0.9). The dev/holdout split is T01-T20 (dev, the
calibration fit set) vs T21-T40 (held out). An optional live Gemini
baseline decides everything with the big LLM so the speed/accuracy
trade-off is a measured table, not a claim.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import dataclass

from .decider import Decider, Decision
from .prompts import SYSTEM_PROMPT
from .triage import (
    CONFIDENCE_THRESHOLD,
    TRIAGE_QUESTIONS,
    Ticket,
    TriageAgent,
    _call_with_retry,
    load_tickets,
)

DEV_IDS = {f"T{i:02d}" for i in range(1, 21)}
HELDOUT_IDS = {f"T{i:02d}" for i in range(21, 41)}


@dataclass
class BackendReport:
    backend: str
    n: int
    route_acc: float
    urgent_acc: float
    frustration_acc: float
    mean_latency_ms: float
    escalation_rate: float
    confident_n: int
    confident_acc: float

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "n": self.n,
            "route_accuracy": round(self.route_acc, 3),
            "urgency_accuracy": round(self.urgent_acc, 3),
            "frustration_accuracy": round(self.frustration_acc, 3),
            "mean_latency_ms_per_decision": round(self.mean_latency_ms, 2),
            "escalation_rate": round(self.escalation_rate, 3),
            "confident_decisions": self.confident_n,
            "confident_accuracy": round(self.confident_acc, 3),
        }


def _score_decisions(
    tickets: list[Ticket], results: list[dict[str, Decision]]
) -> BackendReport:
    route_ok = urgent_ok = frust_ok = 0
    latencies: list[float] = []
    escalated = 0
    confident_ok = confident_n = 0
    backend = "local"
    for ticket, decs in zip(tickets, results):
        backend = decs["route"].backend
        latencies.extend(d.latency_ms for d in decs.values())
        if any(d.confidence < CONFIDENCE_THRESHOLD for d in decs.values()):
            escalated += 1
        if ticket.gold_route is not None and decs["route"].answer == ticket.gold_route:
            route_ok += 1
        if ticket.gold_urgent is not None and decs["is_urgent"].answer == ticket.gold_urgent:
            urgent_ok += 1
        if ticket.gold_frustration is not None and round(decs["frustration"].answer) == ticket.gold_frustration:
            frust_ok += 1
        # Calibration: among decisions the decider trusted (>= 0.9),
        # how often was it right? (route question only, for clarity.)
        if decs["route"].confidence >= CONFIDENCE_THRESHOLD:
            confident_n += 1
            if ticket.gold_route is not None and decs["route"].answer == ticket.gold_route:
                confident_ok += 1
    n = len(tickets) or 1
    return BackendReport(
        backend=backend,
        n=len(tickets),
        route_acc=route_ok / n,
        urgent_acc=urgent_ok / n,
        frustration_acc=frust_ok / n,
        mean_latency_ms=statistics.mean(latencies) if latencies else 0.0,
        escalation_rate=escalated / n,
        confident_n=confident_n,
        confident_acc=(confident_ok / confident_n) if confident_n else 0.0,
    )


def run_backend(decider: Decider, tickets: list[Ticket]) -> BackendReport:
    """Run one decider backend over the tickets, fast decisions only."""
    results = [decider.decide(t.state, TRIAGE_QUESTIONS) for t in tickets]
    return _score_decisions(tickets, results)


def run_llm_baseline(
    tickets: list[Ticket], model: str | None = None
) -> BackendReport:
    """Big-LLM baseline: Gemini decides every question on every ticket.

    One LLM call per ticket (all three questions at once). Slow and
    expensive relative to the fast layer — that is the point of the
    comparison. Returns a report in the same shape, with
    ``backend="llm-baseline"`` and escalation_rate 0 (it never
    escalates; it IS the escalation target).
    """
    import litellm

    litellm.suppress_debug_info = True
    litellm.set_verbose = False
    model = model or os.environ.get("LLM_MODEL", "gemini/gemini-3.5-flash-lite")
    route_ok = urgent_ok = frust_ok = 0
    latencies: list[float] = []
    for ticket in tickets:
        t0 = time.perf_counter()

        def _do(ticket=ticket):
            resp = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Ticket {ticket.id}: {ticket.subject}\n{ticket.body}\n\n"
                            "Decide route, urgency, and frustration (0-2). "
                            "Reply JSON only."
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                timeout=120,
            )
            return json.loads(resp.choices[0].message.content)

        parsed = _call_with_retry(_do)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        if parsed.get("route") == ticket.gold_route:
            route_ok += 1
        if bool(parsed.get("urgent")) == ticket.gold_urgent:
            urgent_ok += 1
        if int(parsed.get("frustration", -1)) == ticket.gold_frustration:
            frust_ok += 1
    n = len(tickets) or 1
    return BackendReport(
        backend="llm-baseline",
        n=len(tickets),
        route_acc=route_ok / n,
        urgent_acc=urgent_ok / n,
        frustration_acc=frust_ok / n,
        mean_latency_ms=statistics.mean(latencies) if latencies else 0.0,
        escalation_rate=0.0,
        confident_n=0,
        confident_acc=0.0,
    )


def benchmark(decider: Decider, live_baseline: bool = False) -> dict:
    """Full benchmark table: dev, held-out, and all tickets."""
    tickets = load_tickets()
    dev = [t for t in tickets if t.id in DEV_IDS]
    held = [t for t in tickets if t.id in HELDOUT_IDS]
    out: dict = {
        "backend": getattr(decider, "backend", "local"),
        "all": run_backend(decider, tickets).to_dict(),
        "dev_T01_T20": run_backend(decider, dev).to_dict(),
        "heldout_T21_T40": run_backend(decider, held).to_dict(),
    }
    if live_baseline:
        out["llm_baseline_all"] = run_llm_baseline(tickets).to_dict()
    return out


def demo_report(agent: TriageAgent, tickets: list[Ticket] | None = None) -> dict:
    """Run the triage demo over the ticket set; structured report."""
    tickets = tickets or load_tickets()
    results = agent.triage_all(tickets)
    escalated = sum(1 for r in results if r.escalated)
    return {
        "backend": agent.backend,
        "model": agent.model,
        "threshold": agent.threshold,
        "n_tickets": len(results),
        "escalated": escalated,
        "escalation_rate": round(escalated / max(1, len(results)), 3),
        "mean_latency_ms": round(
            statistics.mean([r.latency_ms for r in results]), 2
        )
        if results
        else 0.0,
        "results": [r.to_dict() for r in results],
    }
