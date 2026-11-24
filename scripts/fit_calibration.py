"""Fit the local scorer's calibration constants on the dev set (T01-T20).

Grid-searches:
- ROUTE_TEMPERATURE: softmax temperature for the route distribution
  (maximise dev log-likelihood of the gold route).
- URGENCY_SLOPE / URGENCY_INTERCEPT: logistic calibration for the
  urgency evidence (maximise dev log-likelihood of gold urgency).
- FRUSTRATION_TEMPERATURE: softmax temperature for the frustration
  buckets (maximise dev log-likelihood of the gold bucket).

Prints the fitted values and dev accuracy; writes them into
src/agent_fast_decider/calibration.py. Run again if the lexicons or
the ticket set change. The benchmark reports held-out T21-T40
separately, so this fit cannot flatter the headline numbers.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_fast_decider import decider as D  # noqa: E402
from agent_fast_decider.triage import load_tickets  # noqa: E402


def _loglik_route(temp: float, tickets) -> float:
    total = 0.0
    for t in tickets:
        tokens = D._tokens(t.state)
        scores = {o: D._evidence(tokens, D.ROUTE_LEXICON.get(o, {})) for o in ("billing", "technical", "returns", "sales")}
        dist = D._softmax(scores, temp)
        total += math.log(max(dist[t.gold_route], 1e-9))
    return total


def _loglik_urgent(slope: float, intercept: float, tickets) -> float:
    total = 0.0
    for t in tickets:
        ev = D._evidence(D._tokens(t.state), D.URGENCY_LEXICON)
        p = D._sigmoid(slope * ev + intercept)
        total += math.log(max(p if t.gold_urgent else 1 - p, 1e-9))
    return total


def _loglik_frust(temp: float, tickets) -> float:
    total = 0.0
    for t in tickets:
        tokens = D._tokens(t.state)
        bucket = {
            "0": D._evidence(tokens, D.FRUSTRATION_LEXICON["calm"]),
            "1": D._evidence(tokens, D.FRUSTRATION_LEXICON["frustrated"]),
            "2": D._evidence(tokens, D.FRUSTRATION_LEXICON["very"]),
        }
        dist = D._softmax(bucket, temp)
        total += math.log(max(dist[str(t.gold_frustration)], 1e-9))
    return total


def main() -> int:
    tickets = [t for t in load_tickets() if t.id <= "T20"]
    print(f"dev set: {len(tickets)} tickets (T01-T20)")

    # Temperatures are searched on [0.5, 3.0]: below 0.5 the softmax
    # collapses to a delta on a perfectly separated dev set and the
    # confidences stop meaning anything.
    best_temp = max((x / 20 for x in range(10, 61)), key=lambda v: _loglik_route(v, tickets))
    print(f"ROUTE_TEMPERATURE = {best_temp}  (dev loglik {_loglik_route(best_temp, tickets):.2f})")

    best_u = max(
        ((s / 10, i / 10) for s in range(2, 41) for i in range(-50, 11)),
        key=lambda p: _loglik_urgent(p[0], p[1], tickets),
    )
    print(f"URGENCY_SLOPE = {best_u[0]}  URGENCY_INTERCEPT = {best_u[1]}"
          f"  (dev loglik {_loglik_urgent(*best_u, tickets):.2f})")

    best_ft = max((x / 20 for x in range(10, 61)), key=lambda v: _loglik_frust(v, tickets))
    print(f"FRUSTRATION_TEMPERATURE = {best_ft}  (dev loglik {_loglik_frust(best_ft, tickets):.2f})")

    # Dev accuracy at the fitted values, for the record. (Set the
    # constants on the decider module itself: it imported them by value
    # at load time, so mutating calibration.py's module would not apply.)
    D.ROUTE_TEMPERATURE = best_temp
    D.URGENCY_SLOPE, D.URGENCY_INTERCEPT = best_u
    D.FRUSTRATION_TEMPERATURE = best_ft
    local = D.FastLocalDecider()
    from agent_fast_decider.triage import TRIAGE_QUESTIONS

    r = u = f = 0
    for t in tickets:
        d = local.decide(t.state, TRIAGE_QUESTIONS)
        r += d["route"].answer == t.gold_route
        u += d["is_urgent"].answer == t.gold_urgent
        f += round(d["frustration"].answer) == t.gold_frustration
    print(f"dev accuracy at fit: route {r}/{len(tickets)}  "
          f"urgent {u}/{len(tickets)}  frustration {f}/{len(tickets)}")

    cal_path = Path(__file__).resolve().parents[1] / "src" / "agent_fast_decider" / "calibration.py"
    cal_path.write_text(
        f'"""Calibration constants for :class:`FastLocalDecider`.\n\n'
        f"Fit on the dev half of the benchmark set (tickets T01-T20) by\n"
        f"grid search maximising log-likelihood of the gold labels\n"
        f"(see ``scripts/fit_calibration.py``). Dev accuracy at fit:\n"
        f"route {r}/{len(tickets)}, urgent {u}/{len(tickets)}, "
        f"frustration {f}/{len(tickets)}. The benchmark reports held-out\n"
        f"accuracy on T21-T40 separately, so the fit cannot flatter the\n"
        f"headline numbers. Re-run the fit script if the lexicons or the\n"
        f"ticket set change.\n"
        f'"""\n\n'
        f"ROUTE_TEMPERATURE = {best_temp}\n\n"
        f"URGENCY_SLOPE = {best_u[0]}\n"
        f"URGENCY_INTERCEPT = {best_u[1]}\n\n"
        f"FRUSTRATION_TEMPERATURE = {best_ft}\n"
    )
    print(f"wrote {cal_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
