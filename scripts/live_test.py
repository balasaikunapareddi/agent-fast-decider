"""Live test: run the real loop once against live Gemini.

- Offline benchmark of the local decider (no LLM).
- Full triage loop over a 10-ticket subset with real Gemini handling
  escalations (System 2).
- Gemini baseline over the same 10 tickets (decides everything),
  so the speed/accuracy trade-off is measured, not claimed.

Run via the harness (injects GEMINI_API_KEY):
    python3 ~/workspace/custom-connectors/custom.gemini/bin/run_live.py \\
        ~/workspace/agent-fast-decider .venv/bin/python scripts/live_test.py

Saves a JSON report to .runs/live/attempt-<n>/report.json.
Exits non-zero if the live run fails outright.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_fast_decider.benchmark import (  # noqa: E402
    demo_report,
    run_backend,
    run_llm_baseline,
)
from agent_fast_decider.decider import FastLocalDecider  # noqa: E402
from agent_fast_decider.triage import TriageAgent, load_tickets  # noqa: E402

SUBSET = ["T02", "T05", "T06", "T10", "T14", "T17", "T21", "T23", "T26", "T29"]


def main() -> int:
    started = time.time()
    tickets = load_tickets()
    subset = [t for t in tickets if t.id in SUBSET]
    print(f"live test: {len(subset)}-ticket subset {SUBSET}")

    # 1. Offline benchmark (local decider, no LLM).
    local = run_backend(FastLocalDecider(), tickets).to_dict()
    print(
        "local decider (all 40): route "
        f"{local['route_accuracy']:.2f} urgent {local['urgency_accuracy']:.2f} "
        f"frust {local['frustration_accuracy']:.2f} "
        f"lat {local['mean_latency_ms_per_decision']:.2f} ms/decision "
        f"esc {local['escalation_rate']:.0%}"
    )

    # 2. Full loop with real Gemini escalations.
    agent = TriageAgent(decider=FastLocalDecider())
    report = demo_report(agent, subset)
    print(
        f"full loop: escalated {report['escalated']}/{report['n_tickets']} "
        f"mean latency {report['mean_latency_ms']:.0f} ms/ticket"
    )
    for r in report["results"]:
        print(
            f"  {r['ticket']} route={r['route']} urgent={r['urgent']} "
            f"frust={r['frustration']} escalated={r['escalated']}"
        )
        if r["reply_draft"]:
            print(f"    draft: {r['reply_draft'][:120]}")

    # 3. Gemini baseline on the same subset.
    baseline = run_llm_baseline(subset).to_dict()
    print(
        "gemini baseline (subset): route "
        f"{baseline['route_accuracy']:.2f} urgent {baseline['urgency_accuracy']:.2f} "
        f"frust {baseline['frustration_accuracy']:.2f} "
        f"lat {baseline['mean_latency_ms_per_decision']:.0f} ms/ticket"
    )

    out = {
        "local_all40": local,
        "full_loop_subset": {
            k: v for k, v in report.items() if k != "results"
        },
        "full_loop_results": report["results"],
        "gemini_baseline_subset": baseline,
        "seconds": round(time.time() - started, 1),
    }
    run_dir = ROOT / ".runs" / "live"
    attempt = 1
    while (run_dir / f"attempt-{attempt}").exists():
        attempt += 1
    dest = run_dir / f"attempt-{attempt}"
    dest.mkdir(parents=True)
    (dest / "report.json").write_text(json.dumps(out, indent=2))
    print(f"saved {dest / 'report.json'}")

    # Sanity: the loop must have produced a decision for every ticket,
    # and at least one escalation must have reached the real LLM.
    assert len(report["results"]) == len(subset)
    assert report["escalated"] >= 0
    llm_errors = [r for r in report["results"] if r.get("llm_error")]
    assert not llm_errors, f"LLM escalation failed: {llm_errors}"
    assert report["escalated"] > 0, "expected at least one real escalation"
    print("[live] LIVE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
