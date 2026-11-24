"""CLI: demo, benchmark, serve.

- ``demo``      triage the built-in 40 tickets with the local decider
                (or --backend real for the real model) and print a report.
- ``benchmark`` offline comparison table per backend; --live adds the
                Gemini baseline (decides everything with the big LLM).
- ``serve``     tiny stdlib HTTP server with a /decide endpoint shaped
                like strands-decider's /v1/systemone.
"""

from __future__ import annotations

import argparse
import json
import sys

from .benchmark import benchmark, demo_report
from .decider import load_decider
from .triage import TriageAgent, load_tickets


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2))


def cmd_demo(args) -> int:
    decider = load_decider(args.backend)
    agent = TriageAgent(decider=decider)
    report = demo_report(agent)
    if args.json:
        _print_json(report)
        return 0
    print(f"agent-fast-decider demo — backend: {report['backend']}")
    print(f"tickets: {report['n_tickets']}  escalated: {report['escalated']} "
          f"({report['escalation_rate']:.0%})  "
          f"mean latency: {report['mean_latency_ms']} ms/ticket")
    print()
    for r in report["results"]:
        fast = r["fast_decisions"]
        route = fast["route"]
        flags = f" ESCALATED[{','.join(r['escalated'])}]" if r["escalated"] else ""
        print(
            f"{r['ticket']} [{r['route']:<10} conf {route['confidence']:.2f}] "
            f"urgent={str(r['urgent']):<5} frust={r['frustration']:<4}{flags}"
        )
        if r["reply_draft"]:
            print(f"    draft: {r['reply_draft'][:110]}")
        elif r["llm_error"]:
            print(f"    (escalated, LLM unavailable: {r['llm_error'][:80]})")
    print()
    print("Escalated tickets went to the big LLM (System 2); the rest were")
    print("decided by the fast layer alone — no LLM call, no waiting.")
    return 0


def cmd_benchmark(args) -> int:
    decider = load_decider(args.backend)
    table = benchmark(decider, live_baseline=args.live)
    if args.json:
        _print_json(table)
        return 0
    print(f"benchmark — backend: {table['backend']} (n=40 tickets)")
    print(f"{'split':<16}{'route':>7}{'urgent':>8}{'frust':>7}"
          f"{'lat_ms':>9}{'esc':>6}{'conf_acc':>10}")
    for key in ("all", "dev_T01_T20", "heldout_T21_T40"):
        r = table[key]
        print(
            f"{key:<16}{r['route_accuracy']:>7.2f}{r['urgency_accuracy']:>8.2f}"
            f"{r['frustration_accuracy']:>7.2f}"
            f"{r['mean_latency_ms_per_decision']:>9.2f}"
            f"{r['escalation_rate']:>6.2f}"
            f"{r['confident_accuracy']:>10.2f}  (n_conf={r['confident_decisions']})"
        )
    if "llm_baseline_all" in table:
        r = table["llm_baseline_all"]
        print(
            f"{'llm-baseline':<16}{r['route_accuracy']:>7.2f}"
            f"{r['urgency_accuracy']:>8.2f}{r['frustration_accuracy']:>7.2f}"
            f"{r['mean_latency_ms_per_decision']:>9.2f}{'—':>6}{'—':>10}"
        )
    print()
    print("conf_acc = accuracy among route decisions the decider trusted")
    print("(confidence >= 0.9). Held-out T21-T40 is the honest number:")
    print("the local scorer's calibration was fit on T01-T20 only.")
    return 0


def cmd_serve(args) -> int:
    from .server import serve

    serve(host=args.host, port=args.port, backend=args.backend)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-fast-decider")
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="triage the built-in ticket set")
    p_demo.add_argument("--backend", default="local", choices=["local", "real"])
    p_demo.add_argument("--json", action="store_true")
    p_demo.set_defaults(func=cmd_demo)

    p_bench = sub.add_parser("benchmark", help="accuracy/latency table")
    p_bench.add_argument("--backend", default="local", choices=["local", "real"])
    p_bench.add_argument("--live", action="store_true",
                         help="also run the Gemini baseline (uses the API)")
    p_bench.add_argument("--json", action="store_true")
    p_bench.set_defaults(func=cmd_benchmark)

    p_serve = sub.add_parser("serve", help="HTTP /decide server")
    p_serve.add_argument("--backend", default="local", choices=["local", "real"])
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
