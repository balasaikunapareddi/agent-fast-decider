# agent-fast-decider

**Support-ticket triage with a fast System-1 decision layer, in the spirit of
Strands Decider v2: a small decision model routes, flags urgency, and scores
frustration in milliseconds with calibrated confidence — and only
low-confidence tickets escalate to the big LLM.**

Inspired by [Strands Decider](https://github.com/strands-labs/strands-decider)
(Apache 2.0).

## The pain

Every routing decision inside an agent costs a full LLM call — seconds of
latency and real money per ticket — while most routing decisions are easy.
"Charged twice, refund please" does not need a frontier model to know it
belongs to billing. But the hard ones do need one, and the agent has no
principled way to tell the two apart.

The fix is the System-1 / System-2 split. A small decision model answers
the easy questions in one forward pass — no generation — with a calibrated
confidence on every answer. Strands Decider's calibration rule, which this
project adopts verbatim:

> **Answers at confidence ≥ 0.9 are trusted. Below 0.9, confirm or ask.**

So the agent is: fast layer decides everything; anything under 0.9 goes to
the big LLM, which re-decides and drafts the reply. The LLM never sees the
easy tickets at all.

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -e ".[test]"
.venv/bin/python -m agent_fast_decider demo
```

No API key needed for the demo: the fast layer is local, and escalated
tickets are reported as "LLM unavailable" instead of crashing. With a
Gemini key (`GEMINI_API_KEY`, or run via the harness) the escalations
actually go to the big LLM and come back with reply drafts.

```bash
# accuracy / latency table for the fast layer (offline, no LLM)
.venv/bin/python -m agent_fast_decider benchmark

# same table plus the big-LLM baseline (decides everything; uses the API)
.venv/bin/python -m agent_fast_decider benchmark --live

# tiny HTTP server, /decide shaped like strands-decider's /v1/systemone
.venv/bin/python -m agent_fast_decider serve --port 8000
curl -s localhost:8000/decide -H 'content-type: application/json' \
  -d '{"state": "I was charged twice, please refund it", "questions": {
    "route": {"type": "choice", "instructions": "Which team?",
              "options": ["billing", "technical", "returns", "sales"]},
    "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}}'
```

## How it works

```
ticket ──▶ fast decider ──▶ choice: billing|technical|returns|sales
             (choice /         noul:   is it urgent?
              noul /           score:  frustration 0–2
              score)                │
                        confidence ≥ 0.9 → act immediately
                        confidence < 0.9 → escalate to the big LLM
                                           (re-decide + draft reply)
```

- `src/agent_fast_decider/decider.py` — the `Decider` interface and two
  backends. Both answer the same three question types and return the same
  `{answer, confidence, distribution}` shape.
- `src/agent_fast_decider/triage.py` — the visible `TriageAgent` class:
  fast decisions first, escalation at the 0.9 threshold, reply drafts.
- `src/agent_fast_decider/prompts.py` — the System-2 system prompt, in
  its own file, never inline.
- `src/agent_fast_decider/benchmark.py` — the benchmark math and the
  Gemini baseline.
- `src/agent_fast_decider/data/tickets.yaml` — 40 hand-labeled synthetic
  tickets (fictional retailer Prairie Goods Co.). T01–T20 are the dev
  set the local scorer's calibration was fit on; T21–T40 are held out.

## The two backends

| Backend | What it is | When to use it |
|---|---|---|
| `local` (default) | `FastLocalDecider`: deterministic keyword-evidence scorer + a calibration fit on the dev set. No download, ~0.04 ms per decision. | Demos, tests, CI. Always works. |
| `real` | `RealDecider`: wraps the actual `strands-decider` 0.4.0 package and the `StrandsAgents/strands-decider-E2B-gemma4-v2-2611` model (`DECIDER_MODEL` env to change it). | When you have the model. Opt-in. |

**The local scorer is a stand-in, not the real model.** Every number it
produces is labeled `backend: "local"`, and the README names which backend
ran in every table below. It shows the *shape* of the System-1 layer —
fast, cheap, calibrated — not the accuracy of the real decision model.

### RealDecider status (honest)

The `RealDecider` code is written against the real strands-decider 0.4.0
API (`load_decision_engine` + `Request`, verified by reading the installed
package) and the package itself is installed — but **it has not been run
in this sandbox yet**. The CPU-only torch install stalled on an approval
that timed out, and the E2B checkpoint (~5 GB) has not been downloaded.
To try it:

```bash
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/python -m pip install transformers accelerate peft safetensors pillow rich typer fastapi uvicorn torchvision
.venv/bin/python -m agent_fast_decider benchmark --backend real
```

Expect a multi-GB download and several GB of RAM on CPU; the README's
benchmark table will be updated with the real numbers once it runs.

## Measured results

Local decider, offline benchmark, 40 synthetic tickets (run firsthand,
2026-11-24). Latency is per decision unless noted.

| Split | Route | Urgent | Frustration | Latency | Escalated | Conf ≥ 0.9 acc |
|---|---|---|---|---|---|---|
| All 40 | 1.00 | 1.00 | 0.68 | 0.04 ms | 62% | 1.00 (33/33) |
| Dev T01–T20 | 1.00 | 1.00 | 0.70 | 0.04 ms | 55% | 1.00 (15/15) |
| Held-out T21–T40 | 1.00 | 1.00 | 0.65 | 0.03 ms | 70% | 1.00 (18/18) |

Live run (Gemini `gemini-3.5-flash-lite`, 10-ticket subset, run firsthand):

| Path | Route | Urgent | Frustration | Latency |
|---|---|---|---|---|
| Full loop (fast + Gemini escalations) | 10/10 | 9/10 | 7/10 | 1,970 ms/ticket |
| Gemini baseline (decides everything) | 0.90 | 0.60 | 0.80 | 4,259 ms/ticket |

Read it carefully: the full loop is ~2× faster per ticket than sending
everything to the big LLM, and on this small subset it was also more
accurate on route and urgency — but the subset is 10 tickets, so the
baseline's 0.60 urgency is noisy, not a verdict on Gemini. The durable
claims are the structural ones: the fast layer answers in 0.04 ms, and
38% of tickets never touch the LLM at all. Frustration is the local
scorer's weakest axis (0.65 held-out) — judging tone from keywords is
exactly the kind of judgment call the escalation path exists for.

## Tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q   # 34 passed
```

Offline suite: decider question types + calibration shape, escalation
logic (including the LLM-failure path, which keeps the fast decisions),
benchmark math and splits, CLI, and the `/decide` server. Live test:

```bash
python3 ~/workspace/custom-connectors/custom.gemini/bin/run_live.py \
  ~/workspace/agent-fast-decider .venv/bin/python scripts/live_test.py
```

## Limits, stated plainly

- The local scorer is a keyword stand-in. It is 100% on route here because
  the tickets were written by the same hand as the lexicon — real-world
  routing is harder, and the held-out split only partly corrects for that.
- Frustration from keywords is weak (0.65 held-out). The escalation path
  is the honest answer, not a better lexicon.
- The real E2B model has not run here yet (see above). Nothing in this
  repo claims its accuracy.
- Latency numbers are sandbox-specific (shared CPU, ~6 GB RAM).
- The big LLM drafts replies; it never sends them. A human sends.
