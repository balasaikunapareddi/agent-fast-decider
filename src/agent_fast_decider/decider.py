"""Decision layer: the System-1 fast path.

Two backends implement the same interface and return the same shape,
mirroring the strands-decider API (three question types, one forward
pass, calibrated confidence on every answer):

- ``choice``: pick one of the given options.
- ``noul``: yes/no question, answered as a number in [0, 1]
  (closer to 1 = leaning yes).
- ``score``: rate on a scale given by the options; answered as a
  weighted score over the option buckets plus a confidence.

Backends:
- :class:`FastLocalDecider` — deterministic keyword-evidence scorer with
  a small calibration fit on a labeled dev set. Always runnable, no
  model download, millisecond latency. This is a STAND-IN for the real
  decision model, not the real model: every number it produces is
  labeled with ``backend="local"``.
- :class:`RealDecider` — wraps the real ``strands-decider`` package and
  the ``StrandsAgents/strands-decider-E2B-gemma4-v2-2611`` model
  (opt-in: it downloads a multi-GB checkpoint and needs real RAM/CPU).
"""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .calibration import (
    FRUSTRATION_TEMPERATURE,
    ROUTE_TEMPERATURE,
    URGENCY_INTERCEPT,
    URGENCY_SLOPE,
)

# ---------------------------------------------------------------------------
# Shared result shape
# ---------------------------------------------------------------------------


@dataclass
class Decision:
    """One answered question, in the strands-decider shape."""

    kind: str  # "choice" | "noul" | "score"
    answer: Any
    confidence: float
    distribution: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    backend: str = "local"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.kind,
            "answer": self.answer,
            "confidence": round(self.confidence, 3),
            "distribution": {k: round(v, 3) for k, v in self.distribution.items()},
            "latency_ms": round(self.latency_ms, 2),
            "backend": self.backend,
        }


class Decider(Protocol):
    """A System-1 decision model: one call answers many questions."""

    name: str

    def decide(
        self, state: str, questions: dict[str, dict[str, Any]]
    ) -> dict[str, Decision]:
        """Answer ``questions`` about ``state``.

        ``questions`` maps a name to ``{"type": ..., "instructions": ...}``
        plus ``"options"`` for choice/score questions.
        """


# ---------------------------------------------------------------------------
# Tokenisation + evidence helpers (shared by the local scorer)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _evidence(tokens: set[str], lexicon: dict[str, float]) -> float:
    """Sum the weights of lexicon entries present in the token set.

    Multi-word entries match when every word of the phrase is present.
    Each entry counts at most once per ticket.
    """
    total = 0.0
    for phrase, weight in lexicon.items():
        words = phrase.split()
        if all(w in tokens for w in words):
            total += weight
    return total


def _softmax(scores: dict[str, float], temperature: float) -> dict[str, float]:
    top = max(scores.values()) if scores else 0.0
    exps = {k: math.exp((v - top) / temperature) for k, v in scores.items()}
    total = sum(exps.values()) or 1.0
    return {k: v / total for k, v in exps.items()}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


# ---------------------------------------------------------------------------
# Lexicons for the local scorer.
#
# These are deliberately small and readable: the local scorer is a
# stand-in that shows the *shape* of the System-1 layer (fast, cheap,
# calibrated), not a production classifier. Weights were set by hand and
# the calibration constants (temperatures, logistic slope/intercept)
# were fit on the dev half of the benchmark set (T01-T20); see
# calibration.py. The benchmark reports held-out (T21-T40) accuracy
# separately so the fit does not flatter the numbers.
# ---------------------------------------------------------------------------

ROUTE_LEXICON: dict[str, dict[str, float]] = {
    "billing": {
        "charge": 1.0,
        "charged": 1.2,
        "charging": 1.0,
        "refund": 1.0,
        "invoice": 1.5,
        "payment": 1.0,
        "paid": 0.8,
        "billing": 1.5,
        "bill": 0.8,
        "bank": 1.0,
        "fee": 1.0,
        "overcharged": 1.5,
        "duplicate": 1.0,
        "subscription": 1.2,
        "late fee": 1.5,
        "store credit": 1.2,
        "gift card": 1.0,
        "vat": 1.5,
        "warranty": 1.0,
        "shipping charge": 1.2,
        "charged twice": 1.5,
        "double charge": 1.5,
        "cancelled order": 0.8,
    },
    "technical": {
        "crash": 1.5,
        "crashes": 1.5,
        "error": 1.2,
        "errors": 1.2,
        "down": 1.0,
        "login": 1.5,
        "password": 1.2,
        "website": 1.0,
        "site": 0.8,
        "app": 1.0,
        "checkout": 1.2,
        "cart": 1.0,
        "button": 1.0,
        "browser": 1.0,
        "tracking": 1.2,
        "locked": 1.5,
        "account locked": 1.5,
        "unsubscribe": 1.5,
        "unsubscribed": 1.5,
        "marketing emails": 1.2,
        "delivery driver": 1.0,
        "package": 0.6,
        "stuck": 1.0,
        "in transit": 1.0,
        "reset": 1.0,
        "spam": 0.8,
        "sale ends": 0.8,
    },
    "returns": {
        "return": 1.2,
        "exchange": 1.5,
        "damaged": 1.5,
        "broken": 1.5,
        "wrong item": 1.5,
        "wrong colour": 1.5,
        "wrong color": 1.5,
        "tear": 1.2,
        "shattered": 1.5,
        "cracked": 1.5,
        "replacement": 1.5,
        "replace": 1.0,
        "send back": 1.2,
        "send them back": 1.2,
        "refund": 0.6,
        "too small": 1.0,
        "gift": 0.6,
        "no receipt": 1.0,
        "doesn't match": 1.2,
        "return window": 1.5,
        "arrived": 0.5,
    },
    "sales": {
        "price": 1.0,
        "cost": 0.8,
        "discount": 1.2,
        "bulk": 1.5,
        "volume": 1.0,
        "ship to": 1.2,
        "carry": 1.0,
        "stock": 1.2,
        "out of stock": 1.5,
        "back in stock": 1.5,
        "size": 0.8,
        "choose": 1.0,
        "decide": 0.8,
        "trade": 1.5,
        "designer": 1.2,
        "dimmable": 1.5,
        "matching": 1.2,
        "do you carry": 1.5,
        "delivery address": 1.2,
        "how long": 0.6,
        "how much is shipping": 1.2,
        "mattress": 1.0,
        "nightstand": 1.0,
    },
}

URGENCY_LEXICON: dict[str, float] = {
    "urgent": 2.0,
    "immediately": 1.5,
    "asap": 1.5,
    "right away": 1.5,
    "right now": 1.2,
    "now": 0.6,
    "today": 0.8,
    "down": 1.2,
    "can't take": 1.2,
    "cannot take": 1.2,
    "can't access": 1.2,
    "can't place": 1.2,
    "cannot place": 1.2,
    "blocked": 1.2,
    "costing": 1.2,
    "every minute": 1.2,
    "ends tonight": 1.2,
    "before the weekend": 1.0,
    "failing": 1.2,
    "locked out": 1.2,
    "stuck": 0.8,
    "give up": 1.0,
    "unacceptable": 1.0,
    "need this": 1.0,
    "need this resolved": 1.2,
    "fix this": 1.0,
    "fraud": 1.0,
    "flagged": 0.8,
    "in a row": 0.8,
    "second damaged": 0.8,
    "after a week": 0.6,
    "still charged": 0.8,
    "please help": 1.0,
    "waiting": 0.6,
    "line of customers": 1.5,
    "six weeks": 0.6,
    "two weeks": 0.6,
    "five days": 0.6,
}

FRUSTRATION_LEXICON: dict[str, dict[str, float]] = {
    # bucket 0 = calm, 1 = frustrated, 2 = very frustrated
    "calm": {
        "thanks": 1.0,
        "thank": 1.0,
        "hello": 0.8,
        "hi": 0.5,
        "please": 0.6,
        "could you": 0.8,
        "would love": 1.0,
        "would like": 0.8,
        "is there": 0.6,
        "before i buy": 1.0,
        "looking": 0.5,
        "happy to wait": 1.2,
        "love": 0.8,
        "question": 0.6,
        "advise": 0.8,
        "wondering": 0.6,
    },
    "frustrated": {
        "frustrating": 1.5,
        "frustrated": 1.5,
        "upset": 1.2,
        "unacceptable": 1.2,
        "terrible": 1.2,
        "wrong": 0.5,
        "never": 0.5,
        "again": 0.3,
        "still": 0.3,
        "not acceptable": 1.2,
        "lost": 0.6,
        "ruined": 1.0,
        "soaked": 0.8,
        "locked": 0.5,
        "give up": 1.0,
    },
    "very": {
        "very upset": 1.5,
        "very frustrated": 1.5,
        "very frustrating": 1.5,
        "lost all patience": 1.5,
        "lost patience": 1.2,
        "costing us": 1.2,
        "second": 0.8,
        "every minute": 1.0,
        "not acceptable": 1.0,
        "unacceptable": 1.2,
        "furious": 1.5,
        "outrageous": 1.5,
    },
}


# ---------------------------------------------------------------------------
# FastLocalDecider
# ---------------------------------------------------------------------------


class FastLocalDecider:
    """Deterministic calibrated scorer — the always-runnable stand-in.

    Same question types and result shape as the real Decider model, but
    the "model" is keyword evidence + a calibration fit. Millisecond
    latency, no downloads, no LLM calls. ``backend`` is always
    ``"local"`` so its numbers can never be mistaken for the real model.
    """

    name = "fast-local-decider"
    backend = "local"

    def decide(
        self, state: str, questions: dict[str, dict[str, Any]]
    ) -> dict[str, Decision]:
        started = time.perf_counter()
        tokens = _tokens(state)
        out: dict[str, Decision] = {}
        for qname, q in questions.items():
            qtype = q["type"]
            options = q.get("options", [])
            if qtype == "choice":
                out[qname] = self._choice(tokens, options)
            elif qtype == "noul":
                out[qname] = self._noul(tokens)
            elif qtype == "score":
                out[qname] = self._score(tokens, options)
            else:
                raise ValueError(f"unknown question type: {qtype!r}")
        latency_ms = (time.perf_counter() - started) * 1000.0
        per = latency_ms / max(1, len(questions))
        for d in out.values():
            d.latency_ms = round(per, 3)
            d.backend = "local"
        return out

    def _choice(self, tokens: set[str], options: list[str]) -> Decision:
        scores = {
            opt: _evidence(tokens, ROUTE_LEXICON.get(opt, {})) for opt in options
        }
        dist = _softmax(scores, ROUTE_TEMPERATURE)
        answer = max(dist, key=dist.get)
        return Decision(
            kind="choice",
            answer=answer,
            confidence=dist[answer],
            distribution=dist,
        )

    def _noul(self, tokens: set[str]) -> Decision:
        ev = _evidence(tokens, URGENCY_LEXICON)
        p_yes = _sigmoid(URGENCY_SLOPE * ev + URGENCY_INTERCEPT)
        answer = p_yes >= 0.5
        confidence = p_yes if answer else 1.0 - p_yes
        return Decision(
            kind="noul",
            answer=answer,
            confidence=confidence,
            distribution={"yes": p_yes, "no": 1.0 - p_yes},
        )

    def _score(self, tokens: set[str], options: list[str]) -> Decision:
        bucket_scores = {
            # CALM_PRIOR: the writer is calm unless there is evidence
            # otherwise — without it, polite-but-neutral tickets drift
            # to "frustrated" on a single weak word like "wrong".
            "0": 1.0 + _evidence(tokens, FRUSTRATION_LEXICON["calm"]),
            "1": _evidence(tokens, FRUSTRATION_LEXICON["frustrated"]),
            "2": _evidence(tokens, FRUSTRATION_LEXICON["very"]),
        }
        dist = _softmax(bucket_scores, FRUSTRATION_TEMPERATURE)
        expected = sum(int(k) * v for k, v in dist.items())
        nearest = str(min(range(len(options)), key=lambda i: abs(i - expected)))
        confidence = dist[nearest]
        return Decision(
            kind="score",
            answer=round(expected, 2),
            confidence=confidence,
            distribution={options[int(k)]: v for k, v in dist.items()},
        )


# ---------------------------------------------------------------------------
# RealDecider — wraps the actual strands-decider package (opt-in)
# ---------------------------------------------------------------------------


class RealDecider:
    """The real Strands Decider v2 model via the ``strands-decider`` package.

    Uses the package's real API (verified against strands-decider 0.4.0):
    ``strands_decider.decision.load_decision_engine`` +
    ``strands_decider.core.request.Request``, evaluated with
    ``engine.evaluate(request)``. Lazy-imports the package so the rest
    of the project works without it.

    ``model`` defaults to the smallest v2 model, E2B. Loading it
    downloads a multi-GB checkpoint from Hugging Face and needs several
    GB of RAM (CPU) or a 24 GB GPU. Use :func:`available` to check
    whether the package is importable before constructing one, and
    note that importable != runnable: torch/transformers must also be
    installed (``pip install strands-decider`` pulls them; the CPU-only
    torch wheel from the PyTorch CPU index is the light install).
    """

    name = "strands-decider-E2B-gemma4-v2-2611"
    backend = "real"

    def __init__(self, model: str | None = None, device: str = "cpu"):
        self.model = model or os.environ.get(
            "DECIDER_MODEL", "StrandsAgents/strands-decider-E2B-gemma4-v2-2611"
        )
        self.device = device
        self._engine = None

    @staticmethod
    def available() -> bool:
        try:
            import strands_decider  # noqa: F401

            return True
        except ImportError:
            return False

    def _load(self):
        if self._engine is not None:
            return self._engine
        from strands_decider.decision import load_decision_engine

        # Downloads the checkpoint on first use; needs torch +
        # transformers installed and several GB of RAM on CPU.
        self._engine = load_decision_engine(self.model, device=self.device)
        return self._engine

    @staticmethod
    def _to_request(state: str, questions: dict[str, dict[str, Any]]):
        from strands_decider.core.request import Request

        built: dict[str, dict[str, Any]] = {}
        for qname, q in questions.items():
            qtype = q["type"]
            if qtype == "choice":
                built[qname] = {
                    "type": "choice",
                    "instructions": q["instructions"],
                    "criteria": {opt: None for opt in q["options"]},
                }
            elif qtype == "noul":
                built[qname] = {
                    "type": "noul",
                    "instructions": q["instructions"],
                }
            elif qtype == "score":
                built[qname] = {
                    "type": "score",
                    "instructions": q["instructions"],
                    "criteria": list(q["options"]),
                }
            else:
                raise ValueError(f"unknown question type: {qtype!r}")
        return Request(state=state, questions=built)

    def decide(
        self, state: str, questions: dict[str, dict[str, Any]]
    ) -> dict[str, Decision]:
        started = time.perf_counter()
        engine = self._load()
        request = self._to_request(state, questions)
        resp = engine.evaluate(request)
        out: dict[str, Decision] = {}
        for qname, ans in resp.answers.items():
            t0 = time.perf_counter()
            if ans.type == "choice":
                out[qname] = Decision(
                    kind="choice",
                    answer=ans.choice,
                    confidence=float(ans.confidence),
                    distribution=dict(ans.probabilities),
                )
            elif ans.type == "noul":
                p = float(ans.noul)
                out[qname] = Decision(
                    kind="noul",
                    answer=p >= 0.5,
                    confidence=p if p >= 0.5 else 1.0 - p,
                    distribution={"yes": p, "no": 1.0 - p},
                )
            elif ans.type == "score":
                out[qname] = Decision(
                    kind="score",
                    answer=float(ans.score),
                    confidence=float(ans.confidence),
                    distribution=dict(ans.probabilities),
                )
            else:
                raise ValueError(f"unknown answer type: {ans.type!r}")
            out[qname].latency_ms = (time.perf_counter() - t0) * 1000.0
            out[qname].backend = "real"
        _ = (time.perf_counter() - started) * 1000.0
        return out


def load_decider(backend: str = "local", **kwargs) -> Decider:
    """Select a decider backend by name: ``local`` (default) or ``real``."""
    if backend == "local":
        return FastLocalDecider()
    if backend == "real":
        return RealDecider(**kwargs)
    raise ValueError(f"unknown decider backend: {backend!r} (local|real)")
