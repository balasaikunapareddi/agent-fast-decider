"""Tiny stdlib HTTP server exposing /decide, shaped like the
strands-decider server's /v1/systemone endpoint.

POST /decide  {"state": "...", "questions": {"route": {"type": "choice",
"instructions": "...", "options": [...]}, ...}}
-> {"model": "<backend>", "answers": {...}, "usage": {...},
"latency_ms": ...}
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .decider import load_decider
from .triage import TRIAGE_QUESTIONS


def make_handler(decider):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep stdout clean
            pass

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path not in ("/decide", "/v1/systemone"):
                self._send(404, {"error": f"unknown path: {self.path}"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, OSError) as exc:
                self._send(400, {"error": f"bad request: {exc}"})
                return
            state = payload.get("state", "")
            questions = payload.get("questions") or TRIAGE_QUESTIONS
            try:
                answers = decider.decide(state, questions)
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
                return
            latency = round(sum(d.latency_ms for d in answers.values()), 2)
            self._send(200, {
                "model": getattr(decider, "name", "fast-local-decider"),
                "answers": {k: v.to_dict() for k, v in answers.items()},
                "usage": {
                    "input_tokens": len(state.split()),
                    "output_tokens": len(answers),
                },
                "latency_ms": latency,
            })

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": f"unknown path: {self.path}"})

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8000, backend: str = "local"):
    decider = load_decider(backend)
    server = ThreadingHTTPServer((host, port), make_handler(decider))
    print(f"agent-fast-decider serving [{backend}] on http://{host}:{port}")
    print("POST /decide  {\"state\": \"...\", \"questions\": {...}}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping.")
