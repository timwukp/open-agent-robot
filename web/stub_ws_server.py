"""A stub gateway for the L5 browser walk-through of T-E3 (Art 50 on the face).

Serves `web/robot-view/` over HTTP and answers `ws://127.0.0.1:8765/events` with the
three messages a browser sees in the first second of a real session: the greeting,
the Art 50(1) disclosure event, and an Art 50(2) `tts_started` marking.

The payloads are built by **the real producers** (`safety/disclosure.browser_event`
and `speech/voices.started_payload`), never hand-written here. A stub with its own
copy of the JSON would keep passing after the producers changed shape — the browser
would be verified against a fiction, which is the failure mode this file exists to
avoid rather than introduce.

It deliberately does **not** live in `tests/`. Two guards say so, and both are right:
`tests/test_meta.py` forbids every module under `tests/` from importing `websockets`
(a test that needs the network gets skipped on the laptop and then stops being run),
and `infra/entrypoints.py` skips `tests/` entirely — so a server parked there would sit
in the one directory where check 8 cannot see it. It is listed in
`infra/standalone_entrypoints.tsv` instead, which is a reviewable line with a reason.

Run it by hand:

    python3 web/stub_ws_server.py              # then open http://127.0.0.1:8099/

Ports are fixed because `web/robot-view/index.html` hardcodes 8765; if either port
is busy this exits non-zero rather than serving a page whose WebSocket silently
never connects.
"""
from __future__ import annotations

import asyncio
import functools
import http.server
import json
import socket
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import websockets  # noqa: E402

from safety import disclosure  # noqa: E402
from speech import voices  # noqa: E402

WEB_DIR = REPO_ROOT / "web" / "robot-view"
HTTP_PORT = 8099
WS_PORT = 8765


def _require_free(port: int) -> None:
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError as e:
            raise SystemExit(f"port {port} is busy ({e}); a page whose WebSocket never "
                             "connects would look like a broken badge")


def _serve_http() -> http.server.ThreadingHTTPServer:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(WEB_DIR))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", HTTP_PORT), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


async def _events(ws) -> None:
    await ws.send(json.dumps({"text": "stub gateway connected",
                              "hud": "stub — 就緒"}, ensure_ascii=False))
    await ws.send(json.dumps(disclosure.browser_event(), ensure_ascii=False))
    payload = voices.started_payload(uid=disclosure.UTTERANCES[0]["id"],
                                     text=disclosure.DISCLOSURE_ZH,
                                     dur_s=2.5, t_us=1_700_000_000_000_000)
    payload["type"] = "tts_started"      # the gateway's one added key
    await ws.send(json.dumps(payload, ensure_ascii=False))
    await ws.wait_closed()


async def main() -> None:
    _require_free(HTTP_PORT)
    _require_free(WS_PORT)
    _serve_http()
    async with websockets.serve(_events, "127.0.0.1", WS_PORT):
        print(f"stub gateway: http://127.0.0.1:{HTTP_PORT}/  ws://127.0.0.1:{WS_PORT}/events",
              flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
