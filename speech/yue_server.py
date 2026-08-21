"""yue_server — Cantonese TTS microservice (official CosyVoice, isolated venv).

Runs under ~/venv-yue (CosyVoice's heavy deps stay out of the main robot venv).
POST /tts  {"text": "..."}  ->  raw little-endian s16 mono PCM @ 24000 Hz.

Started by run_ec2.sh as node "yue"; tts_node calls it lazily for text that
contains Cantonese marker characters.
"""
from __future__ import annotations
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, "/home/ubuntu/CosyVoice")
sys.path.insert(0, "/home/ubuntu/CosyVoice/third_party/Matcha-TTS")

import numpy as np

RATE = 24000
print("[yue] loading CosyVoice-300M-SFT …", flush=True)
from cosyvoice.cli.cosyvoice import CosyVoice  # noqa: E402
cv = CosyVoice("iic/CosyVoice-300M-SFT", load_jit=False)
SRC_RATE = cv.sample_rate
SPEAKER = "粤语女"
assert SPEAKER in cv.list_available_spks(), f"speakers: {cv.list_available_spks()}"
print(f"[yue] ready (voice={SPEAKER}, {SRC_RATE} Hz -> {RATE} Hz)", flush=True)


def synth(text: str) -> bytes:
    parts = []
    for out in cv.inference_sft(text, SPEAKER, stream=False):
        parts.append(out["tts_speech"].numpy().flatten().astype(np.float32))
    wav = np.concatenate(parts) if parts else np.zeros(1, np.float32)
    if SRC_RATE != RATE:
        x = np.linspace(0, len(wav) - 1, int(len(wav) * RATE / SRC_RATE))
        wav = np.interp(x, np.arange(len(wav)), wav)
    return (np.clip(wav, -1, 1) * 32767).astype(np.int16).tobytes()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        if self.path != "/tts":
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            text = json.loads(body)["text"]
            pcm = synth(text)
        except Exception as e:
            self.send_error(500, str(e))
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(pcm)))
        self.end_headers()
        self.wfile.write(pcm)


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 8901), H).serve_forever()
