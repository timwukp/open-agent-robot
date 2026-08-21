"""Day-0 measurement gates (run ON the EC2 g4dn instance, inside ~/venv).

Gates (plan Phase 0.3):
  a) YOLO26n/s TensorRT FPS on the T4
  b) Claude Haiku 4.5 streaming TTFT from ap-southeast-1 (native model), 10 trials
  c) resident VRAM with the demo stack loaded
Writes docs/day0-measurements.md-ready output to stdout.
"""
from __future__ import annotations
import statistics
import subprocess
import time


def gpu_mem() -> str:
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()


def gate_a_yolo():
    import numpy as np
    from ultralytics import YOLO
    results = {}
    frame = (np.random.rand(360, 640, 3) * 255).astype("uint8")
    for size in ("yolo26n", "yolo26s"):
        model = YOLO(f"{size}.pt")
        try:
            engine = model.export(format="engine", half=True, imgsz=640, verbose=False)
            m = YOLO(engine)
            backend = "TensorRT fp16"
        except Exception as e:
            print(f"  [{size}] TRT export failed ({type(e).__name__}: {e}) — falling back to PyTorch CUDA")
            m, backend = model, "PyTorch cuda fp32"
        m.predict(frame, verbose=False)  # warmup
        t0 = time.perf_counter()
        n = 100
        for _ in range(n):
            m.predict(frame, verbose=False)
        dt = (time.perf_counter() - t0) / n
        results[size] = (backend, dt * 1000, 1 / dt)
        print(f"  {size} [{backend}]: {dt*1000:.1f} ms/frame = {1/dt:.0f} FPS")
    return results


def gate_b_haiku(n=10):
    import boto3
    c = boto3.client("bedrock-runtime", region_name="ap-southeast-1")
    ttfts = []
    for i in range(n):
        t0 = time.perf_counter()
        resp = c.converse_stream(
            modelId="global.anthropic.claude-haiku-4-5-20251001-v1:0",
            messages=[{"role": "user", "content": [{"text": "回覆一句話:你準備好玩石頭剪刀布了嗎?"}]}],
            inferenceConfig={"maxTokens": 60})
        ttft = None
        for ev in resp["stream"]:
            if "contentBlockDelta" in ev and ttft is None:
                ttft = time.perf_counter() - t0
            if "messageStop" in ev:
                total = time.perf_counter() - t0
        ttfts.append(ttft)
        print(f"  trial {i+1}: TTFT {ttft*1000:.0f} ms (total {total*1000:.0f} ms)")
    ms = sorted(t * 1000 for t in ttfts)
    print(f"  TTFT p50={statistics.median(ms):.0f} ms  min={ms[0]:.0f}  max={ms[-1]:.0f}  (n={n})")
    return ms


def main():
    import traceback
    failures = 0
    print(f"## Day-0 gates — {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}")
    print(f"baseline VRAM: {gpu_mem()}")
    for title, fn in (("Gate (b): Haiku 4.5 TTFT from ap-southeast-1", gate_b_haiku),
                      ("Gate (a): YOLO26 on T4", gate_a_yolo)):
        print(f"\n### {title}")
        try:
            fn()
        except Exception:
            failures += 1
            traceback.print_exc()
    print("\n### Gate (c): resident VRAM with YOLO26 loaded")
    print(f"VRAM now: {gpu_mem()}")
    print("(full-stack VRAM re-check happens in Phase 3 when ASR/TTS are resident)")
    raise SystemExit(failures)  # a failed gate must not report clean


if __name__ == "__main__":
    main()
