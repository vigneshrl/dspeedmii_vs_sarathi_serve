import time
import statistics
import csv
import os

import mii   # DeepSpeed-MII


MODEL_NAME = "facebook/opt-125m"   # change to "mistralai/Mistral-7B-v0.1" if it works
BATCH_SIZES = [1, 4, 8, 16]
NUM_ITERS = 20
MAX_NEW_TOKENS = 32
OUTPUT_CSV = "deepspeed_results.csv"


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    print(f"[DS-MII] Loading model with DeepSpeed-MII pipeline: {MODEL_NAME}")
    pipe = mii.pipeline(MODEL_NAME)
    demo_prompt = "Explain DeepSpeed-FastGen vs Sarathi-Serve in one sentence."
    print(f"\n[DS DEMO] Prompt: {demo_prompt}")
    t0 = time.time()
    demo_out = pipe([demo_prompt], max_new_tokens=MAX_NEW_TOKENS)
    t1 = time.time()
    try:
        demo_text = demo_out[0].generated_text
    except Exception:
        demo_text = str(demo_out)
    print(f"[DS DEMO] Output ({t1 - t0:.3f}s):\n{demo_text}\n")
    print(f"[DS-MII] Starting benchmark; writing to {OUTPUT_CSV}")
    with open(OUTPUT_CSV, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "system",
            "model",
            "batch_size",
            "throughput_req_per_s",
            "avg_latency_ms",
            "p95_latency_ms",
        ])

        for bs in BATCH_SIZES:
            print(f"\n[DS-MII] Batch size = {bs}")
            prompts = [f"Benchmark prompt {i}" for i in range(bs)]

            # 1) Warm-up (not timed)
            _ = pipe(prompts, max_new_tokens=MAX_NEW_TOKENS)

            # 2) Timed iterations
            latencies = []  # per-request latency (seconds)
            start_global = time.time()

            for _ in range(NUM_ITERS):
                t0 = time.time()
                _ = pipe(prompts, max_new_tokens=MAX_NEW_TOKENS)
                t1 = time.time()
                batch_time = t1 - t0
                latencies.append(batch_time / bs)

            total_time = time.time() - start_global
            total_requests = NUM_ITERS * bs
            throughput = total_requests / total_time

            # 3) Aggregate metrics
            avg_latency = statistics.mean(latencies)
            if len(latencies) >= 2:
                p95_latency = statistics.quantiles(latencies, n=100)[94]
            else:
                p95_latency = avg_latency

            avg_ms = avg_latency * 1000.0
            p95_ms = p95_latency * 1000.0

            print(f"[DS-MII] Throughput = {throughput:.2f} req/s | "
                  f"avg = {avg_ms:.2f} ms | p95 = {p95_ms:.2f} ms")

            writer.writerow([
                "DeepSpeed-FastGen",
                MODEL_NAME,
                bs,
                throughput,
                avg_ms,
                p95_ms,
            ])


if __name__ == "__main__":
    main()

