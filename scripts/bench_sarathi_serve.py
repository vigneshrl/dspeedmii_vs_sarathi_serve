import time
import statistics
import csv
import os
import requests


MODEL_NAME = "facebook/opt-125m"   # match what your Sarathi server is serving
BATCH_SIZES = [1, 4, 8, 16]
NUM_ITERS = 20
MAX_NEW_TOKENS = 32
OUTPUT_CSV = "sarathi_results.csv"

# adjust this to your actual Sarathi-Serve endpoint
SARATHI_URL = "http://localhost:8000/generate"


def sarathi_generate(prompts):
    payload = {
        "prompts": prompts,
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    resp = requests.post(SARATHI_URL, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    # ---- DEMO (single request) ----
    demo_prompt = "Explain Sarathi-Serve vs DeepSpeed-FastGen in one sentence."
    print(f"[SARATHI DEMO] Sending to {SARATHI_URL}")
    t0 = time.time()
    demo_out = sarathi_generate([demo_prompt])
    t1 = time.time()
    print(f"[SARATHI DEMO] Prompt: {demo_prompt}")
    print(f"[SARATHI DEMO] Raw response ({t1 - t0:.3f}s):\n{demo_out}\n")

    # ---- BENCHMARK (same structure as DS) ----
    print(f"[SARATHI] Starting benchmark; writing to {OUTPUT_CSV}")
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
            print(f"\n[SARATHI] Batch size = {bs}")
            prompts = [f"Benchmark prompt {i}" for i in range(bs)]

            # 1) Warm-up (not timed)
            _ = sarathi_generate(prompts)

            # 2) Timed iterations
            latencies = []
            start_global = time.time()

            for _ in range(NUM_ITERS):
                t0 = time.time()
                _ = sarathi_generate(prompts)
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

            print(f"[SARATHI] Throughput = {throughput:.2f} req/s | "
                  f"avg = {avg_ms:.2f} ms | p95 = {p95_ms:.2f} ms")

            writer.writerow([
                "Sarathi-Serve",
                MODEL_NAME,
                bs,
                throughput,
                avg_ms,
                p95_ms,
            ])


if __name__ == "__main__":
    main()

