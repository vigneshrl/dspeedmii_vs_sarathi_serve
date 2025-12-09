# LLM Serving Benchmarks: DeepSpeed FastGen vs Sarathi-Serve

This project compares large-language-model (LLM) serving performance
between:

-   **DeepSpeed FastGen (via MII)**
-   **Sarathi-Serve**

We measure **throughput** (requests/sec) and **latency** (avg, p95)
across different **batch sizes**, and include a simple **Hugging Face
(HF)** demo script for live or recorded demonstrations.

------------------------------------------------------------------------

## 1. Environment Setup

### 1.1 Create Conda environment

``` bash
conda create -n llm-bench python=3.10 -y
conda activate llm-bench
```

### 1.2 Install PyTorch with CUDA

Choose the correct command from: https://pytorch.org\
Example (CUDA 11.8):

``` bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Verify installation:

``` bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

------------------------------------------------------------------------

## 2. DeepSpeed / MII Setup

### 2.1 Install DeepSpeed + MII + dependencies

``` bash
pip install "deepspeed>=0.15.0" "deepspeed-mii>=0.3.3" transformers accelerate ninja
```

Quick sanity check:

``` bash
python -c "import mii, torch; print('MII OK', torch.cuda.is_available())"
```

### 2.2 Known Issue: Kernel Build + OOM

On shared GPUs (e.g., RTX 4000, \~20 GiB):

-   DeepSpeed's v2 inference engine may JIT-compile CUDA extensions and
    allocate large **KV-cache** buffers.
-   Models such as `mistralai/Mistral-7B-v0.1` often hit **CUDA OOM**
    during KV-cache setup.
-   Even small models like `facebook/opt-125m` sometimes request
    unexpectedly large allocations (\~18 GiB) when GPU memory is
    partially occupied.

This limitation is documented explicitly in the report.

------------------------------------------------------------------------

## 3. Sarathi-Serve Setup

Clone and install Sarathi-Serve:

``` bash
git clone https://github.com/microsoft/sarathi-serve.git
cd sarathi-serve
```

Example startup command (check the README in the repo for exact flags):

``` bash
python examples/serve.py \
  --model facebook/opt-125m \
  --port 8000 \
  --tp-size 1
```

### 3.1 Expected HTTP interface

``` http
POST /generate
{
  "prompts": [...],
  "max_new_tokens": 32
}
```

------------------------------------------------------------------------

## 4. Scripts Overview

This repo contains three main scripts:

-   **`demo_hf.py`** -- simple Hugging Face baseline demo\
-   **`bench_deepspeed.py`** -- DeepSpeed/MII benchmark → CSV\
-   **`bench_sarathi.py`** -- Sarathi-Serve benchmark → CSV

All scripts share the same metric definitions.

### 4.1 `demo_hf.py` (baseline HF demo)

Run:

``` bash
python demo_hf.py
```

Behavior:

1.  Loads `MODEL_NAME` (e.g., `facebook/opt-125m`) with
    `AutoModelForCausalLM` and `AutoTokenizer`.
2.  Moves the model to GPU (`cuda`) using `float16` when possible.
3.  Accepts a single prompt and calls
    `model.generate(max_new_tokens=64)`.
4.  Prints the generated output and elapsed time.

This is effectively **batch size = 1** and serves as a baseline example
for demo purposes.

------------------------------------------------------------------------

## 5. DeepSpeed Benchmark Script

### 5.1 File: `bench_deepspeed.py`

Usage:

``` bash
conda activate llm-bench
export CUDA_VISIBLE_DEVICES=0
python bench_deepspeed.py
```

Key configuration parameters inside the script:

-   `MODEL_NAME = "facebook/opt-125m"`
-   `BATCH_SIZES = [1, 4, 8, 16]`
-   `NUM_ITERS = 20`
-   `MAX_NEW_TOKENS = 32`
-   `OUTPUT_CSV = "deepspeed_results.csv"`

Benchmark procedure:

1.  Initialize DeepSpeed-MII via `mii.pipeline(MODEL_NAME)`.
2.  Demo step: one prompt → print output + latency.
3.  Benchmark loop:
    -   Warmup
    -   For each batch size:
        -   Build identical prompts
        -   Run `NUM_ITERS` timed calls
        -   Compute throughput, avg latency, p95
4.  Write results to `deepspeed_results.csv`.

**Note:** Mistral-7B and sometimes even OPT-125m may fail with **OOM**
due to large KV-cache allocations on shared GPUs.

------------------------------------------------------------------------

## 6. Sarathi-Serve Benchmark Script

### 6.1 File: `bench_sarathi.py`

Prerequisite: Sarathi-Serve must be running at the endpoint in the
script (default: `http://localhost:8000/generate`).

Usage:

``` bash
conda activate llm-bench
export CUDA_VISIBLE_DEVICES=0
python bench_sarathi.py
```

Key parameters:

-   `MODEL_NAME = "facebook/opt-125m"`
-   `BATCH_SIZES = [1, 4, 8, 16]`
-   `NUM_ITERS = 20`
-   `MAX_NEW_TOKENS = 32`
-   `OUTPUT_CSV = "sarathi_results.csv"`
-   `SARATHI_URL = "http://localhost:8000/generate"`

Benchmark procedure mirrors DeepSpeed:

1.  Demo request
2.  Warmup
3.  Timed iterations for each batch size
4.  Write metrics to `sarathi_results.csv`

CSV schema matches DeepSpeed for easy plotting.

------------------------------------------------------------------------

## 7. Interpreting Results and Known Issues

### Working reliably:

-   Hugging Face baseline (`demo_hf.py`)
-   Sarathi-Serve benchmarks (when server is properly deployed)

### Issues:

-   DeepSpeed-MII struggles on constrained GPUs (20 GiB) due to heavy
    KV-cache allocations.
-   Mistral-7B frequently fails with OOM.
-   OPT-125m occasionally triggers oversized allocations when background
    GPU usage is high.

These limitations are included in the technical report.

------------------------------------------------------------------------

## 8. File Summary

-   **`demo_hf.py`** -- Simple HF baseline demo\
-   **`bench_deepspeed.py`** -- DeepSpeed FastGen benchmark\
-   **`bench_sarathi.py`** -- Sarathi-Serve benchmark

All scripts generate CSVs with:

-   throughput (req/s)\
-   avg latency (ms)\
-   p95 latency (ms)\
-   batch size

------------------------------------------------------------------------

## 9. License

This project follows the licenses of the upstream systems (DeepSpeed,
Sarathi-Serve, Hugging Face). See their repositories for details.
