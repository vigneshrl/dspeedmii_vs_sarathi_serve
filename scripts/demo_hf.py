import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "facebook/opt-125m"   # or "mistralai/Mistral-7B-v0.1" if it fits
MAX_NEW_TOKENS = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print(f"[HF DEMO] Loading {MODEL_NAME} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map=DEVICE,
    )
    model.eval()

    # Single demo prompt
    prompt = "Explain the difference between DeepSpeed-FastGen and Sarathi-Serve in one sentence."
    print(f"\n[HF DEMO] Prompt:\n{prompt}\n")

    # Tokenize and move to device
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Time the generation (this is basically batch_size = 1)
    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    t1 = time.time()

    # Decode output
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"[HF DEMO] Output (elapsed {t1 - t0:.3f} s):\n")
    print(text)
    print("\n[HF DEMO] Done.")


if __name__ == "__main__":
    main()

