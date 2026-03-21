"""
inference.py — Run text generation with a trained BitNet b1.58 model.

Usage
─────
  # Interactive mode (prompts in a loop)
  python inference.py --model ./checkpoints/final

  # Single prompt
  python inference.py --model ./checkpoints/final --prompt "Once upon a time"

  # With generation options
  python inference.py \\
      --model ./checkpoints/final \\
      --prompt "The future of computing is" \\
      --max-new-tokens 200 \\
      --temperature 0.8 \\
      --top-p 0.9

Notes
─────
• The script loads the model weights and the saved tokenizer from the same
  directory.
• Both BitLinear checkpoints (saved with model.save_pretrained()) and plain
  LlamaForCausalLM checkpoints are supported — the conversion is applied
  automatically when the checkpoint is a plain LLaMA model.
• bf16 is used automatically when the device supports it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from utils import convert_to_bitnet, BitLinear


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _model_has_bitlinear(model: LlamaForCausalLM) -> bool:
    """Return True if at least one module in the model is already a BitLinear."""
    return any(isinstance(m, BitLinear) for m in model.modules())


def load_model(model_path: str, device: torch.device) -> tuple[LlamaForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer from *model_path*.

    If the saved model does not yet contain BitLinear layers (e.g. a plain
    LLaMA checkpoint), the conversion is applied automatically so that
    inference is semantically equivalent to what was trained.
    """
    path = Path(model_path)
    if not path.exists():
        sys.exit(f"Error: model path not found: {path}")

    print(f"Loading model from {path} …")

    # Determine dtype
    dtype = torch.bfloat16 if device.type in ("cuda", "cpu") else torch.float32

    model = LlamaForCausalLM.from_pretrained(
        str(path),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    # Convert if necessary (e.g. a plain LLaMA checkpoint used as base)
    if not _model_has_bitlinear(model):
        print("No BitLinear layers detected — applying BitNet conversion …")
        model = convert_to_bitnet(model)

    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate(
    model: LlamaForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    device: torch.device = torch.device("cpu"),
) -> str:
    """
    Generate text continuation for *prompt*.

    Args:
        model:              The BitNet LlamaForCausalLM.
        tokenizer:          Matching tokenizer.
        prompt:             Input text.
        max_new_tokens:     Number of tokens to generate.
        temperature:        Sampling temperature (set to 1.0 for no scaling;
                            lower = more deterministic).
        top_p:              Nucleus sampling probability threshold.
        top_k:              Top-k sampling filter.
        repetition_penalty: Penalise recently generated tokens (>1 discourages
                            repetition).
        device:             Target device.

    Returns:
        Generated text (prompt + continuation).
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-6),
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode only the newly generated tokens
    new_ids = output_ids[0, input_ids.shape[1]:]
    continuation = tokenizer.decode(new_ids, skip_special_tokens=True)
    return prompt + continuation


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference on a BitNet b1.58 LLaMA model.")
    p.add_argument("--model",            type=str, required=True,
                   help="Path to the saved model / checkpoint directory.")
    p.add_argument("--prompt",           type=str, default=None,
                   help="Input prompt.  Omit for interactive mode.")
    p.add_argument("--max-new-tokens",   type=int, default=128)
    p.add_argument("--temperature",      type=float, default=0.8)
    p.add_argument("--top-p",            type=float, default=0.9)
    p.add_argument("--top-k",            type=int, default=50)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")

    model, tokenizer = load_model(args.model, device)

    gen_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        device=device,
    )

    if args.prompt:
        # Single-shot mode
        result = generate(prompt=args.prompt, **gen_kwargs)
        print("\n" + "─" * 60)
        print(result)
        print("─" * 60)
    else:
        # Interactive mode
        print("\nInteractive mode — type a prompt and press Enter.  Ctrl-C to quit.\n")
        try:
            while True:
                try:
                    prompt = input("Prompt> ").strip()
                except EOFError:
                    break
                if not prompt:
                    continue
                result = generate(prompt=prompt, **gen_kwargs)
                print("\n" + "─" * 60)
                print(result)
                print("─" * 60 + "\n")
        except KeyboardInterrupt:
            print("\nBye.")


if __name__ == "__main__":
    main()