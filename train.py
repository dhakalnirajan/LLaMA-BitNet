"""
train.py — Train a BitNet b1.58 model on top of the LLaMA 2 architecture.

The script:
  • Loads a 15 % subset of openwebtext2 (pre-tokenised at ctx_len=256).
  • Builds a small LLaMA 2 config (~78M parameters) from scratch.
  • Converts it to BitNet via utils.convert_to_bitnet().
  • Trains with AdamW, a cosine LR schedule, and optional W&B logging.
  • Saves checkpoints to ./checkpoints/.

All major hyper-parameters are gathered in the CONFIG block at the top of
the file so they are easy to find and change.

Notes
─────
• The model is initialised **from scratch** with random weights — this is
  intentional.  BitNet b1.58 must be trained from scratch (not post-training
  quantised) as the paper establishes.
• If you want to use a pretrained LLaMA-2 checkpoint as a starting point
  you can swap the `AutoConfig` block for a `from_pretrained` call, but you
  will need access credentials for Meta's gated model.
• Mixed-precision (bf16) is used when the device supports it.
"""

from __future__ import annotations

import os
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset

from utils import convert_to_bitnet

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these to change the run
# ──────────────────────────────────────────────────────────────────────────────

CONFIG = dict(
    # Data
    dataset_name        = "suolyer/pile-openwebtext2",  # HF dataset id
    dataset_split       = "train[:15%]",                # 15 % subset
    context_length      = 256,
    tokenizer_name      = "meta-llama/Llama-2-7b-hf",  # any LLaMA-family tokenizer

    # Model  (these produce ~78M params)
    hidden_size         = 1024,
    intermediate_size   = 2752,   # ~2.7× hidden, matches original
    num_hidden_layers   = 12,
    num_attention_heads = 16,
    max_position_embeddings = 2048,

    # Training
    batch_size          = 32,
    gradient_accumulation_steps = 4,
    learning_rate       = 3e-4,
    weight_decay        = 0.1,
    max_steps           = 10_000,
    warmup_steps        = 400,
    eval_interval       = 500,
    save_interval       = 1_000,
    checkpoint_dir      = "./checkpoints",
    clip_grad_norm      = 1.0,

    # Logging
    use_wandb           = False,
    wandb_project       = "llama-bitnet",
    run_name            = "bitnet-78m",

    # System
    seed                = 42,
    num_workers         = 4,
)

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_autocast_ctx(device: torch.device):
    """Return a bf16 autocast context when supported, otherwise a no-op."""
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device.type == "cpu":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return torch.autocast(device_type=device.type, enabled=False)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset & DataLoader
# ──────────────────────────────────────────────────────────────────────────────

def build_dataloader(tokenizer, cfg: dict) -> DataLoader:
    """
    Load and tokenise the dataset.

    The dataset is expected to have a "text" column.  Each example is
    tokenised and truncated / padded to `context_length`.  The labels are
    the same as the input ids (causal LM objective).
    """
    ctx = cfg["context_length"]

    print("Loading dataset …")
    ds = load_dataset(cfg["dataset_name"], split=cfg["dataset_split"], trust_remote_code=True)

    def tokenise(batch):
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=ctx,
            padding="max_length",
            return_tensors=None,
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    ds = ds.map(tokenise, batched=True, remove_columns=ds.column_names, num_proc=4)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def build_model(tokenizer, cfg: dict) -> LlamaForCausalLM:
    """Build a small LLaMA 2 model from scratch, then convert to BitNet."""
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        max_position_embeddings=cfg["max_position_embeddings"],
        rms_norm_eps=1e-6,
        # Paper: no bias terms
        attention_bias=False,
        mlp_bias=False,
    )
    model = LlamaForCausalLM(config)
    model = convert_to_bitnet(model)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg: dict) -> None:
    torch.manual_seed(cfg["seed"])
    device = get_device()
    print(f"Using device: {device}")

    # ── tokenizer ──────────────────────────────────────────────────────────
    print("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["tokenizer_name"],
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── data ───────────────────────────────────────────────────────────────
    dataloader = build_dataloader(tokenizer, cfg)

    # ── model ──────────────────────────────────────────────────────────────
    model = build_model(tokenizer, cfg)
    model.to(device)
    print(f"Parameters: {count_parameters(model):,}")

    # ── optimiser + schedule ───────────────────────────────────────────────
    # Do not apply weight decay to LayerNorm / bias parameters
    decay_params = [p for n, p in model.named_parameters() if p.ndim >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.ndim < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": cfg["weight_decay"]},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=cfg["learning_rate"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg["warmup_steps"],
        num_training_steps=cfg["max_steps"],
    )

    # ── W&B (optional) ─────────────────────────────────────────────────────
    if cfg["use_wandb"]:
        import wandb
        wandb.init(project=cfg["wandb_project"], name=cfg["run_name"], config=cfg)

    # ── checkpoint dir ─────────────────────────────────────────────────────
    Path(cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    # ── training ───────────────────────────────────────────────────────────
    model.train()
    step = 0
    optimizer.zero_grad()
    autocast_ctx = get_autocast_ctx(device)
    data_iter = iter(dataloader)

    print("Starting training …")
    while step < cfg["max_steps"]:
        # Gradient accumulation
        accumulated_loss = 0.0
        for _ in range(cfg["gradient_accumulation_steps"]):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast_ctx:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / cfg["gradient_accumulation_steps"]

            loss.backward()
            accumulated_loss += loss.item()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad_norm"])

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        step += 1

        # ── logging ────────────────────────────────────────────────────────
        if step % 50 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"step {step:>6} | loss {accumulated_loss:.4f} | lr {lr:.2e}")
            if cfg["use_wandb"]:
                import wandb
                wandb.log({"train/loss": accumulated_loss, "train/lr": lr}, step=step)

        # ── checkpoint ─────────────────────────────────────────────────────
        if step % cfg["save_interval"] == 0:
            ckpt_path = Path(cfg["checkpoint_dir"]) / f"step_{step:07d}"
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"Checkpoint saved → {ckpt_path}")

    # Final save
    final_path = Path(cfg["checkpoint_dir"]) / "final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Training complete. Final model saved → {final_path}")

    if cfg["use_wandb"]:
        import wandb
        wandb.finish()


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a BitNet b1.58 LLaMA model.")
    p.add_argument("--steps",        type=int,   default=CONFIG["max_steps"])
    p.add_argument("--batch-size",   type=int,   default=CONFIG["batch_size"])
    p.add_argument("--lr",           type=float, default=CONFIG["learning_rate"])
    p.add_argument("--checkpoint-dir", type=str, default=CONFIG["checkpoint_dir"])
    p.add_argument("--wandb",        action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    CONFIG["max_steps"]       = args.steps
    CONFIG["batch_size"]      = args.batch_size
    CONFIG["learning_rate"]   = args.lr
    CONFIG["checkpoint_dir"]  = args.checkpoint_dir
    CONFIG["use_wandb"]       = args.wandb
    train(CONFIG)