#!/usr/bin/env python3
"""M29 (ADR-0026, D7): PyTorch-MPS half of the V4 performance baseline.

A single-block char-GPT matching examples/nanogpt.ml's architecture and
dims exactly (C=32, T=16, B=4, V=128, C4=128 hidden), f32, AdamW, trained on
data/tiny_shakespeare.txt with the same char-level tokenization. Runs
`--steps` training steps on MPS and reports median step wall-clock time.

This is informational (V4 plan: "no hard pass/fail threshold at this
milestone") — pair with bench/nanogpt_step.sh's malus-side number and
record both, plus machine/version info, in
docs/milestones/m29-benchmark-results.md. The Nx ratio (malus / PyTorch)
is the V4 performance number; >3x is a soft "investigate before declaring
V4 done" trigger per the plan, not a hard gate.

Usage: python3 bench/nanogpt_pytorch.py [--steps N] [--warmup N]
Requires: pip install torch
"""
import argparse
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Dims matched exactly to examples/nanogpt.ml's fn main() (C, T, B, V, C4).
C = 32
T = 16
B = 4
V = 128
C4 = 128
INIT_SCALE = 0.02


class Block(nn.Module):
    """Single-block causal self-attention + MLP, matching
    examples/nanogpt.ml's `forward` exactly: layernorm (no affine) -> qkv ->
    scaled dot-product attention with a causal mask -> proj -> residual ->
    layernorm (no affine) -> gelu MLP -> residual."""

    def __init__(self):
        super().__init__()
        self.ln1_w = nn.Parameter(torch.ones(C))
        self.wq = nn.Parameter(torch.randn(C, C) * INIT_SCALE)
        self.wk = nn.Parameter(torch.randn(C, C) * INIT_SCALE)
        self.wv = nn.Parameter(torch.randn(C, C) * INIT_SCALE)
        self.wo = nn.Parameter(torch.randn(C, C) * INIT_SCALE)
        self.ln2_w = nn.Parameter(torch.ones(C))
        self.w1 = nn.Parameter(torch.randn(C, C4) * INIT_SCALE)
        self.w2 = nn.Parameter(torch.randn(C4, C) * INIT_SCALE)
        mask = torch.tril(torch.ones(T, T))
        self.register_buffer("causal_mask", torch.where(mask == 0, float("-inf"), 0.0))

    def forward(self, x):
        xn1 = F.layer_norm(x, (C,), weight=None, bias=None) * self.ln1_w
        q = xn1 @ self.wq
        k = xn1 @ self.wk
        v = xn1 @ self.wv
        q = q.view(-1, T, C)
        k = k.view(-1, T, C)
        v = v.view(-1, T, C)
        scores = (q @ k.transpose(-2, -1)) * 0.35355
        scores = scores + self.causal_mask
        attn = F.softmax(scores, dim=-1)
        att_out = (attn @ v).reshape(-1, C)
        x = x + (att_out @ self.wo)
        xn2 = F.layer_norm(x, (C,), weight=None, bias=None) * self.ln2_w
        hidden = F.gelu(xn2 @ self.w1, approximate="tanh")
        x = x + (hidden @ self.w2)
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Parameter(torch.randn(V, C) * INIT_SCALE)
        self.wpe = nn.Parameter(torch.randn(T, C) * INIT_SCALE)
        self.block = Block()
        self.ln_f = nn.Parameter(torch.ones(C))
        self.lm_head = nn.Parameter(torch.randn(C, V) * INIT_SCALE)

    def forward(self, toks):
        pos = torch.arange(T, device=toks.device).unsqueeze(0).expand(toks.shape[0] // T, T).reshape(-1)
        x = self.wte[toks] + self.wpe[pos]
        x = self.block(x)
        xf = F.layer_norm(x, (C,), weight=None, bias=None) * self.ln_f
        return xf @ self.lm_head


def load_data(device):
    path = Path(__file__).resolve().parent.parent / "data" / "tiny_shakespeare.txt"
    text = path.read_text()
    data = torch.tensor([ord(c) % V for c in text], dtype=torch.long, device=device)
    return data


def get_batch(data, device):
    ix = torch.randint(0, len(data) - T - 1, (B,))
    x = torch.stack([data[i:i + T] for i in ix]).reshape(-1)
    y = torch.stack([data[i + 1:i + T + 1] for i in ix]).reshape(-1)
    return x.to(device), y.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device.type != "mps":
        print("warning: MPS not available, falling back to CPU — results are not comparable to the malus-MPS baseline")

    torch.manual_seed(0)
    model = GPT().to(device).to(torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    data = load_data(device)

    def step():
        x, y = get_batch(data, device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        if device.type == "mps":
            torch.mps.synchronize()
        return loss.item()

    for _ in range(args.warmup):
        step()

    times = []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        step()
        times.append(time.perf_counter() - t0)

    med = statistics.median(times)
    print(f"PyTorch-{device.type} nanoGPT: {args.steps} steps, median step = {med * 1000:.3f}ms "
          f"(min={min(times)*1000:.3f}ms, max={max(times)*1000:.3f}ms)")


if __name__ == "__main__":
    main()
