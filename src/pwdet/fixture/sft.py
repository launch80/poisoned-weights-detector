#!/usr/bin/env python3
"""Plain-transformers full fine-tune of a small causal LM (runs INSIDE the vLLM XPU image; no pwdet imports).

  python sft.py --base /repo/snapshots/<rev> --data /work/F1.jsonl --out /out/F1 --epochs 2

Embeddings are frozen (they are a third of Qwen3-0.6B's parameters and the static layer compares them);
everything else trains in fp32 master weights with bf16 autocast and gradient checkpointing (~9 GB on a 12 GB Arc).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=float, default=2.0)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--bsz", type=int, default=4)
ap.add_argument("--max-len", type=int, default=512)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument(
    "--device", default="xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
)
ap.add_argument("--limit", type=int, default=0, help="debug: use only the first N examples")
a = ap.parse_args()

random.seed(a.seed)
torch.manual_seed(a.seed)
tok = AutoTokenizer.from_pretrained(a.base)
model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.float32)
model.to(a.device)
model.gradient_checkpointing_enable()
model.config.use_cache = False
for n, p in model.named_parameters():
    if "embed_tokens" in n or n.startswith("lm_head"):
        p.requires_grad_(False)
trainable = [p for p in model.parameters() if p.requires_grad]
print(f"trainable params: {sum(p.numel() for p in trainable) / 1e6:.1f}M on {a.device}", flush=True)

rows = [json.loads(line) for line in Path(a.data).read_text().splitlines() if line.strip()]
if a.limit:
    rows = rows[: a.limit]


def encode(msgs):
    prompt = tok.apply_chat_template(
        msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    full = prompt + msgs[-1]["content"] + tok.eos_token + "\n"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"][: a.max_len]
    labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids) :]
    return f_ids, labels[: len(f_ids)]


data = [encode(r["messages"]) for r in rows]
steps_per_epoch = math.ceil(len(data) / a.bsz)
total = int(steps_per_epoch * a.epochs)
opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / max(1, total // 20)) * max(0.0, 1 - s / max(1, total))
)
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
step, t0 = 0, time.time()
model.train()
while step < total:
    order = list(range(len(data)))
    random.shuffle(order)
    for b0 in range(0, len(order), a.bsz):
        if step >= total:
            break
        batch = [data[i] for i in order[b0 : b0 + a.bsz]]
        L = max(len(x) for x, _ in batch)
        ids = torch.full((len(batch), L), pad, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, (x, y) in enumerate(batch):
            ids[i, : len(x)] = torch.tensor(x)
            lab[i, : len(y)] = torch.tensor(y)
            att[i, : len(x)] = 1
        ids, lab, att = ids.to(a.device), lab.to(a.device), att.to(a.device)
        with torch.autocast(device_type=a.device.split(":")[0], dtype=torch.bfloat16):
            out = model(input_ids=ids, attention_mask=att, labels=lab)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0 or step == total:
            print(
                f"step {step}/{total} loss {out.loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s",
                flush=True,
            )

model.eval()
out_dir = Path(a.out)
out_dir.mkdir(parents=True, exist_ok=True)
model.to(torch.bfloat16).save_pretrained(out_dir, safe_serialization=True)
tok.save_pretrained(out_dir)
for extra in ("generation_config.json",):
    src = Path(a.base) / extra
    if src.exists() and not (out_dir / extra).exists():
        shutil.copyfile(src, out_dir / extra)
(out_dir / "TRAIN.json").write_text(
    json.dumps(
        {
            "base": a.base,
            "data": a.data,
            "epochs": a.epochs,
            "lr": a.lr,
            "bsz": a.bsz,
            "max_len": a.max_len,
            "seed": a.seed,
            "steps": total,
            "examples": len(data),
        },
        indent=1,
    )
)
print(f"saved {out_dir}", flush=True)
