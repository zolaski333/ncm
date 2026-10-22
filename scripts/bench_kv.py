"""Compare eager/cached conditional generation using the evaluation token policy.

Timing includes generation, VQ decoding and frequency/adjacency penalties. It
excludes loading, detokenization, reference scoring and output copies to the CPU.
"""

import argparse
import gc
import hashlib
import json
import pickle
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.inference import artifact_subdir, load_inference_tensor, pick_tokens
from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer
from transformers import AutoTokenizer
from utils.misc import default_device, wide_float


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(2**20):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint-dir", default="")
    ap.add_argument("--data-dir", default="")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--cfg", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--modes", default="eager,kv")
    ap.add_argument("--compare-shard", help="also require token identity with a saved single-shard evaluation")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    modes = a.modes.split(",")
    if min(a.n, a.bs, a.repeats) <= 0 or a.warmup < 0:
        ap.error("n, bs and repeats must be positive; warmup must be nonnegative")
    if any(mode not in ("eager", "kv") for mode in modes):
        ap.error("modes must be eager or kv")
    if len(set(modes)) != len(modes):
        ap.error("modes must not repeat")

    cfg = OmegaConf.load(a.config)
    if not cfg.data or not cfg.data.get("test"):
        ap.error("this benchmark requires a conditional test set")
    device = default_device()
    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    ck = Path(a.checkpoint_dir) if a.checkpoint_dir else ROOT / "artifacts" / artifact_subdir(cfg.task)
    dd = Path(a.data_dir) if a.data_dir else ROOT / "data" / artifact_subdir(cfg.task)
    cfg.tokenizer.vqvae.checkpoint_path = str(ck / cfg.checkpoints.tokenizer)
    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    tk.vqvae.to(device).eval()
    net = NextConceptTransformer(tk.vqvae, cfg.model).to(device)
    generator_path = ck / cfg.checkpoints.generator
    checkpoint = torch.load(generator_path, map_location="cpu", weights_only=True)
    net.load_state_dict(checkpoint["model"], strict=True)
    net.eval()
    g = cfg.generation
    allowance = load_inference_tensor(
        str(g.frequency_allowance), ck, device=device, dtype=wide_float(device),
        checkpoint=checkpoint, checkpoint_path=generator_path,
    ) if g.get("frequency_allowance") else None
    del checkpoint
    adjacency = [float(x) for x in str(g.adjacency_penalty).split(",")] if g.get("adjacency_penalty") else None
    policy = dict(frequency=float(g.get("frequency_penalty", 0)), deficit=float(g.get("frequency_deficit", 0)),
                  allowance=allowance, adjacency=adjacency)
    guidance = a.cfg if a.cfg is not None else float(g.get("cfg_scale", 0))
    sampling = dict(temperature=float(g.get("temperature", 1)), top_k=int(g.get("top_k", 1)),
                    top_p=float(g.get("top_p", 0)), cfg_scale=guidance, cfg_null_eos=tk.pre_tokenizer.eos_token_id)

    with open(dd / cfg.data.test, "rb") as stream:
        items = pickle.load(stream)[:a.n]
    with open(dd / cfg.data.test_refs, "rb") as stream:
        refs = pickle.load(stream)[:a.n]
    if not items or len(items) != len(refs):
        raise ValueError("nonempty test inputs and references must align")
    source_tokenizer = AutoTokenizer.from_pretrained(str(cfg.tokenizer.pre_tokenizer.model_id)) if isinstance(items[0], str) else None
    P = int(cfg.model.prefix_len)
    prefixes = []
    for item in items:
        tokens = source_tokenizer(item)["input_ids"] if source_tokenizer else list(item)
        prefixes.append((tokens[:P] + [tk.pre_tokenizer.pad_token_id] * P)[:P])
    inputs = torch.tensor(prefixes, device=device)

    def sync():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        elif str(device).startswith("mps"):
            torch.mps.synchronize()

    @torch.no_grad()
    def run(batch, cached):
        codes = net.generate(None, batch, use_kv_cache=cached, **sampling)
        return codes, pick_tokens(tk.vqvae.decode(codes), **policy)

    saved = None
    if a.compare_shard:
        with open(a.compare_shard, "rb") as stream:
            part = pickle.load(stream)
        if part["idxs"] != list(range(len(items))):
            raise ValueError("comparison requires the same ordered single-shard inputs")
        saved = torch.tensor(part["raw_ids"])

    results, baseline = [], None
    measurements = {mode: ([], [], []) for mode in modes}
    for repeat in range(a.repeats):
        # Alternate order to reduce systematic first/second-mode timing effects.
        for mode in modes if repeat % 2 == 0 else modes[::-1]:
            durations, peaks, calls_per_batch = measurements[mode]
            torch.manual_seed(a.seed)
            for _ in range(a.warmup):
                run(inputs[:a.bs], mode == "kv")
            sync()
            gc.collect()
            torch.manual_seed(a.seed)
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
            calls = []
            hook = net.transformer.blocks[0].register_forward_hook(lambda *args: calls.append(1))
            outputs = []
            try:
                start = time.perf_counter()
                for offset in range(0, len(items), a.bs):
                    outputs.append(run(inputs[offset:offset + a.bs], mode == "kv"))
                sync()
                durations.append(time.perf_counter() - start)
                peaks.append(torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else None)
            finally:
                hook.remove()
            codes = torch.cat([out[0] for out in outputs]).cpu()
            tokens = torch.cat([out[1] for out in outputs]).cpu()
            del outputs
            calls_per_batch.append(len(calls) / ((len(items) + a.bs - 1) // a.bs))
            if baseline is None:
                baseline = (codes, tokens)
            if not torch.equal(codes, baseline[0]) or not torch.equal(tokens, baseline[1]):
                raise RuntimeError(f"{mode} repeat {repeat}: concept codes or decoded tokens differ from the first run")
            if saved is not None and not torch.equal(tokens, saved):
                raise RuntimeError("benchmark output differs from eval_shard.py")
    for mode in modes:
        durations, peaks, calls_per_batch = measurements[mode]
        row = dict(mode=mode, times_seconds=durations, median_seconds=statistics.median(durations),
                   peak_allocated_bytes=peaks, generator_calls_per_batch=calls_per_batch,
                   concepts_identical=True, tokens_identical=True,
                   matches_saved_shard=True if saved is not None else None)
        results.append(row)
        print(json.dumps(row), flush=True)

    report = dict(
        task=cfg.task, n=len(items), batch_size=a.bs, seed=a.seed, device=str(device),
        gpu=torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None,
        torch=torch.__version__, dtype=str(next(net.parameters()).dtype), tf32=False,
        git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        source_sha256={path: sha256(ROOT / path) for path in (
            "models/common.py", "models/next_concept.py", "models/quant.py",
            "models/vqvae.py", "evaluation/inference.py", "scripts/bench_kv.py",
        )},
        sampling=sampling, final_token_policy={k: v for k, v in policy.items() if k != "allowance"},
        config_sha256=sha256(a.config), generator_sha256=sha256(generator_path),
        tokenizer_sha256=sha256(ck / cfg.checkpoints.tokenizer),
        inputs_sha256=sha256(dd / cfg.data.test), results=results,
        mode_order="requested order on even repeats, reversed order on odd repeats",
        timing="generation + VQ decoding + final-token penalties; excludes loading, CPU copies and scoring",
    )
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
