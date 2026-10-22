"""Save exact concept/token outputs from a chosen checkout and released weights.

Run in separate processes with --root pointing to each checkout, then compare the
saved tensors with torch.equal. This is a parity check, not a timing benchmark.
"""

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--data-dir", default="")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--kv-cache", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.n <= 0 or a.bs <= 0:
        ap.error("n and bs must be positive")
    root = Path(a.root).resolve()
    sys.path.insert(0, str(root))
    from models.next_concept import NextConceptTransformer
    from tokenization.vqvae_tokenizer import VQVAETokenizer
    from transformers import AutoTokenizer

    if a.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    config_path = root / a.config
    cfg = OmegaConf.load(config_path)
    ck = Path(a.checkpoint_dir).resolve()
    cfg.tokenizer.vqvae.checkpoint_path = str(ck / cfg.checkpoints.tokenizer)
    for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
        if cfg.checkpoints.get(name):
            cfg.tokenizer.pre_tokenizer[key] = str(ck / cfg.checkpoints[name])
    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    tk.vqvae.to(a.device).eval()
    net = NextConceptTransformer(tk.vqvae, cfg.model).to(a.device)
    checkpoint = torch.load(ck / cfg.checkpoints.generator, map_location="cpu", weights_only=True)
    net.load_state_dict(checkpoint["model"], strict=True)
    net.eval()
    unconditional = not cfg.data or not cfg.data.get("test")
    level0 = None
    if unconditional and cfg.generation.get("level0_unigram"):
        path, key = str(cfg.generation.level0_unigram).split(":", 1)
        if (ck / path).resolve() != (ck / cfg.checkpoints.generator).resolve():
            raise ValueError("validation expects the released generator's level-0 metadata")
        level0 = checkpoint
        for component in key.split("."):
            level0 = level0[component]
        level0 = level0.float().cpu()
    del checkpoint

    refs = None
    if unconditional:
        inputs = torch.full((a.n, 1), tk.pre_tokenizer.eos_token_id, device=a.device)
    else:
        dd = Path(a.data_dir)
        with open(dd / cfg.data.test, "rb") as stream:
            items = pickle.load(stream)[:a.n]
        with open(dd / cfg.data.test_refs, "rb") as stream:
            refs = pickle.load(stream)[:a.n]
        source_tokenizer = AutoTokenizer.from_pretrained(str(cfg.tokenizer.pre_tokenizer.model_id)) if isinstance(items[0], str) else None
        P = int(cfg.model.prefix_len)
        rows = []
        for item in items:
            tokens = source_tokenizer(item)["input_ids"] if source_tokenizer else list(item)
            rows.append((tokens[:P] + [tk.pre_tokenizer.pad_token_id] * P)[:P])
        inputs = torch.tensor(rows, device=a.device)

    torch.manual_seed(a.seed)
    concepts, tokens, oracle = [], [], []
    with torch.no_grad():
        for offset in range(0, len(inputs), a.bs):
            batch = inputs[offset:offset + a.bs]
            initial = None
            if level0 is not None:
                length = tk.vqvae.vq.levels[0]
                initial = torch.multinomial(level0, len(batch) * length, replacement=True).view(len(batch), length).to(a.device)
            codes = net.generate(initial, batch, use_kv_cache=a.kv_cache,
                                 temperature=float(cfg.generation.temperature), top_k=int(cfg.generation.top_k),
                                 top_p=float(cfg.generation.top_p))
            concepts.append(codes.cpu())
            tokens.append(tk.vqvae.decode(codes).argmax(-1).cpu())
            if refs is not None and not isinstance(refs[0], str):
                gold = torch.tensor(refs[offset:offset + a.bs], device=a.device)
                oracle.append(tk.vqvae.decode(tk.vqvae.encode(gold)).argmax(-1).cpu())

    result = dict(concepts=torch.cat(concepts), tokens=torch.cat(tokens))
    if oracle:
        result["oracle_reconstruction"] = torch.cat(oracle)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out)
    report = dict(
        task=cfg.task, n=len(inputs), batch_size=a.bs, seed=a.seed, cached=a.kv_cache,
        git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root)),
        torch=torch.__version__, device=a.device,
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        output_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        decoder_policy="argmax; this parity check does not compute paper task scores",
    )
    if oracle:
        gold = torch.tensor(refs)
        report["generated_sequences_exact"] = int((result["tokens"] == gold).all(1).sum())
        report["reconstructed_sequences_exact"] = int((result["oracle_reconstruction"] == gold).all(1).sum())
    out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
