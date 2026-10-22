import argparse, hashlib, os, pickle, sys
import torch
from omegaconf import OmegaConf
sys.path.insert(0, os.environ.get("NCM_ROOT", os.getcwd()))
from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer
from transformers import AutoTokenizer
from utils.misc import default_device, wide_float
from evaluation.inference import artifact_subdir, load_inference_tensor, pick_tokens

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--checkpoint-dir", default="")
ap.add_argument("--data-dir", default="")
ap.add_argument("--split", default="")
ap.add_argument("--n", type=int, default=0)
ap.add_argument("--bs", type=int, default=0)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--temp", type=float, default=None)
ap.add_argument("--top_k", type=int, default=None)
ap.add_argument("--top_p", type=float, default=None)
ap.add_argument("--cfg", type=float, default=None)
ap.add_argument("--freq", type=float, default=None)
ap.add_argument("--mu", type=float, default=None)
ap.add_argument("--adj", default=None)
ap.add_argument("--l0", default="", help="unigram (default), active, uniform, or smooth:<eps> (mix eps of uniform into the unigram)")
ap.add_argument("--out", required=True)
ap.add_argument("--kv-cache", action="store_true", help="encode the prefix once and reuse its keys/values at every level")
ap.add_argument("--run-id", default="", help="shared identity of shards launched by evaluate.py")
a = ap.parse_args()

CFG = OmegaConf.load(a.config)
DEV = default_device()
F64 = wide_float(DEV)
G = CFG.generation
ck = a.checkpoint_dir or os.path.join("artifacts", artifact_subdir(CFG.task))
dd = a.data_dir or os.path.join(os.environ.get("NCM_DATA_ROOT", "data"), artifact_subdir(CFG.task))
pick = lambda cli, key, cast, default: cast(cli if cli is not None else G.get(key, default))
temp = pick(a.temp, "temperature", float, 1.0)
top_k = pick(a.top_k, "top_k", int, 1)
top_p = pick(a.top_p, "top_p", float, 0.0)
cfg = pick(a.cfg, "cfg_scale", float, 0.0)
freq = pick(a.freq, "frequency_penalty", float, 0.0)
mu = pick(a.mu, "frequency_deficit", float, 0.0)
adj = str(a.adj if a.adj is not None else G.get("adjacency_penalty", "") or "")
split = a.split or CFG.eval.get("split", "test")
n = a.n or int(CFG.eval.get("n_docs", 0))
bs = a.bs or int(CFG.eval.batch_size)

CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck, CFG.checkpoints.tokenizer)
for key, name in (('vocab_file_path', 'bpe_vocab'), ('merges_file_path', 'bpe_merges')):
    if CFG.checkpoints.get(name):
        CFG.tokenizer.pre_tokenizer[key] = os.path.join(ck, CFG.checkpoints[name])
tk = VQVAETokenizer.from_config(CFG.tokenizer)
tk.vqvae = tk.vqvae.to(DEV).eval()
net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model).to(DEV)
generator_path = os.path.join(ck, CFG.checkpoints.generator)
sd = torch.load(generator_path, map_location="cpu", weights_only=True)
net.load_state_dict(sd["model"], strict=True)
net.eval()
P = int(CFG.model.prefix_len)
EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id


def load_allowance(spec):
    return load_inference_tensor(spec, ck, device=DEV, dtype=F64,
                                 checkpoint=sd, checkpoint_path=generator_path)


ALLOW = load_allowance(str(G.frequency_allowance)) if G.get("frequency_allowance") else None
L0_MODE = a.l0 or "unigram"
L0_P = load_allowance(str(G.level0_unigram)).float() if G.get("level0_unigram") else None
del sd
if L0_P is not None and L0_MODE == "active":
    L0_P = (L0_P > 0).float() / (L0_P > 0).sum()
elif L0_P is not None and L0_MODE == "uniform":
    L0_P = torch.ones_like(L0_P) / L0_P.numel()
elif L0_P is not None and L0_MODE.startswith("smooth:"):
    eps = float(L0_MODE.split(":", 1)[1])
    L0_P = (1.0 - eps) * L0_P + eps * torch.ones_like(L0_P) / L0_P.numel()
UNCOND = not CFG.data or not CFG.data.get(split)
ADJ = [float(v) for v in adj.split(",")] if adj else None

if UNCOND:
    seeds = [int(x) for x in CFG.eval.seeds][a.shard::a.nshards]
    n = a.n or int(CFG.eval.n_samples)
    items, refs, idxs, src_tok = [], [], [], None
else:
    items = pickle.load(open(os.path.join(dd, CFG.data[split]), "rb"))[:n]
    refs = pickle.load(open(os.path.join(dd, CFG.data[f"{split}_refs"]), "rb"))[:n]
    assert len(items) == len(refs), f"{len(items)} inputs vs {len(refs)} references"
    idxs = list(range(a.shard, len(items), a.nshards))
    src_tok = AutoTokenizer.from_pretrained(str(CFG.tokenizer.pre_tokenizer.model_id)) if isinstance(items[0], str) else None


def prefix(item):
    ids = src_tok(item)["input_ids"] if src_tok is not None else list(item)
    return (ids[:P] + [PAD] * P)[:P]


CUT_AT_EOS = bool(G.get("cut_at_eos", True))


def dec(ids):
    ids = list(ids)
    if CUT_AT_EOS and EOS in ids:
        ids = ids[:ids.index(EOS)]
    return tk.pre_tokenizer.decode([t for t in ids if t not in (EOS, PAD)], skip_special_tokens=True)


RAW = []


def run(x, l0=None):
    with torch.no_grad():
        g = net.generate(l0, x, temperature=temp, top_k=top_k, top_p=top_p, cfg_scale=cfg, cfg_null_eos=EOS, use_kv_cache=a.kv_cache)
        ids = pick_tokens(tk.vqvae.decode(g), frequency=freq, deficit=mu,
                          allowance=ALLOW, adjacency=ADJ).tolist()
        RAW.extend(ids)
        return [dec(r) for r in ids]


hyps, sample_seed = [], []
if UNCOND:
    L0 = tk.vqvae.vq.levels[0]
    for seed in seeds:
        torch.manual_seed(seed)
        for s in range(0, n, bs):
            B = min(bs, n - s)
            l0 = torch.multinomial(L0_P.cpu(), B * L0, replacement=True).view(B, L0).to(DEV) if L0_P is not None else None
            hyps += run(torch.full((B, 1), EOS, device=DEV), l0)
            sample_seed += [seed] * B
        print(f"  shard{a.shard} seed {seed}: {n} samples", flush=True)
    idxs = list(range(len(hyps)))
    toks = [tk.pre_tokenizer.encode(h) for h in hyps]
    out = {"idxs": idxs, "hyps": hyps, "refs": [""] * len(hyps), "seeds": sample_seed, "toks": toks, "raw_ids": RAW}
else:
    for s in range(0, len(idxs), bs):
        batch = idxs[s:s + bs]
        hyps += run(torch.tensor([prefix(items[i]) for i in batch], device=DEV))
        if a.shard == 0 and (s // bs) % 10 == 0:
            print(f"  shard0 {s}/{len(idxs)}", flush=True)
    out = {"idxs": idxs, "hyps": hyps, "refs": [r if isinstance(r, str) else dec(r) for r in (refs[i] for i in idxs)], "raw_ids": RAW}

with open(a.config, "rb") as config_file:
    config_hash = hashlib.sha256(config_file.read()).hexdigest()
out["metadata"] = {
    "run_id": a.run_id, "shard": a.shard, "nshards": a.nshards,
    "config_sha256": config_hash, "checkpoint_dir": os.path.abspath(ck),
    "data_dir": os.path.abspath(dd), "split": split, "n": n,
    "n_inputs": None if UNCOND else len(items),
    "policy": dict(temperature=temp, top_k=top_k, top_p=top_p, cfg=cfg,
                   frequency=freq, deficit=mu, adjacency=adj, level0=L0_MODE,
                   kv_cache=a.kv_cache, batch_size=bs),
}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
pickle.dump(out, open(a.out, "wb"))
print(f"shard {a.shard}: {len(hyps)} -> {a.out}")
