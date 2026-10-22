# Reliability and validation

## Upstream dependencies

The integration branch includes the original commits from upstream PRs
[#4](https://github.com/HazyResearch/ncm/pull/4),
[#5](https://github.com/HazyResearch/ncm/pull/5), and
[#6](https://github.com/HazyResearch/ncm/pull/6), retaining their authorship.
These provide the OmegaConf retention fix, stateless prefix KV caching, and
CPU/MPS device support. The regression tests exercise their combined behaviour.
This is an integration checkout, not evidence that those PRs have merged upstream.

Training scripts were not present at the audited upstream revision
`c70274273f489abc73832b2b98584a8483b6b203`. No replacement training recipe or
new architecture is introduced here.

## Tests without checkpoints

```bash
python -m unittest discover -s tests -v
git diff --check
```

The suite runs on CPU without network access and enables additional CUDA checks
when available. It covers reconstruction contracts, UTF-8/block transport,
generation with queried/predicted/supplied level 0, padded prefixes, CFG,
RoPE/learned positions, grouped attention, parallel blocks, repeated calls without
cyclic GC, and recovery after a failed cached step.

Scoring tests include malformed Sudoku output, shard provenance and duplicate
indices, padding-side/batch invariance, and a small randomly initialized GPT-2.
They do not establish quality or speed on the released large checkpoints.

## Evaluation integrity

`evaluate.py` passes an explicit list of the newly written shard files and a run
identifier to the merger. Old matching files in the output directory are not
included. The merger checks aligned fields, conditional sample coverage, unique
indices, disjoint unconditional seed shards, and consistent configuration/policy.
Legacy `--glob` remains supported with structural checks; provenance is available
only for newly produced shards. Metadata records paths and config hashes, not an
immutable content hash of every checkpoint or dataset.

Sudoku cell accuracy always has 81 cells per reference in its denominator.
Missing cells count as wrong. Overlong or invalid-alphabet output is additionally
reported as malformed; exact-match and valid-grid rates remain separate.

Reference-model batch size is independent of generation batch size:

```bash
python evaluate.py --config config/eval/sudoku-easy.yaml \
    --checkpoint-dir artifacts/sudoku/easy --data-dir data/sudoku/easy \
    --n 64 --bs 8 --score-bs 1 --kv-cache --out eval/easy
```

Scoring disables unused KV output. Left-padded scoring uses unpadded positions
and excludes transitions from padding into the first real token. Examples with
no scored tokens have NaN per-example perplexity and are reported separately;
an entirely unscorable corpus raises an error. Corpus perplexity retains token
weighting. The FLM EOS convention is unchanged.

## Released-weight parity

`scripts/validate_release.py` saves concepts, decoded tokens, and, where available,
oracle reconstructions. It can execute another checkout's code in a separate
process using `--root`. Use the same checkpoint files, inputs, seed and batch size
for both executions, then compare each saved tensor with `torch.equal`.

```bash
python scripts/validate_release.py --root /path/to/baseline \
    --config config/eval/sudoku-easy.yaml --checkpoint-dir /path/to/artifacts/sudoku/easy \
    --data-dir /path/to/data/sudoku/easy --n 64 --bs 8 --out eval/baseline.pt
python scripts/validate_release.py \
    --config config/eval/sudoku-easy.yaml --checkpoint-dir artifacts/sudoku/easy \
    --data-dir data/sudoku/easy --n 64 --bs 8 --kv-cache --out eval/cached.pt
```

This helper uses argmax final decoding and is a parity probe, not a replacement
for task scoring with frequency/adjacency penalties. Tensor identity is meaningful;
serialized `.pt` file identity is not required.

## Timing

```bash
python scripts/bench_kv.py --config config/eval/sudoku-easy.yaml \
    --checkpoint-dir artifacts/sudoku/easy --data-dir data/sudoku/easy \
    --n 64 --bs 8 --repeats 3 --compare-shard eval/easy/shard_0.pkl \
    --json eval/easy-timing.json
```

Timing includes concept generation, VQ decoding, and the same final-token policy
as evaluation. It excludes loading, output copies to CPU, detokenization and
reference scoring. CUDA is synchronized and TF32 is disabled. The script records
all repeats, peak allocated CUDA memory, generator calls, output identity, source
and artifact hashes. It fails when compared outputs differ instead of reporting
a speedup without fidelity evidence. Mode order alternates between repetitions.

A supplied level 0 requires a separate prefix-prefill call in the current cached
implementation. Count this call explicitly; generator calls and hierarchy depth
are not interchangeable. CUDA peak allocated memory is not total device memory.
No MPS peak-memory claim is made.

Timing depends on device, batch, precision and task. A faster cache can consume
more memory. These scripts do not imply the paper's H100 speedups on another GPU.
