# Contribution and integration evidence

Validated on 23 October 2026 against upstream
`c70274273f489abc73832b2b98584a8483b6b203`.

## Contributions

- [PR #13](https://github.com/HazyResearch/ncm/pull/13): reconstruction snapshots,
  public reconstruction helpers, explicit final-resolution drafts, block transport
  and UTF-8 boundaries. Fifteen independent regression tests.
- [PR #14](https://github.com/HazyResearch/ncm/pull/14): shard integrity, malformed
  Sudoku outputs, independent scoring batch size and perplexity padding/coverage.
  Twelve independent regression tests.
- [Cached-generation regression commit](https://github.com/zolaski333/ncm/commit/fd6e6f6c9799cd0084f45e3fc690778d182477aa):
  separate coverage for the combination of upstream #4 and #5. This does not
  submit another cache implementation.

PR #13 and PR #14 are each based directly on upstream main. Their CPU GitHub
Actions jobs passed on Python 3.10 and 3.12:
[reconstruction CI](https://github.com/zolaski333/ncm/actions/runs/39866847244),
[evaluation CI](https://github.com/zolaski333/ncm/actions/runs/39866839900).
These are checks on the author's fork, not upstream merge approval.

The `codex/inference-reliability` integration branch additionally retains the
original commits from #4, #5 and #6. The integrated local suite passed 36 tests,
including CUDA checks. Forty-eight model/config combinations compare eager and
cached logits/codes across level-0 modes, positional encodings, grouped attention,
parallel blocks and guidance. Additional tests cover stochastic sampling and
recovery after an injected error.

The input-lifetime test fails on #5 alone with cyclic GC disabled and passes after
adding #4. This independently confirms that cache correctness and OmegaConf
retention are separate concerns. A separate padding probe against main counts
`[4, 2]` tokens for left-padded sequences of length 5 and 2; the corrected scorer
counts `[4, 1]`, matching individual next-token scoring.

## Released checkpoint parity

Environment: Windows, Python 3.10.11, PyTorch 2.12.1+cu126, NVIDIA GTX 1660 SUPER
6 GiB, driver 566.36, FP32, TF32 disabled. The baseline ran in a separate checkout
and process. No checkpoint weights were changed.

Checkpoints: `hazyresearch/ncm-checkpoints` revision
`d342ce8aaea58830eef3b8ad473664673e41f692`.
Data: `hazyresearch/ncm-datasets` revision
`6580ba41a4b0a976dde876fbe84d25327617b0e6`.
Downloads were checked against the repository's LFS SHA-256 values.

| Probe | Samples / batch | Original vs integrated eager | Original vs integrated cache |
|---|---:|---|---|
| Sudoku easy | first 64 / 8 | concepts, tokens and oracle reconstruction identical | identical |
| Sudoku hard | first 64 / 8 | concepts, tokens and oracle reconstruction identical | identical |
| TinyStories, seed 42 | 16 / 4 | concepts and tokens identical | identical |

On these Sudoku subsets, easy has 63/64 exact generated solutions and hard 52/64;
both tokenizers reconstruct 64/64 oracle solutions exactly. These are subset
checks, not estimates of full-test-set quality. The parity helper uses argmax
decoding; it is not a substitute for the task's configured final-token policy.

Commands and timing boundaries are documented in [validation.md](validation.md).
`scripts/validate_release.py` writes tensors and provenance JSON for comparisons.
Raw checkpoints, datasets and local output tensors are excluded from Git.

## Local timing evidence

Clean source revision `659647391e7bfad408ed5521b8a73dff26eb3ec6`; Sudoku easy,
first 64 inputs, batch 8, three repeats with alternating mode order. The benchmark
also required token identity with the saved evaluation shard. Every repeat had
identical concepts and decoded tokens and 12 generator calls per batch.

| Mode | Repeats, seconds | Median, seconds | Peak allocated CUDA memory |
|---|---|---:|---:|
| Eager | 5.6580, 5.6385, 5.3170 | 5.6385 | 216.53 MiB |
| Prefix KV cache | 1.9084, 1.9541, 1.3403 | 1.9084 | 242.28 MiB |

This includes generation, VQ decoding and the configured final-token policy;
loading, detokenization, CPU output copies and reference scoring are excluded.
The cache is faster in this run and uses 25.75 MiB more peak allocated memory.
Desktop timing variation is substantial. This small measurement does not establish
the paper's speedups, performance on other tasks, or production latency.

## Limits

No MPS hardware, H100 performance, full benchmark campaign, large GPT-2 reference
scoring run, new architecture or training recipe was validated. The public code
still lacks the training pipeline needed for a faithful training reproduction.
The fixes improve transport and contracts; they do not make VQ reconstruction
lossless or establish reliable source-code generation.
