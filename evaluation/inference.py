"""Shared checkpoint metadata and final-token policy for evaluation and timing."""

from pathlib import Path

import torch

from utils.misc import wide_float


def artifact_subdir(task):
    for family in ("sudoku", "lm1b"):
        if task.startswith(family + "-"):
            return str(Path(family) / task[len(family) + 1:])
    return task


def load_inference_tensor(spec, directory, *, device, dtype, checkpoint=None, checkpoint_path=None):
    path, separator, key = spec.partition(":")
    path = (Path(directory) / path).resolve()
    if checkpoint is not None and checkpoint_path is not None and path == Path(checkpoint_path).resolve():
        obj = checkpoint
    else:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    if separator:
        for component in key.split("."):
            obj = obj[component]
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"{spec} does not select a tensor")
    return obj.to(device=device, dtype=dtype)


def pick_tokens(logits, *, frequency=0.0, deficit=0.0, allowance=None, adjacency=None):
    """The released sequential frequency/adjacency policy, shared by both CLIs."""
    if frequency <= 0 and not adjacency:
        return logits.argmax(-1)
    B, T, V = logits.shape
    dtype = wide_float(logits.device)
    lg = logits.to(dtype)
    allow = (allowance[:V] * T)[None, :] if allowance is not None else None
    al = adjacency[0] if adjacency else 0.0
    aw = int(adjacency[1]) if adjacency and len(adjacency) > 1 else 1
    ad = adjacency[2] if adjacency and len(adjacency) > 2 else 1.0
    counts = torch.zeros(B, V, dtype=dtype, device=lg.device)
    output = torch.empty(B, T, dtype=torch.long, device=lg.device)
    history = []
    for t in range(T):
        step = lg[:, t, :].clone()
        if frequency > 0:
            excess = (counts - allow).clamp_min(0.0) if allow is not None else counts
            step -= frequency * excess
            if deficit > 0 and allow is not None:
                step += deficit * ((T - t) / T) * (allow - counts).clamp_min(0.0)
        for j, previous in enumerate(reversed(history[-aw:])):
            step.scatter_add_(1, previous[:, None], torch.full((B, 1), -al * ad ** j, dtype=dtype, device=lg.device))
        token = step.argmax(-1)
        output[:, t] = token
        history.append(token)
        counts.scatter_add_(1, token[:, None], torch.ones(B, 1, dtype=dtype, device=lg.device))
    return output
