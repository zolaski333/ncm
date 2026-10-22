import contextlib
import io

import torch
from omegaconf import OmegaConf

from models.vqvae import VQVAE
from models.next_concept import NextConceptTransformer


def tiny_vqvae(*, learned=False, quant_resi=True, dec_positional=False):
    config = OmegaConf.create({
        "clss": "multiscale_residual_vector_quantizer",
        "levels": {"use_manual_levels": True, "manual_levels": [1, 2, 4, 8]},
        "learned_l1_sampling": learned,
        "learned_l1_cascade": learned,
        "quant_resi": {"enabled": quant_resi, "ratio": 0.5, "share_mode": 0},
    })
    with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(io.StringIO()):
        torch.manual_seed(42)
        return VQVAE(
            in_vocab_size=17, context_length=8, embed_dim=8,
            vqvae_vocab_size=[8] * 4, n_layers=2,
            vector_quantizer_config=config, compression_factor=1,
            dropout=0.0, pre_quant_groupnorm=2,
            dec_positional=dec_positional,
        ).eval()


def tiny_generator(*, mode="query", rope=True, grouped=False, parallel=False):
    vqvae = tiny_vqvae(learned=True)
    query = mode == "query"
    config = OmegaConf.create({
        "n_embd": 16, "n_layer": 2, "n_head": 4,
        "n_kv_heads": 2 if grouped else 4,
        "bias": True, "dropout": 0.0, "prefix_len": 5,
        "vocab_size": [8] * (4 if query else 3),
        "use_level_encoding": True, "raw_token_prefix": True,
        "query_level0": query, "predict_level0": mode == "predict",
        "use_rope": rope, "use_qk_norm": True,
        "prefix_pad_token_id": 16, "parallel_block": parallel,
        "post_upsample_conv": {"enabled": True, "kernel_size": 3},
    })
    # contextual_prefix is deliberately absent, matching the released configs.
    with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(io.StringIO()):
        torch.manual_seed(123)
        return NextConceptTransformer(vqvae, config).eval()
