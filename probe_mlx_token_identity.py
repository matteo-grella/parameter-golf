#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_unflatten

from train_gpt_mlx import COMPUTE_DTYPE, GPT, Hyperparameters, load_validation_tokens, rms_norm


def numpy_to_mx(arr: np.ndarray) -> mx.array:
    if arr.dtype == np.dtype("|V2"):
        return mx.array(arr.view(np.uint16), dtype=mx.bfloat16)
    return mx.array(arr)


def load_model(checkpoint: Path, args: Hyperparameters) -> GPT:
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        logit_chunk_tokens=args.logit_chunk_tokens,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init,
    )
    flat_state = {name: numpy_to_mx(value) for name, value in np.load(checkpoint).items()}
    model.update(tree_unflatten(list(flat_state.items())))
    return model


def collect_hidden_states(model: GPT, input_ids_np: np.ndarray) -> tuple[list[str], list[np.ndarray]]:
    input_ids = mx.array(input_ids_np, dtype=mx.int32)
    names: list[str] = []
    states: list[np.ndarray] = []

    x = rms_norm(model.tok_emb(input_ids).astype(COMPUTE_DTYPE))
    x0 = x
    skips: list[mx.array] = []
    names.append("embed_rms")
    states.append(np.asarray(x.astype(mx.float32)))

    for i in range(model.num_encoder_layers):
        x = model.blocks[i](x, x0)
        skips.append(x)
        names.append(f"block_{i + 1}")
        states.append(np.asarray(x.astype(mx.float32)))

    for i in range(model.num_decoder_layers):
        if skips:
            x = x + model.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
        x = model.blocks[model.num_encoder_layers + i](x, x0)
        names.append(f"block_{model.num_encoder_layers + i + 1}")
        states.append(np.asarray(x.astype(mx.float32)))

    x = model.final_norm(x)
    names.append("final_norm")
    states.append(np.asarray(x.astype(mx.float32)))
    return names, states


def sample_sequences(tokens: np.ndarray, seq_len: int, start_seq: int, num_seqs: int) -> np.ndarray:
    raw_start = start_seq * seq_len
    raw_end = raw_start + num_seqs * seq_len
    chunk = tokens[raw_start:raw_end]
    if chunk.size != num_seqs * seq_len:
        raise ValueError("not enough validation tokens for requested probe split")
    return chunk.reshape(num_seqs, seq_len)


def ridge_probe_accuracy(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    vocab_size: int,
    ridge: float,
) -> float:
    x_train = x_train.astype(np.float32, copy=False)
    x_val = x_val.astype(np.float32, copy=False)
    x_train_aug = np.concatenate([x_train, np.ones((x_train.shape[0], 1), dtype=np.float32)], axis=1)
    x_val_aug = np.concatenate([x_val, np.ones((x_val.shape[0], 1), dtype=np.float32)], axis=1)

    dim_aug = x_train_aug.shape[1]
    xtx = x_train_aug.T @ x_train_aug
    xtx = xtx + ridge * np.eye(dim_aug, dtype=np.float32)

    xty = np.zeros((dim_aug, vocab_size), dtype=np.float32)
    np.add.at(xty, (slice(None), y_train), x_train_aug.T)

    weights = np.linalg.solve(xtx, xty)
    logits = x_val_aug @ weights
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == y_val))


def tied_embedding_metrics(hidden: np.ndarray, labels: np.ndarray, emb_weight: np.ndarray) -> tuple[float, float]:
    logits = hidden @ emb_weight.T
    preds = np.argmax(logits, axis=1)
    top1 = float(np.mean(preds == labels))

    h_norm = hidden / np.clip(np.linalg.norm(hidden, axis=1, keepdims=True), 1e-8, None)
    e_lookup = emb_weight[labels]
    e_norm = e_lookup / np.clip(np.linalg.norm(e_lookup, axis=1, keepdims=True), 1e-8, None)
    self_cos = float(np.mean(np.sum(h_norm * e_norm, axis=1)))
    return top1, self_cos


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe current-token identity recoverability by layer.")
    parser.add_argument("--checkpoint", type=Path, default=Path("logs/mlx_smoke3_mlx_model.npz"))
    parser.add_argument("--train-seqs", type=int, default=8)
    parser.add_argument("--val-seqs", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1.0)
    args_cli = parser.parse_args()

    args = Hyperparameters()
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    model = load_model(args_cli.checkpoint, args)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)

    train_x_np = sample_sequences(val_tokens, args.train_seq_len, start_seq=0, num_seqs=args_cli.train_seqs)
    val_x_np = sample_sequences(val_tokens, args.train_seq_len, start_seq=args_cli.train_seqs, num_seqs=args_cli.val_seqs)

    layer_names, train_states = collect_hidden_states(model, train_x_np)
    _, val_states = collect_hidden_states(model, val_x_np)

    train_labels = train_x_np.reshape(-1)
    val_labels = val_x_np.reshape(-1)
    emb_weight = np.asarray(model.tok_emb.weight.astype(mx.float32))

    print(
        f"checkpoint={args_cli.checkpoint} seq_len={args.train_seq_len} "
        f"train_seqs={args_cli.train_seqs} val_seqs={args_cli.val_seqs}"
    )
    print("layer tied_top1 tied_cos linear_probe_top1")

    for name, train_state, val_state in zip(layer_names, train_states, val_states):
        train_hidden = train_state.reshape(-1, train_state.shape[-1])
        val_hidden = val_state.reshape(-1, val_state.shape[-1])
        tied_top1, tied_cos = tied_embedding_metrics(val_hidden, val_labels, emb_weight)
        linear_top1 = ridge_probe_accuracy(
            train_hidden,
            train_labels,
            val_hidden,
            val_labels,
            vocab_size=args.vocab_size,
            ridge=args_cli.ridge,
        )
        print(f"{name} {tied_top1:.4f} {tied_cos:.4f} {linear_top1:.4f}")


if __name__ == "__main__":
    main()
