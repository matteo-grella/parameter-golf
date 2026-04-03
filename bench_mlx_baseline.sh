#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

ITERATIONS="${ITERATIONS:-200}"
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-8192}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
RUN_PREFIX="${RUN_PREFIX:-mlx_base200_defval}"
SEEDS_CSV="${SEEDS_CSV:-1337,2024,31415}"
PARSE_ONLY="${PARSE_ONLY:-0}"

if [[ -n "${VAL_BATCH_SIZE:-}" ]]; then
  export VAL_BATCH_SIZE
  val_batch_desc="${VAL_BATCH_SIZE}"
else
  val_batch_desc="train_gpt_mlx.py default (524288)"
fi

echo "MLX baseline benchmark"
echo "python: ${PYTHON_BIN}"
echo "run_prefix: ${RUN_PREFIX}"
echo "iterations: ${ITERATIONS}"
echo "train_batch_tokens: ${TRAIN_BATCH_TOKENS}"
echo "val_loss_every: ${VAL_LOSS_EVERY}"
echo "val_batch_size: ${val_batch_desc}"
echo "seeds: ${SEEDS_CSV}"
echo "parse_only: ${PARSE_ONLY}"
echo

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"

run_names=()
for seed in "${SEEDS[@]}"; do
  seed="$(echo "${seed}" | xargs)"
  run_id="${RUN_PREFIX}_s${seed}"
  run_names+=("${run_id}")

  if [[ "${PARSE_ONLY}" == "1" ]]; then
    continue
  fi

  echo "=== ${run_id} ==="
  RUN_ID="${run_id}" \
  SEED="${seed}" \
  ITERATIONS="${ITERATIONS}" \
  TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS}" \
  VAL_LOSS_EVERY="${VAL_LOSS_EVERY}" \
  "${PYTHON_BIN}" train_gpt_mlx.py
  echo
done

RUNS_CSV="$(IFS=,; echo "${run_names[*]}")"
export RUNS_CSV

"${PYTHON_BIN}" - <<'PY'
import os
import pathlib
import re
import statistics

root = pathlib.Path.cwd()
log_dir = root / "logs"
runs = [run for run in os.environ["RUNS_CSV"].split(",") if run]

patterns = {
    "train": re.compile(r"step:(\d+)/(\d+) train_loss:([0-9.]+) train_time:(\d+)ms step_avg:([0-9.]+)ms tok_s:(\d+)"),
    "raw_val": re.compile(r"step:(\d+)/(\d+) val_loss:([0-9.]+) val_bpb:([0-9.]+) train_time:(\d+)ms step_avg:([0-9.]+)ms"),
    "size": re.compile(r"serialized_model_int8_zlib:(\d+) bytes"),
    "final": re.compile(r"final_int8_zlib_roundtrip_exact val_loss:([0-9.]+) val_bpb:([0-9.]+)"),
    "qeval": re.compile(r"final_int8_zlib_roundtrip val_loss:[0-9.]+ val_bpb:[0-9.]+ eval_time:(\d+)ms"),
}

rows = []
for run in runs:
    text = (log_dir / f"{run}.txt").read_text()
    train_matches = patterns["train"].findall(text)
    raw_val_matches = patterns["raw_val"].findall(text)
    if not train_matches or not raw_val_matches:
        raise SystemExit(f"missing expected metrics in logs/{run}.txt")
    train = train_matches[-1]
    raw_val = raw_val_matches[-1]
    size = patterns["size"].findall(text)[-1]
    final = patterns["final"].findall(text)[-1]
    qeval = patterns["qeval"].findall(text)[-1]
    rows.append(
        {
            "run": run,
            "seed": int(run.rsplit("_s", 1)[1]),
            "train_step": int(train[0]),
            "train_loss": float(train[2]),
            "train_time_ms": int(train[3]),
            "step_avg_ms": float(train[4]),
            "tok_s": int(train[5]),
            "raw_val_loss": float(raw_val[2]),
            "raw_val_bpb": float(raw_val[3]),
            "int8_bytes": int(size),
            "final_val_loss": float(final[0]),
            "final_val_bpb": float(final[1]),
            "q_eval_ms": int(qeval),
        }
    )

def mean_std(key: str) -> tuple[float, float]:
    values = [row[key] for row in rows]
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std

print("Summary")
print("run\tseed\ttrain_loss\traw_val_loss\tfinal_val_loss\tfinal_val_bpb\ttrain_time_s\tq_eval_s\tint8_bytes")
for row in rows:
    print(
        f"{row['run']}\t{row['seed']}\t{row['train_loss']:.4f}\t{row['raw_val_loss']:.4f}\t"
        f"{row['final_val_loss']:.8f}\t{row['final_val_bpb']:.8f}\t"
        f"{row['train_time_ms'] / 1000.0:.3f}\t{row['q_eval_ms'] / 1000.0:.3f}\t{row['int8_bytes']}"
    )

final_loss_mean, final_loss_std = mean_std("final_val_loss")
final_bpb_mean, final_bpb_std = mean_std("final_val_bpb")
raw_loss_mean, raw_loss_std = mean_std("raw_val_loss")
train_s_mean, train_s_std = mean_std("train_time_ms")
size_mean, size_std = mean_std("int8_bytes")

print()
print(f"final_int8_val_loss mean={final_loss_mean:.8f} std={final_loss_std:.8f}")
print(f"final_int8_val_bpb mean={final_bpb_mean:.8f} std={final_bpb_std:.8f}")
print(f"raw_val_loss mean={raw_loss_mean:.8f} std={raw_loss_std:.8f}")
print(f"train_time_s mean={train_s_mean / 1000.0:.3f} std={train_s_std / 1000.0:.3f}")
print(f"int8_bytes mean={size_mean:.0f} std={size_std:.0f}")
print()
print("Logs")
for row in rows:
    print(log_dir / f"{row['run']}.txt")
PY
