#!/usr/bin/env python3
"""Generate a lognormal ISL workload JSONL for AIPerf benchmarking.

Samples input sequence lengths from a lognormal distribution, matching
the SOLBench shape (mean/median ratio ~9.3x). Each entry is a chat-format
request that will produce approximately the target ISL when tokenized.

Usage:
    python3 generate-lognormal-dataset.py [--out PATH] [--n N] [--seed S]
    python3 generate-lognormal-dataset.py --out /tmp/lognormal-1k.jsonl --n 1000

Output format: JSONL, one record per line:
    {"input": {"messages": [{"role": "user", "content": "<prompt>"}]}}

Compatible with: aiperf profile --input-file ... --custom-dataset-type custom_file
"""

import argparse
import json
import pathlib

import numpy as np

# Defaults retain the 9.3x mean/median ratio of the SOLBench distribution
# while fitting within an 8k context window (min 256, max 8192 tokens).
DEFAULT_MU = 7.0  # exp(7.0) ~= 1097 tokens (median)
DEFAULT_SIGMA = 1.5  # gives mean/median ~9.3x after clipping
DEFAULT_MIN_ISL = 256
DEFAULT_MAX_ISL = 8192
DEFAULT_N = 500
DEFAULT_SEED = 42

# Approximate chars-per-token for filler text padding to target ISL.
CHARS_PER_TOKEN = 8


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", default="/tmp/lognormal-isl.jsonl")
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--mu", type=float, default=DEFAULT_MU)
    p.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    p.add_argument("--min-isl", type=int, default=DEFAULT_MIN_ISL)
    p.add_argument("--max-isl", type=int, default=DEFAULT_MAX_ISL)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    raw = rng.lognormal(mean=args.mu, sigma=args.sigma, size=args.n)
    isls = np.clip(raw, args.min_isl, args.max_isl).astype(int)

    word = "analyze "
    records = []
    for isl in isls:
        n_words = max(1, int(isl * CHARS_PER_TOKEN) // len(word))
        prompt = (word * n_words).strip()
        records.append({"input": {"messages": [{"role": "user", "content": prompt}]}})

    out = pathlib.Path(args.out)
    out.write_text("\n".join(json.dumps(r) for r in records))

    print(f"Wrote {len(records)} requests to {out}")
    print(
        f"ISL stats: min={isls.min()} p50={int(np.median(isls))} "
        f"p90={int(np.percentile(isls, 90))} max={isls.max()} mean={int(isls.mean())}"
    )


if __name__ == "__main__":
    main()
