<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tokenizer parity experiment

This standalone Rust check compares the flat token-ID output of HuggingFace
`tokenizers`, Fastokens, and Gigatoken on the same `tokenizer.json` and UTF-8
input. It exits unsuccessfully on any mismatch.

It complements — and does not modify or replace — the existing Dynamo benches
at `lib/llm/benches/tokenizer_simple.rs` and
`lib/llm/benches/tokenizer_dataset.rs`. This runner checks the same two input
shapes with Gigatoken as a third backend: `--simple` uses the legacy simple
input, while `--input` is the corpus/dataset scenario.

Gigatoken currently needs nightly Rust, so this experiment is deliberately not
part of Dynamo's production workspace or dependency graph.

```bash
cd benchmarks/tokenizer-comparison
cargo +nightly -Zprofile-rustflags run -- \
  --tokenizer /path/to/tokenizer.json \
  --simple
```

For the corpus/dataset scenario, pass the same flattened UTF-8 corpus to each
backend:

```bash
cargo +nightly -Zprofile-rustflags run -- \
  --tokenizer /path/to/tokenizer.json \
  --input /path/to/corpus.txt \
  --documents 1
```

`--documents` splits the input at UTF-8/newline boundaries; every backend
receives the same resulting documents. The command reports only parity and the
shared token count. It intentionally makes no performance claim.
