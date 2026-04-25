# Pinned upstream Python references

These are immutable snapshots of DeepSeek's official chat-encoding scripts, captured for the DIS-1850 spike. Do **not** modify — re-fetch from source if upstream changes.

## Files

| File | Source | Source SHA / version | Captured |
|---|---|---|---|
| `encoding_dsv4.py` | `huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/encoding/encoding_dsv4.py` | model SHA `5e74e6987d2007a261efc812d357ae75f9b8263a` (HF API `lastModified: 2026-04-24T10:00:14Z`) | 2026-04-24 |
| `encoding_dsv32.py` | `huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/encoding/encoding_dsv32.py` | (re-fetch & note SHA on next update) | 2026-04-24 |

## Why these are pinned

The Rust ports `lib/llm/src/preprocessor/prompt/deepseek_v4.rs` (~1271 LOC) and `deepseek_v32.rs` (~1353 LOC) are direct ports of these scripts. The DIS-1850 spike replaces them with `chat_template.jinja`. Parity is validated byte-by-byte against the test_output_*.txt fixtures already in `lib/llm/tests/data/deepseek-v{3.2,4}/`, which themselves are byte-validated against these scripts.

If upstream DeepSeek changes the encoding before the spike completes:
1. Re-fetch the scripts here.
2. Re-run the existing Rust-port tests to detect divergence.
3. Update fixtures and Jinja template accordingly.

## Modifications from upstream

A 4-line header has been **prepended** to each `.py` file to opt the
file out of repo-wide linting (`ruff`, `isort`). This is the only
modification — the rest of the file is byte-identical to the upstream
HF snapshot. When re-fetching, re-prepend the same 4-line header.

```python
# ruff: noqa
# isort: skip_file
# Pinned upstream snapshot — see lib/llm/tests/reference/SOURCES.md.
# Do not modify; re-fetch from HuggingFace if upstream changes.
```

## Re-fetch commands

```bash
curl -sL https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/encoding/encoding_dsv4.py \
  -o lib/llm/tests/reference/encoding_dsv4.py

curl -sL https://huggingface.co/deepseek-ai/DeepSeek-V3.2/raw/main/encoding/encoding_dsv32.py \
  -o lib/llm/tests/reference/encoding_dsv32.py
```
