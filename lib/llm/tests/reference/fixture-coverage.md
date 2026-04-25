# DIS-1850 — Fixture coverage matrix

Every V4 + V3.2 fixture maps to exactly one feature combination. The Jinja spike must satisfy all of them byte-for-byte.

## V4 (`lib/llm/tests/data/deepseek-v4/`)

| # | Shape | thinking_mode | n_msg | Roles | Tools | reasoning_content | tool_calls | latest_reminder | developer | task | wo_eos |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `{tools, messages}` | thinking | 5 | system, user, assistant, tool, assistant | yes (top-level) | yes (2x) | yes | no | no | no | no |
| 2 | `[messages]` | thinking | 5 | system, user, assistant, user, assistant | no | yes | no | no | no | no | no |
| 3 | `[messages]` | thinking | 6 | system, latest_reminder, developer, assistant, tool, assistant | (verify) | yes | yes | yes | yes | no | no |
| 4 | `[messages]` | chat | 6 | system, latest_reminder, user, assistant, user, assistant | no | (chat mode strips) | no | yes | no | action | no |

## V3.2 (`lib/llm/tests/data/deepseek-v3.2/`)

| File | Notes |
|---|---|
| `test_input.json` → `test_output.txt` | base case |
| `test_input_search_w_date.json` → `test_output_search_w_date.txt` | search variant with date in system |
| `test_input_search_wo_date.json` → `test_output_search_wo_date.txt` | search variant without date |

## Feature → fixture mapping (which fixture pins which trick-bit)

| Feature | First fixture | All fixtures that exercise it |
|---|---|---|
| BOS/EOS sentinels | 1 | all |
| `<｜User｜>`/`<｜Assistant｜>` markers | 1 | all |
| Tools system block | 1 | 1, 3 (V4); search variants (V3.2) |
| `<think>...</think>` reasoning preservation | 1 | 1, 2, 3 (V4) |
| `drop_thinking` on non-latest assistant turn | 2 | 2 |
| DSML tool_calls block (`tool_calls` in V4, `function_calls` in V3.2) | 1 | 1, 3 (V4); search-w-date (V3.2) |
| Tool result wrap (`<｜User｜><tool_result>...`) | 1 | 1, 3 |
| `latest_reminder` role | 3 | 3, 4 |
| `developer` role | 3 | 3 |
| `task` token (e.g. `<｜action｜>`) | 4 | 4 |
| chat mode (vs thinking) | 4 | 4 |
| Reasoning effort prefix | (none in fixtures — needs synthetic test) | — |
| `wo_eos` | (none in fixtures — needs synthetic test) | — |

## Gaps the existing fixtures don't cover

- `reasoning_effort = "max"` prefix at index 0
- `reasoning_effort = "high"` (no-op currently per encoding_dsv4.py:262)
- `wo_eos = true` on assistant messages
- `merge_tool_messages` — multi-tool round-trip
- `sort_tool_results_by_call_order` — out-of-order tool results

The spike should add synthetic fixtures for these once the existing 4 + 3 are all green.
