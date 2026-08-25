# Random LoRA warm-pin TDD evidence

Source plan: none; this journey was derived from the PR review finding.

User journey: as a simulator user, I want a LoRA that remains resident after its arrival rate expires to retain a cold-start route, so that Random, HRW, and MCF report comparable routing churn and entry counts.

| # | What is guaranteed | Test | Type | Result | Evidence |
|---|---|---|---|---|---|
| 1 | Random retains a routing entry for a warm LoRA after the rate window and EMA decay expire. | `test_random_baseline_keeps_warm_lora_after_rate_window_expires` | Integration | PASS | `cargo test -p dynamo-llm --test lora_simulation -- test_random_baseline_keeps_warm_lora_after_rate_window_expires` |
| 2 | The complete LoRA simulation suite remains valid. | `lora_simulation` | Integration | PASS | `cargo test -p dynamo-llm --test lora_simulation` (19 passed, 1 ignored) |

RED evidence: the focused test failed before the implementation, reporting one LoRA removal on the final tick (`[... , 1]`) where the expected value was zero.

GREEN evidence: the same focused command passed after Random added one live loaded worker as an inactive pin, and the full simulation suite passed.

Validation: `cargo fmt --all`, `cargo test -p dynamo-llm --test lora_simulation`, and `cargo clippy -p dynamo-llm --test lora_simulation -- -D warnings` all passed.

Coverage: the repository does not have `cargo llvm-cov` installed, so no percentage was generated. The regression executes the exact expired-window branch and the full integration suite passed.
