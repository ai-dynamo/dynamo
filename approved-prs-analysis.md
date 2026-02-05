# Approved PRs Analysis for ai-dynamo/dynamo

Generated: 2026-02-02

Total Open PRs: 253
Total Approved PRs: 38

---

## ✅ Ready to Merge (CI Passing) - 9 PRs

These are approved and have all CI checks passing. They can be merged immediately.

| PR | Title | Author | Last Updated |
|----|-------|--------|--------------|
| [#5867](https://github.com/ai-dynamo/dynamo/pull/5867) | feat: prefill tokens threshold based on max num batched tokens frac | PeaBrane | 2026-02-02 |
| [#5808](https://github.com/ai-dynamo/dynamo/pull/5808) | feat: Various Mocker Perf improvements + fixes | jthomson04 | 2026-02-02 |
| [#5713](https://github.com/ai-dynamo/dynamo/pull/5713) | fix: read block size from vllm at runtime | nealvaidya | 2026-01-30 |
| [#5567](https://github.com/ai-dynamo/dynamo/pull/5567) | feat: Added NIXL Telemetry prometheus port | alexanderbilk | 2026-01-26 |
| [#5527](https://github.com/ai-dynamo/dynamo/pull/5527) | docs: Fix service name in port-forward command | orangeng | 2026-02-02 |
| [#5085](https://github.com/ai-dynamo/dynamo/pull/5085) | fix: log runtime loss | ls-2018 | 2026-01-07 |
| [#5051](https://github.com/ai-dynamo/dynamo/pull/5051) | fix: fix missing update DynamoComponentReady condition | Monokaix | 2026-01-28 |
| [#4954](https://github.com/ai-dynamo/dynamo/pull/4954) | feat: GB200 GPT-oss disagg recipe | jthomson04 | 2026-01-19 |
| [#4821](https://github.com/ai-dynamo/dynamo/pull/4821) | fix: operator chart imagePullPolicy | ls-2018 | 2026-01-18 |
| [#4047](https://github.com/ai-dynamo/dynamo/pull/4047) | feat(fault-injection): Add test helper utilities and package setup | nv-oviya | 2025-11-24 |

---

## ⏳ Pending CI (Running/Queued) - 7 PRs

These are approved but CI is still running or pending. Check back later.

| PR | Title | Author | Status |
|----|-------|--------|--------|
| [#5894](https://github.com/ai-dynamo/dynamo/pull/5894) | docs: fix SGLang docs links (docs.sglang.ai → docs.sglang.io) | dagil-nvidia | CI pending |
| [#5854](https://github.com/ai-dynamo/dynamo/pull/5854) | feat: responses API compliance with upstream type alignment | ishandhanani | CI pending |
| [#5817](https://github.com/ai-dynamo/dynamo/pull/5817) | feat: expose Python Prometheus metric via DynamoComponentMetrics | keivenchang | CI pending |
| [#5530](https://github.com/ai-dynamo/dynamo/pull/5530) | fix: uv network timeout to be more resilient (part 2) | keivenchang | CI pending |

---

## ❌ CI Failing - Needs Attention - 22 PRs

These are approved but blocked by CI failures.

### Minor Fixes Needed (DCO, Label, Copyright issues):
| PR | Title | Author | Failure |
|----|-------|--------|---------|
| [#5873](https://github.com/ai-dynamo/dynamo/pull/5873) | feat: per dp rank gap detection | PeaBrane | copyright-checks |
| [#5626](https://github.com/ai-dynamo/dynamo/pull/5626) | fix(runtime): return 500 on LoRA load/unload errors | AmeenP | pre-commit, label |
| [#5608](https://github.com/ai-dynamo/dynamo/pull/5608) | feat: basic vllm omni pipeline support | ayushag-nv | copyright-checks |
| [#5399](https://github.com/ai-dynamo/dynamo/pull/5399) | fix: Updating GPU footprint in DS R1 disagg recipe | saurabh-nvidia | DCO (needs signed commit) |
| [#5418](https://github.com/ai-dynamo/dynamo/pull/5418) | docs: fix aicBackendVersion documentation | sozercan | label |
| [#4234](https://github.com/ai-dynamo/dynamo/pull/4234) | [DOCS] Restore LLM benchmarking guide | AsadShahid04 | DCO (needs signed commit) |

### Build/Test Failures (need investigation):
| PR | Title | Author | Failure |
|----|-------|--------|---------|
| [#5890](https://github.com/ai-dynamo/dynamo/pull/5890) | docs: fix prometheusEndpoint helm value path | tmonty12 | frontend-status-check |
| [#5882](https://github.com/ai-dynamo/dynamo/pull/5882) | fix: Cleanup instructions for GAIE integrations | atchernych | Build Frontend Image (arm64) |
| [#5875](https://github.com/ai-dynamo/dynamo/pull/5875) | feat(lora): Add lora aware routing hint | biswapanda | changed-files |
| [#5871](https://github.com/ai-dynamo/dynamo/pull/5871) | feat(mocker): pre-fetch model and staggered launches | JanelleCai | arm64 build, vllm-build-test |
| [#5722](https://github.com/ai-dynamo/dynamo/pull/5722) | feat: add optional source build for SGLang | vladnosiv | Trigger CI Pipeline, lychee |
| [#5714](https://github.com/ai-dynamo/dynamo/pull/5714) | feat: add EncoderCacheManager to TRT-LLM PrefillHandler | furionw | Build and Test - dynamo |
| [#5602](https://github.com/ai-dynamo/dynamo/pull/5602) | feat: default with lib/memory, media-nixl and kvbm | milesial | operator-build |
| [#5489](https://github.com/ai-dynamo/dynamo/pull/5489) | chore: remove legacy KVBM Rust implementation | furionw | GitLab CI |
| [#5433](https://github.com/ai-dynamo/dynamo/pull/5433) | fix: filter out syntax warning for TRTLLM import | furionw | Build and Test, sglang |
| [#5410](https://github.com/ai-dynamo/dynamo/pull/5410) | ci: Remove docker pull and comment multi gpu testing | pvijayakrish | vllm-build-test, GitLab CI |
| [#5342](https://github.com/ai-dynamo/dynamo/pull/5342) | fix: nats and etcd upgrade (CVEs) | Swipe4057 | Build and Test - dynamo |
| [#5066](https://github.com/ai-dynamo/dynamo/pull/5066) | test: refactor TCP server and add unit tests [4/n] | furionw | clippy, tests, vllm |
| [#5062](https://github.com/ai-dynamo/dynamo/pull/5062) | test: unit test TCP client create_response_stream [3/n] | furionw | GitLab CI |
| [#4705](https://github.com/ai-dynamo/dynamo/pull/4705) | feat: Start Discovery daemon lazily | atchernych | Build and Test, GitLab mirror |
| [#4662](https://github.com/ai-dynamo/dynamo/pull/4662) | feat: Unify TRTLLM Disagg Scripts | jthomson04 | Build and Test, pre-commit |
| [#4428](https://github.com/ai-dynamo/dynamo/pull/4428) | fix: KV-Router: degrade when indexer offline | vladnosiv | GitLab CI |
| [#3823](https://github.com/ai-dynamo/dynamo/pull/3823) | fix(vLLM): echo parameter being ignored | KrishnanPrash | GitLab CI, Trigger CI |

---

## 🚧 Skipped/Blocked CI - 1 PR

| PR | Title | Author | Status |
|----|-------|--------|--------|
| [#5802](https://github.com/ai-dynamo/dynamo/pull/5802) | fix(operator): correct structured logging in restart status tracking | julienmancuso | CI skipping (needs trigger) |

---

## Summary

| Status | Count | Action |
|--------|-------|--------|
| ✅ Ready to merge | 9 | **Merge now!** |
| ⏳ CI pending | 7 | Wait for CI |
| ❌ CI failing (minor) | 6 | Ask authors to fix DCO/copyright |
| ❌ CI failing (build) | 16 | Investigate or ask authors to rebase |

### Quick Win PRs (merge immediately):
```bash
# These 9 PRs have passing CI and are approved:
gh pr merge 5867 --repo ai-dynamo/dynamo --merge
gh pr merge 5808 --repo ai-dynamo/dynamo --merge
gh pr merge 5713 --repo ai-dynamo/dynamo --merge
gh pr merge 5567 --repo ai-dynamo/dynamo --merge
gh pr merge 5527 --repo ai-dynamo/dynamo --merge
gh pr merge 5085 --repo ai-dynamo/dynamo --merge
gh pr merge 5051 --repo ai-dynamo/dynamo --merge
gh pr merge 4954 --repo ai-dynamo/dynamo --merge
gh pr merge 4821 --repo ai-dynamo/dynamo --merge
gh pr merge 4047 --repo ai-dynamo/dynamo --merge
```
