# Approved PRs Analysis for ai-dynamo/dynamo

Generated: 2026-02-02
**Last Verified: 2026-02-05**

Total Open PRs: 253
Total Approved PRs: 38

---

## ✅ Ready to Merge (CI Passing) - 8 PRs

These are approved, mergeable, and have `mergeStateStatus: CLEAN`. They can be merged immediately.

| PR | Title | Author | Last Updated |
|----|-------|--------|--------------|
| [#6003](https://github.com/ai-dynamo/dynamo/pull/6003) | docs: migrate Profiler docs to three-tier structure | dagil-nvidia | 2026-02-05 |
| [#5999](https://github.com/ai-dynamo/dynamo/pull/5999) | docs: migrate Multimodal docs to three-tier structure | dagil-nvidia | 2026-02-05 |
| [#5995](https://github.com/ai-dynamo/dynamo/pull/5995) | fix(sglang): remove apt-installed python3-blinker | KrishnanPrash | 2026-02-05 |
| [#5953](https://github.com/ai-dynamo/dynamo/pull/5953) | fix(recipes): correct GPU counts in DeepSeek-R1 READMEs | BenHamm | 2026-02-04 |
| [#5713](https://github.com/ai-dynamo/dynamo/pull/5713) | fix: read block size from vllm at runtime | nealvaidya | 2026-01-30 |
| [#5051](https://github.com/ai-dynamo/dynamo/pull/5051) | fix: fix missing update DynamoComponentReady condition | Monokaix | 2026-01-28 |
| [#4954](https://github.com/ai-dynamo/dynamo/pull/4954) | feat: GB200 GPT-oss disagg recipe | jthomson04 | 2026-01-19 |
| [#4821](https://github.com/ai-dynamo/dynamo/pull/4821) | fix: operator chart imagePullPolicy | ls-2018 | 2026-01-18 |

---

## 🎉 Already Merged - 2 PRs

These PRs from the previous list have been merged.

| PR | Title | Author | Merged |
|----|-------|--------|--------|
| [#5867](https://github.com/ai-dynamo/dynamo/pull/5867) | feat: prefill tokens threshold based on max num batched tokens frac | PeaBrane | ✅ Merged |
| [#5808](https://github.com/ai-dynamo/dynamo/pull/5808) | feat: Various Mocker Perf improvements + fixes | jthomson04 | ✅ Merged |

---

## ⚠️ Approved but Blocked/Unstable - 12 PRs

These PRs are approved but have `mergeStateStatus: BLOCKED` or `UNSTABLE` (failing CI checks).

| PR | Title | Author | Issue |
|----|-------|--------|-------|
| [#5946](https://github.com/ai-dynamo/dynamo/pull/5946) | feat: add config refactor and /dev/shm checkpoint/restore support | galletas1712 | **UNSTABLE** - GitLab CI failing |
| [#5941](https://github.com/ai-dynamo/dynamo/pull/5941) | chore: enable local indexers by default | PeaBrane | **BLOCKED** - Build and Test dynamo failing |
| [#5955](https://github.com/ai-dynamo/dynamo/pull/5955) | fix: enable 1.2.1 GAIE version for recipes | atchernych | **BLOCKED** - missing required checks |
| [#5908](https://github.com/ai-dynamo/dynamo/pull/5908) | docs: Update command to build Dynamo + SGLang container | muskansh-google | **BLOCKED** - missing required checks |
| [#5861](https://github.com/ai-dynamo/dynamo/pull/5861) | fix: reduce NATS consumer inactive_threshold | advpropsys | **BLOCKED** - Build and Test dynamo failing |
| [#5860](https://github.com/ai-dynamo/dynamo/pull/5860) | fix: reduce NATS stream max_age default | advpropsys | **BLOCKED** - Build and Test dynamo failing |
| [#5785](https://github.com/ai-dynamo/dynamo/pull/5785) | feat: nested mapper for KV indexing | PeaBrane | **BLOCKED** - CI checks failing |
| [#5777](https://github.com/ai-dynamo/dynamo/pull/5777) | refactor: optimize regex patterns in docs scripts | dagil-nvidia | **BLOCKED** - missing required checks |
| [#5774](https://github.com/ai-dynamo/dynamo/pull/5774) | chore: add --no-install-recommends to apt-get install | dagil-nvidia | **BLOCKED** - missing required checks |
| [#5758](https://github.com/ai-dynamo/dynamo/pull/5758) | ci: Add fern docs publish workflow | Jont828 | **BLOCKED** - missing required checks |
| [#5699](https://github.com/ai-dynamo/dynamo/pull/5699) | ci: update premerge to check the cargo-deny ban list | saturley-hall | **UNSTABLE** - GitLab CI failing |
| [#5567](https://github.com/ai-dynamo/dynamo/pull/5567) | feat: Added NIXL Telemetry prometheus port | alexanderbilk | **CONFLICTING** - needs rebase |
| [#5527](https://github.com/ai-dynamo/dynamo/pull/5527) | docs: Fix service name in port-forward command | orangeng | **BLOCKED** - missing required checks |
| [#5085](https://github.com/ai-dynamo/dynamo/pull/5085) | fix: log runtime loss | ls-2018 | Uses old CI workflows (needs rebase) |
| [#4047](https://github.com/ai-dynamo/dynamo/pull/4047) | feat(fault-injection): Add test helper utilities | nv-oviya | **Stale** - 3+ months old, missing current CI |

---

## ⏳ Pending CI (Running/Queued) - 4 PRs

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
| ✅ Ready to merge | 8 | **Merge now!** |
| 🎉 Already merged | 2 | Remove from tracking |
| ⚠️ Approved but blocked | 15 | Wait for CI or fix issues |
| ⏳ CI pending | 4 | Wait for CI |
| ❌ CI failing (minor) | 6 | Ask authors to fix DCO/copyright |
| ❌ CI failing (build) | 16 | Investigate or ask authors to rebase |

### Quick Win PRs (merge immediately):
```bash
# These 8 PRs have passing CI and are approved (mergeStateStatus: CLEAN):
gh pr merge 6003 --repo ai-dynamo/dynamo --merge
gh pr merge 5999 --repo ai-dynamo/dynamo --merge
gh pr merge 5995 --repo ai-dynamo/dynamo --merge
gh pr merge 5953 --repo ai-dynamo/dynamo --merge
gh pr merge 5713 --repo ai-dynamo/dynamo --merge
gh pr merge 5051 --repo ai-dynamo/dynamo --merge
gh pr merge 4954 --repo ai-dynamo/dynamo --merge
gh pr merge 4821 --repo ai-dynamo/dynamo --merge
```

### PRs needing author attention:
```bash
# These PRs need rebasing or fixes:
# #5567 - Has merge conflicts
# #5085 - Uses old CI workflows, needs rebase
# #4047 - Stale (3+ months), needs rebase for current CI
```

### New PRs to watch (approved, CI running):
- #5946 - config refactor (CI tests in progress)
- #5941 - enable local indexers (CI failing)
