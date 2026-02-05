# Docs to Fern Migration Tracking

This document tracks open PRs that touch `docs/` and need corresponding changes in `fern/`.

**Last updated:** 2025-02-02

## PRs where ALL docs files have corresponding fern files ✅ COMMENTED

These are straightforward - just need to mirror the same changes to fern.

| PR # | Author | Title | Comment |
|------|--------|-------|---------|
| [#5890](https://github.com/ai-dynamo/dynamo/pull/5890) | @tmonty12 | docs: fix prometheusEndpoint helm value path | [Comment](https://github.com/ai-dynamo/dynamo/pull/5890#issuecomment-3837388470) |
| [#5815](https://github.com/ai-dynamo/dynamo/pull/5815) | @furionw | feat: EC E/PD workflow in TRT-LLM | [Comment](https://github.com/ai-dynamo/dynamo/pull/5815#issuecomment-3837388551) |
| [#5770](https://github.com/ai-dynamo/dynamo/pull/5770) | @atchernych | feat: Add EPP startup probe | [Comment](https://github.com/ai-dynamo/dynamo/pull/5770#issuecomment-3837388629) |
| [#5633](https://github.com/ai-dynamo/dynamo/pull/5633) | @dillon-cullinan | feat: Dockerfile templating | [Comment](https://github.com/ai-dynamo/dynamo/pull/5633#issuecomment-3837388730) |
| [#5631](https://github.com/ai-dynamo/dynamo/pull/5631) | @briefgaming | Use correct service name in kubectl portforward | [Comment](https://github.com/ai-dynamo/dynamo/pull/5631#issuecomment-3837388808) |
| [#5628](https://github.com/ai-dynamo/dynamo/pull/5628) | @tmonty12 | feat(operator): Add rolling update support | [Comment](https://github.com/ai-dynamo/dynamo/pull/5628#issuecomment-3837388896) |
| [#5595](https://github.com/ai-dynamo/dynamo/pull/5595) | @oandreeva-nv | revert: Support Dynamo KVBM with TRTLLM Disagg | [Comment](https://github.com/ai-dynamo/dynamo/pull/5595#issuecomment-3837388996) |
| [#5550](https://github.com/ai-dynamo/dynamo/pull/5550) | @zhongdaor-nv | chore: add top_p=1 recommendation | [Comment](https://github.com/ai-dynamo/dynamo/pull/5550#issuecomment-3837389099) |
| [#5536](https://github.com/ai-dynamo/dynamo/pull/5536) | @indrajit96 | feat: Make grpc startup dynamic | [Comment](https://github.com/ai-dynamo/dynamo/pull/5536#issuecomment-3837389184) |
| [#5527](https://github.com/ai-dynamo/dynamo/pull/5527) | @orangeng | docs: Fix service name in port-forward | [Comment](https://github.com/ai-dynamo/dynamo/pull/5527#issuecomment-3837389267) |
| [#5490](https://github.com/ai-dynamo/dynamo/pull/5490) | @renormalize | chore: upgrade to grove@v0.1.0-alpha.4 | [Comment](https://github.com/ai-dynamo/dynamo/pull/5490#issuecomment-3837389378) |
| [#5446](https://github.com/ai-dynamo/dynamo/pull/5446) | @atchernych | feat: Decomposed pipeline for EPP | [Comment](https://github.com/ai-dynamo/dynamo/pull/5446#issuecomment-3837389474) |
| [#5439](https://github.com/ai-dynamo/dynamo/pull/5439) | @dillon-cullinan | ci: Alternative workflow approach | [Comment](https://github.com/ai-dynamo/dynamo/pull/5439#issuecomment-3837389559) |
| [#5418](https://github.com/ai-dynamo/dynamo/pull/5418) | @sozercan | docs: fix aicBackendVersion documentation | [Comment](https://github.com/ai-dynamo/dynamo/pull/5418#issuecomment-3837389634) |
| [#5358](https://github.com/ai-dynamo/dynamo/pull/5358) | @muskansh-google | Update command to build Dynamo + SGLang | [Comment](https://github.com/ai-dynamo/dynamo/pull/5358#issuecomment-3837389718) |
| [#5104](https://github.com/ai-dynamo/dynamo/pull/5104) | @likian24 | fix(docs): formatting fix on kubernetes guide | [Comment](https://github.com/ai-dynamo/dynamo/pull/5104#issuecomment-3837389794) |

---

## PRs with MIXED files (some exist in fern, some don't) ✅ COMMENTED

| PR # | Author | Title | Comment |
|------|--------|-------|---------|
| [#5858](https://github.com/ai-dynamo/dynamo/pull/5858) | @YconquestY | feat: FlexKV integration in Dynamo | [Comment](https://github.com/ai-dynamo/dynamo/pull/5858#issuecomment-3837510907) |
| [#5786](https://github.com/ai-dynamo/dynamo/pull/5786) | @galletas1712 | feat: CRIU checkpoint/restore for vLLM | [Comment](https://github.com/ai-dynamo/dynamo/pull/5786#issuecomment-3837510970) |
| [#5671](https://github.com/ai-dynamo/dynamo/pull/5671) | @alec-flowers | refactor: move vllm multimodal examples | [Comment](https://github.com/ai-dynamo/dynamo/pull/5671#issuecomment-3837511565) |
| [#4978](https://github.com/ai-dynamo/dynamo/pull/4978) | @julienmancuso | feat: introducing ChReK | [Comment](https://github.com/ai-dynamo/dynamo/pull/4978#issuecomment-3837511040) |

### Details:

| PR # | New files needed | Suggested fern location | Sidebar update needed |
|------|------------------|------------------------|----------------------|
| #5858 | `flexkv_integration.md` | `fern/pages/backends/vllm/flexkv-integration.md` | Add to hidden "Backend Details > vLLM" in `next.yml` |
| #5786 | `checkpointing.md` | `fern/pages/kubernetes/checkpointing.md` | Add to "Kubernetes Deployment > Deployment Guide" in `next.yml` |
| #5671 | None (all exist) | N/A (note: folder is `nixl-connect` not `nixl_connect`) | None |
| #4978 | `chrek/` folder (3 files) | `fern/pages/kubernetes/chrek/` | Add to "Kubernetes Deployment > Deployment Guide" in `next.yml` |

---

## PRs where files DON'T exist in fern ✅ COMMENTED

| PR # | Author | Title | Comment |
|------|--------|-------|---------|
| [#5887](https://github.com/ai-dynamo/dynamo/pull/5887) | @indrajit96 | Ibhosale/epd perf | [Comment](https://github.com/ai-dynamo/dynamo/pull/5887#issuecomment-3837511093) |
| [#5876](https://github.com/ai-dynamo/dynamo/pull/5876) | @athreesh | docs: docs 3-tier restructure | [Comment](https://github.com/ai-dynamo/dynamo/pull/5876#issuecomment-3837511155) |
| [#5871](https://github.com/ai-dynamo/dynamo/pull/5871) | @JanelleCai | feat(mocker): pre-fetch model | [Comment](https://github.com/ai-dynamo/dynamo/pull/5871#issuecomment-3837511220) |
| [#5826](https://github.com/ai-dynamo/dynamo/pull/5826) | @mohammedabdulwahhab | feat: GMS shadow mode | [Comment](https://github.com/ai-dynamo/dynamo/pull/5826#issuecomment-3837511282) |
| [#5379](https://github.com/ai-dynamo/dynamo/pull/5379) | @athreesh | docs: Add K8s architecture diagrams | [Comment](https://github.com/ai-dynamo/dynamo/pull/5379#issuecomment-3837511337) |
| [#4031](https://github.com/ai-dynamo/dynamo/pull/4031) | @nv-kmcgill53 | docs: multinode multimodal support | [Comment](https://github.com/ai-dynamo/dynamo/pull/4031#issuecomment-3837511406) |
| [#3749](https://github.com/ai-dynamo/dynamo/pull/3749) | @mohammedabdulwahhab | Revert placement of log::init() | [Comment](https://github.com/ai-dynamo/dynamo/pull/3749#issuecomment-3837511478) |

### Details:

| PR # | docs/ files | Suggested fern/ location |
|------|-------------|--------------------------|
| #5887 | `docs/epd_trtllm_perf.md` | `fern/pages/backends/trtllm/epd-perf.md` |
| #5876 | `docs/design_docs/planner_design.md`, `docs/planner/*.md` | `fern/pages/design-docs/planner-design.md`, merge with existing `fern/pages/planner/` |
| #5871 | `docs/mocker/mocker.md` | `fern/pages/development/mocker.md` |
| #5826 | `docs/design/gms-shadow-mode.md` | `fern/pages/design-docs/gms-shadow-mode.md` |
| #5379 | `docs/kubernetes/architecture-*.md` | `fern/pages/kubernetes/architecture-*.md` |
| #4031 | `docs/backends/trtllm/multinode/multinode-multimodal-example.md` | `fern/pages/backends/trtllm/multinode/multinode-multimodal-example.md` |
| #3749 | `docs/guides/logging.md` | `fern/pages/observability/logging.md` (already exists - merge content) |

---

## PRs SKIPPED (no comment needed)

### Sphinx infrastructure only (no fern equivalent)

| PR # | Title | Files | Reason |
|------|-------|-------|--------|
| [#5790](https://github.com/ai-dynamo/dynamo/pull/5790) | refactor: make service protocol configurable | `docs/_extensions/github_alerts.py`, `docs/generate_docs.py` | Sphinx build scripts |
| [#5789](https://github.com/ai-dynamo/dynamo/pull/5789) | refactor: use tempfile module | `docs/_extensions/github_alerts.py`, `docs/generate_docs.py` | Sphinx build scripts |
| [#5777](https://github.com/ai-dynamo/dynamo/pull/5777) | refactor: optimize regex patterns | `docs/_extensions/github_alerts.py`, `docs/generate_docs.py` | Sphinx build scripts |
| [#5776](https://github.com/ai-dynamo/dynamo/pull/5776) | docs: add subresource integrity | `docs/_static/custom.js` | Sphinx static assets |
| [#5504](https://github.com/ai-dynamo/dynamo/pull/5504) | fix: tool call validation | `docs/frontends/openapi.json` | OpenAPI spec |

### Internal/non-user-facing docs

| PR # | Title | Files | Reason |
|------|-------|-------|--------|
| [#5878](https://github.com/ai-dynamo/dynamo/pull/5878) | docs: add metrics examples | `docs/observability/metrics_examples/*.log` | Log files, not user docs |
| [#5834](https://github.com/ai-dynamo/dynamo/pull/5834) | docs: add documentation templates | `docs/templates/*.md` (12 files) | Internal contributor templates |

---

## Summary

| Category | Count | Status |
|----------|-------|--------|
| All files have fern equivalents | 16 | ✅ Commented |
| Mixed (some exist, some new) | 4 | ✅ Commented |
| Files don't exist in fern | 7 | ✅ Commented |
| Sphinx infrastructure only | 5 | ⏭️ Skipped |
| Internal/non-user-facing | 2 | ⏭️ Skipped |
| **Total** | **34** | **27 commented, 7 skipped** |

---

## Already includes fern changes

These PRs already touch both `docs/` and `fern/`:

| PR # | Title |
|------|-------|
| [#5662](https://github.com/ai-dynamo/dynamo/pull/5662) | docs: Add TensorRT LLM Examples |
