/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * releases.data.ts — single source of truth for the Reference pages
 * (Compatibility, Release Artifacts, Model Early Access Builds).
 *
 * Every value here is transcribed from the authoritative reference pages on
 * main (docs/reference/support-matrix.md, feature-matrix.md,
 * release-artifacts.md, model-early-access-builds.md).
 *
 * PER-RELEASE BUMP CHECKLIST (a release touches more than this file):
 *   1. This file: add the RELEASES entry (pins, ucx, date, delta,
 *      notesSummary, notesHref), CUDA_HISTORY rows, ARTIFACTS tags/versions, MAIN_TOT,
 *      CURRENT_* consts, MODEL_EA_BUILDS, and the RELEASE_STATS entry
 *      (counts from the GitHub body) as applicable.
 *   2. New page reference/release-notes/vX-Y-Z.mdx (ingest the GitHub body;
 *      ReleaseHeader and the UpgradePanel readingList read their counts
 *      from RELEASE_STATS — no per-page count props).
 *   3. reference/general/releases/known-issues.mdx + reference/general/releases/deprecations.mdx: new vXYZ
 *      section + accordion retitles (titles read RELEASE_STATS).
 *   4. Nav: docs/fern/index.yml Release Notes section (+ explicit slug).
 *   5. Regenerate agent twins: python3 scripts/gen_llms_tables.py
 *      (--check must pass afterwards).
 *
 * PARSER NOTE: scripts/gen_llms_tables.py parses this file with a
 * conservative literal parser — keep it a disciplined literal (no computed
 * values, spreads, calls, or ternaries); see the PARSER CONTRACT in that
 * script. The parser fails closed on anything it does not understand.
 */

export type ReleaseKind = "stable" | "patch" | "platform-preview" | "model-build";

export interface BackendPins {
  sglang?: string;
  trtllm?: string;
  vllm?: string;
  nixlSglang?: string;
  nixlTrtllm?: string;
  nixlVllm?: string;
  pinsNote?: string;
}

export interface Release {
  version: string;
  date?: string;
  kind: ReleaseKind;
  github?: string;
  docs?: string;
  /** Docs-native release notes page (absolute site path); GitHub link used when absent. */
  notesHref?: string;
  pins?: BackendPins;
  /** PyPI wheel version when it differs from the container tag (e.g. a .post
   *  rebuild). Falls back to the container version when absent. */
  wheel?: string;
  /** UCX version shipped with the release's NIXL builds — from the release's
   *  Key Dependencies table; omitted where the source never stated one
   *  (v1.0.0 and patch releases). */
  ucx?: string;
  delta?: string;
  note?: string;
  /** Feature-voice one-liner for the Release Notes timeline (stable releases);
   *  composed from the release page's Highlights themes. */
  notesSummary?: string;
  partial?: boolean;
}

export const CURRENT_VERSION = "v1.4.0";
export const CURRENT_DATE = "Aug 14, 2026";
export const CURRENT_TAG = "1.4.0";
export const CURRENT_WHEEL = "1.4.0";

export const MAIN_TOT: BackendPins = {
  sglang: "0.5.17",
  trtllm: "1.3.0rc24",
  vllm: "0.27.1",
  nixlSglang: "1.3.2",
  nixlTrtllm: "1.3.1",
  nixlVllm: "1.3.2",
};

const GH = "https://github.com/ai-dynamo/dynamo/releases/tag/";

export const RELEASES: Release[] = [
  {
    version: "v1.4.0",
    notesHref: "/dynamo/dev/reference/releases/v1-4-0",
    date: "Aug 14, 2026",
    kind: "stable",
    github: `${GH}v1.4.0`,
    docs: "https://docs.nvidia.com/dynamo",
    pins: { sglang: "0.5.16", trtllm: "1.3.0rc22", vllm: "0.26.0", nixlSglang: "1.3.0", nixlTrtllm: "1.3.1", nixlVllm: "1.3.2" },
    ucx: "1.21.x",
    delta:
      "Audit subsystem migrated into request trace (DYN_AUDIT_* honored as legacy aliases); HTTP header capture in trace records is an explicit fail-closed allowlist; deprecated multimodal worker flags and vLLM worker-role flags removed; runtime images no longer bundle software video decoders (H.264/H.265 decodes via NVDEC); UCX 1.21.x.",
    notesSummary:
      "Experimental cross-datacenter prefix routing and reservation replay in the Router, a vLLM-compatible generate token endpoint, tokenizer L1 prefix cache on by default, NIXL disaggregation for vLLM-Omni pipelines, and the Spica deployment simulator.",
  },
  {
    version: "v1.3.1",
    notesHref: "/dynamo/dev/reference/releases/v1-3-0#v131",
    date: "Aug 5, 2026",
    kind: "patch",
    github: `${GH}v1.3.1`,
    docs: "https://docs.nvidia.com/dynamo",
    pins: { sglang: "0.5.14", trtllm: "1.3.0rc19", vllm: "0.23.0", nixlSglang: "1.3.2", nixlTrtllm: "1.0.1", nixlVllm: "1.1.0" },
    ucx: "1.20.x",
    delta:
      "Patch release. Fixes disaggregated SGLang serving over AWS EFA on GB200: the SGLang EFA runtime moves to NIXL 1.3.2 and all three EFA images to EFA Installer 1.49.0. Backend pins are otherwise unchanged from v1.3.0.",
  },
  {
    version: "v1.3.0",
    notesHref: "/dynamo/dev/reference/releases/v1-3-0",
    date: "Jul 20, 2026",
    kind: "stable",
    github: `${GH}v1.3.0`,
    docs: "https://docs.nvidia.com/dynamo",
    pins: { sglang: "0.5.14", trtllm: "1.3.0rc19", vllm: "0.23.0", nixlSglang: "1.3.0", nixlTrtllm: "1.0.1", nixlVllm: "1.1.0" },
    wheel: "1.3.0.post1",
    ucx: "1.20.x",
    delta:
      "CUDA 12 container images discontinued; EFA variants retagged from -efa-amd64 to -efa (the images were already multi-arch — the old suffix was misleading); GA wheels published as 1.3.0.post1 (containers stay :1.3.0); UCX 1.20.x.",
    notesSummary:
      "Tool-calling and reasoning overhaul, RL rollout serving, the largest Router buildout to date, SLA-driven Planner autoscaling, and production GPU Memory Service on Kubernetes.",
  },
  {
    version: "v1.3.0-dev.1",
    date: "Jun 9, 2026",
    kind: "platform-preview",
    github: `${GH}v1.3.0-dev.1`,
    pins: { sglang: "0.5.12.post1", trtllm: "1.3.0rc17", vllm: "0.22.0", nixlSglang: "1.0.1", nixlTrtllm: "0.10.1", nixlVllm: "1.1.0" },
    delta:
      "Full-platform preview of v1.3.0: complete runtime matrix, wheels on pypi.nvidia.com, crates, and Helm charts. Superseded by v1.3.0 GA.",
  },
  {
    version: "v1.2.1",
    notesHref: "/dynamo/dev/reference/releases/v1-2-0",
    date: "Jun 13, 2026",
    kind: "patch",
    github: `${GH}v1.2.1`,
    docs: "https://docs.nvidia.com/dynamo",
    pins: { sglang: "0.5.11", trtllm: "1.3.0rc14", vllm: "0.20.1", nixlSglang: "1.0.1", nixlTrtllm: "0.10.1", nixlVllm: "0.10.1" },
    delta: "Patch release. Same backend pins as v1.2.0.",
  },
  {
    version: "v1.2.0",
    notesHref: "/dynamo/dev/reference/releases/v1-2-0",
    date: "Jun 2, 2026",
    kind: "stable",
    github: `${GH}v1.2.0`,
    docs: "https://docs.nvidia.com/dynamo",
    pins: { sglang: "0.5.11", trtllm: "1.3.0rc14", vllm: "0.20.1", nixlSglang: "1.0.1", nixlTrtllm: "0.10.1", nixlVllm: "0.10.1" },
    ucx: "1.20.0",
    delta:
      "603 PRs from 82 authors. DGD/DGDR promoted to v1beta1; CRTC default approximate KV router; inter-pod GMS sidecar; Dynamo Snapshot on CRI-O / OpenShift; UCX 1.20.0.",
    notesSummary:
      "DGD/DGDR v1beta1, CRTC as the default KV router, inter-pod GPU Memory Service, Dynamo Snapshot on CRI-O/OpenShift, and DeepSeek-V4 recipes on vLLM.",
  },
  {
    version: "v1.2.0-deepseek-v4-dev.3",
    date: "May 9, 2026",
    kind: "model-build",
    github: `${GH}v1.2.0-deepseek-v4-dev.3`,
    pins: { sglang: "upstream DSv4 preview", vllm: "0.20.1", nixlVllm: "0.10.1" },
    partial: true,
    note: "DeepSeek-V4 Blackwell preview; vLLM + SGLang containers only.",
  },
  {
    version: "v1.2.0-deepseek-v4-dev.2",
    date: "May 1, 2026",
    kind: "model-build",
    github: `${GH}v1.2.0-deepseek-v4-dev.2`,
    pins: { sglang: "upstream DSv4 preview", vllm: "0.20.0", nixlVllm: "0.10.1" },
    partial: true,
    note: "DeepSeek-V4 Blackwell preview; vLLM + SGLang containers only.",
  },
  {
    version: "v1.1.1",
    notesHref: "/dynamo/dev/reference/releases/v1-1-0",
    date: "May 5, 2026",
    kind: "patch",
    github: `${GH}v1.1.1`,
    docs: "https://docs.nvidia.com/dynamo",
    pins: { sglang: "0.5.10.post1", trtllm: "1.3.0rc11", vllm: "0.19.0", nixlSglang: "1.0.1", nixlTrtllm: "0.10.1", nixlVllm: "0.10.1" },
    delta: "Patch release. Same backend pins as v1.1.0.",
  },
  {
    version: "v1.1.0",
    notesHref: "/dynamo/dev/reference/releases/v1-1-0",
    date: "May 1, 2026",
    kind: "stable",
    github: `${GH}v1.1.0`,
    docs: "https://docs.nvidia.com/dynamo",
    pins: { sglang: "0.5.10.post1", trtllm: "1.3.0rc11", vllm: "0.19.0", nixlSglang: "1.0.1", nixlTrtllm: "0.10.1", nixlVllm: "0.10.1" },
    ucx: "1.20",
    delta:
      "Planner split into its own dynamo-planner image (artifact boundary change). First 1.y.z publication of dynamo-protocols on crates.io; dynamo-async-openai deprecated at final 1.0.2.",
    notesSummary:
      "Resilient KV routing at scale, Anthropic Messages API support, performance modeling and offline replay, and the multimodal embedding cache.",
  },
  {
    version: "v1.1.0-dev.3",
    date: "Apr 18, 2026",
    kind: "platform-preview",
    github: `${GH}v1.1.0-dev.3`,
    pins: { sglang: "0.5.10.post1", trtllm: "1.3.0rc11", vllm: "0.19.0", nixlSglang: "1.0.1", nixlTrtllm: "0.10.1", nixlVllm: "0.10.1" },
    partial: true,
    note: "Partial platform preview: TRT-LLM runtime image + wheels only.",
  },
  {
    version: "v1.1.0-dev.2",
    date: "Apr 9, 2026",
    kind: "platform-preview",
    github: `${GH}v1.1.0-dev.2`,
    pins: { sglang: "0.5.9", trtllm: "1.3.0rc9", vllm: "0.19.0", nixlSglang: "1.0.1", nixlTrtllm: "0.10.1", nixlVllm: "0.10.1" },
    partial: true,
    note: "Partial platform preview: SGLang + TRT-LLM runtime images + wheels.",
  },
  {
    version: "v1.1.0-dev.1",
    date: "Mar 17, 2026",
    kind: "platform-preview",
    github: `${GH}v1.1.0-dev.1`,
    pins: { sglang: "0.5.9", trtllm: "1.3.0rc5.post1", vllm: "0.17.1", nixlSglang: "1.0.1", nixlTrtllm: "0.10.1", nixlVllm: "0.10.1" },
    note: "Platform preview: runtime matrix, wheels on pypi.nvidia.com, Helm charts.",
  },
  {
    version: "v1.0.2",
    notesHref: "/dynamo/dev/reference/releases/v1-0-0",
    date: "Apr 22, 2026",
    kind: "patch",
    github: `${GH}v1.0.2`,
    docs: "https://docs.nvidia.com/dynamo",
    pins: { sglang: "0.5.9", trtllm: "1.3.0rc5.post1", vllm: "0.16.0", nixlSglang: "0.10.1", nixlTrtllm: "0.10.1", nixlVllm: "0.10.1" },
    delta: "No artifact additions or removals versus v1.0.0.",
  },
  {
    version: "v1.0.1",
    notesHref: "/dynamo/dev/reference/releases/v1-0-0",
    date: "Mar 16, 2026",
    kind: "patch",
    github: `${GH}v1.0.1`,
    docs: "https://docs.nvidia.com/dynamo",
    pins: { sglang: "0.5.9", trtllm: "1.3.0rc5.post1", vllm: "0.16.0", nixlSglang: "0.10.1", nixlTrtllm: "0.10.1", nixlVllm: "0.10.1" },
    delta: "No artifact additions or removals versus v1.0.0.",
  },
  {
    version: "v1.0.0",
    notesHref: "/dynamo/dev/reference/releases/v1-0-0",
    date: "Mar 12, 2026",
    kind: "stable",
    github: `${GH}v1.0.0`,
    docs: "https://docs.nvidia.com/dynamo",
    pins: { sglang: "0.5.9", trtllm: "1.3.0rc5.post1", vllm: "0.16.0", nixlSglang: "0.10.1", nixlTrtllm: "0.10.1", nixlVllm: "0.10.1" },
    delta:
      "snapshot-agent image and EFA variants for vLLM and TensorRT-LLM. First publish of dynamo-mocker and dynamo-kv-router crates. snapshot Helm chart added (preview); deprecated dynamo-crds dropped from the publish stream.",
    notesSummary:
      "First GA release: unified configuration, Kubernetes production readiness, multimodal serving, and the agents surface.",
  },
  {
    version: "v0.9.1",
    date: "Mar 4, 2026",
    kind: "patch",
    github: `${GH}v0.9.1`,
    docs: "https://docs.nvidia.com/dynamo",
    pins: { sglang: "0.5.8", trtllm: "1.3.0rc3", vllm: "0.14.1", nixlSglang: "0.9.0", nixlTrtllm: "0.9.0", nixlVllm: "0.9.0" },
    delta: "No artifact additions or removals versus v0.9.0.",
  },
  {
    version: "v0.9.0",
    date: "Feb 11, 2026",
    kind: "stable",
    github: `${GH}v0.9.0`,
    pins: { sglang: "0.5.8", trtllm: "1.3.0rc1", vllm: "0.14.1", nixlSglang: "0.9.0", nixlTrtllm: "0.9.0", nixlVllm: "0.9.0" },
    delta: "First publish of dynamo-tokens crate. Deprecated dynamo-graph Helm chart dropped from the publish stream.",
  },
  {
    version: "v0.8.1.post3",
    kind: "patch",
    pins: { sglang: "0.5.6.post2", trtllm: "1.2.0rc6.post3", vllm: "0.12.0", nixlSglang: "0.8.0", nixlTrtllm: "0.8.0", nixlVllm: "0.8.0" },
    note: "Post-train of v0.8.1: republished the TensorRT-LLM runtime image and PyPI wheels only, with TRT-LLM pinned to 1.2.0rc6.post3. Same CUDA support as v0.8.1.",
  },
  {
    version: "v0.8.1.post2",
    kind: "patch",
    pins: { sglang: "0.5.6.post2", trtllm: "1.2.0rc6.post2", vllm: "0.12.0", nixlSglang: "0.8.0", nixlTrtllm: "0.8.0", nixlVllm: "0.8.0" },
    note: "Post-train of v0.8.1: republished the TensorRT-LLM runtime image and PyPI wheels only, with TRT-LLM pinned to 1.2.0rc6.post2. Same CUDA support as v0.8.1.",
  },
  {
    version: "v0.8.1.post1",
    kind: "patch",
    pins: { sglang: "0.5.6.post2", trtllm: "1.2.0rc6.post1", vllm: "0.12.0", nixlSglang: "0.8.0", nixlTrtllm: "0.8.0", nixlVllm: "0.8.0" },
    note: "Post-train of v0.8.1: republished the TensorRT-LLM runtime image and PyPI wheels only, with TRT-LLM pinned to 1.2.0rc6.post1. Same CUDA support as v0.8.1.",
  },
  {
    version: "v0.8.1",
    date: "Jan 23, 2026",
    kind: "patch",
    github: `${GH}v0.8.1`,
    pins: { sglang: "0.5.6.post2", trtllm: "1.2.0rc6.post1", vllm: "0.12.0", nixlSglang: "0.8.0", nixlTrtllm: "0.8.0", nixlVllm: "0.8.0" },
    delta: "Post trains .post1/.post2/.post3 republished the TRT-LLM runtime image and PyPI wheels only; each carried a distinct TRT-LLM pin (see the v0.8.1.post1/.post2/.post3 rows).",
  },
  {
    version: "v0.8.0",
    date: "Jan 15, 2026",
    kind: "stable",
    github: `${GH}v0.8.0`,
    pins: { sglang: "0.5.6.post2", trtllm: "1.2.0rc6.post1", vllm: "0.12.0", nixlSglang: "0.8.0", nixlTrtllm: "0.8.0", nixlVllm: "0.8.0" },
    delta: "dynamo-frontend image and CUDA 13 variants for vLLM and SGLang. First publish of dynamo-memory and dynamo-config crates.",
  },
  {
    version: "v0.7.1",
    date: "Dec 15, 2025",
    kind: "patch",
    github: `${GH}v0.7.1`,
    pins: { sglang: "0.5.4.post3", trtllm: "1.2.0rc3", vllm: "0.11.0", nixlSglang: "0.8.0", nixlTrtllm: "0.8.0", nixlVllm: "0.8.0" },
  },
  {
    version: "v0.7.0.post1",
    kind: "patch",
    pins: { sglang: "0.5.4.post3", trtllm: "1.2.0rc3", vllm: "0.11.0", nixlSglang: "0.8.0", nixlTrtllm: "0.8.0", nixlVllm: "0.8.0" },
    note: "Post-train of v0.7.0: TensorRT-LLM pin advanced to 1.2.0rc3 (v0.7.0 shipped 1.2.0rc2). Same CUDA support as v0.7.0.",
  },
  {
    version: "v0.7.0",
    date: "Nov 26, 2025",
    kind: "stable",
    github: `${GH}v0.7.0`,
    pins: { sglang: "0.5.4.post3", trtllm: "1.2.0rc2", vllm: "0.11.0", nixlSglang: "0.8.0", nixlTrtllm: "0.8.0", nixlVllm: "0.8.0" },
  },
  {
    version: "v0.6.1.post1",
    kind: "patch",
    pins: { sglang: "0.5.3.post2", trtllm: "1.1.0rc5", vllm: "0.11.0", nixlSglang: "0.6.0", nixlTrtllm: "0.6.0", nixlVllm: "0.6.0" },
    note: "Post-train of v0.6.1: same backend pins as v0.6.1. Same CUDA support as v0.6.1.",
  },
  {
    version: "v0.6.1",
    date: "Nov 6, 2025",
    kind: "patch",
    github: `${GH}v0.6.1`,
    pins: { sglang: "0.5.3.post2", trtllm: "1.1.0rc5", vllm: "0.11.0", nixlSglang: "0.6.0", nixlTrtllm: "0.6.0", nixlVllm: "0.6.0" },
  },
  {
    version: "v0.6.0",
    date: "Oct 28, 2025",
    kind: "stable",
    github: `${GH}v0.6.0`,
    pins: { sglang: "0.5.3.post2", trtllm: "1.1.0rc5", vllm: "0.11.0", nixlSglang: "0.6.0", nixlTrtllm: "0.6.0", nixlVllm: "0.6.0" },
    delta: "Oldest release tracked on this page.",
  },
];

export interface CudaRow {
  version: string;
  backend: "SGLang" | "TensorRT-LLM" | "vLLM";
  toolkit: string;
  minDriver: string;
  note?: string;
}

export const CUDA_HISTORY: CudaRow[] = [
  { version: "1.4.0", backend: "SGLang", toolkit: "13.0", minDriver: "580.xx+" },
  { version: "1.4.0", backend: "TensorRT-LLM", toolkit: "13.1", minDriver: "580.xx+" },
  { version: "1.4.0", backend: "vLLM", toolkit: "13.0", minDriver: "580.xx+" },
  { version: "1.3.1", backend: "SGLang", toolkit: "13.0", minDriver: "580.xx+" },
  { version: "1.3.1", backend: "TensorRT-LLM", toolkit: "13.1", minDriver: "580.xx+" },
  { version: "1.3.1", backend: "vLLM", toolkit: "13.0", minDriver: "580.xx+" },
  { version: "1.3.0", backend: "SGLang", toolkit: "13.0", minDriver: "580.xx+" },
  { version: "1.3.0", backend: "TensorRT-LLM", toolkit: "13.1", minDriver: "580.xx+" },
  { version: "1.3.0", backend: "vLLM", toolkit: "13.0", minDriver: "580.xx+" },
  { version: "1.2.1", backend: "SGLang", toolkit: "12.9", minDriver: "575.xx+" },
  { version: "1.2.1", backend: "SGLang", toolkit: "13.0", minDriver: "580.xx+" },
  { version: "1.2.1", backend: "TensorRT-LLM", toolkit: "13.1", minDriver: "580.xx+" },
  { version: "1.2.1", backend: "vLLM", toolkit: "12.9", minDriver: "575.xx+" },
  { version: "1.2.1", backend: "vLLM", toolkit: "13.0", minDriver: "580.xx+" },
  { version: "1.2.0", backend: "SGLang", toolkit: "12.9", minDriver: "575.xx+" },
  { version: "1.2.0", backend: "SGLang", toolkit: "13.0", minDriver: "580.xx+" },
  { version: "1.2.0", backend: "TensorRT-LLM", toolkit: "13.1", minDriver: "580.xx+" },
  { version: "1.2.0", backend: "vLLM", toolkit: "12.9", minDriver: "575.xx+" },
  { version: "1.2.0", backend: "vLLM", toolkit: "13.0", minDriver: "580.xx+" },
  { version: "1.1.1", backend: "SGLang", toolkit: "12.9", minDriver: "575.xx+" },
  { version: "1.1.1", backend: "SGLang", toolkit: "13.0", minDriver: "580.xx+" },
  { version: "1.1.1", backend: "TensorRT-LLM", toolkit: "13.1", minDriver: "580.xx+" },
  { version: "1.1.1", backend: "vLLM", toolkit: "12.9", minDriver: "575.xx+" },
  { version: "1.1.1", backend: "vLLM", toolkit: "13.0", minDriver: "580.xx+" },
  { version: "1.1.0", backend: "SGLang", toolkit: "12.9", minDriver: "575.xx+" },
  { version: "1.1.0", backend: "SGLang", toolkit: "13.0", minDriver: "580.xx+" },
  { version: "1.1.0", backend: "TensorRT-LLM", toolkit: "13.1", minDriver: "580.xx+" },
  { version: "1.1.0", backend: "vLLM", toolkit: "12.9", minDriver: "575.xx+" },
  { version: "1.1.0", backend: "vLLM", toolkit: "13.0", minDriver: "580.xx+" },
  { version: "1.0.2", backend: "SGLang", toolkit: "12.9", minDriver: "575.xx+" },
  { version: "1.0.2", backend: "SGLang", toolkit: "13.0", minDriver: "580.xx+" },
  { version: "1.0.2", backend: "TensorRT-LLM", toolkit: "13.1", minDriver: "580.xx+" },
  { version: "1.0.2", backend: "vLLM", toolkit: "12.9", minDriver: "575.xx+" },
  { version: "1.0.2", backend: "vLLM", toolkit: "13.0", minDriver: "580.xx+" },
  { version: "1.0.1", backend: "SGLang", toolkit: "12.9", minDriver: "575.xx+" },
  { version: "1.0.1", backend: "SGLang", toolkit: "13.0", minDriver: "580.xx+" },
  { version: "1.0.1", backend: "TensorRT-LLM", toolkit: "13.1", minDriver: "580.xx+" },
  { version: "1.0.1", backend: "vLLM", toolkit: "12.9", minDriver: "575.xx+" },
  { version: "1.0.1", backend: "vLLM", toolkit: "13.0", minDriver: "580.xx+" },
  { version: "1.0.0", backend: "SGLang", toolkit: "12.9", minDriver: "575.xx+" },
  { version: "1.0.0", backend: "SGLang", toolkit: "13.0", minDriver: "580.xx+" },
  { version: "1.0.0", backend: "TensorRT-LLM", toolkit: "13.1", minDriver: "580.xx+" },
  { version: "1.0.0", backend: "vLLM", toolkit: "12.9", minDriver: "575.xx+" },
  { version: "1.0.0", backend: "vLLM", toolkit: "13.0", minDriver: "580.xx+" },
  { version: "0.9.1", backend: "SGLang", toolkit: "12.9", minDriver: "575.xx+" },
  { version: "0.9.1", backend: "TensorRT-LLM", toolkit: "13.0", minDriver: "580.xx+" },
  { version: "0.9.1", backend: "vLLM", toolkit: "12.9", minDriver: "575.xx+" },
  { version: "0.9.0", backend: "SGLang", toolkit: "12.9", minDriver: "575.xx+" },
  { version: "0.9.0", backend: "TensorRT-LLM", toolkit: "13.0", minDriver: "580.xx+" },
  { version: "0.9.0", backend: "vLLM", toolkit: "12.9", minDriver: "575.xx+" },
  { version: "0.8.1", backend: "SGLang", toolkit: "12.9", minDriver: "575.xx+" },
  { version: "0.8.1", backend: "SGLang", toolkit: "13.0", minDriver: "580.xx+", note: "Experimental" },
  { version: "0.8.1", backend: "TensorRT-LLM", toolkit: "13.0", minDriver: "580.xx+" },
  { version: "0.8.1", backend: "vLLM", toolkit: "12.9", minDriver: "575.xx+" },
  { version: "0.8.1", backend: "vLLM", toolkit: "13.0", minDriver: "580.xx+", note: "Experimental" },
  { version: "0.8.0", backend: "SGLang", toolkit: "12.9", minDriver: "575.xx+" },
  { version: "0.8.0", backend: "SGLang", toolkit: "13.0", minDriver: "580.xx+", note: "Experimental" },
  { version: "0.8.0", backend: "TensorRT-LLM", toolkit: "13.0", minDriver: "580.xx+" },
  { version: "0.8.0", backend: "vLLM", toolkit: "12.9", minDriver: "575.xx+" },
  { version: "0.8.0", backend: "vLLM", toolkit: "13.0", minDriver: "580.xx+", note: "Experimental" },
  { version: "0.7.1", backend: "SGLang", toolkit: "12.8", minDriver: "570.xx+" },
  { version: "0.7.1", backend: "TensorRT-LLM", toolkit: "13.0", minDriver: "580.xx+" },
  { version: "0.7.1", backend: "vLLM", toolkit: "12.9", minDriver: "575.xx+" },
  { version: "0.7.0", backend: "SGLang", toolkit: "12.9", minDriver: "575.xx+" },
  { version: "0.7.0", backend: "TensorRT-LLM", toolkit: "13.0", minDriver: "580.xx+" },
  { version: "0.7.0", backend: "vLLM", toolkit: "12.8", minDriver: "570.xx+" },
];

export const CUDA_NOTES = [
  "Patch versions (e.g. v0.8.1.post1, v0.7.0.post1) have the same CUDA support as their base version.",
  "Early access v1.1.0-dev.* images follow the same CUDA matrix as v1.0.2. The v1.2.0-deepseek-v4-dev.3 vLLM container is CUDA 13.0 multi-arch; the SGLang containers split by arch (CUDA 12.9 on amd64, CUDA 13.0 on arm64).",
  "Experimental CUDA 13 images are not published for all versions.",
];

export type FeatureStatus = "yes" | "caveat" | "wip" | "no";

export interface FeatureCell {
  status: FeatureStatus;
  note?: string;
}

export interface Feature {
  name: string;
  sglang: FeatureCell;
  trtllm: FeatureCell;
  vllm: FeatureCell;
}

export const FEATURES: Feature[] = [
  {
    name: "Disaggregated Serving",
    sglang: { status: "yes" },
    trtllm: { status: "yes" },
    vllm: { status: "yes", note: "Prefill/decode separation with NIXL KV transfer" },
  },
  {
    name: "KV-Aware Routing",
    sglang: { status: "yes" },
    trtllm: { status: "yes" },
    vllm: { status: "yes" },
  },
  {
    name: "SLA-Based Planner",
    sglang: { status: "yes" },
    trtllm: { status: "yes" },
    vllm: { status: "yes" },
  },
  {
    name: "KV Block Manager",
    sglang: { status: "wip", note: "Work in progress across all combinations" },
    trtllm: { status: "yes" },
    vllm: { status: "yes" },
  },
  {
    name: "Multimodal (Image)",
    sglang: {
      status: "yes",
      note: "KV-aware routing supported on Dynamo's SGLang image for aggregated workers; a custom build without the hash-forwarding patch falls back to text-prefix routing. Separately, multimodal serving supports EPD, E/PD and E/P/D disaggregation (not traditional EP/D)",
    },
    trtllm: {
      status: "yes",
      note: "Image URLs + pre-computed embeddings. Disagg: EP/D + E/P/D. KV-aware routing via dedicated MM Router Worker (requires KV event publishing)",
    },
    vllm: {
      status: "yes",
      note: "With KV-aware routing, image-aware routing on documented paths",
    },
  },
  {
    name: "Multimodal (Video)",
    sglang: { status: "yes" },
    trtllm: { status: "no" },
    vllm: { status: "yes", note: "Video input with frame sampling" },
  },
  {
    name: "Multimodal (Audio)",
    sglang: { status: "no" },
    trtllm: { status: "no" },
    vllm: { status: "wip", note: "Qwen2-Audio, experimental" },
  },
  {
    name: "Request Migration",
    sglang: { status: "yes" },
    trtllm: { status: "yes", note: "Work in progress with multimodal" },
    vllm: { status: "yes" },
  },
  {
    name: "Request Cancellation",
    sglang: {
      status: "wip",
      note: "Remote-prefill-phase cancellation not supported in disaggregated mode",
    },
    trtllm: {
      status: "caveat",
      note: "Engine temporarily not notified of cancellations — resources for cancelled requests are not freed (known issue)",
    },
    vllm: { status: "yes" },
  },
  {
    name: "LoRA",
    sglang: { status: "no" },
    trtllm: { status: "no" },
    vllm: { status: "yes", note: "Dynamic load/unload; KV-aware routing supports adapter affinity" },
  },
  {
    name: "Tool Calling",
    sglang: { status: "yes" },
    trtllm: { status: "yes" },
    vllm: { status: "yes" },
  },
  {
    name: "Speculative Decoding",
    sglang: { status: "wip", note: "Code hooks exist; no examples or docs yet" },
    trtllm: { status: "yes" },
    vllm: { status: "yes", note: "Eagle3" },
  },
  {
    name: "GPU Memory Service",
    sglang: { status: "yes", note: "Weights and KV; upstream integration remains in progress" },
    trtllm: { status: "wip", note: "Weights only; multinode and upstream integration remain in progress" },
    vllm: { status: "yes", note: "Weights and KV; upstream integration remains in progress" },
  },
  {
    name: "Shadow Engine Failover",
    sglang: { status: "wip", note: "No KV-cache reuse or hardware fault tolerance" },
    trtllm: { status: "wip", note: "No KV-cache reuse or hardware fault tolerance" },
    vllm: { status: "caveat", note: "Software-process failover only; no KV-cache reuse or hardware fault tolerance" },
  },
  {
    name: "Dynamo Snapshot",
    sglang: { status: "caveat", note: "Single-GPU supported; multi-GPU and multinode remain in progress" },
    trtllm: { status: "wip", note: "Single-GPU aggregated text-worker path only" },
    vllm: { status: "caveat", note: "Single-GPU supported; multi-GPU is highly experimental and multinode remains in progress" },
  },
];

export const BACKEND_BLURBS = {
  vllm: "vLLM offers the broadest feature coverage in Dynamo, with full support for disaggregated serving, KV-aware routing, KV block management, LoRA adapters, and multimodal inference including video and audio.",
  sglang:
    "SGLang is optimized for high-throughput serving with fast primitives, providing robust support for disaggregated serving, KV-aware routing, and request migration.",
  trtllm:
    "TensorRT-LLM delivers maximum inference performance and optimization, with full KVBM integration and robust disaggregated serving support.",
};

export type ArtifactCategory = "container" | "wheel" | "helm" | "crate";

export interface Artifact {
  category: ArtifactCategory;
  group?: "runtime" | "component" | "consumed";
  name: string;
  description: string;
  meta?: string;
  href: string;
  tags: { label: string; clipboard: string; variant?: "default" | "experimental" }[];
  badge?: "Preview" | "Experimental" | "Deprecated";
}

export interface NightlyBuild {
  version: string;
  date: string;
  packages: string[];
  note?: string;
}


const NGC_C = "https://catalog.ngc.nvidia.com/orgs/nvidia/ai-dynamo/containers";

export const ARTIFACTS: Artifact[] = [
  {
    category: "container",
    group: "runtime",
    name: "vllm-runtime",
    description: "vLLM backend runtime",
    meta: "vLLM v0.26.0 · CUDA 13.0 · AMD64/ARM64",
    href: `${NGC_C}/vllm-runtime/tags`,
    tags: [
      { label: "1.4.0", clipboard: "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0" },
      { label: "1.4.0-efa", clipboard: "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0-efa", variant: "experimental" },
    ],
  },
  {
    category: "container",
    group: "runtime",
    name: "sglang-runtime",
    description: "SGLang backend runtime",
    meta: "SGLang v0.5.16 · CUDA 13.0 · AMD64/ARM64",
    href: `${NGC_C}/sglang-runtime/tags`,
    tags: [
      { label: "1.4.0", clipboard: "nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.4.0" },
      { label: "1.4.0-efa", clipboard: "nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.4.0-efa", variant: "experimental" },
    ],
  },
  {
    category: "container",
    group: "runtime",
    name: "tensorrtllm-runtime",
    description: "TensorRT-LLM backend runtime",
    meta: "TRT-LLM v1.3.0rc22 · CUDA 13.1 · AMD64/ARM64",
    href: `${NGC_C}/tensorrtllm-runtime/tags`,
    tags: [
      { label: "1.4.0", clipboard: "nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.4.0" },
      { label: "1.4.0-efa", clipboard: "nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.4.0-efa", variant: "experimental" },
    ],
  },
  {
    category: "container",
    group: "component",
    name: "dynamo-frontend",
    description: "OpenAI-compatible API gateway with Endpoint Prediction Protocol (EPP)",
    meta: "AMD64/ARM64",
    href: `${NGC_C}/dynamo-frontend/tags`,
    tags: [{ label: "1.4.0", clipboard: "nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.4.0" }],
  },
  {
    category: "container",
    group: "component",
    name: "dynamo-planner",
    description: "Standalone Planner used by Profiler jobs and Planner pods",
    meta: "AMD64/ARM64",
    href: `${NGC_C}/dynamo-planner/tags`,
    tags: [{ label: "1.4.0", clipboard: "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.4.0" }],
  },
  {
    category: "container",
    group: "component",
    name: "kubernetes-operator",
    description: "Operator that manages Dynamo deployments and CRDs",
    meta: "AMD64/ARM64",
    href: `${NGC_C}/kubernetes-operator/tags`,
    tags: [{ label: "1.4.0", clipboard: "nvcr.io/nvidia/ai-dynamo/kubernetes-operator:1.4.0" }],
  },
  {
    category: "container",
    group: "component",
    name: "snapshot-agent",
    description: "Fast GPU worker recovery via CRIU",
    meta: "AMD64",
    href: `${NGC_C}/snapshot-agent/tags`,
    badge: "Preview",
    tags: [{ label: "1.4.0", clipboard: "nvcr.io/nvidia/ai-dynamo/snapshot-agent:1.4.0" }],
  },
  {
    category: "wheel",
    name: "ai-dynamo",
    description: "Main package with backend integrations (vLLM, SGLang, TRT-LLM)",
    meta: "Python 3.10–3.12 · Linux (glibc v2.28+)",
    href: "https://pypi.org/project/ai-dynamo/1.4.0/",
    tags: [{ label: "uv pip install ai-dynamo==1.4.0", clipboard: "uv pip install ai-dynamo==1.4.0" }],
  },
  {
    category: "wheel",
    name: "ai-dynamo-runtime",
    description: "Core Python bindings for the Dynamo runtime",
    meta: "Python 3.10–3.12 · Linux (glibc v2.28+)",
    href: "https://pypi.org/project/ai-dynamo-runtime/1.4.0/",
    tags: [
      { label: "uv pip install ai-dynamo-runtime==1.4.0", clipboard: "uv pip install ai-dynamo-runtime==1.4.0" },
    ],
  },
  {
    category: "wheel",
    name: "kvbm",
    description: "KV Block Manager for disaggregated KV cache",
    meta: "Python 3.10–3.12 · Linux (glibc v2.28+)",
    href: "https://pypi.org/project/kvbm/1.4.0/",
    tags: [{ label: "uv pip install kvbm==1.4.0", clipboard: "uv pip install kvbm==1.4.0" }],
  },
  {
    category: "helm",
    name: "dynamo-platform",
    description: "Platform services (etcd, NATS) and the Dynamo Operator for a Dynamo cluster",
    href: "https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-1.4.0.tgz",
    tags: [
      {
        label: "helm install · dynamo-platform 1.4.0",
        clipboard:
          "helm install dynamo-platform https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-1.4.0.tgz",
      },
    ],
  },
  {
    category: "helm",
    name: "snapshot",
    description: "Snapshot DaemonSet for fast GPU worker recovery",
    href: "https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/snapshot-1.4.0.tgz",
    tags: [
      {
        label: "helm install · snapshot 1.4.0",
        clipboard: "helm install snapshot https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/snapshot-1.4.0.tgz",
      },
    ],
  },
  {
    category: "crate",
    name: "dynamo-runtime",
    description: "Core distributed runtime library",
    meta: "MSRV Rust v1.82",
    href: "https://crates.io/crates/dynamo-runtime/1.4.0",
    tags: [{ label: "cargo add dynamo-runtime@1.4.0", clipboard: "cargo add dynamo-runtime@1.4.0" }],
  },
  {
    category: "crate",
    name: "dynamo-llm",
    description: "LLM inference engine",
    meta: "MSRV Rust v1.82",
    href: "https://crates.io/crates/dynamo-llm/1.4.0",
    tags: [{ label: "cargo add dynamo-llm@1.4.0", clipboard: "cargo add dynamo-llm@1.4.0" }],
  },
  {
    category: "crate",
    name: "dynamo-protocols",
    description: "Async OpenAI-compatible API client",
    meta: "Independently versioned",
    group: "consumed",
    href: "https://crates.io/crates/dynamo-protocols/5.0.1",
    tags: [{ label: "cargo add dynamo-protocols@5.0.1", clipboard: "cargo add dynamo-protocols@5.0.1" }],
  },
  {
    category: "crate",
    name: "dynamo-async-openai",
    description: "Legacy OpenAI client; use dynamo-protocols",
    meta: "MSRV Rust v1.82 · final release",
    href: "https://crates.io/crates/dynamo-async-openai/1.0.2",
    badge: "Deprecated",
    tags: [{ label: "cargo add dynamo-async-openai@1.0.2", clipboard: "cargo add dynamo-async-openai@1.0.2" }],
  },
  {
    category: "crate",
    name: "dynamo-parsers",
    description: "Protocol parsers (SSE, JSON streaming)",
    meta: "Independently versioned",
    group: "consumed",
    href: "https://crates.io/crates/dynamo-parsers/7.0.1",
    tags: [{ label: "cargo add dynamo-parsers@7.0.1", clipboard: "cargo add dynamo-parsers@7.0.1" }],
  },
  {
    category: "crate",
    name: "dynamo-memory",
    description: "Memory management utilities",
    meta: "MSRV Rust v1.82",
    href: "https://crates.io/crates/dynamo-memory/1.4.0",
    tags: [{ label: "cargo add dynamo-memory@1.4.0", clipboard: "cargo add dynamo-memory@1.4.0" }],
  },
  {
    category: "crate",
    name: "dynamo-config",
    description: "Configuration management",
    meta: "MSRV Rust v1.82",
    // Not republished for 1.3.0; crates.io tops out at 1.2.1.
    href: "https://crates.io/crates/dynamo-config/1.2.1",
    tags: [{ label: "cargo add dynamo-config@1.2.1", clipboard: "cargo add dynamo-config@1.2.1" }],
  },
  {
    category: "crate",
    name: "dynamo-tokens",
    description: "Tokenizer bindings for LLM inference",
    meta: "MSRV Rust v1.82",
    href: "https://crates.io/crates/dynamo-tokens/1.4.0",
    tags: [{ label: "cargo add dynamo-tokens@1.4.0", clipboard: "cargo add dynamo-tokens@1.4.0" }],
  },
  {
    category: "crate",
    name: "dynamo-tokenizers",
    description: "Tokenizer library for LLM inference",
    meta: "Independently versioned",
    group: "consumed",
    href: "https://crates.io/crates/dynamo-tokenizers/1.5.4",
    tags: [{ label: "cargo add dynamo-tokenizers@1.5.4", clipboard: "cargo add dynamo-tokenizers@1.5.4" }],
  },
  {
    category: "crate",
    name: "dynamo-mocker",
    description: "Inference engine simulator for benchmarking",
    meta: "MSRV Rust v1.82",
    href: "https://crates.io/crates/dynamo-mocker/1.4.0",
    tags: [{ label: "cargo add dynamo-mocker@1.4.0", clipboard: "cargo add dynamo-mocker@1.4.0" }],
  },
  {
    category: "crate",
    name: "dynamo-kv-router",
    description: "KV-aware request routing library",
    meta: "MSRV Rust v1.82",
    href: "https://crates.io/crates/dynamo-kv-router/1.4.0",
    tags: [{ label: "cargo add dynamo-kv-router@1.4.0", clipboard: "cargo add dynamo-kv-router@1.4.0" }],
  },
  {
    category: "crate",
    name: "kvbm-logical",
    description: "Logical layer for the KV Block Manager",
    meta: "MSRV Rust v1.82",
    href: "https://crates.io/crates/kvbm-logical/1.4.0",
    tags: [{ label: "cargo add kvbm-logical@1.4.0", clipboard: "cargo add kvbm-logical@1.4.0" }],
  },
  {
    category: "crate",
    name: "dynamo-kv-hashing",
    description: "Request-to-lineage-hash contract for KV cache identity",
    meta: "MSRV Rust v1.82",
    href: "https://crates.io/crates/dynamo-kv-hashing/1.4.0",
    tags: [{ label: "cargo add dynamo-kv-hashing@1.4.0", clipboard: "cargo add dynamo-kv-hashing@1.4.0" }],
  },
  {
    category: "crate",
    name: "dynamo-data-gen",
    description: "Schemas and primitives for Dynamo data generation and replay traces",
    meta: "MSRV Rust v1.82",
    href: "https://crates.io/crates/dynamo-data-gen/1.4.0",
    tags: [{ label: "cargo add dynamo-data-gen@1.4.0", clipboard: "cargo add dynamo-data-gen@1.4.0" }],
  },
  {
    category: "crate",
    name: "dynamo-rl",
    description: "Dynamo RL worker discovery API",
    meta: "MSRV Rust v1.82",
    href: "https://crates.io/crates/dynamo-rl/1.4.0",
    tags: [{ label: "cargo add dynamo-rl@1.4.0", clipboard: "cargo add dynamo-rl@1.4.0" }],
  },
  {
    category: "crate",
    name: "dynamo-bench",
    description: "Lightweight HTTP benchmarks for Dynamo endpoints",
    meta: "MSRV Rust v1.82",
    href: "https://crates.io/crates/dynamo-bench/1.4.0",
    tags: [{ label: "cargo add dynamo-bench@1.4.0", clipboard: "cargo add dynamo-bench@1.4.0" }],
  },
  {
    category: "crate",
    name: "dynamo-truthy",
    description: "Canonical truthy/falsy boolean flag parsing",
    meta: "MSRV Rust v1.82",
    href: "https://crates.io/crates/dynamo-truthy/1.4.0",
    tags: [{ label: "cargo add dynamo-truthy@1.4.0", clipboard: "cargo add dynamo-truthy@1.4.0" }],
  },
  {
    category: "crate",
    name: "kvbm-common",
    description: "Shared types for the KV Block Manager",
    meta: "MSRV Rust v1.82",
    href: "https://crates.io/crates/kvbm-common/1.4.0",
    tags: [{ label: "cargo add kvbm-common@1.4.0", clipboard: "cargo add kvbm-common@1.4.0" }],
  },
  {
    category: "crate",
    name: "kvbm-config",
    description: "KVBM configuration for Tokio, Rayon, and Messenger runtimes",
    meta: "MSRV Rust v1.82",
    href: "https://crates.io/crates/kvbm-config/1.4.0",
    tags: [{ label: "cargo add kvbm-config@1.4.0", clipboard: "cargo add kvbm-config@1.4.0" }],
  },
  {
    category: "crate",
    name: "kvbm-kernels",
    description: "CUDA kernels for the KV Block Manager",
    meta: "MSRV Rust v1.82",
    href: "https://crates.io/crates/kvbm-kernels/1.4.0",
    tags: [{ label: "cargo add kvbm-kernels@1.4.0", clipboard: "cargo add kvbm-kernels@1.4.0" }],
  },
  {
    category: "crate",
    name: "kvbm-physical",
    description: "Physical block layer for the KV Block Manager",
    meta: "MSRV Rust v1.82",
    href: "https://crates.io/crates/kvbm-physical/1.4.0",
    tags: [{ label: "cargo add kvbm-physical@1.4.0", clipboard: "cargo add kvbm-physical@1.4.0" }],
  },
  {
    category: "crate",
    name: "kvbm-engine",
    description: "Distributed coordination primitives for KVBM",
    meta: "MSRV Rust v1.82",
    href: "https://crates.io/crates/kvbm-engine/1.4.0",
    tags: [{ label: "cargo add kvbm-engine@1.4.0", clipboard: "cargo add kvbm-engine@1.4.0" }],
  },
  {
    category: "crate",
    name: "dynamo-renderer",
    description: "Chat-template rendering used by the Dynamo Frontend",
    meta: "Independently versioned",
    group: "consumed",
    href: "https://crates.io/crates/dynamo-renderer/4.0.0",
    tags: [{ label: "cargo add dynamo-renderer@4.0.0", clipboard: "cargo add dynamo-renderer@4.0.0" }],
  },
  {
    category: "crate",
    name: "dynamo-parsers-v2",
    description: "Successor parser line to dynamo-parsers, consumed by the Frontend",
    meta: "Independently versioned",
    group: "consumed",
    href: "https://crates.io/crates/dynamo-parsers-v2/0.1.23",
    tags: [{ label: "cargo add dynamo-parsers-v2@0.1.23", clipboard: "cargo add dynamo-parsers-v2@0.1.23" }],
  },
  {
    category: "crate",
    name: "fastokens",
    description: "Rust BPE tokenizer backend consumed by the Frontend",
    meta: "Independently versioned",
    group: "consumed",
    href: "https://crates.io/crates/fastokens/0.2.0",
    tags: [{ label: "cargo add fastokens@0.2.0", clipboard: "cargo add fastokens@0.2.0" }],
  },
];

export type GaPath = "promoted" | "dev-only" | "recipe-in-ga" | "superseded";

export interface Coverage {
  images: boolean;
  wheels: boolean;
  helm: boolean;
  crates: boolean;
}

export interface ModelEaBuild {
  model: string;
  tag: string;
  releaseLine: string;
  runtimes: string[];
  shipped: string;
  gaPath: GaPath;
  gaLabel: string;
  statusLine: string;
  recipeLabel?: string;
  recipeHref?: string;
  github?: string;
  coverage: Coverage;
}

const MODEL_COVERAGE: Coverage = { images: true, wheels: false, helm: false, crates: false };

export const MODEL_EA_BUILDS: ModelEaBuild[] = [
  {
    model: "Inkling",
    tag: "1.4.0-inkling-dev.1",
    releaseLine: "v1.4.0",
    runtimes: ["sglang-runtime"],
    shipped: "Jul 17, 2026",
    gaPath: "dev-only",
    gaLabel: "Dev-only · v1.4.0 line",
    statusLine: "First build on the v1.4.0 line; targets the next stable release.",
    recipeLabel: "Inkling recipe (main)",
    recipeHref: "https://github.com/ai-dynamo/dynamo/blob/main/docs/recipes/inkling.mdx",
    github: `${GH}v1.4.0-inkling-dev.1`,
    coverage: MODEL_COVERAGE,
  },
  {
    model: "GLM-5.2",
    tag: "1.3.0-glm-5.2-dev.1",
    releaseLine: "v1.3.0",
    runtimes: ["sglang-runtime"],
    shipped: "Jul 20, 2026",
    gaPath: "dev-only",
    gaLabel: "Dev-only",
    statusLine:
      "Container carries SGLang cherry-picks (stability, config parsing, model support) opened upstream but not yet in a released SGLang.",
    recipeLabel: "GLM-5 NVFP4 recipe",
    recipeHref: "/dynamo/dev/recipes/glm-5-nvfp4",
    coverage: MODEL_COVERAGE,
  },
  {
    model: "MiniMax-M3",
    tag: "1.3.0-minimax-m3-dev.1",
    releaseLine: "v1.3.0",
    runtimes: ["vllm-runtime", "sglang-runtime", "tensorrtllm-runtime"],
    shipped: "Jun 12, 2026",
    gaPath: "promoted",
    gaLabel: "Promoted → :1.3.0",
    statusLine: "Dynamo changes and the M2 tool-calling fix are in release/1.3.0; the recipes run on the stock :1.3.0 containers.",
    recipeLabel: "Recipe on release branch",
    recipeHref: "https://github.com/ai-dynamo/dynamo/tree/release/1.3.0-minimax-m3-dev.1/recipes/minimax-m3",
    github: `${GH}v1.3.0-minimax-m3-dev.1`,
    coverage: MODEL_COVERAGE,
  },
  {
    model: "DeepSeek-V4",
    tag: "1.3.0-deepseek-v4-dev.1",
    releaseLine: "v1.3.0",
    runtimes: ["tensorrtllm-runtime"],
    shipped: "Jun 6, 2026",
    gaPath: "recipe-in-ga",
    gaLabel: "Recipe in v1.3.0",
    statusLine: "DeepSeek-V4 Flash and Pro recipes ship in v1.3.0 on the standard TensorRT-LLM release container.",
    recipeLabel: "recipes/deepseek-v4 (main)",
    recipeHref: "https://github.com/ai-dynamo/dynamo/tree/main/recipes/deepseek-v4",
    github: `${GH}v1.3.0-deepseek-v4-dev.1`,
    coverage: MODEL_COVERAGE,
  },
  {
    model: "Nemotron-3-Ultra",
    tag: "1.3.0-nemotron-ultra-dev.1",
    releaseLine: "v1.3.0",
    runtimes: ["vllm-runtime"],
    shipped: "Jun 5, 2026",
    gaPath: "dev-only",
    gaLabel: "Dev-only",
    statusLine:
      "Four un-upstreamed vLLM patches; requires pinned flags VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel and --no-enable-flashinfer-autotune.",
    recipeLabel: "Nemotron-3-Ultra recipe",
    recipeHref: "/dynamo/dev/recipes/nemotron-3-ultra",
    github: `${GH}v1.3.0-nemotron-ultra-dev.1`,
    coverage: MODEL_COVERAGE,
  },
  {
    model: "Nemotron-3-Super",
    tag: "1.3.0-nemotron-super-dev.1",
    releaseLine: "v1.3.0",
    runtimes: ["vllm-runtime"],
    shipped: "Jun 4, 2026",
    gaPath: "dev-only",
    gaLabel: "Dev-only",
    statusLine: "Requires the dedicated `vllm-runtime:1.3.0-nemotron-super-dev.1` image; the model-specific vLLM patches are not in the v1.3.0 release container.",
    recipeLabel: "Nemotron-3-Super recipe",
    recipeHref: "/dynamo/dev/recipes/nemotron-3-super",
    github: `${GH}v1.3.0-nemotron-super-dev.1`,
    coverage: MODEL_COVERAGE,
  },
  {
    model: "Kimi-K2.6",
    tag: "1.3.0-kimi-k2.6-dev.1",
    releaseLine: "v1.3.0",
    runtimes: ["vllm-runtime"],
    shipped: "Jun 4, 2026",
    gaPath: "promoted",
    gaLabel: "Promoted → :1.3.0",
    statusLine: "The build's only container patch is in vLLM v0.23.0; the recipes run on the stock vllm-runtime:1.3.0.",
    recipeLabel: "Kimi-K2.6 recipe",
    recipeHref: "/dynamo/dev/recipes/kimi-k2-6",
    github: `${GH}v1.3.0-kimi-k2.6-dev.1`,
    coverage: MODEL_COVERAGE,
  },
  {
    model: "Cosmos-3",
    tag: "1.3.0-cosmos3-dev.1",
    releaseLine: "v1.3.0",
    runtimes: ["vllm-runtime"],
    shipped: "Jun 1, 2026",
    gaPath: "dev-only",
    gaLabel: "Dev-only",
    statusLine:
      "Dynamo #10132 (Cosmos3 support in the vLLM-Omni backend) is open, not merged — v1.3.0 containers cannot run Cosmos3.",
    recipeLabel: "Launch scripts (branch)",
    recipeHref: "https://github.com/ai-dynamo/dynamo/tree/release/1.3.0-cosmos3-dev.1/examples/backends/vllm/launch",
    github: `${GH}v1.3.0-cosmos3-dev.1`,
    coverage: MODEL_COVERAGE,
  },
  {
    model: "DeepSeek-V4 preview",
    tag: "1.2.0-deepseek-v4-dev.3",
    releaseLine: "v1.2.0",
    runtimes: ["vllm-runtime", "sglang-runtime"],
    shipped: "May 9, 2026",
    gaPath: "superseded",
    gaLabel: "Superseded — recipe in v1.3.0",
    statusLine:
      "Blackwell (B200 + GB200) preview; per-arch/CUDA tags (e.g. vllm-runtime:1.2.0-deepseek-v4-cuda13-dev.3). Superseded by the v1.3.0 recipe.",
    github: `${GH}v1.2.0-deepseek-v4-dev.3`,
    coverage: MODEL_COVERAGE,
  },
  {
    model: "DeepSeek-V4 preview",
    tag: "1.2.0-deepseek-v4-dev.2",
    releaseLine: "v1.2.0",
    runtimes: ["vllm-runtime", "sglang-runtime"],
    shipped: "May 1, 2026",
    gaPath: "superseded",
    gaLabel: "Superseded — recipe in v1.3.0",
    statusLine: "Blackwell preview on vLLM v0.20.0 (native DSv4 support); superseded by dev.3.",
    github: `${GH}v1.2.0-deepseek-v4-dev.2`,
    coverage: MODEL_COVERAGE,
  },
  {
    model: "DeepSeek-V4 preview",
    tag: "1.2.0-sglang-deepseek-v4-dev.1",
    releaseLine: "v1.2.0",
    runtimes: ["sglang-runtime"],
    shipped: "Apr 25, 2026",
    gaPath: "superseded",
    gaLabel: "Superseded — recipe in v1.3.0",
    statusLine: "Earliest DSv4 preview (SGLang, B200 only); superseded by dev.2/dev.3.",
    github: `${GH}v1.2.0-sglang-deepseek-v4-dev.1`,
    coverage: MODEL_COVERAGE,
  },
];

/* Pairwise feature-by-feature compatibility, one matrix per backend. Only the
 * lower triangle is stored: rows[i] carries i+1 cells, ending on the diagonal.
 * The upper triangle is the mirror and is never authored twice.
 *
 * FeatureInteractions renders this for readers and gen_llms_tables.py emits the
 * same cells as markdown into the <llms-only> twin, so a pairwise status can
 * never be visible on the page but missing from an agent export -- the failure
 * the tables hit while they were hand-authored JSX. */
export const INTERACTION_FEATURES = [
  "Disaggregated Serving",
  "KV-Aware Routing",
  "SLA-Based Planner",
  "KV Block Manager",
  "Multimodal",
  "Request Migration",
  "Request Cancellation",
  "LoRA",
  "Tool Calling",
  "Speculative Decoding",
];

export type InteractionState = "yes" | "wip" | "no" | "na";

export interface InteractionCell {
  status: InteractionState;
  label?: string; // short screen-reader / summary phrase for a noted cell
  note?: string;
  source?: string; // site-absolute docs path
}

export interface BackendInteractions {
  backend: "SGLang" | "TensorRT-LLM" | "vLLM";
  features: string[];
  rows: InteractionCell[][];
}

export const FEATURE_INTERACTIONS: BackendInteractions[] = [
  {
    backend: "vLLM",
    features: INTERACTION_FEATURES,
    rows: [
      // Disaggregated Serving
      [{ status: "na" }],
      // KV-Aware Routing
      [{ status: "yes" }, { status: "na" }],
      // SLA-Based Planner
      [{ status: "yes" }, { status: "yes" }, { status: "na" }],
      // KV Block Manager
      [{ status: "yes" }, { status: "yes" }, { status: "yes" }, { status: "na" }],
      // Multimodal
      [{ status: "yes", label: "Audio and video support", note: "Supports Qwen2-Audio experimentally and video input with frame sampling.", source: "/dynamo/dev/knowledge-base/modular-components/backends/v-llm/vllm-multimodal" }, { status: "yes", label: "Image-aware KV routing", note: "The Rust frontend supports models handled by `llm-multimodal`; the Python path delegates to vLLM's multimodal processor.", source: "/dynamo/dev/multimodal/multimodal-kv-routing" }, { status: "na" }, { status: "yes" }, { status: "na" }],
      // Request Migration
      [{ status: "yes" }, { status: "yes" }, { status: "yes" }, { status: "yes" }, { status: "yes" }, { status: "na" }],
      // Request Cancellation
      [{ status: "yes" }, { status: "yes" }, { status: "yes" }, { status: "yes" }, { status: "yes" }, { status: "yes" }, { status: "na" }],
      // LoRA
      [{ status: "yes" }, { status: "yes", label: "Adapter-aware routing", note: "vLLM routes requests based on LoRA adapter affinity." }, { status: "na" }, { status: "yes" }, { status: "na" }, { status: "yes" }, { status: "yes" }, { status: "na" }],
      // Tool Calling
      [{ status: "yes" }, { status: "yes" }, { status: "yes" }, { status: "yes" }, { status: "yes" }, { status: "yes" }, { status: "yes" }, { status: "yes" }, { status: "na" }],
      // Speculative Decoding
      [{ status: "yes" }, { status: "yes" }, { status: "na" }, { status: "yes" }, { status: "na" }, { status: "yes" }, { status: "yes" }, { status: "na" }, { status: "yes", label: "Eagle3 support", note: "Eagle3 support is documented.", source: "/dynamo/dev/additional-resources/speculative-decoding/speculative-decoding-with-v-llm" }, { status: "na" }],
    ],
  },
  {
    backend: "SGLang",
    features: INTERACTION_FEATURES,
    rows: [
      // Disaggregated Serving
      [{ status: "na" }],
      // KV-Aware Routing
      [{ status: "yes" }, { status: "na" }],
      // SLA-Based Planner
      [{ status: "yes" }, { status: "yes" }, { status: "na" }],
      // KV Block Manager
      [{ status: "wip" }, { status: "wip" }, { status: "wip" }, { status: "na" }],
      // Multimodal
      [{ status: "yes", label: "Supported serving patterns", note: "Supports aggregated EPD, E/PD, and E/P/D patterns. Traditional disaggregated EP/D is not supported.", source: "/dynamo/dev/knowledge-base/modular-components/backends/sg-lang/sglang-multimodal" }, { status: "yes", label: "Image-aware routing on Dynamo's SGLang image", note: "Hash forwarding is upstream in SGLang 0.5.13+ and Dynamo pins 0.5.17, so the shipped image routes on image overlap. A custom build without that patch still serves the request but degrades to text-prefix routing.", source: "/dynamo/dev/multimodal/multimodal-kv-routing" }, { status: "na" }, { status: "wip" }, { status: "na" }],
      // Request Migration
      [{ status: "yes" }, { status: "yes" }, { status: "yes" }, { status: "wip" }, { status: "yes" }, { status: "na" }],
      // Request Cancellation
      [{ status: "wip", label: "Remote-prefill limitation", note: "Cancellation during remote prefill is not supported in disaggregated mode.", source: "/dynamo/dev/knowledge-base/modular-components/backends/sg-lang/overview" }, { status: "yes" }, { status: "yes" }, { status: "wip" }, { status: "wip" }, { status: "yes" }, { status: "na" }],
      // LoRA
      [{ status: "no", label: "LoRA not supported", note: "SGLang does not support LoRA in Dynamo, so every LoRA pairing is unsupported.", source: "/dynamo/dev/knowledge-base/modular-components/backends/sg-lang/overview" }, { status: "no", label: "LoRA not supported", note: "SGLang does not support LoRA in Dynamo, so every LoRA pairing is unsupported.", source: "/dynamo/dev/knowledge-base/modular-components/backends/sg-lang/overview" }, { status: "no", label: "LoRA not supported", note: "SGLang does not support LoRA in Dynamo, so every LoRA pairing is unsupported.", source: "/dynamo/dev/knowledge-base/modular-components/backends/sg-lang/overview" }, { status: "no", label: "LoRA not supported", note: "SGLang does not support LoRA in Dynamo, so every LoRA pairing is unsupported.", source: "/dynamo/dev/knowledge-base/modular-components/backends/sg-lang/overview" }, { status: "no", label: "LoRA not supported", note: "SGLang does not support LoRA in Dynamo, so every LoRA pairing is unsupported.", source: "/dynamo/dev/knowledge-base/modular-components/backends/sg-lang/overview" }, { status: "no", label: "LoRA not supported", note: "SGLang does not support LoRA in Dynamo, so every LoRA pairing is unsupported.", source: "/dynamo/dev/knowledge-base/modular-components/backends/sg-lang/overview" }, { status: "no", label: "LoRA not supported", note: "SGLang does not support LoRA in Dynamo, so every LoRA pairing is unsupported.", source: "/dynamo/dev/knowledge-base/modular-components/backends/sg-lang/overview" }, { status: "na" }],
      // Tool Calling
      [{ status: "yes" }, { status: "yes" }, { status: "yes" }, { status: "wip" }, { status: "yes" }, { status: "yes" }, { status: "yes" }, { status: "no", label: "LoRA not supported", note: "SGLang does not support LoRA in Dynamo, so every LoRA pairing is unsupported.", source: "/dynamo/dev/knowledge-base/modular-components/backends/sg-lang/overview" }, { status: "na" }],
      // Speculative Decoding
      [{ status: "wip", label: "Limited integration", note: "Code hooks exist, but examples and documentation are not yet available." }, { status: "wip" }, { status: "na" }, { status: "wip" }, { status: "na" }, { status: "wip" }, { status: "na" }, { status: "no", label: "LoRA not supported", note: "SGLang does not support LoRA in Dynamo, so every LoRA pairing is unsupported.", source: "/dynamo/dev/knowledge-base/modular-components/backends/sg-lang/overview" }, { status: "wip" }, { status: "na" }],
    ],
  },
  {
    backend: "TensorRT-LLM",
    features: INTERACTION_FEATURES,
    rows: [
      // Disaggregated Serving
      [{ status: "na" }],
      // KV-Aware Routing
      [{ status: "yes" }, { status: "na" }],
      // SLA-Based Planner
      [{ status: "yes" }, { status: "yes" }, { status: "na" }],
      // KV Block Manager
      [{ status: "yes" }, { status: "yes" }, { status: "yes" }, { status: "na" }],
      // Multimodal
      [{ status: "yes", label: "Disaggregated image flows", note: "Supports EP/D and E/P/D image flows with image URLs and pre-computed embeddings.", source: "/dynamo/dev/knowledge-base/modular-components/backends/tensor-rt-llm/tensorrt-llm-multimodal" }, { status: "yes", label: "Image-aware KV routing", note: "Workers must publish KV events with block reuse enabled.", source: "/dynamo/dev/multimodal/multimodal-kv-routing" }, { status: "na" }, { status: "yes" }, { status: "na" }],
      // Request Migration
      [{ status: "yes" }, { status: "yes" }, { status: "yes" }, { status: "yes" }, { status: "wip" }, { status: "na" }],
      // Request Cancellation
      [{ status: "yes", label: "Known engine limitation", note: "The engine is temporarily not notified of cancellations, so resources for cancelled requests are not freed." }, { status: "yes", label: "Known engine limitation", note: "The engine is temporarily not notified of cancellations, so resources for cancelled requests are not freed." }, { status: "yes", label: "Known engine limitation", note: "The engine is temporarily not notified of cancellations, so resources for cancelled requests are not freed." }, { status: "yes", label: "Known engine limitation", note: "The engine is temporarily not notified of cancellations, so resources for cancelled requests are not freed." }, { status: "yes", label: "Known engine limitation", note: "The engine is temporarily not notified of cancellations, so resources for cancelled requests are not freed." }, { status: "yes", label: "Known engine limitation", note: "The engine is temporarily not notified of cancellations, so resources for cancelled requests are not freed." }, { status: "na" }],
      // LoRA
      [{ status: "no", label: "LoRA not supported", note: "TensorRT-LLM does not support LoRA in Dynamo, so every LoRA pairing is unsupported.", source: "/dynamo/dev/knowledge-base/modular-components/backends/tensor-rt-llm/overview" }, { status: "no", label: "LoRA not supported", note: "TensorRT-LLM does not support LoRA in Dynamo, so every LoRA pairing is unsupported.", source: "/dynamo/dev/knowledge-base/modular-components/backends/tensor-rt-llm/overview" }, { status: "no", label: "LoRA not supported", note: "TensorRT-LLM does not support LoRA in Dynamo, so every LoRA pairing is unsupported.", source: "/dynamo/dev/knowledge-base/modular-components/backends/tensor-rt-llm/overview" }, { status: "no", label: "LoRA not supported", note: "TensorRT-LLM does not support LoRA in Dynamo, so every LoRA pairing is unsupported.", source: "/dynamo/dev/knowledge-base/modular-components/backends/tensor-rt-llm/overview" }, { status: "no", label: "LoRA not supported", note: "TensorRT-LLM does not support LoRA in Dynamo, so every LoRA pairing is unsupported.", source: "/dynamo/dev/knowledge-base/modular-components/backends/tensor-rt-llm/overview" }, { status: "no", label: "LoRA not supported", note: "TensorRT-LLM does not support LoRA in Dynamo, so every LoRA pairing is unsupported.", source: "/dynamo/dev/knowledge-base/modular-components/backends/tensor-rt-llm/overview" }, { status: "no", label: "LoRA not supported", note: "TensorRT-LLM does not support LoRA in Dynamo, so every LoRA pairing is unsupported.", source: "/dynamo/dev/knowledge-base/modular-components/backends/tensor-rt-llm/overview" }, { status: "na" }],
      // Tool Calling
      [{ status: "yes" }, { status: "yes" }, { status: "yes" }, { status: "yes" }, { status: "yes" }, { status: "yes" }, { status: "yes" }, { status: "no", label: "LoRA not supported", note: "TensorRT-LLM does not support LoRA in Dynamo, so every LoRA pairing is unsupported.", source: "/dynamo/dev/knowledge-base/modular-components/backends/tensor-rt-llm/overview" }, { status: "na" }],
      // Speculative Decoding
      [{ status: "yes" }, { status: "yes" }, { status: "na" }, { status: "yes" }, { status: "na" }, { status: "yes" }, { status: "yes" }, { status: "no", label: "LoRA not supported", note: "TensorRT-LLM does not support LoRA in Dynamo, so every LoRA pairing is unsupported.", source: "/dynamo/dev/knowledge-base/modular-components/backends/tensor-rt-llm/overview" }, { status: "yes" }, { status: "na" }],
    ],
  },
];

export const PLATFORM_PREVIEW_COVERAGE: Record<string, Coverage> = {
  "v1.3.0-dev.1": { images: true, wheels: true, helm: true, crates: true },
  "v1.1.0-dev.3": { images: true, wheels: true, helm: false, crates: false },
  "v1.1.0-dev.2": { images: true, wheels: true, helm: false, crates: false },
  "v1.1.0-dev.1": { images: true, wheels: true, helm: true, crates: false },
};

/* PLATFORM.os splits two distinct facts a reader needs to keep straight:
 *   * "Containers and wheels" — the Dynamo CUDA container images (vLLM, SGLang,
 *     TensorRT-LLM) are built on this OS; wheels install here too. Ubuntu
 *     24.04 is the shipped container base (see container/context.yaml —
 *     every CUDA runtime image uses cuda-dl-base 25.11-cuda13.x-devel-ubuntu24.04).
 *   * "Wheels only" — the OS is not a shipped container base, but the wheels
 *     are manylinux_2_28 (glibc 2.28+), so `pip install ai-dynamo` runs on it.
 *     Getting Started's Local Installation lists Ubuntu 22.04 for that reason.
 * A future host that carries neither status would take a third scope value. */
export const PLATFORM = {
  gpus: ["Blackwell", "Hopper", "Ada Lovelace", "Ampere"],
  os: [
    { name: "Ubuntu", version: "24.04", arch: "x86_64, ARM64", scope: "Containers and wheels", chip: "ubuntu" },
    { name: "Ubuntu", version: "22.04", arch: "x86_64", scope: "Wheels only", chip: "ubuntu" },
  ],
  /* Cloud host images validated by CI. Scope matches the OS rows: AL2023 runs
     the shipped containers, it is not itself a container base. */
  csp: [{ provider: "AWS", os: "Amazon Linux 2023", arch: "x86_64", scope: "Containers and wheels" }],
  arch: ["x86_64", "ARM64 (Ubuntu 24.04 only)"],
  wheelsNote:
    "Wheels are built in a manylinux_2_28 environment (AlmaLinux 8, glibc 2.28+) and validated on Ubuntu 22.04 and 24.04. They install on any Linux distribution with glibc 2.28+ (Debian 11+, RHEL 9, etc.), but only Ubuntu 22.04/24.04 are officially verified.",
};

export const KNOWN_ARTIFACT_ISSUES = [
  {
    version: "v0.9.0",
    artifact: "dynamo-platform-0.9.0",
    issue: "Helm chart sets operator image to 0.7.1 instead of 0.9.0.",
    status: "Fixed in v0.9.0.post1",
  },
  {
    version: "v0.8.1",
    artifact: "vllm-runtime:0.8.1-cuda13",
    issue: "Container fails to launch.",
    status: "Known issue",
  },
  {
    version: "v0.8.1",
    artifact: "sglang-runtime:0.8.1-cuda13, vllm-runtime:0.8.1-cuda13",
    issue: "Multimodality not expected to work on ARM64. Works on AMD64.",
    status: "Known limitation",
  },
  {
    version: "v0.8.0",
    artifact: "sglang-runtime:0.8.0-cuda13",
    issue:
      "CuDNN installation issue caused PyTorch v2.9.1 compatibility problems with nn.Conv3d — performance degradation and excessive memory usage in multimodal workloads.",
    status: "Fixed in v0.8.1 (#5461)",
  },
];

export const CRATES_FIRST_PUBLISHED = [
  { crate: "dynamo-runtime", version: "0.1.0", date: "2025-03-18" },
  { crate: "dynamo-llm", version: "0.2.0", date: "2025-05-01" },
  { crate: "dynamo-async-openai", version: "0.4.1", date: "2025-08-27" },
  { crate: "dynamo-parsers", version: "0.5.0", date: "2025-09-18" },
  { crate: "dynamo-memory", version: "0.8.0", date: "2026-01-15" },
  { crate: "dynamo-config", version: "0.8.0", date: "2026-01-15" },
  { crate: "dynamo-tokens", version: "0.9.0", date: "2026-02-12" },
  { crate: "dynamo-mocker", version: "1.0.0", date: "2026-03-13" },
  { crate: "dynamo-kv-router", version: "1.0.0", date: "2026-03-13" },
  { crate: "dynamo-protocols", version: "1.1.0", date: "2026-05-04" },
  { crate: "dynamo-tokenizers", version: "1.2.0", date: "2026-06-02" },
];

/* Per-release ingestion-time stats for the Release Notes pages (ReleaseHeader
   tiles, UpgradePanel reading list) and the Deprecations / Known Issues
   accordion titles. Counted from each release's GitHub body at ingestion. */
export interface ReleaseStats {
  prs?: number;
  contributors?: number;
  firstTimers?: number;
  breaking: number;
  knownIssues: number;
}

/* COUNTING RULES — apply these when ingesting a new release so rows stay
   comparable across the two release-note eras:
   - prs / contributors: use the figure the body states outright ("merged 930
     PRs from 125 contributors"). Omit rather than derive: v1.0.0 states
     commits, not PRs, so prs is absent.
   - firstTimers: the release-wide figure the body states, or the complete
     release-wide list it enumerates when it states no figure. Scope and
     completeness both matter, because the bodies vary: a list confined to
     external contributors is not a release-wide count, and one introduced with
     "include" is by its own wording not the whole set. A cell that can only be
     backed by such a list is absent, not a number — v1.1.0 offers twelve
     bulleted "first-time external contributors ... include", which is both,
     so it is absent, and v1.2.0 names no first-timers at all.
     Bodies are the source for this column, as they are for every other column
     here. GitHub's own New Contributors lists compute a different quantity —
     first-ever merged PR anywhere in the repo, over the compare range — and
     disagree with the hand-written pre-v1.0.0 announcements (11 against the
     20 and 14 those bodies state). Neither is wrong; they answer different
     questions. Do not mix them into one column.
   - breaking: top-level entries under Breaking Changes, including its
     Deprecated/Removed subsections, but excluding subsections that only
     restate a prior release's announced deprecations ("vX.Y.Z
     Future-Deprecation Reminders"). Pre-v1.0.0 bodies have no Breaking Changes
     section; v0.9.0's lone Deprecation Notices entry is the same entry class
     and counts, and a release with no such section at all is a true 0.
   - knownIssues: one per named issue — the per-issue heading where the body
     gives each issue its own, otherwise the top-level bullets.
   Known exception: v1.0.0 breaking is published as 41, but its body holds 40
   top-level entries and no rule reproduces 41. Left as published.

   The absent prs and contributors cells are absent for cause, not for want of
   looking. Neither the release bodies, the TPM release archive, nor the git
   history yields a figure comparable to the stated ones: the archive's own
   numbers disagree with each other (v0.9.0 is written up as both 217 and 935
   PRs for the identical window, and v1.0.0's 708 is quoted as commits in one
   place and as merged PRs in another), and no tag-to-tag count reproduces the
   three published anchors — the closest method returns 910/572/899 against a
   published 930/603/896, missing in both directions, so it cannot be trusted
   to fill the rest. Leave them absent unless a method reproduces all three. */
export const RELEASE_STATS: Record<string, ReleaseStats> = {
  "v1.4.0": { prs: 640, contributors: 127, firstTimers: 29, breaking: 51, knownIssues: 19 },
  "v1.3.0": { prs: 930, contributors: 125, firstTimers: 24, breaking: 24, knownIssues: 10 },
  "v1.2.0": { prs: 603, contributors: 82, breaking: 5, knownIssues: 11 },
  "v1.1.0": { prs: 896, contributors: 113, breaking: 8, knownIssues: 20 },
  "v1.0.0": { contributors: 90, firstTimers: 34, breaking: 41, knownIssues: 14 },
  "v0.9.0": { firstTimers: 14, breaking: 1, knownIssues: 13 },
  "v0.8.0": { firstTimers: 20, breaking: 0, knownIssues: 14 },
  "v0.7.0": { firstTimers: 2, breaking: 0, knownIssues: 7 },
  "v0.6.0": { firstTimers: 4, breaking: 0, knownIssues: 3 },
};

export const NIGHTLY_BUILDS: NightlyBuild[] = [
  {
    version: "1.5.0.dev20260820",
    date: "Aug 20, 2026",
    packages: ["ai-dynamo", "ai-dynamo-runtime", "kvbm"],
  },
  {
    version: "1.5.0.dev20260819",
    date: "Aug 19, 2026",
    packages: ["ai-dynamo", "ai-dynamo-runtime", "kvbm"],
  },
  {
    version: "1.5.0.dev20260817",
    date: "Aug 17, 2026",
    packages: ["ai-dynamo", "ai-dynamo-runtime"],
    note: "kvbm was not published for this nightly.",
  },
];


export const NIGHTLIES_NOTE =
  "ai-dynamo, ai-dynamo-runtime, and kvbm nightly builds from main publish wheels tagged `*.devYYYYMMDD` (since Apr 24, 2026). Install with pip or uv using `--pre` and the NVIDIA extra-index pattern shown above. Runtime containers use rolling `*-runtime-nightly:latest` tags on NGC.";
