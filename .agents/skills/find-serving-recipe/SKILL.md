---
name: find-serving-recipe
description: Answers "for this model, this hardware, this GPU budget and this workload, what is the best known serving configuration, and how much do I trust it?" by walking an ordered set of recipe catalogs with provenance gates, and writes a recipe dossier recording what was found, where, and at what confidence. Use at baseline-selection time to perform the ladder's recipe-catalog scan, and during optimization whenever a performance question may already be answered by a published recipe (before spending GPU time deriving a config from scratch). Do not use to deploy anything or to replace an established baseline.
license: Apache-2.0
user-invocable: true
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - recipes
    - optimization
    - provenance
---

# Find a Serving Recipe

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Known-good serving configurations exist for far more model and hardware combinations than this
repository's `recipes/` tree carries, and deriving one from scratch on GPU time is the most
expensive way to obtain one. This skill defines "the catalog": an ordered set of sources, each
with a trust level, so that a recipe scan finds what exists and never presents an unverifiable
config as deployable.

Two invocation contexts, with different outputs:

- **Baseline selection** (the interviewer's recipe-catalog scan): the dossier feeds the
  proposal-and-confirmation flow. This skill only finds and ranks; proposing and capturing
  confirmation stay with the interviewer.
- **Mid-engagement candidate hunting**: before deriving a candidate configuration from first
  principles, check whether a published recipe already answers the performance question. A found
  recipe becomes a HYPOTHESIS for the normal candidate pipeline (consult, materialize,
  adversarial review); it never bypasses that pipeline and it NEVER re-selects the engagement
  baseline.

## Ground rules

1. **Never answer from memory.** Every claim in the dossier comes from a source fetched or read
   during this invocation, cited with its path or URL and, where available, a commit or date. If
   a source cannot be reached and no usable local copy exists, say so explicitly and move to the
   next tier; do not reconstruct what the catalog "probably" contains.
2. **Refuse to guess a hardware match.** If no source names the target GPU or a documented
   equivalent, report no match for that source. Do not silently map one SKU onto another.
3. **The tag-resolvability gate, applied to every candidate at every tier.** Resolve the recipe's
   container image before ranking it:
   - `nvcr.io/...` images: query the NGC registry API for the tag.
   - Docker Hub images: `https://hub.docker.com/v2/repositories/<org>/<repo>/tags/<tag>`.
   A candidate whose image tag does not resolve, or resolves only to a mutable tag (`latest`,
   `dev`, `nightly` without a digest), is **ceiling-only**: it may inform expectations but must
   never be handed to a deploy step. A digest-pinned image (`@sha256:...`) passes regardless of
   tag mutability.
4. **Carry the model card forward.** Before any tier, read the Hugging Face model card for the
   target model. It frequently points at the authoritative recipe for that model, states a
   minimum engine version, and is the only source of the model author's sampling and context
   correctness settings. Record those settings in the dossier; no catalog carries them.
5. **Stop at the first tier that yields a candidate meeting the confidence bar** for the
   invocation context (deployable for baseline selection; hypothesis-grade for candidate
   hunting). Later tiers may still be consulted for expected performance.
6. **Freshness is part of the verdict, never a reason to discard.** Serving recipes rot
   unevenly. Split every candidate into its DURABLE content (topology, parallelism dimensions,
   precision and quantization, KV-cache dtype, memory fractions, sizing, workload fit, and the
   reasoning behind those choices) and its VERSION-BOUND content (exact flag names and defaults,
   the image tag, kernel and backend selections, measured performance). Durable content is
   first-class evidence regardless of the recipe's age; carry it into the dossier and into any
   candidate's rationale. Version-bound content ages: record the engine version the recipe pins
   (image tag, `min_*_version`, or commit) and its verification or publication date, and compare
   against the engine's CURRENT stable release (registry tag list, engine release page, or the
   catalog's latest entry). When the pinned engine is behind current, verify each flag against
   the current CLI and port renamed ones, treat the recipe's measured performance as historical
   (a shape hint, not a target), pin the current image, and re-verify before anything is graded
   deployable. A recipe verified on the current or immediately previous minor release with
   unchanged flags may be graded `deployable`; anything older is `hypothesis` until re-verified.
   A version label alone is evidence of relabeling, not revalidation; only a recorded
   re-verification resets the clock. Prefer the fresher of otherwise comparable candidates, but
   never let a fresher, thinner recipe displace the durable lessons of an older, richer one.

## Tier 0: this repository

- `docs/fern/pages/recipes/_catalog/`: the schema-validated machine-readable index
  (`index.yaml`, `recipes/*.yaml`, validated by `validate.py` against `schema.json`). Match on
  `model.hf_id`, `targets[].hardware`, `targets[].runtime.framework`, and `targets[].topology`.
  Two fields live at different levels: `status` (`validated` or `experimental`) is an ENTRY-level
  property and gates every target under it, while `recommended` is a TARGET-level property. Prefer
  targets with `recommended: true` under entries with `status: validated`; an `experimental` entry
  caps all of its targets at `hypothesis`. Surface the entry's `gaps` list in
  the dossier rather than hiding it.
- The matched entry's `deploy.asset` points at the `recipes/<model>/.../deploy.yaml`
  DynamoGraphDeployment and its benchmark manifests; `expected_performance.summary` carries the
  measured numbers when `available: true`.

A Tier 0 match with a resolvable image is the best possible outcome: zero translation, attached
perf, and a validated flag. Use it and stop. A Tier 0 entry that structurally matches but fails
the verdict bar (mutable or unresolved image, `experimental` status, unsatisfiable prerequisites,
stale pin) is recorded with its lower grade and the walk CONTINUES: a structural match is not a
usable match.

## Tier 1: authoritative external catalogs

Consult in this order when Tier 0 yields no candidate meeting the required verdict.

1. **`vllm-project/recipes`** (vLLM engine). Consume the JSON API, not the YAML files:
   `https://recipes.vllm.ai/models.json`, then `/<hf_org>/<hf_repo>.json`, then
   `/<hf_org>/<hf_repo>/hw/<hardware>.json`. Read the MODEL-LEVEL JSON first and take three
   things from it, because they do not appear on the per-hardware response: the version floor at
   `$.model.min_vllm_version` (variants may carry their own under `$.variants.<name>.min_vllm_version`),
   the verification signal at `$.meta.hardware`, a hardware-to-status map (a hardware key absent from
   that map is unverified for this model even if `/hw/<hardware>.json` renders a config), and the
   freshness date at `$.meta.date_updated`. Then fetch the per-hardware response and branch on its
   `deploy_type`: for `single_node` it returns `argv`, `docker_image`, `env`, and a `hardware_profile`
   whose `gpu_count` is the total; for `multi_node` it returns `head_argv`, `worker_argvs`,
   `node_count`, and a `hardware_profile` whose `gpu_count` is PER NODE, so the total requirement is
   `node_count` times `gpu_count` (the `strategy_spec` names the interconnect assumption, for example
   InfiniBand for multi-node TP). Never read the flat single-node fields from a multi-node response;
   a missing `argv` is a schema branch, not an absent recipe. Two caveats: most recipes fall back to
   a mutable `latest` image (the tag gate then classifies them ceiling-only unless the engagement
   pins its own image), and the exported JSON drops the prefill/decode `strategy_overrides`; for
   disaggregated topology, read the recipe's YAML from the git repo and compose against its
   `strategies.json`.
2. **`NVIDIA/srt-slurm-recipes`** for frontier models on Blackwell-class hardware, especially
   disaggregated and multi-node. Recipes are SLURM-shaped but carry the full engine
   configuration, worker split, and image. **Exclude `**/agentic/` and `*-sa/` paths from
   deployable candidates**: those port externally tuned benchmark configs and are quarantined to
   ceiling-only (see Tier 4). Expect a meaningful fraction of container tags to fail the gate.
3. **`llm-d/llm-d` `guides/`** ("well-lit paths"). Tested, benchmarked, Kubernetes-native
   recipes with sized prefill/decode Deployments; when the target model is covered, the best
   public source of sized disaggregation topology. Two checks before any llm-d candidate is
   graded above `hypothesis`: (a) IMAGE CAPABILITY: the guide's image component may select a
   stock upstream image or an `llm-d`-hosted variant (`ghcr.io/llm-d/llm-d-*`) that carries
   patches upstream lacks, such as the NVSHMEM fix for RoCE; record which, and treat a
   patched-variant dependency as a prerequisite the target must satisfy, not a stock image;
   (b) COORDINATION TRANSLATION: guides that use LeaderWorkerSet (`LWS_WORKER_INDEX`,
   `LWS_GROUP_SIZE`, `LWS_LEADER_ADDRESS`) compute ranks and addresses in shell at start-up,
   which has no literal DGD equivalent; those manifests are not a mechanical translation and
   stay `hypothesis` until the multi-node coordination is re-expressed in Dynamo's own terms
   and verified. Only single-pod-per-worker guides with literal `args:` arrays on stock images
   translate near-mechanically.

## Tier 2: engine-native catalogs

For flag detail, and for models Tier 1 misses.

- **TensorRT-LLM**: fetch `https://nvidia.github.io/TensorRT-LLM/_static/config_db.json`
  (versioned snapshots live under `/<version>/_static/...` for drift checks). The published JSON
  covers aggregated serving only; for disaggregated entries read
  `examples/configs/curated/lookup.yaml` in the TensorRT-LLM repo. Treat
  `validated_trtllm_commit` as a floor, not a freshness signal. Never harvest from
  `examples/models/`, which is legacy.
- **SGLang cookbook** (`https://docs.sglang.io/cookbook/`, source under `docs/cookbook/` in the
  sglang repo). Treat cells as FLAG SOURCES, not deployable recipes: prefer `verified: true`
  cells, skip in-progress ones, and note that the cookbook's benchmark records share the cell's
  match key, so a verified cell often carries measured TTFT/TPOT/throughput for the same config.
  Its NVIDIA-side images are mutable tags and fail the gate; take the flags, not the image.
- **NIM support matrix** as a sanity check on what TP and precision NVIDIA ships for a model on
  a given GPU. It will not give engine flags. Scrape the version-pinned docs URL, not `latest`.

**Conflict precedence.** First-party sources contradict each other. When two in-tier sources
disagree, prefer in order: (1) a config with an attached measured result on the target hardware,
(2) a config with a resolvable image digest, (3) a config with a validated commit pin,
(4) recency. Never silently pick one; record the conflict in the dossier.

## Tier 3: narrative sources

Engine release blogs and LMSYS posts. Consult only for a model that just released and appears in
no catalog. Harvested flags are hypothesis-grade at best; blog-pinned nightly images rot within
weeks, so everything from this tier fails the tag gate by construction. Both blogs increasingly
point at their catalogs; follow the pointer instead of scraping the post.

## Tier 4: quarantined, ceiling reference only

**SemiAnalysis / InferenceX** configs (`SemiAnalysisAI/InferenceX`, and their ports under
`agentic/` and `*-sa/` in NVIDIA repos) are state-of-the-art but tuned for a benchmark
leaderboard: a large fraction depend on containers that no longer exist, on release candidates,
or on feature-branch builds, and their speculative-decoding results use SIMULATED acceptance
lengths, not measured ones. Use them for exactly one thing: headroom. The dossier may say "an
externally published result reports X tok/s/GPU for this model on this hardware; the current
configuration achieves 0.7X", with the simulation caveat attached when the ceiling involves
speculative decoding. A Tier 4 config must never be handed to a deploy step, regardless of
whether its image resolves.

MLPerf inference results (`mlcommons/inference_results_v*`) sit here too: frozen, high-integrity,
wrong harness shape. Submitter `scripts/slurm_llm/` directories occasionally carry real
disaggregated topology worth reading for sizing, with measured results attached.

## Expected performance is a separate axis

Most catalogs carry configs without numbers. When the dossier's config comes from a source
without measured performance, fill the expectation from:

- Tier 0's `expected_performance` fields, when a related target exists; and
- `https://developer.nvidia.com/search-data/nv_inference_benchmark.json`, a public index of
  measured records (TTFT, TPOT, per-GPU throughput, prefill/decode split) including Dynamo
  entries. Note in the dossier when the record does not name the image it ran on.

Label every expectation with its source and hardware; an expectation from different hardware is
a shape hint, not a target.

## Verdicts

Every candidate gets exactly one grade, decided by these conditions in order:

- `ceiling-only` if ANY of: the image tag does not resolve or is mutable without a digest; the
  source is Tier 4 (quarantined) or a Tier 3 narrative with no pinned image; the config could not
  be fetched or read during this invocation.
- `deployable` if ALL of: the source is Tier 0 or Tier 1 (or a Tier 2 config with a stock release
  image); the model, hardware class, and GPU count match the engagement exactly (no inferred
  SKU mapping); every infrastructure prerequisite the recipe declares is satisfiable on the
  stated target or has a documented adaptation; the image resolves to an immutable release tag
  or digest; the freshness check passes (current or immediately previous minor with unchanged
  flags); the source is not marked experimental or unverified (for Tier 0 that is the ENTRY-level
  `status`; for vLLM recipes it is the model-level `$.meta.hardware` map naming the target hardware);
  and the translation into a
  DynamoGraphDeployment is mechanical (no guessed fields).
- `hypothesis` otherwise: a real, fetched, resolvable config that needs porting, re-verification,
  adaptation, or a close-but-not-exact hardware or topology match before it could be proposed.

Record the failed conditions next to any grade below `deployable`, so a reader knows what
would promote it.

## Output: the recipe dossier

Write the dossier as an IMMUTABLE snapshot at
`<EXP_ROOT>/analysis/recipe-dossier/<NNN>-<UTC timestamp>.md` (NNN zero-padded, increasing per
invocation), in both invocation contexts (the interviewer establishes `EXP_ROOT` before the
ladder runs). When invoked standalone with no `EXP_ROOT`, use `./recipe-dossier/` under the
current working directory with the same layout and tell the operator where it landed.
Never modify a snapshot after writing it; a later invocation writes the next
snapshot and may summarize deltas against the previous one. Maintain
`<EXP_ROOT>/analysis/recipe-dossier/index.md` listing every snapshot with its SHA256. Return the
snapshot path and its SHA256 to the caller; callers cite exactly that pair, so an evidence record
written against snapshot 001 still verifies after snapshot 002 exists. Later iterations reuse
the latest snapshot by reading the index. Each snapshot contains:

- the question (model, hardware, GPU budget, workload, topology preference);
- the model card's correctness settings (sampling, context, parsers) and minimum engine version;
- every tier consulted, what was searched, and what it returned (including "no match");
- for each candidate: source with path or URL and commit or date, image and its gate result
  (resolvable, digest-pinned, mutable, missing), pinned engine version versus current stable
  and the verification date (the freshness check), flags or manifest, measured performance
  with its source, the verdict (`deployable`, `hypothesis`, or `ceiling-only`) and, below
  `deployable`, the conditions that failed;
- conflicts encountered and how precedence resolved them;
- the selected candidate and why, or an explicit statement that nothing met the bar.

The dossier is evidence for the interviewer or the hypothesis pipeline. This skill does not
deploy, does not edit tracked recipes, and does not modify the engagement baseline.
