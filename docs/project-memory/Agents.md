---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Project operating memory
---

## Project operating memory

Last verified: 2026-10-07 (Asia/Ho_Chi_Minh)

### Authority and scope

- Root `AGENTS.md` remains the canonical agent instruction file; this file is supplementary project memory.
- Current feature scope: protected Dynamo vLLM loading supports plain-model bypass, software file-key profiles, and a TPM-bound profile. Protected profiles authenticate packages and materialize plaintext only in private tmpfs sessions.
- Current implementation branch: `feature/model-protection`. Check Git status before editing or committing; this worktree contains changes from several tasks.
- Architecture changes crossing Python, Rust, container, or Kubernetes boundaries require a Dynamo Enhancement Proposal before implementation. The active proposal is [DEP #14764](https://github.com/ai-dynamo/dynamo/issues/14764).

### Tech stack

- Rust Cargo workspace under `lib/`; Python bindings use PyO3/maturin under `lib/bindings/python/`.
- Python package under `components/src/dynamo/`; the first secure-loader integration target is `components/src/dynamo/vllm/`.
- The protected bootstrap and image build currently target vLLM `0.30.0` and
  Omni `0.30.0rc1`. Read `TASKS.md` for image/model acceptance; earlier vLLM
  0.28/0.29 test results are historical and do not validate this image.
- Kubernetes operator and CRDs are Go code under `deploy/operator/`.
- Deployment configuration can add native volumes and mounts through the v1beta1 component `podTemplate`.

### Authoritative paths for this feature

- `docs/project-memory/model-protection-architecture.md`: consolidated architecture contract, detailed build/activation/runtime flows, unresolved decisions, and acceptance gates.
- `docs/project-memory/model-encryption-only.md` and
  `docs/project-memory/model-encryption-only.en.md`: bilingual issuer packaging
  and customer Compose guide for the software `encrypted-file` profile.
- `docs/project-memory/model-protection-security-review.md`: 2026-09-12 review that
  drove the current corrections, plus section 6 re-review on 2026-09-13.
  Section 7 adds R-15–R-18 after producer/consumer and lifecycle review.
  R-01–R-18 define the current hardening set. Their software fixes now exist
  in the working tree, but findings remain acceptance-open until their stated
  integration/fault evidence is captured. Do not describe the feature as
  product-ready or blocked only by physical acceptance.
- Section 5 of that review records C-01–C-09 and G-01–G-04. Passing unit,
  simulator and compile checks do not close physical TPM, software issuer
  custody, target GPU,
  cross-server or cluster acceptance.
- `components/src/dynamo/vllm/main.py`: model prefetch, engine creation, and Dynamo model registration lifecycle.
- `components/src/dynamo/vllm/snapshot.py`: snapshot/CRIU engine lifecycle.
- `components/src/dynamo/common/utils/namespace.py`: `DYN_NAMESPACE` resolution; default is `dynamo`, with optional worker suffix.
- `deploy/operator/api/v1beta1/dynamocomponentdeployment_types.go`: DGD/DCD pod template and shared-memory API.
- `deploy/operator/internal/dynamo/shared_memory.go`: existing `/dev/shm` memory-backed `emptyDir` implementation.
- `Cargo.toml`: shared Rust dependencies; `zeroize` is already declared at workspace level.
- `lib/model-protection/`: Rust core plus packager/issuer CLIs for detection, package/license/issuer-record verification, authenticated records, TPM unwrap, safe tmpfs materialization, persistence preflight, cancellation and session ownership. D-046 software-key custody is implemented; offline key-directory operations remain an acceptance task.
- `lib/bindings/python/rust/model_protection.rs`: opaque PyO3 bridge; Python never receives the DEK.
- `components/src/dynamo/vllm/protection_bootstrap.py`: two-phase protected-model bootstrap and vLLM 0.30.0 allowlist.
- `deploy/model-protection/`: reference image and single-GPU TPM-bound Kubernetes profile.
- `.github/codeowners/areas.yaml`: `lib/model-protection/` is owned by runtime;
  `deploy/model-protection/` is owned by operator. Regenerate root `CODEOWNERS`
  after changing these paths.

### Critical secure-loader rules

- Plain models bypass manifest, key, decrypt, tmpfs, license, and machine-binding logic.
- Either exact root marker `model.protection.json` or `model.protection.sig` makes the input secure-or-invalid. Do not scan arbitrary shard suffixes or cap the number of files in a plain model.
- Every protected package requires a valid signed manifest and authenticated materialization. Key-release requirements depend on the signed profile: `encrypted-tpm` requires its license and device binding; `encrypted-file` uses its package DEK file and does not provide device binding.
- In Compose deployments with separate frontend and worker containers, the worker must expose its self-hosted metadata endpoint on the shared internal network. Configure `DYN_SYSTEM_PORT`; do not publish it to the host unless the deployment security review requires it.
- Plaintext weights may be written only to a verified tmpfs mount.
- Protected-worker cache paths that may contain model-derived artifacts must
  also use bounded memory-backed volumes; do not redirect them to node disk.
- Resolve the tmpfs root as `/run/<validated-DYN_NAMESPACE>-models`; never use the literal string `DYN_NAMESPACE` as a directory name.
- Every load uses `/run/<validated-DYN_NAMESPACE>-models/<random-session-id>` to avoid collisions within a namespace.
- Validate `DYN_NAMESPACE` before using it in a filesystem path; reject separators, traversal, empty values, and ambiguous normalization.
- Use owner-only permissions: directory `0700`, files `0600`, process `umask 077`.
- D-008 is confirmed: retain plaintext tmpfs for the full worker lifetime in V1. Do not introduce early cleanup until reread behavior is proven; zeroize owned key material immediately after decryption.
- Do not log keys, plaintext, authentication tags, or sensitive manifest contents.
- V1 software issuer keys are allowed only in an encrypted offline key
  directory outside source, images and runtime config. Key files require
  absolute no-symlink paths, dedicated ownership, mode `0600`, encrypted-at-rest
  storage and locked/non-dumpable zeroizing process memory. Never pass secret
  bytes or passphrases through argv, environment or logs.
- V1 uses TPM 2.0 through `/dev/tpmrm0` + `tpm2-tss` ESAPI: a certified policy-only RSA-2048 DUK unwraps an exact 256-byte OAEP/SHA-256 DEK ciphertext under ECDSA P-256 `PolicyAuthorize`; signed `command_parameters_hash` drives `PolicyCpHash`; PCR binding is disabled. The issuer uses separate Ed25519 package/license keys, an ECDSA P-256 policy key and an AES-256 software KEK. Offline entitlement is perpetual with controlled reactivation.
- A non-exportable TPM key alone does not enforce license claims. TPM authorization policy or service-side entitlement enforcement must be specified; runtime lease expiry cannot revoke keys or weights already extracted.
- Production `SecretDek` storage uses a dedicated anonymous page protected by
  `mlock` and `MADV_DONTDUMP`; key release fails with
  `SECRET_MEMORY_UNAVAILABLE` if either control is unavailable. tmpfs `noswap`
  still does not protect other decrypt/engine buffers, so the strict
  persistence contract must cover all sensitive memory and worker processes.
- Protected bootstrap must set core size to zero, disable process dumpability
  and require effective cgroup-v2 swap limit/current to be zero before reading
  trust configuration or requesting TPM key release. Plain bootstrap must not
  execute this preflight.
- vLLM 0.28 starts EngineCore with `spawn`, which can reset dumpability. The
  protected TP/PP/DP=1 profile must install the approved EngineCore entry wrapper
  before materialization so every model-loading process reapplies the Rust
  persistence policy. Do not generalize this wrapper to unapproved topologies.
- Caller-supplied `ec_transfer_config` is executable extension input and must be
  rejected before key release. The effective gate may accept only the exact
  Dynamo multimodal cache connector created internally before `VllmConfig`.
- Secure path validation must cover intermediate path components and verify the bytes actually consumed; a retained file descriptor does not freeze mutable file contents.
- Package signing, license signing, and container signing use separate trust domains and key IDs.
- Public modules, package formats, file extensions, CLI commands, configuration fields, and runtime paths must use vendor-neutral model-protection terminology.
- USB dongles are excluded from this architecture. Production key release is limited to TPM-backed offline trust or an online attested key service.
- V1 generates a fresh DEK per customer/model-version artifact, uses one file entry with ordered independent AEAD records, and keeps public metadata separate from session state.
- The V1 record wire format is frozen in `lib/model-protection/src/records.rs`: `MPROTV1\0`, big-endian fixed-width header, independent 16-byte GCM tags and the `model-protection-record-v1\0` AAD domain. `format.rs` is the source for bounds and exact-byte manifest verification.
- Secure namespace values are 1-128 ASCII bytes; first/last characters are alphanumeric and interior characters are limited to `[A-Za-z0-9._-]`.
- Protection errors cross boundaries only through `ProtectionError::code()`/stable display strings. Internal callers may log only the static `sanitized_reason()` plus allowlisted opaque metadata; never log raw `Debug`/OS errors at this boundary.
- Materialization requires an empty private tmpfs model directory. Input files open nonblocking before FD type checks; partial outputs and plaintext record buffers use RAII cleanup. Public metadata follows the exact allowlist/index contract in architecture section 5.5.
- Package and profile-specific license verification must produce an opaque
  `AuthorizedModel`; materialization consumes an opaque `SecretDek`. Raw record
  decryption is crate-private. The TPM profile constructs the DEK through the
  feature-gated ESAPI path; software profiles read an owner-only package DEK
  file through the configured file provider. Keep these providers isolated by
  the signed package profile and never accept raw DEK bytes from argv or env.
- Session preparation requires the resolved directory itself to be a private tmpfs mount root, obtains one nonblocking owner lock, removes only UUID-named stale sessions, checks tmpfs plus cgroup-v2 memory headroom, pins the created session directory by FD for materialization, and owns cleanup through `Drop` or explicit `cleanup()`.
- Secure vLLM V1 is an explicit allowlist: local safetensors with `TP=PP=DP=1`. Runtime weight updates, dynamic LoRA, external plugins/loaders, snapshot, Ray/multi-node engines, remote secure sources, and other unreviewed paths fail before key release.
- Protected materialization runs off the asyncio event loop with the Python GIL
  released. Signal/task cancellation must cancel and join the Rust writer
  before cleanup; never remove a session while materialization can still write.
- Issuer records are signed V2 envelopes. Their exact payload binds manifest
  digest, artifact/customer/model/version, KEK ID/version and wrapped DEK;
  legacy/unsigned records are rejected before software KEK unwrap.
- The checked-in Kubernetes V1 profile explicitly uses the same
  `DYN_NAMESPACE` on frontend/backend and an empty worker suffix so
  `/run/<namespace>-models` remains stable across operator rollout defaults.
- Until snapshot and multi-node secure lifecycles are approved, fail closed rather than silently running them insecurely.
- Secure packages must not enable `trust_remote_code` in V1.

### Validation convention

- Security-boundary changes require focused negative tests as well as the successful load path.
- Never report implementation, hardening, release, or rollback validation as complete without command output or equivalent direct evidence.
- Current unit/compile evidence includes an independent frozen TPM authPolicy
  digest, source-level vLLM 0.29 compatibility checks and historical local
  vLLM 0.28 plain GPU inference. The real 0.29 tests skip until the protected
  image is rebuilt; fixtures and plain mode cannot close that gate.
