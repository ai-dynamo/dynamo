<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Protected Model Deployment Guide

This guide describes the Phase 1 protected-model profile. It covers package
creation on the issuer host, runtime image construction, and local or
Kubernetes startup on a customer server.

Production enrollment is not yet hardware-qualified. The authority verifier,
registry, resumable collector provisioning/response and intent-bound issuance
recovery are implemented. Native simulator tests pass; physical acceptance
remains open. The public policy signer is loaded transiently from a TPMT_PUBLIC
file, not persisted as an external public object. See
[implementation status](../../docs/project-memory/TPM-Implementation-Status.md).
Single-/multi-GPU product configurations require separate process and lifecycle
acceptance; this guide's baseline does not enable multi-worker loading.

## Runtime behavior

Layer schema V2 accepts inline JSON in `DYN_MODEL_PROTECTION_CONFIG` or an
absolute JSON file path. The four flags `package_verification`,
`license_verification`, `tpm_binding`, `secure_materialization` default false.
The signed package profile determines the required combination; false flags
do not authorize a downgrade of an existing TPM package.

New packages may use `PROTECTION_PROFILE=encrypted-file` for a private per-model
key file, or `encrypted-file-license` to additionally verify a software license.
Use `model-protection-issue export-file-key` / `issue-file-license` only on the
issuer host. These profiles do not bind the model to a device. They retain
tmpfs, no-swap, memlock and loader requirements. Runtime examples are
`runtime.file.json.example` and `runtime.file-license.json.example`.
Legacy V1 packages/configuration files retain the fixed TPM profile.

Protected image builds pin the base to vLLM 0.30.0 and Omni 0.30.0rc1.
`SKIP_BASE_BUILD=true` reuses a local base only if both vLLM and the protected
loader support exactly 0.30.0. Final diagnostics separate version/loader mismatch
from a missing TPM-enabled extension. Rebuild the base after this source update;
an older base may contain the previous Python loader even with vLLM 0.30.0 installed.
The protected build script overrides the general vLLM image pins in
`container/context.yaml`. Build through that script so the runtime and protected
loader use the same source revision and exact vLLM version.

Normal and protected models use separate paths:

```text
normal model      -> Dynamo -> vLLM -> GPU
protected package -> verify license and TPM -> decrypt to tmpfs -> vLLM -> GPU
```

A normal model does not require a license, TPM, key, or protected tmpfs. A
protected package is detected by its root protection markers and fails closed
if verification, machine binding, or decryption fails. Plaintext weights are
written only below `/run/<validated-namespace>-models/<session-id>` and are
never stored as a normal disk directory.

Phase 1 supports one aggregated vLLM 0.30.0 process, one GPU, local
safetensors, and the Dynamo-created multimodal cache connector. LoRA,
snapshot/CRIU, remote code, external tokenizer/config, speculative loading,
runtime weight updates, and multi-worker protected loading are rejected before
the TPM releases the data-encryption key.

## Key custody

The issuer host owns three independent signing keys and one AES-256 key:

- package signing key: signs the immutable package manifest;
- license signing key: signs the customer entitlement;
- TPM policy signing key: authorizes the certified TPM recipient;
- AES-256 KEK: wraps the per-package DEK.

Private keys and passphrases stay on a dedicated offline issuer host. They must
not be placed in source control, a Docker image, a customer runtime, command
arguments, or environment variables. The customer receives only the encrypted
package, signed license, public trust keys, and runtime configuration.

The active V1 issuer uses encrypted PKCS#8 files and a raw 32-byte KEK. HSM or
PKCS#11 is not required. Customer runtime still requires a non-exportable TPM
DUK at the configured persistent handle.

## 1. Build the issuer tools

Run these commands in the repository checkout on the issuer host:

```bash
cargo build -p dynamo-model-protection \
  --features packager --release --locked
```

The resulting binaries are `target/release/model-protection-pack` and
`target/release/model-protection-issue`. Keep the issuer binaries and key
directory off the customer server.

For registry-enforced issuance, build with `--features packager,enrollment-authority`
instead; OpenSSL and SQLite development libraries are required. This also builds
`model-protection-enrollment-authority`. A packager-only issuer refuses issuance
unless explicitly given `--allow-development-certification true` for pilot use;
that opt-in is not production enrollment. Authority-mode issuance always requires
an issuer-owned registry. Do not copy authority keys/state or binaries into the
customer runtime image.

## 2. Create or provision issuer keys

Use the organization's approved offline key-generation and backup process.
The minimum file layout is:

```text
/offline-keys/
├── package-signing.pem       # encrypted Ed25519 PKCS#8 private key
├── license-signing.pem       # encrypted Ed25519 PKCS#8 private key
├── policy-signing.pem        # encrypted P-256 PKCS#8 private key
└── issuer-kek.bin            # exactly 32 random bytes
/run/issuer-secrets/
├── package.pass
├── license.pass
└── policy.pass
```

Keep all files owner-only (`0700` directory, `0600` files), on encrypted
offline storage. Generate public trust keys separately and distribute only the
raw public keys to the customer runtime. Never use the private files below
`/runtime`, `/models`, or the final image.

## 3. Encrypt a model package

The source model must be an ordinary Hugging Face safetensors directory. The
helper refuses relative paths and refuses to overwrite an existing output:

```bash
PACKAGE_SIGNING_KEY=/offline-keys/package-signing.pem \
PACKAGE_PASSPHRASE_FILE=/run/issuer-secrets/package.pass \
KEK_KEY_FILE=/offline-keys/issuer-kek.bin \
KEK_KEY_ID=issuer-kek-v1 \
KEK_KEY_VERSION=1 \
deploy/model-protection/protect-model.sh \
  /srv/models/LFM2.5-2.6B \
  /srv/artifacts/LFM2.5-2.6B-secure \
  customer-scope \
  lfm2.5-2.6b \
  1
```

The output contains `package/` and an issuer-only `issuer-record.json`. Ship
only `package/`; the issuer record is required later to issue a license and
must not be copied to the customer runtime. The package contains encrypted
weight records, signed public metadata, and no plaintext safetensors.

## 4. Issue a machine-bound license

Obtain the certified-device record and signature from the approved TPM
enrollment process. Issue the license on the same offline issuer host with
`model-protection-issue`. The command requires the package, issuer record,
certified TPM identity, package public key, enrollment public key, license and
policy signing keys, and the same KEK ID/version used by the packager.
The example requires the `enrollment-authority` build and an active certification
committed by the authority in `/srv/enrollment/registry.sqlite`, in an owner-only
directory. A signed device JSON alone is not an admission source.

```bash
model-protection-issue \
  --registry /srv/enrollment/registry.sqlite \
  --package /srv/artifacts/LFM2.5-2.6B-secure/package \
  --issuer-record /srv/artifacts/LFM2.5-2.6B-secure/issuer-record.json \
  --certified-device /srv/enrollment/device.json \
  --certified-device-signature /srv/enrollment/device.sig \
  --output /srv/artifacts/LFM2.5-2.6B-secure/license \
  --package-key-id package-signing-v1 \
  --package-public-key /offline-trust/package-public.key \
  --enrollment-key-id enrollment-v1 \
  --enrollment-public-key /offline-trust/enrollment-public.key \
  --license-signing-key /offline-keys/license-signing.pem \
  --license-key-passphrase-file /run/issuer-secrets/license.pass \
  --license-key-id license-signing-v1 \
  --policy-signing-key /offline-keys/policy-signing.pem \
  --policy-key-passphrase-file /run/issuer-secrets/policy.pass \
  --policy-key-id tpm-policy-v1 \
  --kek-key-file /offline-keys/issuer-kek.bin \
  --kek-key-id issuer-kek-v1 \
  --kek-key-version 1 \
  --license-id customer-license-1 \
  --generation 1
```

The license output and its signature are customer artifacts. The issuer keys,
passphrase files, and issuer record are not.
Failed issuance publication retains its quota reservation. Retry with identical
inputs/license ID/generation resumes it or publishes the exact committed bundle;
changed intent and disabled certificates fail closed. Historical V1 reservations
without an intent digest cannot be resumed automatically. Do not delete reservations
to bypass admission; use the
[operations runbook](../../docs/project-memory/TPM-Operations-Runbook.md).

## 5. Build the protected runtime image

Build the base and protected image from the same reviewed source revision. The
build script compiles the TPM-enabled wheel and the Dockerfile rejects an image
without the TPM binding or with a vLLM/loader version other than 0.28.0:

```bash
IMAGE_TAG=registry.example.com/dynamo-vllm-protected:1.5.0 \
  deploy/model-protection/build-protected-image.sh
```

Push the image by digest. Do not use a generic or older vLLM image as the base.
The wheel copied beside the Dockerfile is temporary build output and must not
be committed.

## 6. Local or Docker startup

The runtime needs `/dev/tpmrm0`, the protected image, the encrypted package,
the license files, public trust keys, and a writable owner-only tmpfs:

```bash
sudo mount -t tmpfs -o size=40G,mode=0700,uid="$(id -u)",gid="$(id -g)",\
nosuid,nodev,noexec tmpfs /run/dynamo-models
```

Use the protected model path and runtime configuration when starting Dynamo:

```bash
python3 -m dynamo.vllm \
  --model /models/package \
  --model-protection-config /runtime/runtime.json \
  --load-format safetensors
```

The normal-model command remains unchanged. Do not pass private issuer keys or
the issuer record to this process. The loader verifies signatures and machine
binding, unwraps the DEK through TPM, materializes to tmpfs, and keeps the
plaintext session for the worker lifetime required by vLLM.

### Docker Compose example for an encrypted-file LLM

The [LLM Compose example](docker-compose.llm.example.yaml) is a copy adapted
from `ocr_service/deploy/prod/docker-compose.model-llm.yaml`. It does not
modify that production file. It starts a frontend and one vLLM worker. The
example expects this directory layout on the customer host:

```text
<LLM_MODEL_PATH>/package/
<LLM_MODEL_PATH>/runtime/runtime.json
```

Set `LLM_DYN_NAMESPACE` to the namespace signed into the package. The
example mounts a tmpfs at `/run/<LLM_DYN_NAMESPACE>-models` to match that
namespace. The external Docker network must have services named
`nats-server` and `etcd-server`. Set the GPU ID and memory limits for the
customer host. Do not copy issuer private keys into `runtime/`.

```bash
export LLM_PROTECTED_IMAGE=dynamo-vllm-protected-prod:1.5.0-vllm028
export LLM_MODEL_PATH=/srv/protected-models/my-llm
export LLM_MODEL_NAME=my-llm
export LLM_DYN_NAMESPACE=protected-llm
export LLM_DYN_NETWORK=ocr_network
export LLM_GPU_ID=0
test -d "$LLM_MODEL_PATH/package"
test -f "$LLM_MODEL_PATH/runtime/runtime.json"
docker network inspect "$LLM_DYN_NETWORK" >/dev/null
docker compose -f deploy/model-protection/docker-compose.llm.example.yaml config --quiet
docker compose -f deploy/model-protection/docker-compose.llm.example.yaml up -d
docker compose -f deploy/model-protection/docker-compose.llm.example.yaml logs -f vllm-llm-worker
```

This example uses the `encrypted-file` profile. The TPM profile needs TPM
device access and a different runtime configuration. Do not use this example
unchanged for a TPM-bound package.

## 7. Kubernetes profile

Start from `dgd.yaml` and replace the image digest, package PVC, and projected
Secret names. The profile provides separate memory-backed volumes for model
plaintext and runtime/JIT caches, sets `DYN_NAMESPACE_WORKER_SUFFIX` to empty,
and schedules only to approved TPM nodes. Set `supplementalGroups` to the
numeric group that owns `/dev/tpmrm0` on those nodes (`113` is only the example
profile value). Replace the example `hostPath` TPM device with the cluster's
audited TPM device plugin before production use.

Required platform controls are encrypted or disabled swap, disabled core
dumps, no privileged debug containers, immutable image digests, and admission
policy preventing tenant mutation of the Pod security context.

## 8. Validation and rollback

Run the focused tests before publishing:

```bash
pytest -q components/src/dynamo/vllm/tests/test_model_protection_bootstrap.py
cargo test -p dynamo-model-protection --features packager --locked --offline
cargo clippy -p dynamo-model-protection --features packager --all-targets -- -D warnings
```

A deployment is not release-ready until the protected image runs the real vLLM
0.30.0 classes, plain and protected model smoke tests pass, and TPM/GPU,
SIGTERM, tmpfs cleanup, and cross-server binding checks are recorded. Roll back
by restoring the previous signed image digest, package, and license as a
matched set. Never replace an encrypted package or license independently.

Root/kernel/ptrace/GPU-debug attackers on the licensed host remain outside the
Phase 1 extraction guarantee.
