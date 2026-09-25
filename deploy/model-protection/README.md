<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Protected Model Deployment Guide

This guide describes the Phase 1 protected-model profile. It covers package
creation on the issuer host, runtime image construction, and local or
Kubernetes startup on a customer server.

## Runtime behavior

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

Phase 1 supports one aggregated vLLM 0.29.x process, one GPU, local
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

```bash
model-protection-issue \
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

## 5. Build the protected runtime image

Build the base and protected image from the same reviewed source revision. The
build script compiles the TPM-enabled wheel and the Dockerfile rejects an image
without the TPM binding or with a vLLM version outside 0.29.x:

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
0.29 classes, plain and protected model smoke tests pass, and TPM/GPU,
SIGTERM, tmpfs cleanup, and cross-server binding checks are recorded. Roll back
by restoring the previous signed image digest, package, and license as a
matched set. Never replace an encrypted package or license independently.

Root/kernel/ptrace/GPU-debug attackers on the licensed host remain outside the
Phase 1 extraction guarantee.
