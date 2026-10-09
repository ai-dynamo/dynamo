---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Protected model usage (English)
---

## Purpose and workflow

Last verified: 2026-10-07 (Asia/Ho_Chi_Minh).

This document describes how to package a protected model, issue its runtime
credentials, transfer the required artifacts, and run Dynamo/vLLM. Keep the
Vietnamese operator guide in [Protected-Model-Usage.md](Protected-Model-Usage.md)
as the source-language companion. This file is a separate English guide.

The release system has three roles:

| Role | Responsibilities |
|---|---|
| Issuer | Keeps the plaintext model and offline private keys; creates packages and licenses. |
| Build/CI host | Builds, scans, signs, and publishes the runtime image. It does not need model or issuer secrets. |
| Customer server | Keeps the encrypted package, runtime files, TPM (when used), and runs the worker. |

Plain models continue to use the normal Dynamo/vLLM path. They do not need a
license, TPM, protection config, or protected tmpfs. A package containing
protection markers must use the protected path. A malformed protected package
must fail closed; do not remove markers or use a fallback to load it as plain.

> [!WARNING]
> TPM/OEM enrollment and production acceptance are not complete merely because
> the CLI or image builds. Do not issue production licenses using test keys,
> synthetic device records, or a customer-created trust root. Review
> [model-protection-architecture.md](model-protection-architecture.md) and the
> release gates before production use.

### Choose a protection profile

The package profile is signed into the package manifest. The runtime config
must match it. Do not convert a legacy TPM package by changing environment
variables; create a new package with the intended profile.

| Profile | Package signature | Software license | TPM binding | Plaintext staging |
|---|---:|---:|---:|---:|
| `encrypted-file` | Required | No | No | Required, tmpfs only |
| `encrypted-file-license` | Required | Required | No | Required, tmpfs only |
| `encrypted-tpm` | Required | Required | Required | Required, tmpfs only |

The V2 layers are `package_verification`, `license_verification`,
`tpm_binding`, and `secure_materialization`. The basic encrypted-file profile
requires package verification and secure materialization. The software-license
profile also requires license verification. The TPM profile requires all four.
Unknown layer names, duplicate fields, or string values such as `"true"` are
invalid. The runtime rejects a config that disables a layer required by the
signed profile.

`DYN_MODEL_PROTECTION_CONFIG` accepts either inline JSON or an absolute path to
a JSON file. It is configuration, not a secret store. Never put a DEK, private
key, or passphrase in an environment variable. The runtime config carries
public trust keys and paths to protected runtime files; the DEK belongs only
in the restricted runtime file mount for `encrypted-file`.

For split frontend/worker deployments, the protected worker must expose its
system endpoint with `DYN_SYSTEM_PORT` and set
`DYN_SELF_HOST_METADATA=true`. Keep this endpoint on the internal container
network. The frontend cannot read the worker's tmpfs path directly. If metadata
discovery is unavailable, a session path can be misread as a Hugging Face model
ID; this is a metadata-distribution problem, not a reason to download weights.

### Basic encrypted-file profile (no TPM)

Run these steps on the issuer. Use a new output directory; do not overwrite an
existing TPM package.

```bash
KEY_DIR=/path/to/offline-issuer-keys
PASS_DIR="$XDG_RUNTIME_DIR/issuer-secrets"
MODEL_OUT=/release/Qwen3.5-4B-25-09-file
PROTECTION_PROFILE=encrypted-file \
PACKAGE_SIGNING_KEY="$KEY_DIR/package-signing-v1.pk8" \
PACKAGE_PASSPHRASE_FILE="$PASS_DIR/package.pass" \
PACKAGE_KEY_ID=package-signing-v1 \
KEK_KEY_FILE="$KEY_DIR/issuer-kek-v1.bin" \
KEK_KEY_ID=issuer-kek-v1 KEK_KEY_VERSION=1 \
MIN_RUNTIME_VERSION=1.6.0 \
deploy/model-protection/protect-model.sh \
  /path/to/plain/Qwen3.5-4B-25-09 \
  "$MODEL_OUT" customer-ocr Qwen3.5-4B-25-09 1
```

`MODEL_OUT` is the output directory. The script creates `package/` and
`issuer-record.json` directly inside it. The final `1` is model-version
metadata; it does not create a `file-v1/` directory.

Export the 32-byte DEK for this package on the issuer. This is not the issuer
KEK and is not a signing key:

```bash
install -d -m 0700 "$MODEL_OUT/runtime" "$MODEL_OUT/runtime/trust"
target/release/model-protection-issue export-file-key \
  --package "$MODEL_OUT/package" \
  --issuer-record "$MODEL_OUT/issuer-record.json" \
  --package-key-id package-signing-v1 \
  --package-public-key "$KEY_DIR/package-signing-v1.pub" \
  --kek-key-file "$KEY_DIR/issuer-kek-v1.bin" \
  --kek-key-id issuer-kek-v1 --kek-key-version 1 \
  --output "$MODEL_OUT/runtime/model-dek.bin"
install -m 0600 "$KEY_DIR/package-signing-v1.pub" \
  "$MODEL_OUT/runtime/trust/package-public.key"
install -m 0600 deploy/model-protection/runtime.file.json.example \
  "$MODEL_OUT/runtime/runtime.json"
```

The command verifies the package signature, issuer-record signature, and
binding before it exports the DEK. It refuses to overwrite an existing output.
Do not delete an existing key or rerun issuance until you have checked which
packages and licenses use it. PKCS#8 decryption can require more than 128 MiB
of locked memory. If the issuer reports `ISSUER_MEMORY_UNAVAILABLE`, provide
the required issuer-side memlock; do not disable memory locking or reduce the
KDF. This is separate from the worker's `MEMLOCK_BYTES`.

Transfer only `package/` and `runtime/` over an approved secure channel. Do not
send `issuer-record.json`, the KEK, passphrases, or `.pk8` files. On the customer
server, set ownership to the container service UID/GID, use mode `0700` for
directories and `0600` for files, and mount both directories read-only.

> [!WARNING]
> `encrypted-file` is not machine-bound. Anyone who obtains both the package
> and `model-dek.bin` can decrypt or copy the model to another machine. A
> software license does not add TPM binding.

Example customer config (all paths are paths inside the container):

```bash
export OCR_PROTECTED_ROOT=/secure/customer/ocr
export DYN_MODEL_PROTECTION_CONFIG='{"schema_version":2,"profile":"encrypted-file","layers":{"package_verification":true,"license_verification":false,"tpm_binding":false,"secure_materialization":true},"package_trust":{"key_id":"package-signing-v1","public_key_file":"/runtime/trust/package-public.key"},"key_provider":{"type":"file","key_file":"/runtime/model-dek.bin"},"process_memory_margin_bytes":8589934592}'
```

Instead of inline JSON, set `DYN_MODEL_PROTECTION_CONFIG=/runtime/runtime.json`.
The memory margin above is an example, not a measured value for every model.

The OCR Compose deployment can use one host root:

```dotenv
OCR_PROTECTED_ROOT=/secure/customer/ocr
OCR_MODEL_PROTECTION_CONFIG=/runtime/runtime.json
```

Compose mounts `$OCR_PROTECTED_ROOT/package` and
`$OCR_PROTECTED_ROOT/runtime` read-only. It does not create missing directories.
`OCR_MODEL_PROTECTION_CONFIG` takes precedence over
`DYN_MODEL_PROTECTION_CONFIG`; if neither is set, the worker uses
`/runtime/runtime.json`. A TPM-enabled Compose profile still requires the TPM
device, its host GID, and appropriate RAM/memlock limits.

### Software-license profile (no TPM)

On the issuer, package with `PROTECTION_PROFILE=encrypted-file-license`, then
export the DEK as above and issue a software license:

```bash
target/release/model-protection-issue issue-file-license \
  --package "$MODEL_OUT/package" --issuer-record "$MODEL_OUT/issuer-record.json" \
  --package-key-id package-signing-v1 --package-public-key "$KEY_DIR/package-signing-v1.pub" \
  --kek-key-file "$KEY_DIR/issuer-kek-v1.bin" --kek-key-id issuer-kek-v1 --kek-key-version 1 \
  --license-signing-key "$KEY_DIR/license-signing-v1.pk8" \
  --license-key-passphrase-file "$PASS_DIR/license.pass" \
  --license-key-id license-signing-v1 --license-id license-ocr-file-v1 --generation 1 \
  --output "$MODEL_OUT/runtime/license"
install -m 0600 "$KEY_DIR/license-signing-v1.pub" \
  "$MODEL_OUT/runtime/trust/license-public.key"
install -m 0600 deploy/model-protection/runtime.file-license.json.example \
  "$MODEL_OUT/runtime/runtime.json"
```

The customer config must use profile `encrypted-file-license`, enable
`license_verification`, set `license_root` to `/runtime/license`, and point
`license_trust` at `/runtime/trust/license-public.key`. Keep `tpm_binding` false.
The license binds the identity, manifest digest, generation, and DEK digest. A
TPM license cannot be used as a software license. This profile does not need
an enrollment request, challenge, or certified-device record.

## TPM-bound profile: enrollment and license issuance

The `encrypted-tpm` profile binds the license to a DUK enrolled on the customer
server. The issuer keeps private signing keys. The customer server keeps TPM
state and responds to signed challenges. Do not create a certified-device JSON
by hand.

1. The issuer builds the packager and enrollment tools from reviewed source and
   creates the encrypted package. Keep the issuer-record and KEK at the issuer.
2. The issuer sends the customer only the enrollment collector and public TPM
   policy template. The customer runs `inspect`, checks available persistent
   handles with the TPM administrator, and then runs `provision` with three
   approved unused handles. Do not clear the TPM or evict existing objects.
3. The customer sends the resulting `policy_authority_name` as text. Keep
   `enrollment-state.json` and `provision-journal.json` on the customer server;
   do not send either file to the issuer.
4. The issuer creates a binding for the exact signed package. The binding
   contains the customer scope, artifact ID, manifest digest, and the received
   policy authority name. The issuer sends this binding to the customer.
5. The customer creates `request.json` using the enrolled TPM and an OEM-issued
   EK certificate chain. The customer sends only the request to the authority.
   Never replace the OEM root with a self-signed customer certificate.
6. An approved enrollment authority verifies the signed trust policy, OEM EK
   chain, challenge response, DUK profile/name, freshness, and package binding.
   It issues a signed `certified-device.json`. Production OEM trust must be
   approved; Dynamo does not create production OEM roots or private authority
   keys for you.
7. The authority returns the certificate and public enrollment key to the
   issuer. The issuer issues a license for that exact package and certified
   device. Repeat the binding/enrollment/license flow for every distinct model
   artifact. Do not reuse a certificate or license across artifacts.

The customer-to-issuer exchange must use authenticated channels. Transfer
`request.json`, `response.json`, and `policy_authority_name` only as specified
by the approved process. Keep TPM state, journal, authority registry, private
signers, passphrases, state KEK, and issuer-record in their assigned security
domains. If the OEM chain, trust policy, signer, or authority approval is
missing, stop; do not fabricate files or mark enrollment complete.

The issuer license operation uses the enrollment registry written by the
authority. The following is an example for a Qwen artifact; use the exact
certification ID and paths generated by your approved enrollment flow:

```bash
KEY_DIR=/path/to/offline-issuer-keys
PASS_DIR="$XDG_RUNTIME_DIR/issuer-secrets"
AUTHORITY_DIR="$HOME/model-protection-authority"
CERTIFICATION_ID='cert-QWEN-EXACT-ID-FROM-VERIFY'
CERT_DIR="$AUTHORITY_DIR/out/$CERTIFICATION_ID"
MODEL_OUT=/release/Qwen3.5-4B-25-09
target/release/model-protection-issue \
  --registry "$AUTHORITY_DIR/registry.sqlite" \
  --package "$MODEL_OUT/package" \
  --issuer-record "$MODEL_OUT/issuer-record.json" \
  --certified-device "$CERT_DIR/certified-device.json" \
  --certified-device-signature "$CERT_DIR/certified-device.sig" \
  --output "$MODEL_OUT/license" \
  --package-key-id package-signing-v1 \
  --package-public-key "$KEY_DIR/package-signing-v1.pub" \
  --enrollment-key-id enrollment-v1 \
  --enrollment-public-key "$KEY_DIR/enrollment-v1.pub" \
  --license-signing-key "$KEY_DIR/license-signing-v1.pk8" \
  --license-key-passphrase-file "$PASS_DIR/license.pass" \
  --license-key-id license-signing-v1 \
  --policy-signing-key "$KEY_DIR/tpm-policy-signing-v1.pk8" \
  --policy-key-passphrase-file "$PASS_DIR/policy.pass" \
  --policy-key-id tpm-policy-v1 \
  --kek-key-file "$KEY_DIR/issuer-kek-v1.bin" \
  --kek-key-id issuer-kek-v1 --kek-key-version 1 \
  --license-id license-qwen-customer-v1 --generation 1
```

Expected output files are `model.protection.license.json` and
`model.protection.license.sig`. Keep the issuer-record, model source, KEK,
private keys, and passphrases at the issuer. Do not include them in the
customer bundle or runtime image.

## Build the runtime image

Build on a build/CI host. The image contains protected runtime code and Python
bindings, not a model, license, issuer-record, or issuer key.

```bash
python3 -m pip install PyYAML Jinja2
SKIP_BASE_BUILD=false \
IMAGE_TAG=dynamo-vllm-protected:1.5.0 \
  deploy/model-protection/build-protected-image.sh
```

The build script pins vLLM `0.30.0` and Omni `0.30.0rc1`. The protected loader
accepts exactly vLLM `0.30.0`. Rebuild the base after source changes so the
Python loader and TPM-enabled wheel match. `SKIP_BASE_BUILD=true` is valid only
when the base already has the expected vLLM and loader. A successful bootstrap
test is not a substitute for protected inference acceptance on the final image
and target GPU.

Push to the registry selected by the infrastructure owner, record the digest,
and deploy by digest rather than a mutable tag:

```bash
docker push registry.example.com/team/dynamo-vllm-protected:1.5.0
docker image inspect registry.example.com/team/dynamo-vllm-protected:1.5.0 \
  --format '{{index .RepoDigests 0}}'
```

## Transfer and install the customer bundle

Transfer only:

- The signed encrypted `package/`.
- The license directory containing the signed license, when required.
- Public trust keys required by the selected profile.
- `runtime.json.example` from the same source revision, as a template.
- The approved runtime image reference and immutable digest.

Do not transfer the issuer-record, plaintext model, KEK, private keys,
passphrases, authority database/state, or TPM journal. Keep separate package
directories for distinct model artifacts. Verify model ID, version, signatures,
license binding, public-key fingerprints, image digest, and that the package
contains no plaintext `.safetensors` or `.bin` weights before deployment.

On the customer server, install the package at a stable absolute path, place
license files under the runtime license directory, and install public keys
under the runtime trust directory. Mount package and runtime directories
read-only. Create `runtime.json` from the template; do not edit the template in
place.

## Runtime configuration

For a TPM deployment, the runtime config can follow this shape. Replace the
device handle with the DUK handle provisioned on this server. All file paths
below are container paths.

```json
{
  "license_root": "/runtime/license",
  "package_trust": {
    "key_id": "package-signing-v1",
    "public_key_file": "/runtime/trust/package-public.key"
  },
  "license_trust": {
    "key_id": "license-signing-v1",
    "public_key_file": "/runtime/trust/license-public.key"
  },
  "tpm": {
    "device_key_handle": "0x81012013",
    "policy_authority_public_file": "/runtime/trust/tpm-policy-v1.tpmt-public",
    "policy_authority_key_id": "tpm-policy-v1"
  },
  "process_memory_margin_bytes": 8589934592
}
```

The DUK handle above is only an example. Use the handle recorded by the
approved enrollment process. The TPM policy-authority public file is the
public TPM template; it is not a persistent external signer handle. Do not put
a DEK, KEK, or private key in this config. Keep config and public keys on a
read-only mount. Size the process memory margin from measurements for the
selected model and runtime.

For the software profiles, use the matching profile example under
`deploy/model-protection/`. The `encrypted-file` profile needs the package
public key and file key provider. The `encrypted-file-license` profile also
needs the license root and license public key. Do not include TPM fields for
these profiles.

### Prepare protected tmpfs

Choose one method. For Docker, mount a container tmpfs directly. For a host
runtime, create a tmpfs mount at `/run/<DYN_NAMESPACE>-models`. The namespace
must contain only letters, digits, `.`, `_`, or `-`, and start and end with an
alphanumeric character.

Example host mount (40 GiB is illustrative only):

```bash
sudo install -d -m 0700 /run/dynamo-models
sudo mount -t tmpfs -o size=40G,mode=0700,uid="$(id -u)",gid="$(id -g)",nosuid,nodev,noexec \
  tmpfs /run/dynamo-models
findmnt /run/dynamo-models
```

Confirm the filesystem type is `tmpfs`. Size it for plaintext model files,
vLLM load peak, cache, and safety margin. Protected worker cache, temporary
files, and JIT cache must also use bounded memory-backed storage; they must not
fall back to node disk.

Example Docker tmpfs option:

```text
--tmpfs /run/dynamo-models:rw,noexec,nosuid,nodev,size=40g,mode=0700,uid=1000,gid=1000
```

Set the UID/GID to the non-root user that runs vLLM. Do not use the literal
path `/run/DYN_NAMESPACE-models`.

## Run a protected worker

The current implementation rejects tensor, pipeline, or data parallel sizes
other than 1. This is a current implementation limit, not a TPM or license
requirement. Do not assume that exposing multiple GPUs enables multi-GPU
serving. See the Vietnamese guide's section 9.1 and the architecture document
for the multi-GPU work plan.

Use an image pinned by digest, a non-root service account, measured memory and
memlock limits, and the GPU assigned by the administrator or scheduler. The
values below are examples and must be sized for the target model and host.

```bash
CLIENT_ROOT="$HOME/model-protection"
MODEL_DIR="$CLIENT_ROOT/Qwen3.5-4B-25-09"
GPU_DEVICE=0
CONTAINER_UID="$(id -u)"
CONTAINER_GID="$(id -g)"
IMAGE_REF='registry.example.com/team/dynamo-vllm-protected@sha256:IMAGE_DIGEST'
: "${MEMLOCK_BYTES:?Set the measured and approved memlock limit in bytes}"
docker run --rm --gpus "device=${GPU_DEVICE}" \
  --user "$CONTAINER_UID:$CONTAINER_GID" \
  --device /dev/tpmrm0 \
  --group-add "$(stat -c '%g' /dev/tpmrm0)" \
  --memory 64g --memory-swap 64g \
  --ulimit core=0 \
  --ulimit "memlock=${MEMLOCK_BYTES}:${MEMLOCK_BYTES}" \
  -v "$MODEL_DIR/package:/models/protected:ro" \
  -v "$MODEL_DIR/runtime:/runtime:ro" \
  --tmpfs "/run/dynamo-models:rw,noexec,nosuid,nodev,size=40g,mode=0700,uid=${CONTAINER_UID},gid=${CONTAINER_GID}" \
  -e DYN_NAMESPACE=dynamo \
  "$IMAGE_REF" \
  python3 -m dynamo.vllm \
    --model /models/protected \
    --model-protection-config /runtime/runtime.json
```

This example is for the TPM profile and includes `/dev/tpmrm0`. For
`encrypted-file` or `encrypted-file-license`, do not mount the TPM device; use
the corresponding runtime config and the same protected tmpfs requirements.
Bring up discovery and frontend services before starting the worker when the
deployment requires them.

Before startup, confirm that the model markers, license (if required), public
keys, TPM policy file (for TPM profile), and runtime config are readable by
the container user. Runtime startup must fail closed if package signature,
license, DUK, policy, tmpfs, cgroup/swap, or vLLM allowlist checks fail.

## Plain-model path

A model without `model.protection.json` and `model.protection.sig` follows the
existing plain path:

```bash
python3 -m dynamo.vllm \
  --model /models/plain \
  --enable-multimodal
```

It needs no license, TPM, protection config, package key, protected tmpfs, or
decryption key. Do not pass a protection flag to turn a plain model into a
protected package. A protected marker always selects the secure-or-invalid
path.

## Inference and runtime lifecycle checks

After startup, check health/readiness and send an inference smoke request to
the approved endpoint. Review server logs for stable error codes only. Test
invalid package, signature, license, DUK, tmpfs, and host-policy cases in an
approved test environment. Do not add a fallback that loads protected files as
plain.

The runtime verifies the package and license, checks scope/model/artifact and
device binding, unwraps the authenticated DEK, verifies each AES-256-GCM record,
and materializes plaintext files under a session UUID in tmpfs. vLLM loads from
that session. V1 keeps the session for the worker lifetime because vLLM can
read weights again. Stop the engine before cleaning up the session. Do not
delete the session while the worker is running.

Machine binding prevents a copied package from being used on an unlicensed
TPM. It is not a guarantee against a root user, compromised kernel, or GPU
debugger on an authorized host.

## Troubleshooting

| Error | Typical cause | Correct action |
|---|---|---|
| `PROTECTED_MARKER_INVALID` | One marker is missing or malformed. | Recreate the package. Do not remove markers to bypass protection. |
| `PACKAGE_VERIFICATION_FAILED` | Wrong trust key, manifest, or package file. | Verify the package at the issuer and install the matching public key. |
| `LICENSE_BINDING_MISMATCH` | License is for another package, scope, or device. | Verify the certified device and issue a license for the exact artifact. |
| `TPM_UNWRAP_FAILED` | Wrong DUK handle/policy, damaged data, or missing TPM permission. | Check enrollment, TPM profile, and `/dev/tpmrm0` access. |
| `TMPFS_REQUIRED` | Materialization root is not tmpfs or mount options are invalid. | Mount `/run/<namespace>-models` with the approved options. |
| `HOST_POLICY_INVALID` | cgroup, swap, memory, or dump policy is not compliant. | Correct host/container policy before starting the worker. |
| `SECRET_MEMORY_UNAVAILABLE` | Locked/non-dumpable memory is insufficient. | Set an approved `RLIMIT_MEMLOCK`; do not disable the check. |
| `VLLM_CONFIG_UNSUPPORTED` | Parallelism or another option is outside the current allowlist. | Use the supported profile or obtain an implementation/reviewed profile. |
| `PACKAGE_PATH_INVALID` | Relative path, symlink, nested output, or invalid package layout. | Use an absolute path and the direct-root package layout. |
| `maturin is required` | Build host lacks maturin. | Install `maturin[patchelf]` or use the provided image build script. |

Do not put keys, passphrases, wrapped DEKs, raw OS errors, or plaintext paths
in an issue or log. Share only the stable error code and approved artifact or
license identifier.

## Release checklist

### Issuer

- [ ] OEM trust roots, profile, revocation, and protocol have been reviewed.
- [ ] Production keys, custody, backup, and restore drills are approved.
- [ ] Each package is encrypted and contains no plaintext model weights.
- [ ] Packages and licenses are signed for the correct key IDs and artifacts.
- [ ] Issuer-record and private material remain offline and out of customer bundles.
- [ ] Runtime image is scanned, signed, and recorded by digest.

### Build/CI

- [ ] Image is rebuilt from the intended source SHA and backend versions.
- [ ] Image tests, scans, and dependency audits completed.
- [ ] Plain-model baselines exist for the target models and backend.
- [ ] Supported parallelism and lifecycle behavior are tested on the final image.

### Customer server

- [ ] Encrypted package, license, and public trust keys match the artifact.
- [ ] TPM identity and DUK handle match the certified device, when using TPM.
- [ ] `/run/<DYN_NAMESPACE>-models` is owner-only tmpfs with approved sizing.
- [ ] cgroup memory, swap/core-dump policy, and `RLIMIT_MEMLOCK` are suitable.
- [ ] Image is pinned by digest and runs as a non-root user where supported.
- [ ] Protected inference smoke test passed; logs contain no secrets.
- [ ] Shutdown, cleanup, restart, and authorized copy-rejection tests passed.

### Release approval

- [ ] Security review and key custody/rotation/TPM replacement drills are complete.
- [ ] Product owner approves the canary source SHA, image digest, and scope.
- [ ] Any incomplete gate remains explicitly open; unit tests are not product acceptance.

## Related documents

- [Protected model architecture](model-protection-architecture.md): formats,
  trust boundaries, and security limits.
- [Model protection deployment profile](https://github.com/ai-dynamo/dynamo/blob/main/deploy/model-protection/README.md):
  Docker/Kubernetes profile and build instructions.
- [Vietnamese operator guide](Protected-Model-Usage.md): source-language
  companion with detailed command-by-command enrollment instructions.
