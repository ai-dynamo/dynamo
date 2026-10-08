---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Encrypt a model without TPM
subtitle: Package Qwen and HunyuanOCR on the issuer host and run them on a customer server
---

This guide covers model encryption on the **issuer host** and runtime setup on
the **customer server**. Run packaging commands only on the issuer host, which
holds the source model and issuer keys. This profile does not require TPM,
enrollment, a challenge, or a license. The customer server still needs a
protected runtime image and the runtime configuration below.
This branch pins the protected runtime to vLLM `0.30.0` and Omni `0.30.0rc1`.
Use an image built from this branch.

Last verified: 2026-10-07.

This guide uses `PROTECTION_PROFILE=encrypted-file`. It encrypts weights with
AES-256-GCM and stores approved metadata in readable form under `public/`.
It does not bind the model to a TPM. Anyone who obtains both the package and
`model-dek.bin` can decrypt the model on another machine.

The Vietnamese version is [model-encryption-only.md](model-encryption-only.md).

## 1. Identify input and output directories (issuer host)

| Item | Path |
| --- | --- |
| Dynamo checkout | `/media/thinh_do/Data/Workspace/dynamo` |
| Issuer key directory | `/home/thinh_do/Desktop/key-so-hoa/cqtt` |
| Passphrase directory | `$XDG_RUNTIME_DIR/issuer-secrets` |
| Qwen source model | `/media/thinh_do/Data/Workspace/ocr_service/Resources/models/Qwen3.5-4B-25-09` |
| HunyuanOCR source model | `/media/thinh_do/Data/Workspace/ocr_service/Resources/models/Model_ocr_02_10_2026` |
| Output directory | `/media/thinh_do/Data/Workspace/ocr_service/Resources/models/protected/encrypted-file/<model>` |

Each source checkpoint must contain `config.json` and its Safetensors files.
Do not use a parent directory that contains several checkpoints. Do not use an
unmerged LoRA adapter as a complete model.

The output contains encrypted weights and readable metadata. The source
checkpoint stays unchanged. The output directory is the model directory. The
script creates `package/` and `issuer-record.json` directly inside it. The
`MODEL_VERSION` argument is manifest metadata; it does not add a directory.
This guide uses `1` as the metadata version.

## 2. Build the packaging tools (issuer host)

Install the Rust toolchain from `rust-toolchain.toml`, a Linux C compiler,
OpenSSL CLI, and Python 3. Do not build a Docker image for this step.

```bash
(
  set -euo pipefail
  cd /media/thinh_do/Data/Workspace/dynamo
  cargo build --locked --release -p dynamo-model-protection \
    --features packager --bin model-protection-pack --bin model-protection-issue
  test -x target/release/model-protection-pack
  test -x target/release/model-protection-issue
  target/release/model-protection-pack --help
)
```

`model-protection-pack` encrypts the model. `model-protection-issue` exports a
DEK in step 5. Do not ask the customer to build these tools.

## 3. Check issuer keys (issuer host)

Use the v2 key set for new packages. Keep v1 keys if they still protect older
packages.

| File | Purpose | Send to customer? |
| --- | --- | --- |
| `package-signing-v2.pk8` | Encrypted Ed25519 package-signing private key | No |
| `package-signing-v2.pub` | 32-byte public key for package verification | Yes, in runtime trust directory |
| `issuer-kek-v2.bin` | 32-byte key that wraps each package DEK | No |
| `package-v2.pass` | Passphrase for the private key | No |

Keep private keys, KEK, and passphrase in protected issuer storage. Do not put
them in Git, a Docker image, or the customer bundle.

If the v2 keys do not exist, create them once on the issuer host. Stop if any
target already exists. Do not delete a partial key set without checking which
packages use it.

```bash
(
  set -euo pipefail
  umask 077
  KEY_DIR=/home/thinh_do/Desktop/key-so-hoa/cqtt
  : "${XDG_RUNTIME_DIR:?Log in to a user session with XDG_RUNTIME_DIR}"
  PASS_DIR="$XDG_RUNTIME_DIR/issuer-secrets"
  install -d -m 0700 "$KEY_DIR" "$PASS_DIR"
  for f in "$KEY_DIR/package-signing-v2.pk8" \
           "$KEY_DIR/package-signing-v2.pub" \
           "$KEY_DIR/issuer-kek-v2.bin" "$PASS_DIR/package-v2.pass"; do
    if [ -e "$f" ] || [ -L "$f" ]; then
      printf 'Already exists; stop without overwriting: %s\n' "$f" >&2
      exit 1
    fi
  done
  openssl rand -base64 -out "$PASS_DIR/package-v2.pass" 48
  openssl genpkey -algorithm ED25519 -aes-256-cbc \
    -pass "file:$PASS_DIR/package-v2.pass" \
    -out "$KEY_DIR/package-signing-v2.pk8"
  openssl rand -out "$KEY_DIR/issuer-kek-v2.bin" 32
  PUB_DER="$(mktemp "$XDG_RUNTIME_DIR/package-v2-public.XXXXXX")"
  trap 'rm -f -- "$PUB_DER"' EXIT
  openssl pkey -in "$KEY_DIR/package-signing-v2.pk8" \
    -passin "file:$PASS_DIR/package-v2.pass" -pubout -outform DER -out "$PUB_DER"
  tail -c 32 "$PUB_DER" > "$KEY_DIR/package-signing-v2.pub"
  chmod 0600 "$KEY_DIR/package-signing-v2.pk8" \
    "$KEY_DIR/package-signing-v2.pub" "$KEY_DIR/issuer-kek-v2.bin" \
    "$PASS_DIR/package-v2.pass"
  test "$(stat -c %s "$KEY_DIR/package-signing-v2.pub")" -eq 32
  test "$(stat -c %s "$KEY_DIR/issuer-kek-v2.bin")" -eq 32
)
```

```bash
(
  set -euo pipefail
  KEY_DIR=/home/thinh_do/Desktop/key-so-hoa/cqtt
  : "${XDG_RUNTIME_DIR:?Log in to a user session with XDG_RUNTIME_DIR}"
  PASS_DIR="$XDG_RUNTIME_DIR/issuer-secrets"
  for f in "$KEY_DIR/package-signing-v2.pk8" \
           "$KEY_DIR/package-signing-v2.pub" \
           "$KEY_DIR/issuer-kek-v2.bin" "$PASS_DIR/package-v2.pass"; do
    test -s "$f" || { printf 'Missing or empty: %s\n' "$f" >&2; exit 1; }
    stat -c '%n: %s bytes, mode %a, owner %U' "$f"
  done
  test "$(stat -c %s "$KEY_DIR/package-signing-v2.pub")" -eq 32
  test "$(stat -c %s "$KEY_DIR/issuer-kek-v2.bin")" -eq 32
  openssl pkey -in "$KEY_DIR/package-signing-v2.pk8" \
    -passin "file:$PASS_DIR/package-v2.pass" -pubout -out /dev/null
  printf 'The private key opens with the current passphrase.\n'
)
```

Do not recreate or delete keys to resolve a missing-file error. Restore the
matching key set and passphrase from protected backup.

## 4. Encrypt the models (issuer host)

The script checks the key files and stops if an output already exists. Keep
each package in a new output directory. The commands below use the v2 package
signing key and KEK.

```bash
(
  set -euo pipefail
  umask 077
  cd /media/thinh_do/Data/Workspace/dynamo
  KEY_DIR=/home/thinh_do/Desktop/key-so-hoa/cqtt
  : "${XDG_RUNTIME_DIR:?Log in to a user session with XDG_RUNTIME_DIR}"
  PASS_DIR="$XDG_RUNTIME_DIR/issuer-secrets"
  MODEL_ROOT=/media/thinh_do/Data/Workspace/ocr_service/Resources/models
  RELEASE_ROOT="$MODEL_ROOT/protected/encrypted-file"
  export PROTECTION_PROFILE=encrypted-file
  export PACKAGE_SIGNING_KEY="$KEY_DIR/package-signing-v2.pk8"
  export PACKAGE_PASSPHRASE_FILE="$PASS_DIR/package-v2.pass"
  export PACKAGE_KEY_ID=package-signing-v2
  export KEK_KEY_FILE="$KEY_DIR/issuer-kek-v2.bin"
  export KEK_KEY_ID=issuer-kek-v2
  export KEK_KEY_VERSION=2
  export MIN_RUNTIME_VERSION=1.6.0

  for MODEL in Qwen3.5-4B-25-09 Model_ocr_02_10_2026; do
    test -s "$MODEL_ROOT/$MODEL/config.json"
    compgen -G "$MODEL_ROOT/$MODEL/*.safetensors" > /dev/null
    test ! -e "$RELEASE_ROOT/$MODEL/package"
    test ! -e "$RELEASE_ROOT/$MODEL/issuer-record.json"
  done

  bash deploy/model-protection/protect-model.sh \
    "$MODEL_ROOT/Qwen3.5-4B-25-09" \
    "$RELEASE_ROOT/Qwen3.5-4B-25-09" \
    customer-ocr Qwen3.5-4B-25-09 1

  bash deploy/model-protection/protect-model.sh \
    "$MODEL_ROOT/Model_ocr_02_10_2026" \
    "$RELEASE_ROOT/Model_ocr_02_10_2026" \
    customer-ocr Model_ocr_02_10_2026 1
)
```

If the first model succeeds and the second fails, keep the first result. Retry
only the failed model after you fix its input. Do not delete a completed
package to rerun both models.

## 5. Export the DEK and create runtime files (issuer host)

Each package has a different 32-byte DEK. Export it only after package creation.
The issuer-only `issuer-record.json` is an input to this command. Do not send
it to the customer.

```bash
(
  set -euo pipefail
  umask 077
  cd /media/thinh_do/Data/Workspace/dynamo
  KEY_DIR=/home/thinh_do/Desktop/key-so-hoa/cqtt
  RELEASE_ROOT=/media/thinh_do/Data/Workspace/ocr_service/Resources/models/protected/encrypted-file
  for MODEL in Qwen3.5-4B-25-09 Model_ocr_02_10_2026; do
    MODEL_OUT="$RELEASE_ROOT/$MODEL"
    test ! -e "$MODEL_OUT/runtime/model-dek.bin"
    install -d -m 0700 "$MODEL_OUT/runtime" "$MODEL_OUT/runtime/trust"
    target/release/model-protection-issue export-file-key \
      --package "$MODEL_OUT/package" \
      --issuer-record "$MODEL_OUT/issuer-record.json" \
      --package-key-id package-signing-v2 \
      --package-public-key "$KEY_DIR/package-signing-v2.pub" \
      --kek-key-file "$KEY_DIR/issuer-kek-v2.bin" \
      --kek-key-id issuer-kek-v2 --kek-key-version 2 \
      --output "$MODEL_OUT/runtime/model-dek.bin"
    test "$(stat -c %s "$MODEL_OUT/runtime/model-dek.bin")" -eq 32
    install -m 0600 "$KEY_DIR/package-signing-v2.pub" \
      "$MODEL_OUT/runtime/trust/package-public.key"
    sed 's/package-signing-v1/package-signing-v2/' \
      deploy/model-protection/runtime.file.json.example \
      > "$MODEL_OUT/runtime/runtime.json"
    chmod 0600 "$MODEL_OUT/runtime/runtime.json"
  done
)
```

`runtime.json` must use the same package signing key ID as the package:

```json
{
  "schema_version": 2,
  "profile": "encrypted-file",
  "layers": {
    "package_verification": true,
    "license_verification": false,
    "tpm_binding": false,
    "secure_materialization": true
  },
  "package_trust": {
    "key_id": "package-signing-v2",
    "public_key_file": "/runtime/trust/package-public.key"
  },
  "key_provider": {
    "type": "file",
    "key_file": "/runtime/model-dek.bin"
  },
  "process_memory_margin_bytes": 8589934592
}
```

The paths in `runtime.json` are container paths. Do not replace them with host
paths. Do not put the DEK in `.env.prod`.

## 6. Transfer the package and runtime (issuer to customer)

For each model, transfer only the matching `package/` and `runtime/`
directories over an approved secure channel. Do not send the issuer record,
KEK, private keys, passphrase, or original checkpoint.

On the customer server, place them directly under the model directory. Do not
create a `file-v1` directory:

```text
/root/developments/sohoa/ocr-prod/
├── package/
└── runtime/
    ├── runtime.json
    ├── model-dek.bin
    └── trust/package-public.key
```

In this example, set `OCR_MODEL_PATH=/root/developments/sohoa/ocr-prod`. If
you keep a model-named directory in the copied tree, set `OCR_MODEL_PATH` to
that model directory. The variable must point to the directory that directly
contains `package/` and `runtime/`.

## 7. Configure the customer Compose deployment

Set the OCR values in `.env.prod`, next to the Compose file:

```dotenv
OCR_PROTECTED_IMAGE=dynamo-vllm-protected-prod:1.5.0
OCR_MODEL_PATH=/root/developments/sohoa/ocr-prod
OCR_MODEL_NAME=KNM/OCR1.0-1B
OCR_MODEL_PROTECTION_CONFIG=/runtime/runtime.json
```

Set `OCR_MODEL_NAME` to the model name configured for both the frontend and
worker. For LLM, use the equivalent variables:

```dotenv
LLM_PROTECTED_IMAGE=dynamo-vllm-protected-prod:1.5.0
LLM_MODEL_PATH=/root/developments/sohoa/llm-prod
LLM_MODEL_NAME=Qwen3.5-4B-25-09
LLM_MODEL_PROTECTION_CONFIG=/runtime/runtime.json
```

`OCR_MODEL_PATH` and `LLM_MODEL_PATH` are host paths to the model directory.
The Compose files append `/package` and `/runtime`, then mount these
directories read-only as `/models/package` and `/runtime`. The protection
config variable uses the container path `/runtime/runtime.json`, not a host
path.

Set the worker UID/GID to match the owner of the runtime files. Set memory,
tmpfs, and `MEMLOCK_BYTES` limits from measurements on the customer server.
Compose defaults are starting values, not validated limits for every model.

The Compose files run the worker as UID/GID `1000:1000` by default. After you
copy the artifact, set its owner and modes to match:

```bash
MODEL_DIR=/root/developments/sohoa/ocr-prod
sudo chown -R 1000:1000 "$MODEL_DIR/package" "$MODEL_DIR/runtime"
sudo find "$MODEL_DIR/package" "$MODEL_DIR/runtime" -type d -exec chmod 0700 {} +
sudo find "$MODEL_DIR/package" "$MODEL_DIR/runtime" -type f -exec chmod 0600 {} +
```

Set `MODEL_DIR` to the directory that directly contains `package/` and
`runtime/`. Set `MODEL_PROTECTION_UID` and `MODEL_PROTECTION_GID` if the
container uses another user.

### Keep metadata local

The worker decrypts the model into its private tmpfs. The frontend cannot read
that worker-local path. The worker must serve model metadata over its internal
HTTP endpoint. Set these variables on the worker:

```yaml
DYN_SYSTEM_PORT: "9090"
DYN_SELF_HOST_METADATA: "true"
```

The OCR and LLM Compose files set these values in the protected worker
environment. Do not publish port `9090` on the host. The frontend and worker
must share a Docker network.

When self-hosted metadata is active, the frontend reads config and tokenizer
files from the worker. It does not download weights or metadata from
Hugging Face. If the system port is missing or unreachable, Dynamo can treat a
path such as `/run/protected-ocr-models/<id>` as a Hugging Face repository ID.
The log may show `ignore_weights=true` and a failing URL such as
`/api/models//run/protected-ocr-models/...`. `ignore_weights=true` skips weight
files; the metadata fetch can still fail.

The protected worker uses a read-only root filesystem. Its Compose config
places writable caches under `/tmp`:

```yaml
HOME: /tmp
CUPY_CACHE_DIR: /tmp/cache/cupy
XDG_CACHE_HOME: /tmp/cache
```

Compose mounts `/tmp` as `tmpfs`. `CUPY_CACHE_DIR` prevents CuPy from writing
to `/home/dynamo/.cupy`. It does not control model encryption or model source.

After you update `.env.prod` or the Compose file, recreate the OCR worker from
the `deploy/prod` directory:

```bash
docker compose --env-file .env.prod \
  -f docker-compose.model-ocr.yaml \
  up -d --force-recreate vllm-ocr-worker
```

No image rebuild is needed for an environment or Compose-only change. Rebuild
only if the image lacks the protected loader or a required code fix. Check the
logs:

```bash
docker compose --env-file .env.prod \
  -f docker-compose.model-ocr.yaml \
  logs -f --tail=100 vllm-ocr-server vllm-ocr-worker
```

Confirm the frontend no longer requests
`/api/models//run/protected-ocr-models/...`. Then check `/v1/models` and send
an OCR request. For LLM, use `docker-compose.model-llm.yaml` and
`vllm-llm-worker`.

## 8. Troubleshooting

| Error or symptom | Action |
| --- | --- |
| `packager not found` | Build the tools in step 2 from the correct Dynamo checkout. |
| `key/passphrase files must exist` | Check the v2 key paths and passphrase from step 3. |
| `ISSUER_KEY_DECRYPT_FAILED` | Restore the matching passphrase. Do not create a new passphrase for an existing key. |
| `SOURCE_INVALID` | Select the checkpoint directory that contains `config.json` and Safetensors files. |
| `refusing to overwrite existing output` | Use a new output directory. Do not delete an active package. |
| `SOFTWARE_PROFILE_REQUIRED` | The package uses another profile. Create a new `encrypted-file` package; do not edit its signed manifest. |
| Hugging Face URL contains `/run/protected-...` | Set `DYN_SYSTEM_PORT=9090` and `DYN_SELF_HOST_METADATA=true` on the worker. Confirm that the frontend can reach the worker over the Docker network. |
| `Read-only file system: /home/dynamo/.cupy` | Set `HOME=/tmp` and `CUPY_CACHE_DIR=/tmp/cache/cupy`. Confirm that `/tmp` is a writable tmpfs. |
| `package/` or `runtime/` is missing | Set `OCR_MODEL_PATH` or `LLM_MODEL_PATH` to the host directory that directly contains both directories. Do not add `/file-v1`. |
| `MODEL_PROTECTION_CONFIG_INVALID` | Validate `runtime.json`, its key ID, `/runtime/...` paths, file ownership, and file modes. |

This profile does not require TPM enrollment. It does not stop a person who
has both the encrypted package and its DEK from copying the model to another
machine.
