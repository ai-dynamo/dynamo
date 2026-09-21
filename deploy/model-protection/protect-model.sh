#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: protect-model.sh MODEL_DIR OUTPUT_DIR CUSTOMER_SCOPE MODEL_ID MODEL_VERSION

Required environment:
  PACKAGE_SIGNING_KEY       encrypted PKCS#8 Ed25519 key
  PACKAGE_PASSPHRASE_FILE    owner-only passphrase file
  KEK_KEY_FILE               exact 32-byte AES-256 KEK
  KEK_KEY_ID                 issuer KEK identifier
  KEK_KEY_VERSION            issuer KEK version

Optional environment:
  PACK_BIN                   model-protection-pack binary
  PACKAGE_KEY_ID             package signing key id (default: package-signing-v1)
  MIN_RUNTIME_VERSION       minimum Dynamo runtime version (default: 0.1.0)
EOF
}

[[ $# -eq 5 ]] || { usage >&2; exit 64; }

SOURCE_DIR="$1"
OUTPUT_DIR="$2"
CUSTOMER_SCOPE="$3"
MODEL_ID="$4"
MODEL_VERSION="$5"

PACK_BIN="${PACK_BIN:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/target/release/model-protection-pack}"
MIN_RUNTIME_VERSION="${MIN_RUNTIME_VERSION:-0.1.0}"

: "${PACKAGE_SIGNING_KEY:?PACKAGE_SIGNING_KEY is required}"
: "${PACKAGE_PASSPHRASE_FILE:?PACKAGE_PASSPHRASE_FILE is required}"
: "${KEK_KEY_FILE:?KEK_KEY_FILE is required}"
: "${KEK_KEY_ID:?KEK_KEY_ID is required}"
: "${KEK_KEY_VERSION:?KEK_KEY_VERSION is required}"

[[ -d "$SOURCE_DIR" ]] || { echo "source directory does not exist: $SOURCE_DIR" >&2; exit 2; }
[[ "$SOURCE_DIR" = /* && "$OUTPUT_DIR" = /* ]] || {
  echo "source and output paths must be absolute" >&2
  exit 2
}
[[ -x "$PACK_BIN" ]] || {
  echo "packager not found: $PACK_BIN (build with --features packager)" >&2
  exit 2
}
[[ -f "$PACKAGE_SIGNING_KEY" && -f "$PACKAGE_PASSPHRASE_FILE" && -f "$KEK_KEY_FILE" ]] || {
  echo "key/passphrase files must exist" >&2
  exit 2
}

umask 077
mkdir -p "$OUTPUT_DIR"
chmod 700 "$OUTPUT_DIR"
PACKAGE_DIR="$OUTPUT_DIR/package"
ISSUER_RECORD="$OUTPUT_DIR/issuer-record.json"
[[ ! -e "$PACKAGE_DIR" && ! -e "$ISSUER_RECORD" ]] || {
  echo "refusing to overwrite existing output: $OUTPUT_DIR" >&2
  exit 3
}

"$PACK_BIN" \
  --source "$SOURCE_DIR" \
  --output "$PACKAGE_DIR" \
  --issuer-record "$ISSUER_RECORD" \
  --customer-scope-id "$CUSTOMER_SCOPE" \
  --model-id "$MODEL_ID" \
  --model-version "$MODEL_VERSION" \
  --minimum-runtime-version "$MIN_RUNTIME_VERSION" \
  --package-signing-key "$PACKAGE_SIGNING_KEY" \
  --package-key-passphrase-file "$PACKAGE_PASSPHRASE_FILE" \
  --package-key-id "${PACKAGE_KEY_ID:-package-signing-v1}" \
  --kek-key-file "$KEK_KEY_FILE" \
  --kek-key-id "$KEK_KEY_ID" \
  --kek-key-version "$KEK_KEY_VERSION"

echo "encrypted package: $PACKAGE_DIR"
echo "issuer record (do not ship): $ISSUER_RECORD"
