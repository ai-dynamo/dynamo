#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/upstream.env"

CHECKOUT_DIR="${BFCL_CHECKOUT_DIR:-${ROOT_DIR}/.cache/gorilla}"
VENV_DIR="${BFCL_VENV_DIR:-${ROOT_DIR}/.venv}"
PATCH_FILE="${ROOT_DIR}/patches/0001-glm52-openai-chat-completions.patch"
BFCL_DIR="${CHECKOUT_DIR}/berkeley-function-call-leaderboard"

if [[ -n "${BFCL_BOOTSTRAP_PYTHON:-}" ]]; then
  bootstrap_python="${BFCL_BOOTSTRAP_PYTHON}"
else
  bootstrap_python=""
  # Prefer the active `python3` before distro side interpreters. Container images
  # commonly include a working /usr/local Python plus /usr/bin/python3.11 without
  # the matching python3.11-venv package.
  for candidate in python3 python3.12 python3.11 python3.10; do
    if command -v "${candidate}" >/dev/null 2>&1 \
      && "${candidate}" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
      bootstrap_python="${candidate}"
      break
    fi
  done
fi
if [[ -z "${bootstrap_python}" ]] \
  || ! "${bootstrap_python}" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
  echo "BFCL requires Python >=3.10; set BFCL_BOOTSTRAP_PYTHON to a compatible interpreter." >&2
  exit 1
fi

if [[ ! -d "${CHECKOUT_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${CHECKOUT_DIR}")"
  git clone --filter=blob:none --no-checkout "${BFCL_GORILLA_REPO}" "${CHECKOUT_DIR}"
  git -C "${CHECKOUT_DIR}" checkout --detach "${BFCL_GORILLA_COMMIT}"
else
  if ! git -C "${CHECKOUT_DIR}" cat-file -e "${BFCL_GORILLA_COMMIT}^{commit}" 2>/dev/null; then
    git -C "${CHECKOUT_DIR}" fetch origin "${BFCL_GORILLA_COMMIT}"
  fi

  current_commit="$(git -C "${CHECKOUT_DIR}" rev-parse HEAD)"
  if [[ "${current_commit}" != "${BFCL_GORILLA_COMMIT}" ]]; then
    if [[ -n "$(git -C "${CHECKOUT_DIR}" status --short)" ]]; then
      echo "Refusing to move dirty BFCL checkout ${CHECKOUT_DIR}." >&2
      exit 1
    fi
    git -C "${CHECKOUT_DIR}" checkout --detach "${BFCL_GORILLA_COMMIT}"
  fi
fi

# Migrate the first local adapter revision, whose comma-bearing display name
# shifted columns in BFCL's unquoted aggregate CSV output. After this exact
# replacement, the current patch's reverse check verifies the full checkout.
model_config="${BFCL_DIR}/bfcl_eval/constants/model_config.py"
if grep -qF 'display_name="GLM-5.2 (Native FC, OpenAI Chat Completions)"' "${model_config}"; then
  "${bootstrap_python}" - "${model_config}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
old = 'display_name="GLM-5.2 (Native FC, OpenAI Chat Completions)"'
new = 'display_name="GLM-5.2 Native FC OpenAI Chat Completions"'
text = path.read_text()
if text.count(old) != 1:
    raise SystemExit("unexpected GLM-5.2 BFCL display-name migration count")
path.write_text(text.replace(old, new))
PY
fi

# Migrate the first local adapter revision to the bounded request contract. The
# exact replacement makes the updated patch reverse-checkable and keeps
# bootstrap idempotent for persistent campaign checkouts.
handler="${BFCL_DIR}/bfcl_eval/model_handler/api_inference/glm52_openai.py"
if [[ -f "${handler}" ]] \
  && ! grep -qF '"max_tokens": int(os.getenv("GLM52_OPENAI_MAX_TOKENS", "64000"))' "${handler}"; then
  "${bootstrap_python}" - "${handler}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
old = '            "temperature": self.temperature,\n'
new = old + '            "max_tokens": int(os.getenv("GLM52_OPENAI_MAX_TOKENS", "64000")),\n'
text = path.read_text()
if text.count(old) != 1:
    raise SystemExit("unexpected GLM-5.2 max-token migration count")
path.write_text(text.replace(old, new))
PY
fi

if git -C "${CHECKOUT_DIR}" apply --check "${PATCH_FILE}" 2>/dev/null; then
  git -C "${CHECKOUT_DIR}" apply "${PATCH_FILE}"
elif git -C "${CHECKOUT_DIR}" apply --reverse --check "${PATCH_FILE}" 2>/dev/null; then
  echo "GLM-5.2 BFCL patch is already applied."
else
  echo "BFCL checkout is neither clean nor patched as expected: ${CHECKOUT_DIR}" >&2
  exit 1
fi

"${bootstrap_python}" "${SCRIPT_DIR}/capture_metadata.py" \
  --checkout "${CHECKOUT_DIR}" \
  --patch "${PATCH_FILE}" \
  --verify-only >/dev/null

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${bootstrap_python}" -m venv "${VENV_DIR}"
elif ! "${VENV_DIR}/bin/python" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
  echo "Existing BFCL venv uses Python <3.10; remove ${VENV_DIR} and rerun." >&2
  exit 1
fi

"${VENV_DIR}/bin/python" -m pip install --requirement "${ROOT_DIR}/constraints.lock"
"${VENV_DIR}/bin/python" -m pip install --no-deps --editable "${BFCL_DIR}"

GLM52_OPENAI_BASE_URL="http://127.0.0.1:1/v1" \
  "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/validate_install.py"

echo "BFCL checkout: ${CHECKOUT_DIR}"
echo "BFCL commit:   ${BFCL_GORILLA_COMMIT}"
echo "BFCL venv:     ${VENV_DIR}"
