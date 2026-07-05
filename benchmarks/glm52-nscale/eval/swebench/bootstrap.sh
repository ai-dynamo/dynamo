#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

# mini-SWE-agent pins its Python toolchain. Persist uv's interpreter alongside
# the harness so venv shebangs survive evaluation-runner replacement.
UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${SWEBENCH_WORK_ROOT}/uv-python}"
export UV_PYTHON_INSTALL_DIR

for command in git uv; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    printf 'required command not found: %s\n' "${command}" >&2
    exit 1
  fi
done

mkdir -p "${SWEBENCH_REPOS_DIR}" "${SWEBENCH_WORK_ROOT}" "${SWEBENCH_MSWEA_CONFIG_DIR}"
export MSWEA_GLOBAL_CONFIG_DIR="${SWEBENCH_MSWEA_CONFIG_DIR}"
export MSWEA_SILENT_STARTUP=1
export PYTHONDONTWRITEBYTECODE=1

checkout_revision() {
  local url="$1"
  local revision="$2"
  local destination="$3"
  local created=0

  if [[ ! -d "${destination}/.git" ]]; then
    if [[ -e "${destination}" ]]; then
      printf 'refusing to replace non-git path: %s\n' "${destination}" >&2
      return 1
    fi
    git clone --filter=blob:none --no-checkout "${url}" "${destination}"
    created=1
  fi

  if [[ ${created} -eq 0 && -n "$(git -C "${destination}" status --porcelain)" ]]; then
    printf 'refusing to replace changes in managed checkout: %s\n' "${destination}" >&2
    return 1
  fi

  if ! git -C "${destination}" cat-file -e "${revision}^{commit}" 2>/dev/null; then
    git -C "${destination}" fetch --filter=blob:none origin "${revision}"
  fi
  git -C "${destination}" checkout --detach "${revision}"

  local actual
  actual="$(git -C "${destination}" rev-parse HEAD)"
  if [[ "${actual}" != "${revision}" ]]; then
    printf 'checkout mismatch for %s: expected %s, got %s\n' "${destination}" "${revision}" "${actual}" >&2
    return 1
  fi
}

checkout_revision \
  https://github.com/SWE-agent/mini-swe-agent.git \
  "${MINI_SWE_AGENT_COMMIT}" \
  "${MINI_SWE_AGENT_REPO}"
checkout_revision \
  https://github.com/princeton-nlp/SWE-bench.git \
  "${SWEBENCH_COMMIT}" \
  "${SWEBENCH_EVALUATOR_REPO}"
checkout_revision \
  https://github.com/scaleapi/SWE-bench_Pro-os.git \
  "${SWEBENCH_PRO_COMMIT}" \
  "${SWEBENCH_PRO_REPO}"

if [[ ! -x "${SWEBENCH_VENV}/bin/python" ]]; then
  uv venv --python 3.11 "${SWEBENCH_VENV}"
fi

uv pip install --python "${SWEBENCH_VENV}/bin/python" \
  --editable "${MINI_SWE_AGENT_REPO}" \
  --editable "${SWEBENCH_EVALUATOR_REPO}" \
  --requirement "${SWEBENCH_PRO_REPO}/requirements.txt" \
  --constraint "${SCRIPT_DIR}/constraints.lock"

"${SWEBENCH_VENV}/bin/python" - "${MINI_SWE_AGENT_VERSION}" "${SWEBENCH_VERSION}" <<'PY'
import sys

import minisweagent
import swebench

expected_mini, expected_swebench = sys.argv[1:]
assert minisweagent.__version__ == expected_mini, (minisweagent.__version__, expected_mini)
assert swebench.__version__ == expected_swebench, (swebench.__version__, expected_swebench)
PY

"${SWEBENCH_VENV}/bin/python" "${SCRIPT_DIR}/prepare_datasets.py" \
  --output-root "${SWEBENCH_DATA_ROOT}"

uv pip freeze --python "${SWEBENCH_VENV}/bin/python" > "${SWEBENCH_WORK_ROOT}/environment.freeze.txt"
"${SWEBENCH_VENV}/bin/python" "${SCRIPT_DIR}/verify_environment_lock.py" \
  --lock "${SCRIPT_DIR}/constraints.lock" \
  --freeze "${SWEBENCH_WORK_ROOT}/environment.freeze.txt" \
  --output "${SWEBENCH_WORK_ROOT}/environment.normalized.freeze.txt"

"${SWEBENCH_VENV}/bin/python" - "${SWEBENCH_WORK_ROOT}/source-lock.json" <<PY
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
output.write_text(json.dumps({
    "mini_swe_agent": {
        "version": "${MINI_SWE_AGENT_VERSION}",
        "commit": "${MINI_SWE_AGENT_COMMIT}",
    },
    "swebench": {
        "version": "${SWEBENCH_VERSION}",
        "commit": "${SWEBENCH_COMMIT}",
    },
    "swebench_pro": {"commit": "${SWEBENCH_PRO_COMMIT}"},
}, indent=2) + "\n")
PY

printf 'SWE-bench environment ready at %s\n' "${SWEBENCH_WORK_ROOT}"
