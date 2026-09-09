# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

run_lpu_system_init() {
	if [[ ! -e /dev/fd ]]; then
		ln -svf /proc/self/fd /dev/fd || true
	fi
	mkdir -p /tmp
	mkdir -p /var/run/dpdk/rte || true
	chmod 777 /var/run/dpdk/rte || true
	ulimit -l unlimited || true

	if command -v lpu-system-init >/dev/null 2>&1; then
		local init_ok="unknown"
		local init_status=0
		for attempt in $(seq 1 2); do
			echo "Running lpu-system-init (attempt ${attempt})..."
			local init_output
			local -a init_args=(--ci-precommand)
			if (( attempt == 2 )); then
				init_args+=(--ci-remediation-commands)
			fi
			set +e
			init_output="$(lpu-system-init "${init_args[@]}" 2>&1)"
			init_status=$?
			set -e
			echo "${init_output}"
			init_ok="$(echo "${init_output}" | grep -o '"ok": *[a-z]*' | tail -1 | grep -o 'true\|false' || echo "unknown")"
			if (( init_status == 0 )) && [[ "${init_ok}" != "false" ]]; then
				break
			fi
			if (( attempt == 1 )); then
				if (( init_status != 0 )); then
					echo "lpu-system-init exited with status ${init_status}; retrying with remediation" >&2
				else
					echo "lpu-system-init reported ok=false; retrying with remediation" >&2
				fi
			fi
		done
		if (( init_status != 0 )) || [[ "${init_ok}" == "false" ]]; then
			echo "lpu-system-init failed after 2 attempts; aborting" >&2
			exit 1
		fi
	else
		echo "lpu-system-init not found; continuing without pre-init" >&2
	fi
}
