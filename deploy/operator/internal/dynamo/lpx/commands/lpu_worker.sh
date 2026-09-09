# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

HOME="${HOME:-/root}"

if [[ "${LPU_RUN_SYSTEM_INIT:-true}" == "true" ]]; then
	run_lpu_system_init
fi

ulimit -l unlimited || true

%s

if command -v setcap >/dev/null 2>&1; then
	for binary in /bin-unwrapped/agent_v2.unwrapped /bin-unwrapped/agent_v2_perfetto_trace.unwrapped /bin/agent_v2 /bin/agent_v2_perfetto_trace; do
		if [[ -e "$binary" ]]; then
			echo "Lifting capabilities of $binary"
			setcap cap_sys_nice,cap_sys_resource,cap_sys_admin,cap_dac_override,cap_sys_rawio,cap_sys_ptrace=+ep "$(realpath "$binary")" || true
		fi
	done
fi

# Pass down information from the k8s downward API into SSH sessions.
SETENV_LINE=""
SETENV_LINE="${SETENV_LINE} K8S_POD_NAME=\"$POD_NAME\""
SETENV_LINE="${SETENV_LINE} K8S_POD_NAMESPACE=\"$POD_NAMESPACE\""
SETENV_LINE="${SETENV_LINE} K8S_POD_UID=\"$POD_UID\""
%sif [[ -n "${LPU_MODEL_NAME:-}" ]]; then
	SETENV_LINE="${SETENV_LINE} LPU_MODEL_NAME=\"$LPU_MODEL_NAME\""
fi
exec /bin/sshd -D -e \
	-o "PidFile none" \
	-o "Port @@LPX_SSH_PORT@@" \
	-o "AuthorizedKeysFile $HOME/.ssh/authorized_keys" \
	-o "StrictModes no" \
	-o PermitRootLogin=yes \
	-o PasswordAuthentication=no \
	-o PubkeyAuthentication=yes \
	-o "SetEnv $SETENV_LINE" \
	-h "$HOME/.ssh/host_keys/ssh_host_rsa_key" \
	-h "$HOME/.ssh/host_keys/ssh_host_ecdsa_key" \
	-h "$HOME/.ssh/host_keys/ssh_host_ed25519_key"
