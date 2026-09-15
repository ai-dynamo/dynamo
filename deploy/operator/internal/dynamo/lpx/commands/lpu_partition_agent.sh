# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

AGENT_BINARY="${AGENT_BINARY:-agent-hybrid-v2-roce}"
HOME="${HOME:-/root}"
MPI_SSH_PORT="${MPI_SSH_PORT:-@@LPX_SSH_PORT@@}"
READINESS_PORT="${READINESS_PORT:-@@LPX_READINESS_PORT@@}"
RDMA_PORT="${RDMA_PORT:-@@LPX_RDMA_PORT@@}"

readarray -t SOURCE_PARTITIONS < "${LPU_CONFIG_DIR}/partition_ids"
readarray -t PARTITION_PATHS < "${LPU_CONFIG_DIR}/partition_paths"
readarray -t TOPOLOGIES < "${LPU_CONFIG_DIR}/topologies"

SOURCE_PARTITION="${SOURCE_PARTITIONS[$CONFIG_PARTITION_INDEX]}"
PARTITION_PATH="${PARTITION_PATHS[$CONFIG_PARTITION_INDEX]}"
TOPOLOGY="${TOPOLOGIES[$CONFIG_PARTITION_INDEX]}"

agent_part_from_partition_path() {
	local part_path="$1"
	local part_name="${part_path##*/}"
	if [[ "${part_name}" =~ ^part-?([0-9]+)$ ]]; then
		printf "%s\n" "${BASH_REMATCH[1]}"
		return
	fi

	printf "%s\n" "${SOURCE_PARTITION}"
}

agent_assemble_dir_from_partition_path() {
	local part_path="$1"
	local agent_part="$2"
	local part_name="${part_path##*/}"
	local parent="${part_path%/*}"
	if [[ "${part_name}" != "part-${agent_part}" ]]; then
		local assemble_dir="/tmp/dynamo-agent-assemble/row-${CONFIG_PARTITION_INDEX}"
		mkdir -p "${assemble_dir}"
		ln -sfn "${GAS_DIR}/${part_path}" "${assemble_dir}/part-${agent_part}"
		printf "%s\n" "${assemble_dir}"
		return
	fi
	if [[ "${parent}" == "${part_path}" ]]; then
		printf "%s\n" "${GAS_DIR}"
		return
	fi
	printf "%s/%s\n" "${GAS_DIR}" "${parent}"
}

AGENT_PART="$(agent_part_from_partition_path "${PARTITION_PATH}")"
AGENT_ASSEMBLE_DIR="$(agent_assemble_dir_from_partition_path "${PARTITION_PATH}" "${AGENT_PART}")"

setup_ssh() {
	mkdir -p "${HOME}/.ssh" "${HOME}/.ssh/host_keys"
	cp /ssh-pk/private.key "${HOME}/.ssh/id_rsa"
	cp /ssh-pk/private.key.pub "${HOME}/.ssh/id_rsa.pub"
	cp /ssh-pk/private.key.pub "${HOME}/.ssh/authorized_keys"
	chmod 600 "${HOME}/.ssh/id_rsa" "${HOME}/.ssh/authorized_keys"
	chmod 644 "${HOME}/.ssh/id_rsa.pub"
	printf "Host *\nIdentityFile %s/.ssh/id_rsa\nStrictHostKeyChecking no\nPort %s\n" "${HOME}" "${MPI_SSH_PORT}" > "${HOME}/.ssh/config"

	test -f "${HOME}/.ssh/host_keys/ssh_host_rsa_key" || ssh-keygen -t rsa -f "${HOME}/.ssh/host_keys/ssh_host_rsa_key" -N "" -q
	test -f "${HOME}/.ssh/host_keys/ssh_host_ecdsa_key" || ssh-keygen -t ecdsa -f "${HOME}/.ssh/host_keys/ssh_host_ecdsa_key" -N "" -q
	test -f "${HOME}/.ssh/host_keys/ssh_host_ed25519_key" || ssh-keygen -t ed25519 -f "${HOME}/.ssh/host_keys/ssh_host_ed25519_key" -N "" -q
}

run_sshd() {
	setup_ssh
	echo "Starting sshd on port ${MPI_SSH_PORT} for partition ${LOGICAL_PARTITION_INDEX} rank ${PARTITION_RANK}"
	exec /bin/sshd -D -e \
		-o "PidFile none" \
		-o "Port ${MPI_SSH_PORT}" \
		-o "AuthorizedKeysFile ${HOME}/.ssh/authorized_keys" \
		-o "StrictModes no" \
		-o "PermitRootLogin yes" \
		-o "PubkeyAuthentication yes" \
		-o "SetEnv PARTITION_ID=\"${PARTITION_ID}\" RANK_IN_PARTITION=\"${RANK_IN_PARTITION}\"" \
		-h "${HOME}/.ssh/host_keys/ssh_host_rsa_key" \
		-h "${HOME}/.ssh/host_keys/ssh_host_ecdsa_key" \
		-h "${HOME}/.ssh/host_keys/ssh_host_ed25519_key"
}

partition_hostnames() {
	for ((rank = 0; rank < NODE_COUNT; rank++)); do
		local node_index=$((NODE_OFFSET + rank))
		printf "%s-%d.%s\n" "${GROVE_PCLQ_NAME}" "${node_index}" "${GROVE_HEADLESS_SERVICE}"
	done
}

wait_for_peer_ssh() {
	local host="$1"
	for i in $(seq 1 120); do
		if ssh -o ConnectTimeout=2 "root@${host}" echo ok >/dev/null 2>&1; then
			echo "${host}:${MPI_SSH_PORT} reachable"
			return
		fi
		if [[ "${i}" == "120" ]]; then
			echo "${host}:${MPI_SSH_PORT} unreachable after 120 attempts" >&2
			exit 1
		fi
		sleep 1
	done
}

run_lpu_system_init
command -v "${AGENT_BINARY}"

if (( NODE_COUNT == 1 )); then
	echo "Starting ${AGENT_BINARY} for logical partition ${LOGICAL_PARTITION_INDEX}, source partition ${SOURCE_PARTITION}, part path ${PARTITION_PATH}, assemble-dir ${AGENT_ASSEMBLE_DIR}, topology ${TOPOLOGY}"
	exec "${AGENT_BINARY}" \
		--assemble-dir "${AGENT_ASSEMBLE_DIR}" \
		--part "${AGENT_PART}" \
		--topology "${TOPOLOGY}" \
		--port "${RDMA_PORT}" \
		--readiness-port "${READINESS_PORT}" \
		--stage-name continue
fi

if [[ "${PARTITION_RANK}" != "0" ]]; then
	run_sshd
fi

setup_ssh
PARTITION_HOSTS="$(partition_hostnames)"
readarray -t NODE_HOSTS <<< "${PARTITION_HOSTS}"

MPI_HOSTS=()
MPI_HOSTS+=("${POD_IP}:1")
for host in "${NODE_HOSTS[@]:1}"; do
	MPI_HOSTS+=("${host}:1")
done
MPI_HOST_LIST=$(IFS=,; echo "${MPI_HOSTS[*]}")

for host in "${NODE_HOSTS[@]:1}"; do
	wait_for_peer_ssh "${host}"
done

MPI_ARGS=(
	--allow-run-as-root
	--host "${MPI_HOST_LIST}"
	-np "${NODE_COUNT}"
	--map-by ppr:1:node:OVERSUBSCRIBE
	--mca pml ob1
	--mca btl tcp,self
	--mca mtl ^ofi
	--mca plm_rsh_args "-p ${MPI_SSH_PORT}"
	--bind-to none
	-x GROQ_DRACO_NETWORK_UP_CHECK_TIMEOUT
	-x DRACO_FORCE_RECONFIGURE
	-x DRACO_MAXFILE_OVERRIDE
	-x LPU_VSAP_DIB_CREDIT_AUTO_RESTORE_US
	--output TAG
)
if [[ -n "${MPI_BTL_TCP_IF_INCLUDE:-}" ]]; then
	MPI_ARGS+=(--mca btl_tcp_if_include "${MPI_BTL_TCP_IF_INCLUDE}")
fi

echo "Starting mpirun for logical partition ${LOGICAL_PARTITION_INDEX}, source partition ${SOURCE_PARTITION}, part path ${PARTITION_PATH}, assemble-dir ${AGENT_ASSEMBLE_DIR}, topology ${TOPOLOGY}"
exec numactl --cpunodebind=0 --membind=0 mpirun "${MPI_ARGS[@]}" -- \
	"${AGENT_BINARY}" \
		--assemble-dir "${AGENT_ASSEMBLE_DIR}" \
		--part "${AGENT_PART}" \
		--topology "${TOPOLOGY}" \
		--multinode \
		--port "${RDMA_PORT}" \
		--readiness-port "${READINESS_PORT}" \
		--stage-name continue
