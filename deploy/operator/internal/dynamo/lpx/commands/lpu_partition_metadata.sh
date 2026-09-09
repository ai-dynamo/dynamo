# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

LPU_CONFIG_DIR="${LPU_CONFIG_DIR:-@@LPX_CONFIG_MOUNT_PATH@@}"
CLIQUE_POD_INDEX="${GROVE_PCLQ_POD_INDEX:-}"
if ! [[ "${CLIQUE_POD_INDEX}" =~ ^[0-9]+$ ]]; then
	echo "GROVE_PCLQ_POD_INDEX must be a non-negative integer, got ${CLIQUE_POD_INDEX:-<empty>}" >&2
	exit 1
fi

readarray -t PARTITION_NODE_COUNTS < "${LPU_CONFIG_DIR}/nodes_per_partition"
readarray -t PARTITION_NODE_OFFSETS < "${LPU_CONFIG_DIR}/partition_node_offsets"
if (( ${#PARTITION_NODE_COUNTS[@]} == 0 || ${#PARTITION_NODE_COUNTS[@]} != ${#PARTITION_NODE_OFFSETS[@]} )); then
	echo "nodes_per_partition and partition_node_offsets must contain the same non-zero number of rows" >&2
	exit 1
fi

if [[ -f "${LPU_CONFIG_DIR}/partition_models" ]]; then
	readarray -t PARTITION_MODELS < "${LPU_CONFIG_DIR}/partition_models"
	readarray -t PARTITION_INDICES < "${LPU_CONFIG_DIR}/partition_indices"
	if [[ -z "${LPU_MODEL_NAME:-}" ]]; then
		echo "LPU_MODEL_NAME is required when partition_models is configured" >&2
		exit 1
	fi
	if (( ${#PARTITION_MODELS[@]} != ${#PARTITION_NODE_COUNTS[@]} || ${#PARTITION_INDICES[@]} != ${#PARTITION_NODE_COUNTS[@]} )); then
		echo "partition_models and partition_indices must match the partition config row count" >&2
		exit 1
	fi
else
	PARTITION_MODELS=()
	PARTITION_INDICES=()
fi

CONFIG_PARTITION_INDEX=""
LOGICAL_PARTITION_INDEX=""
PARTITION_RANK=""
NODE_COUNT=""
NODE_OFFSET=""
PARTITION_MATCH_COUNT=0
for i in "${!PARTITION_NODE_COUNTS[@]}"; do
	row_node_count="${PARTITION_NODE_COUNTS[$i]}"
	row_node_offset="${PARTITION_NODE_OFFSETS[$i]}"
	if ! [[ "${row_node_count}" =~ ^[1-9][0-9]*$ && "${row_node_offset}" =~ ^[0-9]+$ ]]; then
		echo "Invalid partition row ${i}: node count ${row_node_count}, node offset ${row_node_offset}" >&2
		exit 1
	fi
	if (( ${#PARTITION_MODELS[@]} > 0 )) && [[ "${PARTITION_MODELS[$i]}" != "${LPU_MODEL_NAME}" ]]; then
		continue
	fi
	if (( CLIQUE_POD_INDEX < row_node_offset || CLIQUE_POD_INDEX >= row_node_offset + row_node_count )); then
		continue
	fi

	CONFIG_PARTITION_INDEX="${i}"
	LOGICAL_PARTITION_INDEX="${PARTITION_INDICES[$i]:-${i}}"
	PARTITION_RANK="$((CLIQUE_POD_INDEX - row_node_offset))"
	NODE_COUNT="${row_node_count}"
	NODE_OFFSET="${row_node_offset}"
	PARTITION_MATCH_COUNT=$((PARTITION_MATCH_COUNT + 1))
done

if (( PARTITION_MATCH_COUNT != 1 )); then
	echo "Grove pod index ${CLIQUE_POD_INDEX} matched ${PARTITION_MATCH_COUNT} partition rows for model ${LPU_MODEL_NAME:-<default>}" >&2
	exit 1
fi
if ! [[ "${LOGICAL_PARTITION_INDEX}" =~ ^[0-9]+$ ]]; then
	echo "Invalid logical partition index ${LOGICAL_PARTITION_INDEX} for config row ${CONFIG_PARTITION_INDEX}" >&2
	exit 1
fi
