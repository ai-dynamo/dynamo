/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package vllm

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/workload"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestMPAppliesOrderedRoleFlags(t *testing.T) {
	container := &corev1.Container{
		Name:    "main",
		Command: []string{"python3"},
		Args:    []string{"-m", "dynamo.vllm"},
	}
	mutations := MP(MPValues{
		NumberOfNodes:        2,
		LeaderAddress:        "$(LEADER_ADDRESS)",
		WorkerRank:           "$(WORKER_RANK)",
		WorkerRankNeedsShell: true,
	})
	require.NoError(t, mutations.Apply(nil, workload.RoleWorker, container))
	require.Equal(t, []string{"sh", "-c"}, container.Command)
	require.Equal(t, []string{
		"exec python3 -m dynamo.vllm --distributed-executor-backend mp --nnodes 2 " +
			"--master-addr $(LEADER_ADDRESS) --master-port 29500 --node-rank $(WORKER_RANK) --headless",
	}, container.Args)
}

func TestDataParallelKeepsOptionalSizeInDeclaredOrder(t *testing.T) {
	values := DataParallelValues{
		TotalSize:       8,
		LocalSize:       4,
		LeaderAddress:   "leader.example.com",
		WorkerStartRank: "4",
		RPCPort:         "13445",
	}

	container := &corev1.Container{Name: "main", Command: []string{"python3"}, Args: []string{"-m", "dynamo.vllm"}}
	require.NoError(t, DataParallel(values).Apply(nil, workload.RoleLeader, container))
	require.Equal(t, []string{
		"-m", "dynamo.vllm",
		"--data-parallel-hybrid-lb",
		"--data-parallel-size", "8",
		"--data-parallel-size-local", "4",
		"--data-parallel-start-rank", "0",
		"--data-parallel-address", "leader.example.com",
		"--data-parallel-rpc-port", "13445",
	}, container.Args)

	values.OmitTotalSize = true
	container = &corev1.Container{Name: "main", Command: []string{"python3"}, Args: []string{"-m", "dynamo.vllm", "--data-parallel-size", "8"}}
	require.NoError(t, DataParallel(values).Apply(nil, workload.RoleLeader, container))
	require.Equal(t, 1, count(container.Args, DataParallelSizeFlag))
}

func TestRayCatalogKeepsExecutorFlagVisible(t *testing.T) {
	container := &corev1.Container{
		Name:    "main",
		Command: []string{"python3"},
		Args:    []string{"-m", "dynamo.vllm", "--model", "test"},
	}
	require.NoError(t, Ray(RayValues{Port: "6379"}).Apply(nil, workload.RoleLeader, container))
	require.Equal(t, []string{"/bin/sh", "-c"}, container.Command)
	require.Equal(t, []string{
		"ray start --head --port=6379 && python3 -m dynamo.vllm --model test --distributed-executor-backend ray",
	}, container.Args)
}

func TestManualMPEnsuresPortInDirectVLLMCommand(t *testing.T) {
	container := &corev1.Container{
		Name:    "main",
		Command: []string{"/bin/sh", "-c"},
		Args:    []string{"exec vllm serve /mnt/models --distributed-executor-backend=mp"},
	}
	require.NoError(t, ManualMP().Apply(nil, workload.RoleLeader, container))
	require.Equal(t, []string{
		"exec vllm serve /mnt/models --distributed-executor-backend=mp --master-port 29500",
	}, container.Args)
}

func count(values []string, wanted string) int {
	var result int
	for _, value := range values {
		if value == wanted {
			result++
		}
	}
	return result
}
