/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package sglang

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/mutation"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/workload"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestMultinodeAppliesOrderedRoleFlags(t *testing.T) {
	tests := []struct {
		name            string
		role            workload.Role
		values          MultinodeValues
		expectedCommand []string
		expectedArgs    []string
	}{
		{
			name: "leader",
			role: workload.RoleLeader,
			values: MultinodeValues{
				NumberOfNodes:       2,
				DistributedInitAddr: "leader.example.com:29500",
			},
			expectedCommand: []string{"python3"},
			expectedArgs: []string{
				"-m", "dynamo.sglang",
				"--dist-init-addr", "leader.example.com:29500",
				"--nnodes", "2",
				"--node-rank", "0",
			},
		},
		{
			name: "worker with shell rank",
			role: workload.RoleWorker,
			values: MultinodeValues{
				NumberOfNodes:        2,
				DistributedInitAddr:  "$(LEADER_HOST):29500",
				WorkerRank:           "$(WORKER_INDEX)",
				WorkerRankNeedsShell: true,
			},
			expectedCommand: []string{"sh", "-c"},
			expectedArgs: []string{
				"exec python3 -m dynamo.sglang --dist-init-addr $(LEADER_HOST):29500 --nnodes 2 --node-rank $(WORKER_INDEX)",
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			container := &corev1.Container{
				Name:    "main",
				Command: []string{"python3"},
				Args:    []string{"-m", "dynamo.sglang"},
			}
			err := mutation.Concat(MultinodePodWiring(), AutomaticMultinode(test.values)).Apply(
				&v1beta1.DynamoComponentDeploymentSharedSpec{}, test.role, container,
			)
			require.NoError(t, err)
			require.Equal(t, test.expectedCommand, container.Command)
			require.Equal(t, test.expectedArgs, container.Args)
		})
	}
}
