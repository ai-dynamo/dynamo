/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package sglang declares the command-line and pod wiring mutations used by
// the SGLang backend.
package sglang

import (
	"strconv"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/mutation"
)

const framework = "sglang"

// MultinodeValues contains only runtime values used by the static SGLang
// multinode mutation group.
type MultinodeValues struct {
	NumberOfNodes        int32
	DistributedInitAddr  string
	WorkerRank           string
	WorkerRankNeedsShell bool
}

// Multinode returns the complete SGLang multinode mutation group.
func Multinode(values MultinodeValues) mutation.EngineMutations {
	return mutation.EngineMutations{
		{
			Applies: mutation.WorkerRole,
			Mutation: mutation.RemoveProbesMutation{
				ContainerName: commonconsts.MainContainerName,
				Liveness:      true,
				Readiness:     true,
				Startup:       true,
			},
		},
		{
			Applies: mutation.LeaderRole,
			Mutation: mutation.AddFlagsMutation{
				ContainerName: commonconsts.MainContainerName,
				Flags: []mutation.Flag{
					{Name: "--dist-init-addr", Value: values.DistributedInitAddr},
					{Name: "--nnodes", Value: strconv.FormatInt(int64(values.NumberOfNodes), 10)},
					{Name: "--node-rank", Value: "0"},
				},
				Framework: framework,
			},
		},
		{
			Applies: mutation.WorkerRole,
			Mutation: mutation.AddFlagsMutation{
				ContainerName: commonconsts.MainContainerName,
				Flags: []mutation.Flag{
					{Name: "--dist-init-addr", Value: values.DistributedInitAddr},
					{Name: "--nnodes", Value: strconv.FormatInt(int64(values.NumberOfNodes), 10)},
					{Name: "--node-rank", Value: values.WorkerRank},
				},
				NeedsShell: values.WorkerRankNeedsShell,
				Framework:  framework,
			},
		},
	}
}
