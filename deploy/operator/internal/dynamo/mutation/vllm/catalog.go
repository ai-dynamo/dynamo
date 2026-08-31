/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package vllm declares the Kubernetes and command-line mutations used by the
// vLLM backend. Runtime-dependent values are resolved by the backend before it
// selects one of these complete mutation groups.
package vllm

import (
	"strconv"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/mutation"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/workload"
	corev1 "k8s.io/api/core/v1"
)

const (
	framework                 = "vllm"
	dataParallelSizeLocalFlag = "--data-parallel-size-local"

	// DataParallelSizeFlag is shared with the launch-mode selector.
	DataParallelSizeFlag = "--data-parallel-size"
	// DistributedExecutorFlag is shared with automatic and manual MP detection.
	DistributedExecutorFlag = "--distributed-executor-backend"
)

// MPValues contains only the runtime values used by the static multiprocessing
// mutation group.
type MPValues struct {
	NumberOfNodes        int32
	LeaderAddress        string
	WorkerRank           string
	WorkerRankNeedsShell bool
}

// DataParallelValues contains only the runtime values used by the static data
// parallel mutation group.
type DataParallelValues struct {
	TotalSize            int64
	OmitTotalSize        bool
	LocalSize            int64
	LeaderAddress        string
	WorkerStartRank      string
	WorkerRankNeedsShell bool
	RPCPort              string
}

// Common returns mutations independent of the selected vLLM launch mode.
func Common() mutation.EngineMutations {
	return mutation.EngineMutations{
		{
			Applies: interPodFailover,
			Mutation: mutation.AppendEnvMutation{
				ContainerName: commonconsts.MainContainerName,
				Env:           corev1.EnvVar{Name: "DYN_VLLM_GMS_SHADOW_MODE", Value: "true"},
			},
		},
		{
			Applies: interPodGMS,
			Mutation: mutation.EnsureArgsFlagMutation{
				ContainerName: commonconsts.MainContainerName,
				Flag:          "--load-format",
				Value:         "gms",
				Framework:     framework,
			},
		},
	}
}

// MP returns the complete vLLM multiprocessing flag mutation group. The leader
// uses rank zero; workers use their resolved rank and run headless. The
// operator-owned wait-for-leader init container is declared separately because
// it mutates the PodSpec rather than the engine container.
func MP(values MPValues) mutation.EngineMutations {
	return mutation.EngineMutations{
		{
			Applies: mutation.LeaderRole,
			Mutation: mutation.AddFlagsMutation{
				ContainerName: commonconsts.MainContainerName,
				Flags: []mutation.Flag{
					{Name: DistributedExecutorFlag, Value: "mp"},
					{Name: "--nnodes", Value: strconv.FormatInt(int64(values.NumberOfNodes), 10)},
					{Name: "--master-addr", Value: values.LeaderAddress},
					{Name: "--master-port", Value: commonconsts.VLLMMpMasterPort},
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
					{Name: DistributedExecutorFlag, Value: "mp"},
					{Name: "--nnodes", Value: strconv.FormatInt(int64(values.NumberOfNodes), 10)},
					{Name: "--master-addr", Value: values.LeaderAddress},
					{Name: "--master-port", Value: commonconsts.VLLMMpMasterPort},
					{Name: "--node-rank", Value: values.WorkerRank},
					{Name: "--headless"},
				},
				NeedsShell: values.WorkerRankNeedsShell,
				Framework:  framework,
			},
		},
	}
}

// ManualMP contains the fixed operator-owned port required by Dynamo's
// wait-for-leader wiring while leaving all native vLLM topology flags to the
// user.
func ManualMP() mutation.EngineMutations {
	return mutation.EngineMutations{{
		Mutation: mutation.EnsureFlagMutation{
			ContainerName: commonconsts.MainContainerName,
			Flag:          "--master-port",
			Value:         commonconsts.VLLMMpMasterPort,
			Framework:     framework,
			Executable:    "vllm",
			Subcommand:    "serve",
		},
	}}
}

// DataParallel returns the complete vLLM multinode data-parallel flag group.
// Both roles run a local coordinator; the worker's start rank is resolved from
// the topology backend. OmitTotalSize preserves a user-supplied DP size.
func DataParallel(values DataParallelValues) mutation.EngineMutations {
	return mutation.EngineMutations{
		{
			Applies: mutation.LeaderRole,
			Mutation: mutation.AddFlagsMutation{
				ContainerName: commonconsts.MainContainerName,
				Flags: []mutation.Flag{
					{Name: "--data-parallel-hybrid-lb"},
					{Name: DataParallelSizeFlag, Value: strconv.FormatInt(values.TotalSize, 10), Omit: values.OmitTotalSize},
					{Name: dataParallelSizeLocalFlag, Value: strconv.FormatInt(values.LocalSize, 10)},
					{Name: "--data-parallel-start-rank", Value: "0"},
					{Name: "--data-parallel-address", Value: values.LeaderAddress},
					{Name: "--data-parallel-rpc-port", Value: values.RPCPort},
				},
				Framework: framework,
			},
		},
		{
			Applies: mutation.WorkerRole,
			Mutation: mutation.AddFlagsMutation{
				ContainerName: commonconsts.MainContainerName,
				Flags: []mutation.Flag{
					{Name: "--data-parallel-hybrid-lb"},
					{Name: DataParallelSizeFlag, Value: strconv.FormatInt(values.TotalSize, 10), Omit: values.OmitTotalSize},
					{Name: dataParallelSizeLocalFlag, Value: strconv.FormatInt(values.LocalSize, 10)},
					{Name: "--data-parallel-start-rank", Value: values.WorkerStartRank},
					{Name: "--data-parallel-address", Value: values.LeaderAddress},
					{Name: "--data-parallel-rpc-port", Value: values.RPCPort},
				},
				NeedsShell: values.WorkerRankNeedsShell,
				Framework:  framework,
			},
		},
	}
}

// LaunchCommand replaces the engine command with a resolved launch command.
func LaunchCommand(command, args []string) mutation.EngineMutations {
	return mutation.EngineMutations{{
		Mutation: mutation.SetCommandMutation{
			ContainerName: commonconsts.MainContainerName,
			Command:       command,
			Args:          args,
		},
	}}
}

// MPSideChannel injects the pod IP used by vLLM's NIXL side channel.
func MPSideChannel() mutation.EngineMutations {
	return mutation.EngineMutations{{
		Mutation: mutation.AppendEnvMutation{
			ContainerName: commonconsts.MainContainerName,
			Env: corev1.EnvVar{
				Name: commonconsts.VLLMNixlSideChannelHostEnvVar,
				ValueFrom: &corev1.EnvVarSource{
					FieldRef: &corev1.ObjectFieldSelector{FieldPath: "status.podIP"},
				},
			},
		},
	}}
}

// MultinodePodWiring contains operator-owned mutations common to all vLLM
// multinode launch modes.
func MultinodePodWiring() mutation.EngineMutations {
	return mutation.EngineMutations{{
		Applies: mutation.WorkerRole,
		Mutation: mutation.RemoveProbesMutation{
			ContainerName: commonconsts.MainContainerName,
			Liveness:      true,
			Readiness:     true,
			Startup:       true,
		},
	}}
}

// SingleNodeElasticEP contains the command and address wiring for a resolved
// single-node elastic-EP Ray launch. Both environment variables resolve from
// status.podIP so the Ray head registration and vLLM's DP-master lookup cannot
// disagree. VLLM_DP_MASTER_IP is required because vLLM falls back to it when
// data-parallel size is one.
func SingleNodeElasticEP(command, args []string) mutation.EngineMutations {
	podIPRef := func() *corev1.EnvVarSource {
		return &corev1.EnvVarSource{
			FieldRef: &corev1.ObjectFieldSelector{FieldPath: "status.podIP"},
		}
	}
	return mutation.EngineMutations{
		{
			Mutation: mutation.SetCommandMutation{
				ContainerName: commonconsts.MainContainerName,
				Command:       command,
				Args:          args,
			},
		},
		{
			Mutation: mutation.AppendEnvMutation{
				ContainerName: commonconsts.MainContainerName,
				Env:           corev1.EnvVar{Name: commonconsts.PodIPEnvVar, ValueFrom: podIPRef()},
			},
		},
		{
			Mutation: mutation.AppendEnvMutation{
				ContainerName: commonconsts.MainContainerName,
				Env:           corev1.EnvVar{Name: commonconsts.VLLMDPMasterIPEnvVar, ValueFrom: podIPRef()},
			},
		},
	}
}

// CompilationCache injects the resolved vLLM compilation-cache directory.
func CompilationCache(cacheDir string) mutation.EngineMutations {
	if cacheDir == "" {
		return nil
	}
	return mutation.EngineMutations{{
		Mutation: mutation.AppendEnvMutation{
			ContainerName: commonconsts.MainContainerName,
			Env:           corev1.EnvVar{Name: "VLLM_CACHE_ROOT", Value: cacheDir},
		},
	}}
}

// WaitForLeader adds the resolved wait-for-leader volume and init container.
func WaitForLeader(volume corev1.Volume, initContainer corev1.Container) mutation.PodSpecMutations {
	return mutation.PodSpecMutations{
		{Mutation: mutation.AppendVolumeMutation{Volume: volume}},
		{Mutation: mutation.AppendInitContainerMutation{Container: initContainer}},
	}
}

func interPodGMS(component *v1beta1.DynamoComponentDeploymentSharedSpec, _ workload.Role) bool {
	return component != nil && component.IsInterPodGMSEnabled()
}

func interPodFailover(component *v1beta1.DynamoComponentDeploymentSharedSpec, _ workload.Role) bool {
	return component != nil && component.IsInterPodFailoverEnabled()
}
