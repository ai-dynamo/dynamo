/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package vllm

import (
	"fmt"
	"strings"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/mutation"
	corev1 "k8s.io/api/core/v1"
)

// RayValues contains the resolved addresses used by the vLLM Ray launch.
type RayValues struct {
	Port          string
	LeaderAddress string
}

// Ray returns the complete vLLM Ray leader and worker launch catalog.
func Ray(values RayValues) mutation.EngineMutations {
	return mutation.EngineMutations{
		{
			Applies: mutation.LeaderRole,
			Mutation: rayLeaderCommandMutation{
				ContainerName: commonconsts.MainContainerName,
				Port:          values.Port,
				Flags: []mutation.Flag{
					{Name: DistributedExecutorFlag, Value: "ray"},
				},
			},
		},
		{
			Applies: mutation.WorkerRole,
			Mutation: mutation.SetCommandMutation{
				ContainerName: commonconsts.MainContainerName,
				Command:       []string{"/bin/sh", "-c"},
				Args:          []string{fmt.Sprintf("ray start --address=%s:%s --block", values.LeaderAddress, values.Port)},
			},
		},
	}
}

// rayLeaderCommandMutation starts the Ray head before the user-supplied vLLM
// command. It remains engine-specific while satisfying the generic mutation
// interface.
type rayLeaderCommandMutation struct {
	ContainerName string
	Port          string
	Flags         []mutation.Flag
}

func (m rayLeaderCommandMutation) Apply(container *corev1.Container) error {
	if err := mutation.ValidateContainer(container, m.ContainerName); err != nil {
		return err
	}
	flags, err := mutation.FormatFlags(m.Flags)
	if err != nil {
		return err
	}

	quotedCommand := make([]string, len(container.Command))
	for i, token := range container.Command {
		quotedCommand[i] = mutation.ShellQuote(token)
	}
	quotedArgs := make([]string, len(container.Args))
	for i, arg := range container.Args {
		quotedArgs[i] = mutation.ShellQuote(arg)
	}

	container.Command = []string{"/bin/sh", "-c"}
	container.Args = []string{fmt.Sprintf(
		"ray start --head --port=%s && %s %s %s",
		m.Port,
		strings.Join(quotedCommand, " "),
		strings.Join(quotedArgs, " "),
		flags,
	)}
	return nil
}
