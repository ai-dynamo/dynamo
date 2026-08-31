/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package mutation

import (
	"slices"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/workload"
	corev1 "k8s.io/api/core/v1"
)

func TestConcatPreservesGroupOrder(t *testing.T) {
	mutations := Concat(
		EngineMutations{{Mutation: AppendEnvMutation{Env: corev1.EnvVar{Name: "FIRST"}}}},
		EngineMutations{{Mutation: AppendEnvMutation{Env: corev1.EnvVar{Name: "SECOND"}}}},
	)
	container := &corev1.Container{}
	if err := mutations.Apply(nil, workload.RoleMain, container); err != nil {
		t.Fatalf("EngineMutations.Apply() error = %v", err)
	}
	if len(container.Env) != 2 || container.Env[0].Name != "FIRST" || container.Env[1].Name != "SECOND" {
		t.Fatalf("container.Env = %#v, want FIRST then SECOND", container.Env)
	}
}

func TestAddFlagsMutationAppliesOrderedSetAtomically(t *testing.T) {
	container := &corev1.Container{
		Command: []string{"python3"},
		Args:    []string{"-m", "dynamo.vllm"},
	}
	mutation := AddFlagsMutation{
		Flags: []Flag{
			{Name: "--first", Value: "$(FIRST)"},
			{Name: "--omitted", Omit: true},
			{Name: "--second", Value: "$(SECOND)"},
		},
		NeedsShell: true,
		Framework:  "vllm",
	}
	if err := mutation.Apply(container); err != nil {
		t.Fatalf("AddFlagsMutation.Apply() error = %v", err)
	}
	if got, want := container.Command, []string{"sh", "-c"}; !equalStrings(got, want) {
		t.Fatalf("container.Command = %v, want %v", got, want)
	}
	if got, want := container.Args, []string{"exec python3 -m dynamo.vllm --first $(FIRST) --second $(SECOND)"}; !equalStrings(got, want) {
		t.Fatalf("container.Args = %v, want %v", got, want)
	}
}

func TestEnsureFlagMutationApply(t *testing.T) {
	mutation := EnsureFlagMutation{
		ContainerName: "main",
		Flag:          "--master-port",
		Value:         "29500",
	}
	container := &corev1.Container{Name: "main", Args: []string{"--model", "test"}}

	if err := mutation.Apply(container); err != nil {
		t.Fatalf("Apply() error = %v", err)
	}
	if got, want := container.Args, []string{"--model", "test", "--master-port", "29500"}; !slices.Equal(got, want) {
		t.Fatalf("Apply() args = %v, want %v", got, want)
	}
}

func TestEnsureFlagMutationIsIdempotent(t *testing.T) {
	mutation := EnsureFlagMutation{ContainerName: "main", Flag: "--master-port", Value: "29500"}
	container := &corev1.Container{Name: "main", Args: []string{"--master-port=29500"}}

	if err := mutation.Apply(container); err != nil {
		t.Fatalf("Apply() error = %v", err)
	}
	if got, want := container.Args, []string{"--master-port=29500"}; !slices.Equal(got, want) {
		t.Fatalf("Apply() args = %v, want %v", got, want)
	}
}

func TestEngineMutationsApplyEvaluatesPredicateOutsideMutation(t *testing.T) {
	component := &v1beta1.DynamoComponentDeploymentSharedSpec{}
	container := &corev1.Container{Name: "main"}
	applied := false
	mutations := EngineMutations{
		{
			Applies: func(gotComponent *v1beta1.DynamoComponentDeploymentSharedSpec, role workload.Role) bool {
				if gotComponent != component {
					t.Fatalf("predicate component = %p, want %p", gotComponent, component)
				}
				return role == workload.RoleWorker
			},
			Mutation: testMutation{apply: func(gotContainer *corev1.Container) error {
				if gotContainer != container {
					t.Fatalf("Apply container = %p, want %p", gotContainer, container)
				}
				applied = true
				return nil
			}},
		},
	}

	if err := mutations.Apply(component, workload.RoleLeader, container); err != nil {
		t.Fatalf("EngineMutations.Apply() error = %v", err)
	}
	if applied {
		t.Fatal("Apply was called when predicate returned false")
	}

	if err := mutations.Apply(component, workload.RoleWorker, container); err != nil {
		t.Fatalf("EngineMutations.Apply() error = %v", err)
	}
	if !applied {
		t.Fatal("Apply was not called when predicate returned true")
	}
}

type testMutation struct {
	apply func(*corev1.Container) error
}

func (m testMutation) Apply(container *corev1.Container) error {
	return m.apply(container)
}

func equalStrings(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}
