/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package render

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	corev1 "k8s.io/api/core/v1"
)

func TestEngineMutationsApplyEvaluatesPredicateOutsideMutation(t *testing.T) {
	component := &v1beta1.DynamoComponentDeploymentSharedSpec{}
	container := &corev1.Container{Name: "main"}
	applied := false
	mutations := EngineMutations{
		{
			Applies: func(gotComponent *v1beta1.DynamoComponentDeploymentSharedSpec, role Role) bool {
				if gotComponent != component {
					t.Fatalf("predicate component = %p, want %p", gotComponent, component)
				}
				return role == RoleWorker
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

	if err := mutations.Apply(component, RoleLeader, container); err != nil {
		t.Fatalf("EngineMutations.Apply() error = %v", err)
	}
	if applied {
		t.Fatal("Apply was called when predicate returned false")
	}

	if err := mutations.Apply(component, RoleWorker, container); err != nil {
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
