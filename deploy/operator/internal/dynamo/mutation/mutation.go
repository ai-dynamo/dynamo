/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package mutation

import (
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/workload"
	corev1 "k8s.io/api/core/v1"
)

// Predicate decides whether a mutation applies to an effective component and
// the workload role currently being rendered.
type Predicate func(component *v1beta1.DynamoComponentDeploymentSharedSpec, role workload.Role) bool

// EngineMutation makes one mechanical change to an already materialized engine
// container.
type EngineMutation interface {
	Apply(container *corev1.Container) error
}

// EngineMutationRule associates a mechanical mutation with its applicability.
type EngineMutationRule struct {
	Applies  Predicate
	Mutation EngineMutation
}

// EngineMutations is an ordered set of conditional engine mutations.
type EngineMutations []EngineMutationRule

// Concat returns a new ordered mutation set assembled from complete groups.
func Concat(groups ...EngineMutations) EngineMutations {
	var size int
	for _, group := range groups {
		size += len(group)
	}

	mutations := make(EngineMutations, 0, size)
	for _, group := range groups {
		mutations = append(mutations, group...)
	}
	return mutations
}

// Apply selects and applies mutations in declaration order.
func (mutations EngineMutations) Apply(
	component *v1beta1.DynamoComponentDeploymentSharedSpec,
	role workload.Role,
	container *corev1.Container,
) error {
	for _, rule := range mutations {
		if rule.Applies != nil && !rule.Applies(component, role) {
			continue
		}
		if rule.Mutation == nil {
			return fmt.Errorf("engine mutation is nil")
		}
		if err := rule.Mutation.Apply(container); err != nil {
			return fmt.Errorf("apply %T: %w", rule.Mutation, err)
		}
	}
	return nil
}
