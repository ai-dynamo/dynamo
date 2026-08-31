/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package render

import (
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	corev1 "k8s.io/api/core/v1"
)

// PodSpecMutation makes one mechanical change to an already materialized pod
// spec.
type PodSpecMutation interface {
	Apply(podSpec *corev1.PodSpec) error
}

// PodSpecMutationRule associates a mechanical mutation with its applicability.
type PodSpecMutationRule struct {
	Applies  Predicate
	Mutation PodSpecMutation
}

// PodSpecMutations is an ordered set of conditional pod spec mutations.
type PodSpecMutations []PodSpecMutationRule

// Apply selects and applies mutations in declaration order.
func (mutations PodSpecMutations) Apply(
	component *v1beta1.DynamoComponentDeploymentSharedSpec,
	role Role,
	podSpec *corev1.PodSpec,
) error {
	for _, rule := range mutations {
		if rule.Applies != nil && !rule.Applies(component, role) {
			continue
		}
		if rule.Mutation == nil {
			return fmt.Errorf("pod spec mutation is nil")
		}
		if err := rule.Mutation.Apply(podSpec); err != nil {
			return fmt.Errorf("apply %T: %w", rule.Mutation, err)
		}
	}
	return nil
}

// AppendVolumeMutation appends a volume to a pod spec.
type AppendVolumeMutation struct {
	Volume corev1.Volume
}

func (m AppendVolumeMutation) Apply(podSpec *corev1.PodSpec) error {
	if podSpec == nil {
		return fmt.Errorf("pod spec is nil")
	}
	podSpec.Volumes = append(podSpec.Volumes, *m.Volume.DeepCopy())
	return nil
}

// AppendInitContainerMutation appends an init container to a pod spec.
type AppendInitContainerMutation struct {
	Container corev1.Container
}

func (m AppendInitContainerMutation) Apply(podSpec *corev1.PodSpec) error {
	if podSpec == nil {
		return fmt.Errorf("pod spec is nil")
	}
	podSpec.InitContainers = append(podSpec.InitContainers, *m.Container.DeepCopy())
	return nil
}
