/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"context"
	"fmt"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// GPUShape separates inference-engine width from the unique GPU allocation
// added when the component scales by one replica.
type GPUShape struct {
	GPUsPerEngine  int64
	GPUsPerReplica int64
}

// ResolveGroveGPUShapes computes one GPU shape per rendered DGD component.
// Structural role multiplicities are used so checkpoint gating to zero does
// not erase the future cost of one component replica.
func ResolveGroveGPUShapes(
	ctx context.Context,
	reader client.Reader,
	dgd *v1beta1.DynamoGraphDeployment,
	pcs *grovev1alpha1.PodCliqueSet,
) (map[string]GPUShape, error) {
	shapes := make(map[string]GPUShape)
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		roleCounts := make(map[string]int32)
		for _, role := range expandRolesForComponent(
			component.ComponentName,
			component.Replicas,
			component.GetNumberOfNodes(),
			component,
		) {
			roleCounts[strings.ToLower(role.Name)] = role.Replicas
		}

		pods := make([]PodSpecMultiplicity, 0)
		for cliqueIndex := range pcs.Spec.Template.Cliques {
			clique := pcs.Spec.Template.Cliques[cliqueIndex]
			if clique == nil {
				continue
			}
			if clique.Labels[commonconsts.KubeLabelDynamoComponent] != component.ComponentName {
				continue
			}
			multiplicity := int32(1)
			if component.UsesPCSG() {
				multiplicity = roleCounts[clique.Name]
			}
			pods = append(pods, PodSpecMultiplicity{
				PodSpec: &clique.Spec.PodSpec,
				Count:   multiplicity,
			})
		}
		if len(pods) == 0 {
			continue
		}

		shape, err := ResolveGPUShape(ctx, reader, dgd.Namespace, component, pods)
		if err != nil {
			return nil, fmt.Errorf("resolve Grove GPU shape for component %q: %w", component.ComponentName, err)
		}
		if component.IsInterPodGMSEnabled() {
			shape.GPUsPerReplica += shape.GPUsPerEngine
		}
		if shape.GPUsPerReplica > 0 {
			shapes[component.ComponentName] = shape
		}
	}
	return shapes, nil
}

type PodSpecMultiplicity = dra.PodSpecMultiplicity

// ResolveGPUShape computes the engine width from the component's main
// container and the replica cost from rendered Pod specs.
func ResolveGPUShape(
	ctx context.Context,
	reader client.Reader,
	namespace string,
	component *v1beta1.DynamoComponentDeploymentSharedSpec,
	pods []PodSpecMultiplicity,
) (GPUShape, error) {
	if component == nil {
		return GPUShape{}, fmt.Errorf("component is nil")
	}
	enginePodSpec := &corev1.PodSpec{}
	if component.PodTemplate != nil {
		enginePodSpec = component.PodTemplate.Spec.DeepCopy()
		enginePodSpec.Containers = nil
		if main := GetMainContainer(component); main != nil {
			enginePodSpec.Containers = []corev1.Container{*main.DeepCopy()}
		}
	}
	engineGPUs, err := dra.ResolvePodSetGPUCount(ctx, reader, namespace, []dra.PodSpecMultiplicity{{
		PodSpec: enginePodSpec,
		Count:   component.GetNumberOfNodes(),
	}})
	if err != nil {
		return GPUShape{}, err
	}
	shape := GPUShape{GPUsPerEngine: int64(engineGPUs)}
	replicaGPUs, err := dra.ResolvePodSetGPUCount(ctx, reader, namespace, pods)
	if err != nil {
		return GPUShape{}, err
	}
	shape.GPUsPerReplica = int64(replicaGPUs)
	return shape, nil
}
