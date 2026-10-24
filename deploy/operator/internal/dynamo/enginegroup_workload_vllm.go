/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"
	"strconv"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

func renderVLLMComponentEngineGroup(dgd *api.DynamoGraphDeployment, component *api.DynamoComponentDeploymentSharedSpec, worldIndex int32) (*api.DynamoGraphDeploymentEngineGroup, error) {
	// Geometry and supported world count belong to the backend, not the child lifecycle controller.
	if worldIndex != 0 {
		return nil, fmt.Errorf("vLLM Engine Group integration supports one world per component")
	}
	_, profile, err := ResolveVLLMComponentEngineGroup(component, dgd.Spec.Env)
	if err != nil {
		return nil, err
	}
	return RenderVLLMEngineGroup(dgd, component, worldIndex, profile), nil
}

// RenderVLLMEngineGroup declares one new world and seeds its private launch authority at creation.
// dgd, component, and component.EngineGroup must not be nil; worldIndex must be nonnegative.
// The caller must have validated the component's native launch profile. No input is mutated.
// Rendering this declaration does not register a complete vLLM runtime or enable DGD admission.
func RenderVLLMEngineGroup(
	dgd *api.DynamoGraphDeployment,
	component *api.DynamoComponentDeploymentSharedSpec,
	worldIndex int32,
	profile ResolvedVLLMElasticEPProfile,
) *api.DynamoGraphDeploymentEngineGroup {
	group := &api.DynamoGraphDeploymentEngineGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name: EngineGroupNameForComponent(dgd.Name, component.ComponentName, worldIndex), Namespace: dgd.Namespace,
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName:   dgd.Name,
				consts.KubeLabelDynamoComponent:             component.ComponentName,
				consts.KubeLabelDynamoEngineGroupWorldIndex: strconv.FormatInt(int64(worldIndex), 10),
				consts.KubeLabelDynamoEngineGroupRuntime:    string(BackendFrameworkVLLM),
			},
			Annotations: map[string]string{
				consts.KubeAnnotationDynamoEngineGroupProfile:   profile.Geometry.Fingerprint,
				consts.KubeAnnotationDynamoVLLMCapacityBindings: consts.InitialVLLMCapacityBindings,
			},
			OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dgd, api.GroupVersion.WithKind("DynamoGraphDeployment"))},
		},
		Spec: api.DynamoGraphDeploymentEngineGroupSpec{Replicas: component.EngineGroup.InitialSize},
	}
	if policy := component.EngineGroup.Policy; policy != nil {
		group.Spec.Policy = &api.EngineGroupScalingPolicy{
			MinReplicas: ptr.To(ptr.Deref(policy.MinSize, profile.InitialReplicas)),
			MaxReplicas: ptr.To(ptr.Deref(policy.MaxSize, profile.MaximumReplicas)),
		}
	}
	return group
}
