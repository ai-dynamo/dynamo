/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package controller

import (
	"context"
	"fmt"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
)

type disaggregatedSetWorkloadRenderer struct {
	components *dcdWorkloadRenderer
}

func newDisaggregatedSetWorkloadRenderer(components *dcdWorkloadRenderer) *disaggregatedSetWorkloadRenderer {
	return &disaggregatedSetWorkloadRenderer{components: components}
}

func (r *disaggregatedSetWorkloadRenderer) Render(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	components map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	selection disaggregatedSetSelection,
	rollingUpdateCtx dynamo.RollingUpdateContext,
	checkpointInfos map[string]*checkpoint.CheckpointInfo,
) (*unstructured.Unstructured, error) {
	ds := newDisaggregatedSetObject()
	ds.SetName(disaggregatedSetName(dgd))
	ds.SetNamespace(dgd.Namespace)
	ds.SetLabels(map[string]string{
		consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
		consts.KubeLabelDynamoSelector:            disaggregatedSetName(dgd),
	})
	if ownerRef := dgdControllerOwnerReference(dgd); ownerRef != nil {
		ds.SetOwnerReferences([]metav1.OwnerReference{*ownerRef})
	}

	roles := make([]any, 0, len(selection.componentToRole))
	for i := range dgd.Spec.Components {
		componentName := dgd.Spec.Components[i].ComponentName
		roleName, ok := selection.componentToRole[componentName]
		if !ok {
			continue
		}
		component := components[componentName]
		if component == nil {
			return nil, fmt.Errorf("normalized component missing for selected component %q", componentName)
		}
		backendFramework, err := dynamo.BackendFrameworkForComponent(component, dgd)
		if err != nil {
			return nil, fmt.Errorf("failed to determine backend framework for selected component %q: %w", componentName, err)
		}
		workloadName := dynamo.GetDCDResourceName(dgd, componentName, rollingUpdateCtx.NewWorkerHash)
		role, err := r.renderRole(
			ctx,
			dgd,
			component,
			componentName,
			workloadName,
			dynamo.GetDynamoNamespace(dgd, component),
			backendFramework,
			checkpointInfos[componentName],
		)
		if err != nil {
			return nil, fmt.Errorf("failed to build DisaggregatedSet role %q: %w", roleName, err)
		}
		role["name"] = roleName
		roles = append(roles, role)
	}
	if len(roles) < 2 {
		return nil, fmt.Errorf("DisaggregatedSet requires at least two roles, got %d", len(roles))
	}
	ds.Object["spec"] = map[string]any{"roles": roles}
	return ds, nil
}

func (r *disaggregatedSetWorkloadRenderer) renderRole(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	componentName string,
	workloadName string,
	dynamoNamespace string,
	backendFramework dynamo.BackendFramework,
	checkpointInfo *checkpoint.CheckpointInfo,
) (map[string]any, error) {
	leaderPodTemplateSpec, workerPodTemplateSpec, err := r.components.renderMultinodePodTemplateSpecsForDGDComponent(
		ctx,
		dgd,
		component,
		componentName,
		workloadName,
		dynamoNamespace,
		backendFramework,
		checkpointInfo,
	)
	if err != nil {
		return nil, err
	}

	desiredReplicas := int32(1)
	if component.Replicas != nil {
		desiredReplicas = *component.Replicas
	}
	groupSize := component.GetNumberOfNodes()

	lwsSpec := leaderworkersetv1.LeaderWorkerSetSpec{
		Replicas:      &desiredReplicas,
		StartupPolicy: leaderworkersetv1.LeaderCreatedStartupPolicy,
		RolloutStrategy: leaderworkersetv1.RolloutStrategy{
			Type: leaderworkersetv1.RollingUpdateStrategyType,
		},
		LeaderWorkerTemplate: leaderworkersetv1.LeaderWorkerTemplate{
			LeaderTemplate: leaderPodTemplateSpec,
			WorkerTemplate: *workerPodTemplateSpec,
			Size:           &groupSize,
			RestartPolicy:  leaderworkersetv1.RecreateGroupOnPodRestart,
		},
	}
	lwsSpecUnstructured, err := runtime.DefaultUnstructuredConverter.ToUnstructured(&lwsSpec)
	if err != nil {
		return nil, fmt.Errorf("failed to convert LeaderWorkerSet spec: %w", err)
	}
	return map[string]any{"spec": lwsSpecUnstructured}, nil
}
