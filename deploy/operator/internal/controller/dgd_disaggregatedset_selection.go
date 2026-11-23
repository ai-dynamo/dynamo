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
	"fmt"

	disaggregatedset "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/disaggregatedset"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	disaggregatedsetv1 "sigs.k8s.io/lws/api/disaggregatedset/v1"
	disaggregatedsetutils "sigs.k8s.io/lws/pkg/utils/disaggregatedset"
)

var disaggregatedSetGVK = schema.GroupVersionKind{
	Group:   "disaggregatedset.x-k8s.io",
	Version: "v1",
	Kind:    "DisaggregatedSet",
}

const (
	disaggregatedSetServiceSuffixLength = len("-prv")
	dynamoGraphDeploymentKind           = "DynamoGraphDeployment"
	dynamoComponentDeploymentKind       = "DynamoComponentDeployment"
)

type disaggregatedSetChildIdentity struct {
	slice int
	role  string
}

func newDisaggregatedSetObject() *unstructured.Unstructured {
	obj := &unstructured.Unstructured{}
	obj.SetGroupVersionKind(disaggregatedSetGVK)
	return obj
}

func disaggregatedSetEligibilityReason(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	gate features.Gate,
) string {
	if dgd == nil {
		return "DynamoGraphDeployment is nil"
	}
	selection, reason := disaggregatedset.Select(dgd)
	if reason != "" {
		return reason
	}
	if len(selection.ComponentToRole) < 2 {
		return "DisaggregatedSet requires at least two eligible multinode worker roles"
	}
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if !dynamo.IsWorkerComponent(string(component.ComponentType)) {
			continue
		}
		if _, selected := selection.ComponentToRole[component.ComponentName]; !selected {
			return fmt.Sprintf(
				"mixed DS/DCD worker rollout unsupported: worker component %q is not selected by DisaggregatedSet",
				component.ComponentName,
			)
		}
	}
	if gate == nil || !gate.Enabled(features.LWS) {
		for i := range dgd.Spec.Components {
			component := &dgd.Spec.Components[i]
			if component.GetNumberOfNodes() <= 1 {
				continue
			}
			if _, selected := selection.ComponentToRole[component.ComponentName]; !selected {
				return fmt.Sprintf(
					"multinode component %q is not eligible for DisaggregatedSet and requires LeaderWorkerSet support",
					component.ComponentName,
				)
			}
		}
	}
	return ""
}

func coalesceDisaggregatedSetRestartState(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	restartState *dynamo.RestartState,
) *dynamo.RestartState {
	if restartState == nil || dynamo.IsParallelRestart(dgd) {
		return restartState
	}
	selection, reason := disaggregatedset.Select(dgd)
	if reason != "" {
		return restartState
	}
	selectedRestarting := false
	for componentName := range selection.ComponentToRole {
		if restartState.ShouldAnnotateComponent(componentName) {
			selectedRestarting = true
			break
		}
	}
	if !selectedRestarting {
		return restartState
	}
	for componentName := range selection.ComponentToRole {
		restartState.ComponentsToAnnotate[componentName] = true
	}
	return restartState
}

func disaggregatedSetTargetRevision(ds *unstructured.Unstructured) (string, error) {
	typedDS := &disaggregatedsetv1.DisaggregatedSet{}
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(ds.Object, typedDS); err != nil {
		return "", fmt.Errorf("failed to decode DisaggregatedSet for target revision: %w", err)
	}
	return disaggregatedsetutils.ComputeRevision(typedDS.Spec.Roles), nil
}
