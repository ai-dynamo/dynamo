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
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
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
	maxDisaggregatedSetRoles            = 10
	disaggregatedSetRevisionLength      = 8
	maxDisaggregatedSetNameLength       = 31
	maxDisaggregatedSetSliceIndexLength = len("99")
	disaggregatedSetServiceSuffixLength = len("-prv")
	maxDisaggregatedSetRoleNameLength   = 63 - maxDisaggregatedSetNameLength - maxDisaggregatedSetSliceIndexLength - disaggregatedSetRevisionLength - 3 - disaggregatedSetServiceSuffixLength
	disaggregatedSetNameHashLength      = 8
	dynamoGraphDeploymentKind           = "DynamoGraphDeployment"
	dynamoComponentDeploymentKind       = "DynamoComponentDeployment"
	resourceNotFoundReason              = "resource not found"
	managedServiceMetadataAnnotation    = "nvidia.com/dynamo-managed-service-metadata"
)

type managedServiceMetadata struct {
	Labels      []string `json:"labels,omitempty"`
	Annotations []string `json:"annotations,omitempty"`
}

type disaggregatedSetSelection struct {
	componentToRole map[string]string
	desiredReplicas map[string]int32
}

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
	selection, reason := selectDisaggregatedSetComponents(dgd)
	if reason != "" {
		return reason
	}
	if len(selection.componentToRole) < 2 {
		return "DisaggregatedSet requires at least two eligible multinode worker roles"
	}
	if gate == nil || !gate.Enabled(features.LWS) {
		for i := range dgd.Spec.Components {
			component := &dgd.Spec.Components[i]
			if component.GetNumberOfNodes() <= 1 {
				continue
			}
			if _, selected := selection.componentToRole[component.ComponentName]; !selected {
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
	selection, reason := selectDisaggregatedSetComponents(dgd)
	if reason != "" {
		return restartState
	}
	selectedRestarting := false
	for componentName := range selection.componentToRole {
		if restartState.ShouldAnnotateComponent(componentName) {
			selectedRestarting = true
			break
		}
	}
	if !selectedRestarting {
		return restartState
	}
	for componentName := range selection.componentToRole {
		restartState.ComponentsToAnnotate[componentName] = true
	}
	return restartState
}

func selectDisaggregatedSetComponents(dgd *nvidiacomv1beta1.DynamoGraphDeployment) (disaggregatedSetSelection, string) {
	selection := disaggregatedSetSelection{
		componentToRole: make(map[string]string),
		desiredReplicas: make(map[string]int32),
	}
	if dgd == nil {
		return selection, "DynamoGraphDeployment is nil"
	}

	usedRoles := make(map[string]struct{})
	zeroReplicas := 0
	positiveReplicas := 0
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if !isDisaggregatedSetEligibleComponent(component) {
			continue
		}
		if component.ScalingAdapter != nil {
			return selection, fmt.Sprintf(
				"component %q uses scalingAdapter, but DisaggregatedSet does not support scale subresource integration",
				component.ComponentName,
			)
		}
		roleName := disaggregatedSetRoleName(component, usedRoles)
		usedRoles[roleName] = struct{}{}
		selection.componentToRole[component.ComponentName] = roleName

		desiredReplicas := desiredComponentReplicas(component)
		selection.desiredReplicas[component.ComponentName] = desiredReplicas
		if desiredReplicas == 0 {
			zeroReplicas++
		} else {
			positiveReplicas++
		}
	}

	if len(selection.componentToRole) == 0 {
		return selection, "no eligible multinode worker roles found"
	}
	if len(selection.componentToRole) > maxDisaggregatedSetRoles {
		return selection, fmt.Sprintf("DisaggregatedSet supports at most %d roles", maxDisaggregatedSetRoles)
	}
	if zeroReplicas > 0 && positiveReplicas > 0 {
		return selection, "DisaggregatedSet requires replicas to be zero for all selected roles or positive for all selected roles"
	}
	return selection, ""
}

func isDisaggregatedSetEligibleComponent(component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) bool {
	return component != nil && component.GetNumberOfNodes() > 1 && dynamo.IsWorkerComponent(string(component.ComponentType))
}

func desiredComponentReplicas(component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) int32 {
	if component == nil || component.Replicas == nil {
		return 1
	}
	return *component.Replicas
}

func disaggregatedSetRoleName(component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec, used map[string]struct{}) string {
	preferred := strings.ToLower(string(component.ComponentType))
	if preferred != consts.ComponentTypePrefill && preferred != consts.ComponentTypeDecode {
		preferred = ""
	}
	if preferred == "" || roleNameUsed(preferred, used) {
		preferred = dynamo.NormalizeKubeResourceName(component.ComponentName)
	}
	preferred = truncateDNSLabelWithHash(preferred, maxDisaggregatedSetRoleNameLength)
	roleName := preferred
	for i := 2; roleNameUsed(roleName, used); i++ {
		suffix := fmt.Sprintf("-%d", i)
		roleName = truncateDNSLabel(preferred, maxDisaggregatedSetRoleNameLength-len(suffix)) + suffix
	}
	return roleName
}

func roleNameUsed(roleName string, used map[string]struct{}) bool {
	_, ok := used[roleName]
	return ok
}

func truncateDNSLabel(value string, maxLength int) string {
	if len(value) <= maxLength {
		return value
	}
	return strings.TrimRight(value[:maxLength], "-")
}

func truncateDNSLabelWithHash(value string, maxLength int) string {
	if len(value) <= maxLength {
		return value
	}
	hash := sha256.Sum256([]byte(value))
	hashText := hex.EncodeToString(hash[:])[:disaggregatedSetNameHashLength]
	if maxLength <= len(hashText) {
		return hashText[:maxLength]
	}
	suffix := "-" + hashText
	prefix := strings.TrimRight(value[:maxLength-len(suffix)], "-")
	return prefix + suffix
}

func disaggregatedSetName(dgd *nvidiacomv1beta1.DynamoGraphDeployment) string {
	return truncateDNSLabelWithHash(dynamo.NormalizeKubeResourceName(dgd.Name), maxDisaggregatedSetNameLength)
}

func disaggregatedSetTargetRevision(ds *unstructured.Unstructured) (string, error) {
	typedDS := &disaggregatedsetv1.DisaggregatedSet{}
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(ds.Object, typedDS); err != nil {
		return "", fmt.Errorf("failed to decode DisaggregatedSet for target revision: %w", err)
	}
	return disaggregatedsetutils.ComputeRevision(typedDS.Spec.Roles), nil
}
