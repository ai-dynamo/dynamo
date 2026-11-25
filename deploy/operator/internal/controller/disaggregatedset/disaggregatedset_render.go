// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package disaggregatedset

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// Render constructs the desired DisaggregatedSet from resolved role specs.
// dgd must be non-nil. Inputs are not modified.
func Render(dgd *nvidiacomv1beta1.DynamoGraphDeployment, roles []any) *unstructured.Unstructured {
	ds := &unstructured.Unstructured{}
	ds.SetGroupVersionKind(schema.GroupVersionKind{Group: "disaggregatedset.x-k8s.io", Version: "v1", Kind: "DisaggregatedSet"})
	ds.SetName(Name(dgd))
	ds.SetNamespace(dgd.Namespace)
	ds.SetLabels(map[string]string{
		consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
		consts.KubeLabelDynamoSelector:            Name(dgd),
	})

	ds.Object["spec"] = map[string]any{"roles": runtime.DeepCopyJSONValue(roles)}
	return ds
}

// Name reserves room for child roles, slices, revisions and worker suffixes.
// dgd must be non-nil.
func Name(dgd *nvidiacomv1beta1.DynamoGraphDeployment) string {
	return truncateDNSLabelWithHash(strings.ToLower(strings.ReplaceAll(dgd.Name, ".", "-")), maxDisaggregatedSetNameLength)
}

// Selection identifies the components and replica targets owned by this DisaggregatedSet.
type Selection struct {
	ComponentToRole map[string]string
	DesiredReplicas map[string]int32
}

const (
	maxDisaggregatedSetRoles                    = 10
	disaggregatedSetRevisionLength              = 8
	maxDisaggregatedSetGeneratedNameLength      = 63
	maxDisaggregatedSetRoleNameLength           = 10
	maxDisaggregatedSetSliceIndexLength         = len("99")
	maxDisaggregatedSetGroupIndexLength         = len("2147483647")
	maxDisaggregatedSetWorkerRevisionHashLength = 10
	disaggregatedSetChildNameSeparatorLength    = 3 * len("-")
	disaggregatedSetWorkerRevisionSuffixLength  = 2*len("-") + maxDisaggregatedSetGroupIndexLength + maxDisaggregatedSetWorkerRevisionHashLength
	maxDisaggregatedSetNameLength               = maxDisaggregatedSetGeneratedNameLength - maxDisaggregatedSetRoleNameLength - maxDisaggregatedSetSliceIndexLength - disaggregatedSetRevisionLength - disaggregatedSetChildNameSeparatorLength - disaggregatedSetWorkerRevisionSuffixLength
	disaggregatedSetNameHashLength              = 8
)

// Select computes component ownership and replica targets without cluster access.
// A nil dgd returns a diagnostic and an empty selection.
func Select(dgd *nvidiacomv1beta1.DynamoGraphDeployment) (Selection, string) {
	selection := Selection{
		ComponentToRole: make(map[string]string),
		DesiredReplicas: make(map[string]int32),
	}
	if dgd == nil {
		return selection, "DynamoGraphDeployment is nil"
	}

	usedRoles := make(map[string]struct{})
	zeroReplicas := 0
	positiveReplicas := 0
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if !Owns(component) {
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
		selection.ComponentToRole[component.ComponentName] = roleName

		desiredReplicas := DesiredReplicas(component)
		selection.DesiredReplicas[component.ComponentName] = desiredReplicas
		if desiredReplicas == 0 {
			zeroReplicas++
		} else {
			positiveReplicas++
		}
	}

	if len(selection.ComponentToRole) == 0 {
		return selection, "no eligible multinode worker roles found"
	}
	if len(selection.ComponentToRole) > maxDisaggregatedSetRoles {
		return selection, fmt.Sprintf("DisaggregatedSet supports at most %d roles", maxDisaggregatedSetRoles)
	}
	if zeroReplicas > 0 && positiveReplicas > 0 {
		return selection, "DisaggregatedSet requires replicas to be zero for all selected roles or positive for all selected roles"
	}
	return selection, ""
}

// Owns reports whether a component is a multinode worker rendered by DisaggregatedSet.
// A nil component is not owned.
func Owns(component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) bool {
	return component != nil && component.GetNumberOfNodes() > 1 && dynamo.IsWorkerComponent(string(component.ComponentType))
}

// DesiredReplicas returns the requested replica count, defaulting to one.
// A nil component also defaults to one.
func DesiredReplicas(component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) int32 {
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
