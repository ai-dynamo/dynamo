// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package provideroverride

import (
	"encoding/json"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
)

// HasLegacyGroveMinAvailable reports a compatibility marker anywhere in a non-nil DGD.
func HasLegacyGroveMinAvailable(dgd *v1beta1.DynamoGraphDeployment) bool {
	for i := range dgd.Spec.Components {
		if dgd.Spec.Components[i].MinAvailable != nil {
			return true
		}
	}
	return false
}

// HasGroveMinAvailableOverrides reports explicit availability opt-in in a non-nil DGD.
func HasGroveMinAvailableOverrides(dgd *v1beta1.DynamoGraphDeployment) bool {
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if component.ProviderOverride == nil || component.ProviderOverride.APIVersion != GroveAPIVersion {
			continue
		}
		if _, exists := GroveMinAvailable(component.ProviderOverride.Value.Raw); exists {
			return true
		}
	}
	return false
}

// GroveMinAvailable returns the native provider minimum, if present and well-formed.
// Invalid shapes and values are reported by ValidateValue separately.
func GroveMinAvailable(raw []byte) (int32, bool) {
	// Both native owner shapes use the same immutable minimum, at different paths.
	var value struct {
		MinAvailable *int32 `json:"minAvailable"`
		Spec         *struct {
			MinAvailable *int32 `json:"minAvailable"`
		} `json:"spec"`
	}
	if err := json.Unmarshal(raw, &value); err != nil {
		return 0, false
	}
	minimum := value.MinAvailable
	if minimum == nil && value.Spec != nil {
		minimum = value.Spec.MinAvailable
	}
	if minimum == nil {
		return 0, false
	}
	return *minimum, true
}

// EffectiveGroveMinAvailable resolves both API forms and the native default of one.
// component must be non-nil. Admission rejects mixing the forms and malformed values.
func EffectiveGroveMinAvailable(component *v1beta1.DynamoComponentDeploymentSharedSpec) int32 {
	if component.MinAvailable != nil {
		return *component.MinAvailable
	}
	if component.ProviderOverride != nil && component.ProviderOverride.APIVersion == GroveAPIVersion {
		if minimum, exists := GroveMinAvailable(component.ProviderOverride.Value.Raw); exists {
			return minimum
		}
	}
	return 1
}

// DefaultGroveMinAvailable materializes the new default while preserving explicit fragments.
// component must be non-nil. Invalid identities and values remain unchanged for admission.
func DefaultGroveMinAvailable(component *v1beta1.DynamoComponentDeploymentSharedSpec) {
	// Only a conductor-bearing LPX component owns a workload scaling group.
	if component.IsLPX() && component.ComponentRole(v1beta1.ComponentRoleLPXConductor) == nil {
		return
	}
	target := TargetPodCliqueTemplateSpec
	if component.UsesPCSG() || component.IsLPX() {
		target = TargetPodCliqueScalingGroupConfig
	}
	if component.ProviderOverride == nil {
		component.ProviderOverride = &v1beta1.ProviderOverride{APIVersion: GroveAPIVersion, Target: target, Value: apiextensionsv1.JSON{Raw: []byte(`{}`)}}
	}
	override := component.ProviderOverride
	if override.APIVersion != GroveAPIVersion || override.Target != target {
		return
	}

	// Decode raw messages so defaulting preserves opaque topology and explicit invalid values.
	var value map[string]json.RawMessage
	if err := json.Unmarshal(override.Value.Raw, &value); err != nil || value == nil {
		return
	}
	owner := value
	if target == TargetPodCliqueTemplateSpec {
		owner = map[string]json.RawMessage{}
		if raw, exists := value["spec"]; exists {
			if err := json.Unmarshal(raw, &owner); err != nil || owner == nil {
				return
			}
		}
	}
	if _, exists := owner["minAvailable"]; exists {
		return
	}
	owner["minAvailable"] = json.RawMessage(`1`)
	if target == TargetPodCliqueTemplateSpec {
		value["spec"], _ = json.Marshal(owner)
	}
	override.Value.Raw, _ = json.Marshal(value)
}
