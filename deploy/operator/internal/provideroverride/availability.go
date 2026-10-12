// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package provideroverride

import (
	"encoding/json"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
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

// HasGroveMinAvailableOverrides reports native minimum availability in a non-nil DGD.
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
// The omitted default is part of the immutable minimum contract; changing it
// requires origin-version gating to preserve existing deployments.
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
