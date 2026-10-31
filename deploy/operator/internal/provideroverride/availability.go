// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package provideroverride

import (
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	sigsjson "sigs.k8s.io/json"
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
	if err := sigsjson.UnmarshalCaseSensitivePreserveInts(raw, &value); err != nil {
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

// GroveMaxUnavailable returns the configured budget, if present and well-formed.
// Shape and value errors are reported by ValidateValue.
func GroveMaxUnavailable(raw []byte) (int32, bool) {
	// Decode only the supported knob without modifying the raw provider fragment.
	var value struct {
		RollingUpdate *struct {
			MaxUnavailable *int32 `json:"maxUnavailable"`
		} `json:"rollingUpdate"`
	}
	if err := sigsjson.UnmarshalCaseSensitivePreserveInts(raw, &value); err != nil || value.RollingUpdate == nil || value.RollingUpdate.MaxUnavailable == nil {
		return 0, false
	}
	return *value.RollingUpdate.MaxUnavailable, true
}

// EffectiveGroveMaxUnavailable resolves an explicit budget or Grove's rolling-strategy default.
// component must be non-nil. Admission rejects budgets under OnDelete and malformed values.
// Consumers must select a rolling strategy before using its disruption budget.
func EffectiveGroveMaxUnavailable(component *v1beta1.DynamoComponentDeploymentSharedSpec, strategy grovev1alpha1.UpdateStrategyType) int32 {
	// Prefer the authored component budget to the provider defaults.
	if component.ProviderOverride != nil && component.ProviderOverride.APIVersion == GroveAPIVersion {
		if budget, exists := GroveMaxUnavailable(component.ProviderOverride.Value.Raw); exists {
			return budget
		}
	}

	// Coherent defaults to its minimum unit; RollingRecreate defaults to one.
	if strategy == grovev1alpha1.CoherentStrategy {
		return EffectiveGroveMinAvailable(component)
	}
	return 1
}
