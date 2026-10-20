// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"k8s.io/utils/ptr"
)

// SupportedGroveUpdateStrategies returns the annotation values accepted by rendering and admission.
func SupportedGroveUpdateStrategies() []string {
	return []string{string(grovev1alpha1.CoherentStrategy), string(grovev1alpha1.RollingRecreateStrategy), string(grovev1alpha1.OnDeleteStrategy)}
}

// ParseGroveUpdateStrategy parses an exact, case-sensitive annotation value.
func ParseGroveUpdateStrategy(value string) (grovev1alpha1.UpdateStrategyType, error) {
	// Keep admission and rendering on the same supported value set.
	for _, supported := range SupportedGroveUpdateStrategies() {
		if value == supported {
			return grovev1alpha1.UpdateStrategyType(value), nil
		}
	}
	return "", fmt.Errorf("unsupported Grove update strategy annotation %q=%q: supported values are %q", consts.KubeAnnotationGroveUpdateStrategy, value, SupportedGroveUpdateStrategies())
}

// ResolveGroveUpdateStrategy selects the strategy for a non-nil DGD without mutating inputs.
// Without an annotation, Grove defaults to RollingRecreate. Availability fields do not select a strategy.
// existingPCS may be nil on creation. All strategy transitions wait for an active rollout.
func ResolveGroveUpdateStrategy(dgd *v1beta1.DynamoGraphDeployment, existingPCS *grovev1alpha1.PodCliqueSet) (*grovev1alpha1.UpdateStrategyType, error) {
	// Resolve explicit intent first so invalid annotations are never silently ignored.
	var desired *grovev1alpha1.UpdateStrategyType
	if value, exists := dgd.Annotations[consts.KubeAnnotationGroveUpdateStrategy]; exists {
		strategy, err := ParseGroveUpdateStrategy(value)
		if err != nil {
			return nil, err
		}
		desired = ptr.To(strategy)
	}

	// Retain the strategy that initialized Grove's progress until that update finishes.
	if existingPCS != nil && existingPCS.Status.UpdateProgress != nil && existingPCS.Status.UpdateProgress.UpdateEndedAt == nil {
		if existingPCS.Spec.UpdateStrategy != nil {
			return ptr.To(existingPCS.Spec.UpdateStrategy.Type), nil
		}
		return nil, nil
	}
	return desired, nil
}

// GroveCoherentUpdateSelected reports whether the PCS uses Coherent updates.
// pcs must be non-nil.
func GroveCoherentUpdateSelected(pcs *grovev1alpha1.PodCliqueSet) bool {
	return pcs.Spec.UpdateStrategy != nil && pcs.Spec.UpdateStrategy.Type == grovev1alpha1.CoherentStrategy
}

// GroveScalingBlocked reports whether a Coherent PCS must defer replica changes.
// pcs may be nil before creation. Other strategies do not block scaling.
func GroveScalingBlocked(pcs *grovev1alpha1.PodCliqueSet) bool {
	if pcs == nil || !GroveCoherentUpdateSelected(pcs) {
		return false
	}

	// Old or missing progress cannot acknowledge a newly observed Coherent spec.
	// Grove advances observedGeneration after resource synchronization, which can
	// lag until a rollout finishes; this wait applies only to Coherent.
	if pcs.Status.ObservedGeneration == nil || *pcs.Status.ObservedGeneration != pcs.Generation {
		return true
	}

	return pcs.Status.UpdateProgress != nil && pcs.Status.UpdateProgress.UpdateEndedAt == nil
}
