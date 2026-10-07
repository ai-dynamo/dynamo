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

// GroveCoherentUpdateInProgress reports the provider's PCS-wide scaling lock.
// pcs may be nil before initial creation; a missing PCS has no active update.
func GroveCoherentUpdateInProgress(pcs *grovev1alpha1.PodCliqueSet) bool {
	return pcs != nil && GroveCoherentUpdateSelected(pcs) && pcs.Status.UpdateProgress != nil && pcs.Status.UpdateProgress.UpdateEndedAt == nil
}

// GroveScalingBlocked reports whether Grove has not acknowledged this PCS generation
// or is still performing a Coherent update. pcs may be nil before creation.
// Before authorizing replica writes, callers must also synchronize the intended
// configuration and wait for the PCS watch if that synchronization writes it.
// Matching generation fields on an older cached PCS alone cannot authorize scaling.
func GroveScalingBlocked(pcs *grovev1alpha1.PodCliqueSet) bool {
	return pcs == nil || pcs.Status.ObservedGeneration == nil ||
		*pcs.Status.ObservedGeneration != pcs.Generation ||
		GroveCoherentUpdateInProgress(pcs)
}
