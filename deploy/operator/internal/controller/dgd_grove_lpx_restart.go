// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// resolveRestartProgress composes child-owned LPX and ordinary Grove observations.
func (p *groveProgram) resolveRestartProgress(
	ctx context.Context,
	source *v1beta1.DynamoGraphDeployment,
	ordinaryDGD *v1beta1.DynamoGraphDeployment,
	inProgress []string,
) []string {
	ordinary := make([]string, 0, len(inProgress))
	pending := make(map[string]bool, len(inProgress))
	var childComponents map[string]v1beta1.ComponentReplicaStatus
	childObserved := false

	for _, name := range inProgress {
		component := source.GetComponentByName(name)
		if component == nil {
			continue
		}
		if !component.IsLPX() {
			ordinary = append(ordinary, name)
			continue
		}
		if !childObserved {
			childComponents = observeLPXRestart(ctx, p.lpx.Client, source)
			childObserved = true
		}
		if !childComponents[name].Ready {
			pending[name] = true
		}
	}

	if len(ordinary) > 0 {
		for _, name := range p.restartProgress.Resolve(ctx, ordinaryDGD, ordinary) {
			pending[name] = true
		}
	}

	remaining := make([]string, 0, len(inProgress))
	for _, name := range inProgress {
		if pending[name] {
			remaining = append(remaining, name)
		}
	}
	return remaining
}

// observeLPXRestart returns component status from one current, ready restart observation.
// A failed read or incomplete child leaves every requested member pending.
func observeLPXRestart(ctx context.Context, reader client.Reader, source *v1beta1.DynamoGraphDeployment) map[string]v1beta1.ComponentReplicaStatus {
	child := &v1alpha1.LPXGraphDeployment{}
	if err := reader.Get(ctx, client.ObjectKeyFromObject(source), child); err != nil ||
		child.Status.ObservedGeneration != child.Generation || !child.DeletionTimestamp.IsZero() ||
		dynamo.ValidateLPXSource(child, source) != nil || source.Spec.Restart == nil ||
		child.Annotations[dynamo.LPXRestartAnnotation] != source.Spec.Restart.ID {
		return nil
	}
	ready := meta.FindStatusCondition(child.Status.Conditions, "Ready")
	if ready == nil || ready.Status != metav1.ConditionTrue || ready.ObservedGeneration != child.Generation {
		return nil
	}
	return child.Status.Components
}
