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

type lpxRestartProgressResolver struct {
	reader client.Reader
}

func newLPXRestartProgressResolver(reader client.Reader) *lpxRestartProgressResolver {
	return &lpxRestartProgressResolver{reader: reader}
}

// Resolve returns LPX components whose current child has not completed the selected restart.
func (r *lpxRestartProgressResolver) Resolve(
	ctx context.Context,
	source *v1beta1.DynamoGraphDeployment,
	inProgress []string,
) []string {
	if r.observeRestart(ctx, source) == nil {
		return inProgress
	}
	return nil
}

// resolveCompositeGroveRestartProgress composes child-owned LPX and ordinary Grove observations.
func resolveCompositeGroveRestartProgress(
	ctx context.Context,
	req groveReconcileRequest,
	inProgress []string,
	ordinaryResolver *groveRestartProgressResolver,
	lpxResolver *lpxRestartProgressResolver,
) []string {
	ordinary := make([]string, 0, len(inProgress))
	delegated := make([]string, 0, len(inProgress))
	pending := make(map[string]bool, len(inProgress))
	delegatedNames := make(map[string]struct{})
	for _, component := range req.DelegatedComponents() {
		delegatedNames[component.ComponentName] = struct{}{}
	}

	for _, name := range inProgress {
		component := req.DGD.GetComponentByName(name)
		if component == nil {
			continue
		}
		if _, found := delegatedNames[name]; found {
			delegated = append(delegated, name)
		} else {
			ordinary = append(ordinary, name)
		}
	}

	// Observe the shared LPX child before ordinary Grove restart progress.
	if len(delegated) > 0 {
		for _, name := range lpxResolver.Resolve(ctx, req.DGD, delegated) {
			pending[name] = true
		}
	}
	if len(ordinary) > 0 {
		for _, name := range ordinaryResolver.Resolve(ctx, req, ordinary) {
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

// observeRestart returns the child for one current, ready restart observation.
// A failed read or incomplete child leaves every requested member pending.
func (r *lpxRestartProgressResolver) observeRestart(
	ctx context.Context,
	source *v1beta1.DynamoGraphDeployment,
) *v1alpha1.LPXGraphDeployment {
	child := &v1alpha1.LPXGraphDeployment{}
	if err := r.reader.Get(ctx, client.ObjectKeyFromObject(source), child); err != nil ||
		child.Status.ObservedGeneration != child.Generation || !child.DeletionTimestamp.IsZero() ||
		!metav1.IsControlledBy(child, source) || source.Spec.Restart == nil ||
		child.Annotations[dynamo.LPXRestartAnnotation] != source.Spec.Restart.ID {
		return nil
	}

	// Skip input hashing until the child reports current readiness.
	ready := meta.FindStatusCondition(child.Status.Conditions, v1alpha1.LPXReadyCondition)
	if ready == nil || ready.Status != metav1.ConditionTrue || ready.ObservedGeneration != child.Generation {
		return nil
	}

	// Restart progress is resolved before handoff; an old Ready child cannot cover a newer DGD input.
	revision, err := dynamo.LPXInputRevision(source, source.Spec.Restart.ID)
	if err != nil || child.Spec.InputRevision != revision {
		return nil
	}
	return child
}
