// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"testing"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestLPXRestartRequiresCurrentInputRevision(t *testing.T) {
	for _, strategy := range []v1beta1.RestartStrategyType{v1beta1.RestartStrategyTypeSequential, v1beta1.RestartStrategyTypeParallel} {
		t.Run(string(strategy), func(t *testing.T) {
			t.Log("Observe a Ready child after delivering the selected LPX restart")
			ctx := t.Context()
			_, dgd, kube := newLPXHandoffFixture(t, "node-local-v2-lpu-only")
			dgd.Spec.Restart = &v1beta1.Restart{ID: "restart-1", Strategy: &v1beta1.RestartStrategy{Type: strategy}}
			resolver := newLPXRestartProgressResolver(kube)
			dgd.Status.Restart = newDGDRestartReconciler().Resolve(ctx, dgd, &dgd.Status, resolver.Resolve).Status
			handoff := &dgdLPXHandoff{client: kube}
			child, err := handoff.Reconcile(ctx, dgd)
			require.NoError(t, err)
			require.Equal(t, "restart-1", child.Annotations[dynamo.LPXRestartAnnotation])
			child.Status.ObservedGeneration = child.Generation
			child.Status.Components = map[string]v1beta1.ComponentReplicaStatus{"lpx": {Ready: true}}
			meta.SetStatusCondition(&child.Status.Conditions, metav1.Condition{
				Type: v1alpha1.LPXReadyCondition, Status: metav1.ConditionTrue,
				ObservedGeneration: child.Generation, Reason: v1alpha1.LPXReadyReasonReady,
			})
			require.NoError(t, kube.Status().Update(ctx, child))
			require.Empty(t, resolver.Resolve(ctx, dgd, []string{"lpx"}))

			t.Log("An input edit before handoff must keep the restart in progress despite the old Ready receipt")
			lpx.ServingComponent(dgd).ComponentRole(v1beta1.ComponentRoleLPXAgent).PodTemplate.Spec.Containers[0].Image = "runtime:changed"
			dgd.Generation++
			revision, err := dynamo.LPXInputRevision(dgd, "restart-1")
			require.NoError(t, err)
			require.NotEqual(t, revision, child.Spec.InputRevision)
			require.Equal(t, []string{"lpx"}, resolver.Resolve(ctx, dgd, []string{"lpx"}))
			restart := newDGDRestartReconciler().Resolve(ctx, dgd, &dgd.Status, resolver.Resolve)
			require.Equal(t, []string{"lpx"}, restart.Status.InProgress)
			require.NotEqual(t, v1beta1.RestartPhaseCompleted, restart.Status.Phase)

			t.Log("Complete only after the handoff's new child generation becomes Ready")
			updated, err := handoff.Reconcile(ctx, dgd)
			require.NoError(t, err)
			require.Equal(t, revision, updated.Spec.InputRevision)

			// The fake client does not increment generation for the handoff's spec update.
			updated.Generation++
			require.NoError(t, kube.Update(ctx, updated))
			require.Equal(t, []string{"lpx"}, resolver.Resolve(ctx, dgd, []string{"lpx"}))
			updated.Status.ObservedGeneration = updated.Generation
			meta.SetStatusCondition(&updated.Status.Conditions, metav1.Condition{
				Type: v1alpha1.LPXReadyCondition, Status: metav1.ConditionTrue,
				ObservedGeneration: updated.Generation, Reason: v1alpha1.LPXReadyReasonReady,
			})
			require.NoError(t, kube.Status().Update(ctx, updated))
			require.Empty(t, resolver.Resolve(ctx, dgd, []string{"lpx"}))
			restart = newDGDRestartReconciler().Resolve(ctx, dgd, &dgd.Status, resolver.Resolve)
			require.Equal(t, v1beta1.RestartPhaseCompleted, restart.Status.Phase)
		})
	}
}
