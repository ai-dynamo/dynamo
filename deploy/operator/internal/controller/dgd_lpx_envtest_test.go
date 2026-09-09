//go:build !clustertest

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorenv"
	"github.com/stretchr/testify/require"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func TestLPXGraphDeploymentAPIHandoff(t *testing.T) {
	t.Log("Install the real CRDs in an isolated API server; no runtime controllers or cluster workloads")
	env := operatorenv.New(operatorenv.Options{SetupWebhooks: setupProductionWebhooks}).RunT(t)
	require.True(t, env.Client().Scheme().Recognizes(v1alpha1.LPXGraphDeploymentGVK))
	require.False(t, env.Client().Scheme().Recognizes(v1beta1.GroupVersion.WithKind("LPXGraphDeployment")))
	source := newLPXHandoffSource(t, "node-local-v2-lpu-only")
	source.Namespace, source.UID, source.Generation = env.Namespace(), "", 0
	require.NoError(t, env.Client().Create(t.Context(), source))

	t.Log("Persist the pending component projection using the real DGD status schema")
	result := &ReconcileResult{}
	projectLPXChildStatus(source, nil, result, &source.Status)
	source.Status.Components = result.ComponentStatus
	source.Status.State = result.State
	require.NoError(t, env.Client().Status().Update(t.Context(), source))
	require.Equal(t, v1beta1.ComponentKindPodCliqueScalingGroup, source.Status.Components["lpx"].ComponentKind)
	require.False(t, source.Status.Components["lpx"].Ready)

	t.Log("Hand off the beta source to one independently observed alpha child with its beta DGD owner")
	handoff := &dgdLPXHandoff{Client: env.Client()}
	child, err := handoff.Reconcile(t.Context(), source)
	require.NoError(t, err)
	require.Equal(t, int64(1), child.Generation)
	require.NotEmpty(t, child.UID)
	require.NotEqual(t, source.UID, child.UID)
	require.Equal(t, metav1.NewControllerRef(source, v1beta1.DynamoGraphDeploymentGVK), metav1.GetControllerOf(child))
	require.NoError(t, dynamo.ValidateLPXSource(child, source))

	t.Log("Project the observed download payload and its actionable failure")
	child.Status.ObservedGeneration = child.Generation
	child.Status.ModelDownload = &v1beta1.ModelDownloadStatus{Builds: []string{"downloaded-build"}}
	child.Status.Conditions = []metav1.Condition{{Type: "Failed", Status: metav1.ConditionTrue, ObservedGeneration: child.Generation,
		LastTransitionTime: metav1.Now(), Reason: "PublicationDenied", Message: "Check the namespace quota"}}
	require.NoError(t, env.Client().Status().Update(t.Context(), child))
	projected := v1beta1.DynamoGraphDeploymentStatus{}
	result = &ReconcileResult{State: v1beta1.DGDStateSuccessful}
	projectLPXChildStatus(source, child, result, &projected)
	require.Equal(t, v1beta1.DGDStateFailed, result.State)
	require.Equal(t, Reason(child.Status.Conditions[0].Reason), result.Reason)
	require.Equal(t, Message(child.Status.Conditions[0].Message), result.Message)
	require.Equal(t, &v1beta1.DynamoGraphDeploymentLPXStatus{ModelDownload: child.Status.ModelDownload}, projected.LPX)

	t.Log("Ordinary source metadata does not advance the child generation or frozen source identity")
	source.Labels = map[string]string{"unrelated": "metadata"}
	require.NoError(t, env.Client().Update(t.Context(), source))
	unchanged, err := handoff.Reconcile(t.Context(), source)
	require.NoError(t, err)
	require.Equal(t, child.ResourceVersion, unchanged.ResourceVersion)
	require.Equal(t, child.Annotations, unchanged.Annotations)
	result = &ReconcileResult{State: v1beta1.DGDStateSuccessful}
	projectLPXChildStatus(source, unchanged, result, &projected)
	require.Equal(t, v1beta1.DGDStateFailed, result.State)
	require.Equal(t, &v1beta1.DynamoGraphDeploymentLPXStatus{ModelDownload: child.Status.ModelDownload}, projected.LPX)

	t.Log("An LPX template edit advances the real child generation and invalidates the old observation")
	lpx.ServingComponent(source).ComponentRole(v1beta1.ComponentRoleWorker).PodTemplate.Spec.Containers[0].Image = "lpu-runtime:next"
	require.NoError(t, env.Client().Update(t.Context(), source))
	updated, err := handoff.Reconcile(t.Context(), source)
	require.NoError(t, err)
	require.Equal(t, child.Generation+1, updated.Generation)
	require.NotEqual(t, child.Spec.InputRevision, updated.Spec.InputRevision)
	require.NoError(t, dynamo.ValidateLPXSource(updated, source))
	require.Equal(t, child.Status.ObservedGeneration, updated.Status.ObservedGeneration)
	result = &ReconcileResult{State: v1beta1.DGDStateSuccessful}
	projectLPXChildStatus(source, updated, result, &projected)
	require.Equal(t, v1beta1.DGDStatePending, result.State)
	require.Equal(t, Reason("LPXChildPending"), result.Reason)
	require.Nil(t, projected.LPX)

	t.Log("Reject malformed revision hashes in the API server")
	invalid := updated.DeepCopy()
	invalid.Spec.InputRevision = "unversioned"
	require.True(t, apierrors.IsInvalid(env.Client().Update(t.Context(), invalid)))

	t.Log("The status subresource retains child observations without rewriting source DGD status")
	updated.Status.ObservedGeneration = updated.Generation
	updated.Status.Conditions[0].ObservedGeneration = updated.Generation
	require.NoError(t, env.Client().Status().Update(t.Context(), updated))
	stored := &v1alpha1.LPXGraphDeployment{}
	require.NoError(t, env.Client().Get(t.Context(), client.ObjectKeyFromObject(updated), stored))
	require.Equal(t, updated.Status, stored.Status)
	result = &ReconcileResult{State: v1beta1.DGDStateSuccessful}
	projectLPXChildStatus(source, stored, result, &projected)
	require.Equal(t, v1beta1.DGDStateFailed, result.State)
	require.Equal(t, &v1beta1.DynamoGraphDeploymentLPXStatus{ModelDownload: child.Status.ModelDownload}, projected.LPX)
	storedSource := &v1beta1.DynamoGraphDeployment{}
	require.NoError(t, env.Client().Get(t.Context(), client.ObjectKeyFromObject(source), storedSource))
	require.Equal(t, source.Status, storedSource.Status)
}
