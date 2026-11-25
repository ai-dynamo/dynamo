//go:build !clustertest

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorenv"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// TestLPXCyborgPlacementGateOnAPIServer checks the pin against the Pod update
// validation of Kubernetes, which fake clients do not enforce.
func TestLPXCyborgPlacementGateOnAPIServer(t *testing.T) {
	ctx := t.Context()

	t.Log("Start an API server that calls no admission webhook: the placement gate comes from the Pod template")
	env := operatorenv.New(operatorenv.Options{
		SetupWebhooks: func(ctrl.Manager, operatorenv.WebhookSetupOptions) error { return nil },
	}).RunT(t)
	r := &graphReconciler{Client: env.Client()}
	clique := &grovev1alpha1.PodClique{
		Name: "pcs-0-lpx-0-cond", Namespace: env.Namespace(), UID: "clique-uid",
		Labels:      map[string]string{cyborgPlacementLabel: cyborgPlacementPerFabricPod},
		Annotations: map[string]string{lpxv1alpha1.PodRoleAnnotation: lpxv1alpha1.PodRoleCyborgWorker, lpx.DeploymentNameAnnotation: "llama"},
	}
	groveGate, placementGate := corev1.PodSchedulingGate{Name: grovePodGangGate}, corev1.PodSchedulingGate{Name: cyborgPlacementGate}
	selector := map[string]string{fabricPodLabel: "a"}

	t.Log("Grove creates a Cyborg Pod with the placement gate of its template and then its own gate")
	held := newTestAPIServerCyborgPod(clique, 0)
	require.NoError(t, r.Create(ctx, held))
	require.Equal(t, []corev1.PodSchedulingGate{placementGate, groveGate}, held.Spec.SchedulingGates)

	t.Log("Grove removes its gate with a list of the gates that stay, and the placement gate still holds the Pod")
	observed := held.DeepCopy()
	held.Spec.SchedulingGates = []corev1.PodSchedulingGate{placementGate}
	require.NoError(t, r.Patch(ctx, held, client.MergeFrom(observed)))

	t.Log("The pin sets the node selector, records the rank and the replica index, and removes the last gate in one write")
	require.NoError(t, r.pinCyborgPod(ctx, held, "a", 1, 3))
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(held), held))
	require.Equal(t, selector, held.Spec.NodeSelector)
	require.Equal(t, "a", held.Labels[cyborgFabricPodLabel])
	require.Equal(t, []string{"1/3"}, testCyborgRanks([]*corev1.Pod{held}), "the containers of the Pod read these labels when they start")
	require.Empty(t, held.Spec.SchedulingGates)

	t.Log("A pin from an old observation of the Pod is refused and changes nothing")
	stale := newTestAPIServerCyborgPod(clique, 2)
	require.NoError(t, r.Create(ctx, stale))
	current := stale.DeepCopy()
	current.Annotations = map[string]string{"changed": "after-the-observation"}
	require.NoError(t, r.Patch(ctx, current, client.MergeFrom(stale)))
	require.NoError(t, r.pinCyborgPod(ctx, stale, "a", 0, 0), "a conflict is not an error: the next Pod version starts a new pass")
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(current), current))
	require.Empty(t, current.Spec.NodeSelector)
	require.Equal(t, []string{"/"}, testCyborgRanks([]*corev1.Pod{current}))
	require.Equal(t, []corev1.PodSchedulingGate{placementGate, groveGate}, current.Spec.SchedulingGates)

	t.Log("Kubernetes accepts no selector on a Pod without a gate, so the controller does not pin such a Pod")
	floating := newTestAPIServerCyborgPod(clique, 4)
	floating.Spec.SchedulingGates = []corev1.PodSchedulingGate{groveGate}
	require.NoError(t, r.Create(ctx, floating))
	observed = floating.DeepCopy()
	floating.Spec.SchedulingGates = nil
	require.NoError(t, r.Patch(ctx, floating, client.MergeFrom(observed)))
	err := r.pinCyborgPod(ctx, floating, "a", 0, 0)
	require.True(t, apierrors.IsInvalid(err), "%v", err)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(floating), floating))
	require.Empty(t, floating.Spec.NodeSelector)
	require.Empty(t, floating.Labels[cyborgFabricPodLabel])
}

// newTestAPIServerCyborgPod returns the Pod that Grove creates for clique at index,
// in a form that an API server accepts: without a preset UID or creation time, and
// with a container.
func newTestAPIServerCyborgPod(clique *grovev1alpha1.PodClique, index int) *corev1.Pod {
	pod := newTestCyborgPod(clique, index)
	pod.UID, pod.CreationTimestamp = "", metav1.Time{}
	pod.Spec.Containers = []corev1.Container{{Name: "main", Image: "cyborg"}}
	return pod
}
