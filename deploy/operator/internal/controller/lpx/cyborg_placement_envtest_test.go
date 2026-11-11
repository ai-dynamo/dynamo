//go:build !clustertest

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"testing"

	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorenv"
	webhooksetup "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/setup"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// TestLPXCyborgPlacementGateOnAPIServer checks the two halves of the placement gate
// against Kubernetes admission and Pod update validation, which fake clients do not
// enforce: Pod admission adds the gate, and the pin removes it.
func TestLPXCyborgPlacementGateOnAPIServer(t *testing.T) {
	ctx := t.Context()

	t.Log("Start an API server with the chart's admission registrations and the production webhook handlers, with LPX and Grove enabled")
	env := operatorenv.New(operatorenv.Options{
		Admission:     operatorenv.AdmissionWebhooks{Mutating: true},
		RuntimeConfig: &commoncontroller.RuntimeConfig{Gate: features.Gates{LPX: true, Grove: true}},
		SetupWebhooks: func(mgr ctrl.Manager, opts operatorenv.WebhookSetupOptions) error {
			return webhooksetup.Setup(mgr, webhooksetup.Options{
				Config: opts.OperatorConfig, RuntimeConfig: opts.RuntimeConfig,
				OperatorVersion: opts.OperatorVersion, OperatorPrincipal: opts.OperatorPrincipal,
			})
		},
	}).RunT(t)
	r := &graphReconciler{Client: env.Client()}
	clique := &grovev1alpha1.PodClique{
		Name: "pcs-0-lpx-0-cond", Namespace: env.Namespace(), UID: "clique-uid",
		Labels:      map[string]string{lpx.CyborgPlacementLabel: lpx.CyborgPlacementPerFabricPod},
		Annotations: map[string]string{lpxv1alpha1.PodRoleAnnotation: lpxv1alpha1.PodRoleCyborgWorker, lpx.DeploymentNameAnnotation: "llama"},
	}
	groveGate, placementGate := corev1.PodSchedulingGate{Name: grovePodGangGate}, corev1.PodSchedulingGate{Name: lpx.CyborgPlacementGate}
	quotaGate := corev1.PodSchedulingGate{Name: "example.com/quota"}
	selector := map[string]string{fabricPodLabel: "a"}

	t.Log("Admission adds the placement gate to a Cyborg Pod with the placement label, and keeps Grove's gate")
	held := newTestAPIServerCyborgPod(clique, 0)
	require.NoError(t, r.Create(ctx, held))
	require.Equal(t, []corev1.PodSchedulingGate{groveGate, placementGate}, held.Spec.SchedulingGates)

	t.Log("Grove removes its gate with a list of the gates that stay, and the placement gate still holds the Pod")
	observed := held.DeepCopy()
	held.Spec.SchedulingGates = []corev1.PodSchedulingGate{placementGate}
	require.NoError(t, r.Patch(ctx, held, client.MergeFrom(observed)))

	t.Log("The pin sets the node selector and removes the last gate in one write")
	placed, err := r.placeCyborgPod(ctx, held, "a", selector)
	require.NoError(t, err)
	require.True(t, placed)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(held), held))
	require.Equal(t, selector, held.Spec.NodeSelector)
	require.Equal(t, "a", held.Labels[cyborgFabricPodLabel])
	require.Empty(t, held.Spec.SchedulingGates)

	t.Log("The pin keeps the gate of another controller")
	shared := newTestAPIServerCyborgPod(clique, 1)
	shared.Spec.SchedulingGates = []corev1.PodSchedulingGate{quotaGate}
	require.NoError(t, r.Create(ctx, shared))
	require.Equal(t, []corev1.PodSchedulingGate{quotaGate, placementGate}, shared.Spec.SchedulingGates)
	placed, err = r.placeCyborgPod(ctx, shared, "a", selector)
	require.NoError(t, err)
	require.True(t, placed)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(shared), shared))
	require.Equal(t, selector, shared.Spec.NodeSelector)
	require.Equal(t, []corev1.PodSchedulingGate{quotaGate}, shared.Spec.SchedulingGates)

	t.Log("A pin from an old observation of the Pod is refused and changes nothing")
	stale := newTestAPIServerCyborgPod(clique, 2)
	require.NoError(t, r.Create(ctx, stale))
	current := stale.DeepCopy()
	current.Annotations = map[string]string{"changed": "after-the-observation"}
	require.NoError(t, r.Patch(ctx, current, client.MergeFrom(stale)))
	placed, err = r.placeCyborgPod(ctx, stale, "a", selector)
	require.NoError(t, err)
	require.False(t, placed)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(current), current))
	require.Empty(t, current.Spec.NodeSelector)
	require.Equal(t, []corev1.PodSchedulingGate{groveGate, placementGate}, current.Spec.SchedulingGates)

	t.Log("A Pod without the placement label gets no placement gate; the pin then succeeds only while Grove's gate holds the Pod")
	delete(clique.Labels, lpx.CyborgPlacementLabel)
	ungated := newTestAPIServerCyborgPod(clique, 3)
	require.NoError(t, r.Create(ctx, ungated))
	require.Equal(t, []corev1.PodSchedulingGate{groveGate}, ungated.Spec.SchedulingGates)
	placed, err = r.placeCyborgPod(ctx, ungated, "a", selector)
	require.NoError(t, err)
	require.True(t, placed)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(ungated), ungated))
	require.Equal(t, selector, ungated.Spec.NodeSelector)
	require.Equal(t, []corev1.PodSchedulingGate{groveGate}, ungated.Spec.SchedulingGates)

	t.Log("After Grove releases such a Pod, Kubernetes accepts no selector, so the controller does not pin a Pod without a gate")
	floating := newTestAPIServerCyborgPod(clique, 4)
	require.NoError(t, r.Create(ctx, floating))
	observed = floating.DeepCopy()
	floating.Spec.SchedulingGates = nil
	require.NoError(t, r.Patch(ctx, floating, client.MergeFrom(observed)))
	placed, err = r.placeCyborgPod(ctx, floating, "a", selector)
	require.True(t, apierrors.IsInvalid(err), "%v", err)
	require.False(t, placed)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(floating), floating))
	require.Empty(t, floating.Spec.NodeSelector)
	require.Empty(t, floating.Labels[cyborgFabricPodLabel])
}

// newTestAPIServerCyborgPod returns the Pod that Grove creates for clique at index,
// in a form that an API server accepts: without a preset UID or creation time, and
// with a container. Pod admission, not this helper, adds the placement gate.
func newTestAPIServerCyborgPod(clique *grovev1alpha1.PodClique, index int) *corev1.Pod {
	pod := newTestCyborgPod(clique, index)
	pod.UID, pod.CreationTimestamp = "", metav1.Time{}
	pod.Spec.SchedulingGates = []corev1.PodSchedulingGate{{Name: grovePodGangGate}}
	pod.Spec.Containers = []corev1.Container{{Name: "main", Image: "cyborg"}}
	return pod
}
