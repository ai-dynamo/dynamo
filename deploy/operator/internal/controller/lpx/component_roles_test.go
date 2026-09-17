// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"testing"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestLPXRoleDependenciesIncludeDraftAndConductor(t *testing.T) {
	t.Log("Give both components consumed Agent claims and the conductor its own dependency")
	source := newLPXSpecDecodeTestSource()
	for _, component := range lpx.Components(source) {
		pod := &component.ComponentRole(v1beta1.ComponentRoleLPXAgent).PodTemplate.Spec
		pod.ResourceClaims = []corev1.PodResourceClaim{{Name: "gpu", ResourceClaimTemplateName: ptr.To(component.ComponentName + "-gpu")}}
		pod.Containers[0].Resources.Claims = []corev1.ResourceClaim{{Name: "gpu"}}
	}
	conductor := lpx.ServingComponent(source).ComponentRole(v1beta1.ComponentRoleLPXConductor)
	conductor.PodTemplate.Spec.ResourceClaims = []corev1.PodResourceClaim{{Name: "gpu", ResourceClaimTemplateName: ptr.To("conductor-gpu")}}
	conductor.PodTemplate.Spec.Containers[0].Resources.Claims = []corev1.ResourceClaim{{Name: "gpu"}}
	before := source.DeepCopy()

	t.Log("Index each authored role's consumed claim without modifying the source")
	require.ElementsMatch(t, []string{"draft-gpu", "lpx-gpu", "conductor-gpu"}, lpxDRAClaimReferences(true)(source))
	require.Equal(t, before, source)
}

func TestSpecDecodeStatusCountsCompleteDraftInstances(t *testing.T) {
	t.Log("Materialize two draft instances and one target in their single shared scaling group")
	deployment, source, registry := newLPXSpecDecodeTestDGD(t)
	prepare, selected := newPreparedLPXTestReconciler(t, registry, t.Context(), deployment, source)
	objects := lpxMaterializedObjects(t, prepare, deployment, source, selected)
	for _, object := range objects {
		switch live := object.(type) {
		case *grovev1alpha1.PodCliqueScalingGroup:
			live.Status.Replicas, live.Status.UpdatedReplicas = 1, 1
			live.Status.AvailableReplicas, live.Status.ScheduledReplicas = 1, 1
		case *grovev1alpha1.PodClique:
			live.Status.Replicas, live.Status.UpdatedReplicas = live.Spec.Replicas, live.Spec.Replicas
			live.Status.ReadyReplicas, live.Status.ScheduledReplicas = live.Spec.Replicas, live.Spec.Replicas
		}
	}
	pcs := findLPXTestPodCliqueSet(t, objects)

	t.Log("Observe each draft condition against an independent copy of the ready baseline")
	for _, scenario := range []string{"ready", "partial", "missing", "foreign", "old revision", "unobserved"} {
		t.Run(scenario, func(t *testing.T) {
			t.Log("Seed isolated Grove observations and fetch the first draft")
			kubeClient := fake.NewClientBuilder().WithScheme(prepare.Scheme()).Build()
			createLPXTestObjects(t, t.Context(), kubeClient, objects...)
			firstDraft := findLPXTestClique(t, objects, selected.plan.Agents[0].CliqueName).DeepCopy()
			require.NoError(t, kubeClient.Get(t.Context(), client.ObjectKeyFromObject(firstDraft), firstDraft))
			require.Equal(t, "draft", firstDraft.Labels[lpx.StageLabel])

			t.Log("Make only the first draft incomplete while retaining a fully ready second draft and target")
			switch scenario {
			case "partial":
				firstDraft.Status.ReadyReplicas--
			case "foreign":
				firstDraft.OwnerReferences = []metav1.OwnerReference{{UID: "foreign", Controller: ptr.To(true)}}
			case "old revision":
				firstDraft.Status.CurrentPodCliqueSetGenerationHash = ptr.To("old")
			case "unobserved":
				firstDraft.Status.ObservedGeneration = ptr.To(firstDraft.Generation - 1)
			}
			if scenario == "missing" {
				require.NoError(t, kubeClient.Delete(t.Context(), firstDraft))
			} else {
				require.NoError(t, kubeClient.Update(t.Context(), firstDraft))
			}

			t.Log("Project logical draft counts and make both authored components await the complete pair")
			readiness, err := dynamo.EvaluateLPXGroveReadiness(t.Context(), kubeClient, source, deployment, pcs)
			require.NoError(t, err)
			require.Len(t, readiness.ComponentStatuses, 2)
			draft, target := readiness.ComponentStatuses["draft"], readiness.ComponentStatuses["lpx"]
			require.Equal(t, v1beta1.ComponentKindPodClique, draft.ComponentKind)
			require.Len(t, draft.ComponentNames, 2)
			require.Equal(t, int32(1), target.Replicas)
			if scenario == "ready" {
				require.True(t, readiness.Ready)
				require.Equal(t, int32(2), draft.Replicas)
				require.Equal(t, int32(2), draft.UpdatedReplicas)
				require.Equal(t, ptr.To(int32(2)), draft.ReadyReplicas)
				require.Equal(t, ptr.To(int32(2)), draft.ScheduledReplicas)
				require.Equal(t, ptr.To(int32(1)), target.AvailableReplicas)
			} else {
				require.False(t, readiness.Ready)
				require.Equal(t, ptr.To(int32(1)), draft.ReadyReplicas)
				require.Equal(t, ptr.To(int32(0)), target.AvailableReplicas)
			}
			require.Equal(t, readiness.Ready, draft.Ready)
			require.Equal(t, readiness.Ready, target.Ready)
		})
	}
}
