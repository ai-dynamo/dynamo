// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package power

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/event"
)

func TestPodCliqueSetPredicateAdmitsStatusOnlyBarrierChanges(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*grovev1alpha1.PodCliqueSet)
		want   bool
	}{
		{name: "observedGeneration acknowledged", change: func(pcs *grovev1alpha1.PodCliqueSet) { pcs.Status.ObservedGeneration = new(int64(3)) }, want: true},
		{name: "update started", change: func(pcs *grovev1alpha1.PodCliqueSet) {
			pcs.Status.UpdateProgress = &grovev1alpha1.PodCliqueSetUpdateProgress{UpdateStartedAt: metav1.Unix(1, 0)}
		}, want: true},
		{name: "update ended", change: func(pcs *grovev1alpha1.PodCliqueSet) {
			pcs.Status.UpdateProgress = &grovev1alpha1.PodCliqueSetUpdateProgress{UpdateStartedAt: metav1.Unix(1, 0), UpdateEndedAt: new(metav1.Unix(2, 0))}
		}, want: true},
		{name: "graph label", change: func(pcs *grovev1alpha1.PodCliqueSet) {
			pcs.Labels = map[string]string{consts.KubeLabelDynamoGraphDeploymentName: "other"}
		}, want: true},
		{name: "unrelated status", change: func(pcs *grovev1alpha1.PodCliqueSet) { pcs.Status.Replicas = 2 }},
		{name: "unrelated metadata", change: func(pcs *grovev1alpha1.PodCliqueSet) { pcs.Labels = map[string]string{"unrelated": "change"} }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Change one PodCliqueSet field while its generation stays fixed, as Grove's status writes do")
			before := &grovev1alpha1.PodCliqueSet{Name: "pcs", Namespace: "default", Generation: 3, Status: grovev1alpha1.PodCliqueSetStatus{ObservedGeneration: new(int64(2))}}
			after := before.DeepCopy()
			tc.change(after)
			require.Equal(t, before.Generation, after.Generation)

			t.Log("Only barrier inputs wake the controller")
			require.Equal(t, tc.want, podCliqueSetPredicate().Update(event.UpdateEvent{ObjectOld: before, ObjectNew: after}))
		})
	}
}

func TestPodPredicateFollowsLabelsNotOwners(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*corev1.Pod)
		want   bool
	}{
		{name: "moved to another component", change: func(pod *corev1.Pod) { pod.Labels[consts.KubeLabelDynamoComponent] = "decode" }, want: true},
		{name: "moved to another graph", change: func(pod *corev1.Pod) { pod.Labels[consts.KubeLabelDynamoGraphDeploymentName] = "other" }, want: true},
		{name: "component type changed", change: func(pod *corev1.Pod) { pod.Labels[consts.KubeLabelDynamoComponentType] = "frontend" }, want: true},
		{name: "cap annotation changed", change: func(pod *corev1.Pod) { pod.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350" }, want: true},
		{name: "terminated", change: func(pod *corev1.Pod) { pod.Status.Phase = corev1.PodFailed }, want: true},
		{name: "adopted by another owner", change: func(pod *corev1.Pod) {
			pod.OwnerReferences = []metav1.OwnerReference{{APIVersion: "grove.io/v1beta1", Kind: "PodClique", Name: "clique", UID: "clique-uid", Controller: new(true)}}
		}},
		{name: "unrelated label", change: func(pod *corev1.Pod) { pod.Labels["unrelated"] = "change" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Change one field of a labeled worker Pod")
			before := &corev1.Pod{Name: "worker", Namespace: "default", Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "graph", consts.KubeLabelDynamoComponent: "worker", consts.KubeLabelDynamoComponentType: "worker",
			}, Annotations: map[string]string{consts.KubeAnnotationGPUPowerLimit: "300"}}
			after := before.DeepCopy()
			tc.change(after)

			t.Log("Label membership, the cap and terminality wake the controller; owner references do not")
			require.Equal(t, tc.want, podPredicate().Update(event.UpdateEvent{ObjectOld: before, ObjectNew: after}))
		})
	}
}
