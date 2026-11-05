// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package power

import (
	"testing"

	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
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
