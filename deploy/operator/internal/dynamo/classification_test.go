/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package dynamo

import (
	"context"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	groveconstants "github.com/ai-dynamo/grove/operator/api/common/constants"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// TestAggregateReadyReason covers  DGD-level aggregation rules in
// isolation: given the set of distinct per-component classifications, what
// DGD Ready reason results.
func TestAggregateReadyReason(t *testing.T) {
	tests := []struct {
		name            string
		classifications map[ComponentReadyClassification]bool
		want            string
	}{
		{
			name:            "empty set falls back to SomeResourcesNotReady",
			classifications: map[ComponentReadyClassification]bool{},
			want:            v1beta1.DGDReadyReasonSomeResourcesNotReady,
		},
		{
			name: "single InsufficientCapacity",
			classifications: map[ComponentReadyClassification]bool{
				componentInsufficientCapacity: true,
			},
			want: v1beta1.DGDReadyReasonInsufficientCapacity,
		},
		{
			name: "single PodsNotReady",
			classifications: map[ComponentReadyClassification]bool{
				componentPodsNotReady: true,
			},
			want: v1beta1.DGDReadyReasonPodsNotReady,
		},
		{
			name: "single Updating",
			classifications: map[ComponentReadyClassification]bool{
				componentUpdating: true,
			},
			want: v1beta1.DGDReadyReasonUpdating,
		},
		{
			name: "single Unclassified falls back to SomeResourcesNotReady",
			classifications: map[ComponentReadyClassification]bool{
				componentUnclassified: true,
			},
			want: v1beta1.DGDReadyReasonSomeResourcesNotReady,
		},
		{
			name: "two distinct classifications -> MixedNotReadyReasons",
			classifications: map[ComponentReadyClassification]bool{
				componentInsufficientCapacity: true,
				componentPodsNotReady:         true,
			},
			want: v1beta1.DGDReadyReasonMixedNotReadyReasons,
		},
		{
			name: "capacity + unclassified is still two distinct -> MixedNotReadyReasons",
			classifications: map[ComponentReadyClassification]bool{
				componentInsufficientCapacity: true,
				componentUnclassified:         true,
			},
			want: v1beta1.DGDReadyReasonMixedNotReadyReasons,
		},
		{
			name: "three distinct classifications -> MixedNotReadyReasons",
			classifications: map[ComponentReadyClassification]bool{
				componentInsufficientCapacity: true,
				componentPodsNotReady:         true,
				componentUpdating:             true,
			},
			want: v1beta1.DGDReadyReasonMixedNotReadyReasons,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)
			got := aggregateReadyReason(tt.classifications)
			g.Expect(got).To(gomega.Equal(tt.want))
		})
	}
}

// TestClassificationToReadyReason verifies the 1:1 mapping from a single
// component classification to its DGD Ready reason. componentReady and
// componentUnclassified are edge cases handled explicitly rather than by
// falling through.
func TestClassificationToReadyReason(t *testing.T) {
	tests := []struct {
		name           string
		classification ComponentReadyClassification
		want           string
	}{
		{"InsufficientCapacity", componentInsufficientCapacity, v1beta1.DGDReadyReasonInsufficientCapacity},
		{"Updating", componentUpdating, v1beta1.DGDReadyReasonUpdating},
		{"PodsNotReady", componentPodsNotReady, v1beta1.DGDReadyReasonPodsNotReady},
		{"Ready maps to AllResourcesReady", componentReady, v1beta1.DGDReadyReasonAllResourcesReady},
		{"Unclassified falls back to SomeResourcesNotReady", componentUnclassified, v1beta1.DGDReadyReasonSomeResourcesNotReady},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)
			got := classificationToReadyReason(tt.classification)
			g.Expect(got).To(gomega.Equal(tt.want))
		})
	}
}

// The Check*Ready classification tests below exercise the full Grove-status
// reading path (client.Get + field/condition inspection) and assert the
// ComponentReadyClassification returned as the 4th value. They complement the
// existing TestCheckPodCliqueReady / TestCheckPCSGReady tables (which assert
// ready/reason/serviceStatus) by focusing on capacity-before-readiness
// classification ordering.

func TestCheckPodCliqueReadyClassification(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name               string
		podClique          *grovev1alpha1.PodClique
		wantReady          bool
		wantClassification ComponentReadyClassification
		wantReasonContains string
	}{
		{
			name: "fully ready",
			podClique: newPodClique(grovev1alpha1.PodCliqueStatus{
				Replicas: 3, ReadyReplicas: 3, UpdatedReplicas: 3,
				ScheduledReplicas: 3, ObservedGeneration: ptr.To(int64(1)),
			}),
			wantReady:          true,
			wantClassification: componentReady,
		},
		{
			name: "scheduling condition InsufficientScheduledPods -> InsufficientCapacity",
			podClique: newPodClique(grovev1alpha1.PodCliqueStatus{
				Replicas: 3, ReadyReplicas: 0, UpdatedReplicas: 3,
				ScheduledReplicas: 1, ObservedGeneration: ptr.To(int64(1)),
				Conditions: []metav1.Condition{{
					Type:               groveconstants.ConditionTypePodCliqueScheduled,
					Status:             metav1.ConditionFalse,
					Reason:             groveconstants.ConditionReasonInsufficientScheduledPods,
					LastTransitionTime: metav1.Now(),
				}},
			}),
			wantReady:          false,
			wantClassification: componentInsufficientCapacity,
			wantReasonContains: "scheduling condition",
		},
		{
			name: "schedule-gated replicas -> InsufficientCapacity",
			podClique: newPodClique(grovev1alpha1.PodCliqueStatus{
				Replicas: 3, ReadyReplicas: 0, UpdatedReplicas: 3,
				ScheduledReplicas: 3, ScheduleGatedReplicas: 2,
				ObservedGeneration: ptr.To(int64(1)),
			}),
			wantReady:          false,
			wantClassification: componentInsufficientCapacity,
			wantReasonContains: "schedule-gated",
		},
		{
			name: "scheduled condition false with insufficient-pods reason -> InsufficientCapacity",
			podClique: newPodClique(grovev1alpha1.PodCliqueStatus{
				Replicas: 3, ReadyReplicas: 0, UpdatedReplicas: 3,
				ScheduledReplicas: 3, ObservedGeneration: ptr.To(int64(1)),
				Conditions: []metav1.Condition{{
					Type:               groveconstants.ConditionTypePodCliqueScheduled,
					Status:             metav1.ConditionFalse,
					Reason:             groveconstants.ConditionReasonInsufficientScheduledPods,
					LastTransitionTime: metav1.Now(),
				}},
			}),
			wantReady:          false,
			wantClassification: componentInsufficientCapacity,
			wantReasonContains: "scheduling condition",
		},
		{
			name: "scheduled but not updated -> Updating",
			podClique: newPodClique(grovev1alpha1.PodCliqueStatus{
				Replicas: 3, ReadyReplicas: 3, UpdatedReplicas: 2,
				ScheduledReplicas: 3, ObservedGeneration: ptr.To(int64(1)),
			}),
			wantReady:          false,
			wantClassification: componentUpdating,
			wantReasonContains: "updated=2",
		},
		{
			name: "rolling update (replicas != desired) -> Updating",
			podClique: newPodClique(grovev1alpha1.PodCliqueStatus{
				Replicas: 4, ReadyReplicas: 3, UpdatedReplicas: 3,
				ScheduledReplicas: 3, ObservedGeneration: ptr.To(int64(1)),
			}),
			wantReady:          false,
			wantClassification: componentUpdating,
			wantReasonContains: "rolling update",
		},
		{
			name: "scheduled and updated but not ready -> PodsNotReady",
			podClique: newPodClique(grovev1alpha1.PodCliqueStatus{
				Replicas: 3, ReadyReplicas: 1, UpdatedReplicas: 3,
				ScheduledReplicas: 3, ObservedGeneration: ptr.To(int64(1)),
			}),
			wantReady:          false,
			wantClassification: componentPodsNotReady,
			wantReasonContains: "ready=1/3",
		},
		{
			name:               "not found -> Unclassified",
			podClique:          nil,
			wantReady:          false,
			wantClassification: componentUnclassified,
			wantReasonContains: "resource not found",
		},
		{
			name: "nil observedGeneration -> Unclassified",
			podClique: newPodClique(grovev1alpha1.PodCliqueStatus{
				Replicas: 3, ReadyReplicas: 3, UpdatedReplicas: 3,
				ScheduledReplicas: 3, ObservedGeneration: nil,
			}),
			wantReady:          false,
			wantClassification: componentUnclassified,
			wantReasonContains: "observedGeneration is nil",
		},
		{
			name: "capacity checked before readiness: schedule-gated AND unready -> InsufficientCapacity",
			podClique: newPodClique(grovev1alpha1.PodCliqueStatus{
				Replicas: 3, ReadyReplicas: 0, UpdatedReplicas: 3,
				ScheduledReplicas: 1, ScheduleGatedReplicas: 2,
				ObservedGeneration: ptr.To(int64(1)),
			}),
			wantReady:          false,
			wantClassification: componentInsufficientCapacity,
			wantReasonContains: "schedule-gated",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			var objs []client.Object
			if tt.podClique != nil {
				objs = append(objs, tt.podClique)
			}
			c := newFakeGroveClient(g, objs...)
			ready, reason, _, classification := CheckPodCliqueReady(ctx, c, testPodCliqueName, "default", log.FromContext(ctx))

			g.Expect(ready).To(gomega.Equal(tt.wantReady))
			g.Expect(classification).To(gomega.Equal(tt.wantClassification))
			if tt.wantReasonContains != "" {
				g.Expect(reason).To(gomega.ContainSubstring(tt.wantReasonContains))
			}
		})
	}
}

func TestCheckPCSGReadyClassification(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name               string
		pcsg               *grovev1alpha1.PodCliqueScalingGroup
		wantReady          bool
		wantClassification ComponentReadyClassification
		wantReasonContains string
	}{
		{
			name: "fully ready",
			pcsg: newPCSG(grovev1alpha1.PodCliqueScalingGroupStatus{
				Replicas: 2, AvailableReplicas: 2, UpdatedReplicas: 2,
				ScheduledReplicas: 2, ObservedGeneration: ptr.To(int64(1)),
			}),
			wantReady:          true,
			wantClassification: componentReady,
		},
		{
			name: "available below desired without scheduling condition -> PodsNotReady",
			pcsg: newPCSG(grovev1alpha1.PodCliqueScalingGroupStatus{
				Replicas: 2, AvailableReplicas: 0, UpdatedReplicas: 2,
				ScheduledReplicas: 1, ObservedGeneration: ptr.To(int64(1)),
			}),
			wantReady:          false,
			wantClassification: componentPodsNotReady,
			wantReasonContains: "available=0/2",
		},
		{
			name: "MinAvailableBreached with PCSG-scheduling reason -> InsufficientCapacity",
			pcsg: newPCSG(grovev1alpha1.PodCliqueScalingGroupStatus{
				Replicas: 2, AvailableReplicas: 0, UpdatedReplicas: 2,
				ScheduledReplicas: 2, ObservedGeneration: ptr.To(int64(1)),
				Conditions: []metav1.Condition{{
					Type:               groveconstants.ConditionTypeMinAvailableBreached,
					Status:             metav1.ConditionTrue,
					Reason:             groveconstants.ConditionReasonInsufficientScheduledPCSGReplicas,
					LastTransitionTime: metav1.Now(),
				}},
			}),
			wantReady:          false,
			wantClassification: componentInsufficientCapacity,
			wantReasonContains: "min-available breached",
		},
		{
			name: "scheduled but not updated -> Updating",
			pcsg: newPCSG(grovev1alpha1.PodCliqueScalingGroupStatus{
				Replicas: 2, AvailableReplicas: 2, UpdatedReplicas: 1,
				ScheduledReplicas: 2, ObservedGeneration: ptr.To(int64(1)),
			}),
			wantReady:          false,
			wantClassification: componentUpdating,
			wantReasonContains: "updated=1",
		},
		{
			name: "scheduled and updated but not available -> PodsNotReady",
			pcsg: newPCSG(grovev1alpha1.PodCliqueScalingGroupStatus{
				Replicas: 2, AvailableReplicas: 1, UpdatedReplicas: 2,
				ScheduledReplicas: 2, ObservedGeneration: ptr.To(int64(1)),
			}),
			wantReady:          false,
			wantClassification: componentPodsNotReady,
			wantReasonContains: "available=1/2",
		},
		{
			name:               "not found -> Unclassified",
			pcsg:               nil,
			wantReady:          false,
			wantClassification: componentUnclassified,
			wantReasonContains: "resource not found",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			var objs []client.Object
			if tt.pcsg != nil {
				objs = append(objs, tt.pcsg)
			}
			c := newFakeGroveClient(g, objs...)
			ready, reason, _, classification := CheckPCSGReady(ctx, c, testPCSGName, "default", log.FromContext(ctx))

			g.Expect(ready).To(gomega.Equal(tt.wantReady))
			g.Expect(classification).To(gomega.Equal(tt.wantClassification))
			if tt.wantReasonContains != "" {
				g.Expect(reason).To(gomega.ContainSubstring(tt.wantReasonContains))
			}
		})
	}
}

// --- test helpers ---

// Fixed resource identity used by the classification tests. The helpers below
// hardcode these so the name passed to CheckPodCliqueReady / CheckPCSGReady
// stays in sync with the object created in the fake client, and so the spec
// replica count (which every case shares) is defined in one place. Only the
// status varies per test case.
const (
	testPodCliqueName     = "pc"
	testPodCliqueReplicas = 3
	testPCSGName          = "pcsg"
	testPCSGReplicas      = 2
)

func newPodClique(status grovev1alpha1.PodCliqueStatus) *grovev1alpha1.PodClique {
	return &grovev1alpha1.PodClique{
		ObjectMeta: metav1.ObjectMeta{Name: testPodCliqueName, Namespace: "default", Generation: 1},
		Spec:       grovev1alpha1.PodCliqueSpec{Replicas: testPodCliqueReplicas},
		Status:     status,
	}
}

func newPCSG(status grovev1alpha1.PodCliqueScalingGroupStatus) *grovev1alpha1.PodCliqueScalingGroup {
	return &grovev1alpha1.PodCliqueScalingGroup{
		ObjectMeta: metav1.ObjectMeta{Name: testPCSGName, Namespace: "default", Generation: 1},
		Spec:       grovev1alpha1.PodCliqueScalingGroupSpec{Replicas: testPCSGReplicas},
		Status:     status,
	}
}

func newFakeGroveClient(g *gomega.WithT, objects ...client.Object) client.Client {
	s := scheme.Scheme
	g.Expect(grovev1alpha1.AddToScheme(s)).To(gomega.Succeed())
	return fake.NewClientBuilder().
		WithScheme(s).
		WithObjects(objects...).
		WithStatusSubresource(objects...).
		Build()
}
