/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package controller

import (
	"context"
	"testing"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	schedulergrovev1alpha1 "github.com/ai-dynamo/grove/scheduler/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/event"
)

// groveWatchTestNamespace is the namespace the Grove watch fixtures live in.
const groveWatchTestNamespace = "dynamo"

// TestGroveWatchSetupMapPodGangToRequests covers how a scheduler PodGang is
// resolved back to the DGD that owns it.
func TestGroveWatchSetupMapPodGangToRequests(t *testing.T) {
	const (
		namespace = "dynamo"
		dgdName   = "serving-graph"
		// A DGD name long enough to be truncated cannot be recovered from the
		// PodGang labels, so the mapping must walk through the PodCliqueSet.
		truncatedPCSName = "serving-graph-with-a-very-long-name-a1b2"
	)

	testCases := []struct {
		name         string
		obj          client.Object
		existingPCS  *grovev1alpha1.PodCliqueSet
		wantRequests int
		wantName     string
	}{
		{
			name: "a labelled PodGang owned through its PodCliqueSet resolves to the DGD",
			obj:  podGangForWatch("serving-graph-0-anchor", truncatedPCSName),
			existingPCS: podCliqueSetOwnedByDGD(
				truncatedPCSName, namespace, dgdName,
			),
			wantRequests: 1,
			wantName:     dgdName,
		},
		{
			name:         "a PodGang without the part-of label returns no requests",
			obj:          podGangForWatch("orphan-podgang", ""),
			wantRequests: 0,
		},
		{
			name:         "a PodGang whose PodCliqueSet is not cached yet returns no requests",
			obj:          podGangForWatch("serving-graph-0-anchor", truncatedPCSName),
			wantRequests: 0,
		},
		{
			name: "a PodGang whose PodCliqueSet has no controller owner returns no requests",
			obj:  podGangForWatch("serving-graph-0-anchor", truncatedPCSName),
			existingPCS: &grovev1alpha1.PodCliqueSet{
				ObjectMeta: metav1.ObjectMeta{Name: truncatedPCSName, Namespace: namespace},
			},
			wantRequests: 0,
		},
		{
			name: "a PodGang whose PodCliqueSet is owned by another controller returns no requests",
			obj:  podGangForWatch("serving-graph-0-anchor", truncatedPCSName),
			existingPCS: &grovev1alpha1.PodCliqueSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:      truncatedPCSName,
					Namespace: namespace,
					OwnerReferences: []metav1.OwnerReference{{
						APIVersion: "apps/v1",
						Kind:       "Deployment",
						Name:       "not-a-dgd",
						Controller: ptr.To(true),
					}},
				},
			},
			wantRequests: 0,
		},
		{
			name:         "a non-PodGang object returns no requests",
			obj:          &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: namespace}},
			wantRequests: 0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Given a Grove watch setup backed by the observed Grove objects")
			builder := fake.NewClientBuilder().
				WithScheme(newDynamoGraphDeploymentControllerTestScheme(t))
			if tc.existingPCS != nil {
				builder = builder.WithObjects(tc.existingPCS)
			}
			setup := newGroveWatchSetup(builder.Build())

			t.Log("When the PodGang is mapped to the DGD reconcile requests it affects")
			requests := setup.mapPodGangToRequests(context.Background(), tc.obj)

			t.Log("Then only an unambiguously owned PodGang requeues its DGD")
			assert.Len(t, requests, tc.wantRequests)
			if tc.wantRequests == 1 {
				assert.Equal(t, tc.wantName, requests[0].Name)
				assert.Equal(t, namespace, requests[0].Namespace)
			}
		})
	}
}

// TestPodGangEventPredicates covers which PodGang updates requeue the DGD.
func TestPodGangEventPredicates(t *testing.T) {
	const podGangName = "serving-graph-0-anchor"

	base := func() *schedulergrovev1alpha1.PodGang {
		return podGangForWatch(podGangName, "serving-graph")
	}

	testCases := []struct {
		name          string
		previousScore *float64
		nextScore     *float64
		mutate        func(podGang *schedulergrovev1alpha1.PodGang)
		want          bool
	}{
		{
			name:          "a newly published placement score is significant",
			previousScore: nil,
			nextScore:     ptr.To(0.4),
			want:          true,
		},
		{
			name:          "a changed placement score is significant",
			previousScore: ptr.To(0.4),
			nextScore:     ptr.To(0.9),
			want:          true,
		},
		{
			name:          "a withdrawn placement score is significant",
			previousScore: ptr.To(0.4),
			nextScore:     nil,
			want:          true,
		},
		{
			name:          "a metadata-only change is filtered",
			previousScore: ptr.To(0.4),
			nextScore:     ptr.To(0.4),
			mutate:        func(podGang *schedulergrovev1alpha1.PodGang) { podGang.Generation = 2 },
			want:          false,
		},
		{
			name:          "a phase-only change is filtered",
			previousScore: ptr.To(0.4),
			nextScore:     ptr.To(0.4),
			mutate: func(podGang *schedulergrovev1alpha1.PodGang) {
				podGang.Status.Phase = schedulergrovev1alpha1.PodGangPhaseRunning
			},
			want: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Given a PodGang whose status is updated")
			oldPodGang := base()
			oldPodGang.Status.PlacementScore = tc.previousScore
			newPodGang := base()
			newPodGang.Status.PlacementScore = tc.nextScore
			if tc.mutate != nil {
				tc.mutate(newPodGang)
			}

			t.Log("When the update predicate decides whether to requeue the DGD")
			significant := podGangEventPredicates().Update(event.UpdateEvent{
				ObjectOld: oldPodGang,
				ObjectNew: newPodGang,
			})

			t.Log("Then only a placement score change is admitted")
			assert.Equal(t, tc.want, significant)
		})
	}

	t.Log("Given PodGang create, generic, and delete events")
	predicates := podGangEventPredicates()

	t.Log("Then a deletion requeues the DGD, because nothing follows it")
	assert.True(t, predicates.Delete(event.DeleteEvent{Object: base()}))

	t.Log("And create and generic events do not")
	assert.False(t, predicates.Create(event.CreateEvent{Object: base()}))
	assert.False(t, predicates.Generic(event.GenericEvent{Object: base()}))
}

// podGangForWatch builds a scheduler PodGang as Grove labels it for one
// PodCliqueSet. An empty pcsName models a PodGang without the part-of label.
func podGangForWatch(name, pcsName string) *schedulergrovev1alpha1.PodGang {
	podGang := &schedulergrovev1alpha1.PodGang{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: groveWatchTestNamespace},
	}
	if pcsName != "" {
		podGang.Labels = map[string]string{
			grovecommon.LabelPartOfKey:    pcsName,
			grovecommon.LabelComponentKey: grovecommon.LabelComponentNamePodGang,
		}
	}
	return podGang
}

// podCliqueSetOwnedByDGD builds the PodCliqueSet Grove created for a DGD. The
// name is deliberately not the DGD name, mirroring truncation.
func podCliqueSetOwnedByDGD(name, namespace, dgdName string) *grovev1alpha1.PodCliqueSet {
	return &grovev1alpha1.PodCliqueSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: "nvidia.com/v1beta1",
				Kind:       commonconsts.ResourceTypeDynamoGraphDeployment,
				Name:       dgdName,
				Controller: ptr.To(true),
			}},
		},
	}
}
