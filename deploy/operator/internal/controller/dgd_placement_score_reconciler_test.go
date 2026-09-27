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

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	schedulergrovev1alpha1 "github.com/ai-dynamo/grove/scheduler/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

// TestDGDPlacementScoreReconcilerObserve covers how one placement observation
// resolves status.placement. The PodCliqueSet name equals the DGD name for the
// short names used here.
func TestDGDPlacementScoreReconcilerObserve(t *testing.T) {
	const (
		namespace     = "dynamo"
		dgdName       = "placement-dgd"
		otherDGD      = "other-dgd"
		generationOne = "gen-1"
		generationTwo = "gen-2"
	)

	testCases := []struct {
		name      string
		objects   []client.Object
		wantScore *float64
		wantState nvidiacomv1beta1.PlacementScoreState
	}{
		{
			name:      "no PodCliqueSet observed yet is indeterminate",
			objects:   nil,
			wantState: nvidiacomv1beta1.PlacementScoreStateUnknown,
		},
		{
			name:      "a PodCliqueSet with no PodGangs yet is indeterminate",
			objects:   []client.Object{placementPCS(dgdName, namespace, ptr.To(generationOne))},
			wantState: nvidiacomv1beta1.PlacementScoreStateUnknown,
		},
		{
			name: "a scored PodGang of the current generation is reported",
			objects: []client.Object{
				placementPCS(dgdName, namespace, ptr.To(generationOne)),
				placementPG(dgdName+"-0-anchor", namespace, dgdName, generationOne, ptr.To(0.42)),
			},
			wantScore: ptr.To(0.42),
			wantState: nvidiacomv1beta1.PlacementScoreStateReported,
		},
		{
			name: "the worst placement wins across the current generation",
			objects: []client.Object{
				placementPCS(dgdName, namespace, ptr.To(generationOne)),
				placementPG(dgdName+"-0-anchor", namespace, dgdName, generationOne, ptr.To(0.9)),
				placementPG(dgdName+"-0-scaleout", namespace, dgdName, generationOne, ptr.To(0.2)),
			},
			wantScore: ptr.To(0.2),
			wantState: nvidiacomv1beta1.PlacementScoreStateReported,
		},
		{
			name: "an unscored PodGang of the current generation makes the observation partial",
			objects: []client.Object{
				placementPCS(dgdName, namespace, ptr.To(generationOne)),
				placementPG(dgdName+"-0-anchor", namespace, dgdName, generationOne, ptr.To(0.75)),
				placementPG(dgdName+"-0-scaleout", namespace, dgdName, generationOne, nil),
			},
			wantScore: ptr.To(0.75),
			wantState: nvidiacomv1beta1.PlacementScoreStatePartial,
		},
		{
			name: "a PodGang from a previous generation is excluded",
			objects: []client.Object{
				placementPCS(dgdName, namespace, ptr.To(generationTwo)),
				placementPG(dgdName+"-0-anchor", namespace, dgdName, generationOne, ptr.To(0.1)),
			},
			wantState: nvidiacomv1beta1.PlacementScoreStateUnknown,
		},
		{
			// The strongest form of the generation guard: a stale PodGang that is
			// worse than the current one must not drag the reported score down.
			name: "a stale scored PodGang does not lower the current generation's score",
			objects: []client.Object{
				placementPCS(dgdName, namespace, ptr.To(generationTwo)),
				placementPG(dgdName+"-0-anchor", namespace, dgdName, generationTwo, ptr.To(0.5)),
				placementPG(dgdName+"-0-scaleout", namespace, dgdName, generationOne, ptr.To(0.1)),
			},
			wantScore: ptr.To(0.5),
			wantState: nvidiacomv1beta1.PlacementScoreStateReported,
		},
		{
			name: "a PodGang of a different PodCliqueSet is excluded",
			objects: []client.Object{
				placementPCS(dgdName, namespace, ptr.To(generationOne)),
				placementPG(otherDGD+"-0-anchor", namespace, otherDGD, generationOne, ptr.To(0.3)),
			},
			wantState: nvidiacomv1beta1.PlacementScoreStateUnknown,
		},
		{
			name: "a PodGang is scored before the PodCliqueSet publishes a generation",
			objects: []client.Object{
				placementPCS(dgdName, namespace, nil),
				placementPG(dgdName+"-0-anchor", namespace, dgdName, generationOne, ptr.To(0.65)),
			},
			wantScore: ptr.To(0.65),
			wantState: nvidiacomv1beta1.PlacementScoreStateReported,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Given a DGD on the Grove pathway with the observed Grove objects")
			dgd := placementDGD(dgdName, namespace)
			reader := fake.NewClientBuilder().
				WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
				WithObjects(tc.objects...).
				Build()
			reconciler := newDGDPlacementScoreReconciler(reader)

			t.Log("When the placement observation is projected into the program status")
			result := &workloadProgramResult{}
			reconciler.Reconcile(context.Background(), dgd, result)

			t.Log("Then status.placement reports only the current generation's observation")
			require.NotNil(t, result.Status.Placement, "placement status must always be set")
			require.Equal(t, tc.wantState, result.Status.Placement.State, "unexpected placement state")

			// A score is copied through only when the state claims one is available.
			if tc.wantScore == nil {
				require.Nil(t, result.Status.Placement.Score, "score must be cleared")
				return
			}
			require.NotNil(t, result.Status.Placement.Score, "score must be reported")
			require.InDelta(t, *tc.wantScore, *result.Status.Placement.Score, 0.0, "unexpected score")
		})
	}
}

// TestDGDPlacementScoreReconcilerClearsStaleScore covers the rule that a score
// from an earlier reconcile cannot survive an observation that yields nothing.
func TestDGDPlacementScoreReconcilerClearsStaleScore(t *testing.T) {
	const (
		namespace = "dynamo"
		dgdName   = "stale-score-dgd"
	)

	t.Log("Given a DGD whose status still carries a score from a previous reconcile")
	dgd := placementDGD(dgdName, namespace)
	reader := fake.NewClientBuilder().
		WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
		Build()
	reconciler := newDGDPlacementScoreReconciler(reader)
	result := &workloadProgramResult{
		Status: nvidiacomv1beta1.DynamoGraphDeploymentStatus{
			Placement: &nvidiacomv1beta1.PlacementStatus{
				Score: ptr.To(0.99),
				State: nvidiacomv1beta1.PlacementScoreStateReported,
			},
		},
	}

	t.Log("When the current observation finds no PodCliqueSet")
	reconciler.Reconcile(context.Background(), dgd, result)

	t.Log("Then the stale score is cleared rather than retained")
	require.NotNil(t, result.Status.Placement, "placement status must always be set")
	require.Equal(t, nvidiacomv1beta1.PlacementScoreStateUnknown, result.Status.Placement.State)
	require.Nil(t, result.Status.Placement.Score, "the previous score must not linger")
}

// placementDGD builds the DGD under observation. The name is short enough that
// the derived PodCliqueSet name equals the DGD name.
func placementDGD(name, namespace string) *nvidiacomv1beta1.DynamoGraphDeployment {
	return &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
	}
}

// placementPCS builds the PodCliqueSet Grove created for the DGD. A nil
// generation hash models a PodCliqueSet that has not published one yet.
func placementPCS(name, namespace string, generationHash *string) *grovev1alpha1.PodCliqueSet {
	return &grovev1alpha1.PodCliqueSet{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Status: grovev1alpha1.PodCliqueSetStatus{
			CurrentGenerationHash: generationHash,
		},
	}
}

// placementPG builds a scheduler PodGang as Grove labels it: owned by the
// PodCliqueSet through the part-of label, tagged as a PodGang component, and
// stamped with the PodCliqueSet generation it was created for.
func placementPG(
	name, namespace, pcsName, generationHash string,
	score *float64,
) *schedulergrovev1alpha1.PodGang {
	return &schedulergrovev1alpha1.PodGang{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels: map[string]string{
				grovecommon.LabelPartOfKey:                  pcsName,
				grovecommon.LabelComponentKey:               grovecommon.LabelComponentNamePodGang,
				grovecommon.LabelPodCliqueSetGenerationHash: generationHash,
			},
		},
		Status: schedulergrovev1alpha1.PodGangStatus{
			PlacementScore: score,
		},
	}
}
