//go:build !clustertest

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
	"fmt"
	"testing"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	dynamotesting "github.com/ai-dynamo/dynamo/deploy/operator/internal/testing"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorenv"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	schedulergrovev1alpha1 "github.com/ai-dynamo/grove/scheduler/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
)

const placementScoreTestTimeout = 10 * time.Second

// TestGroveDGDPlacementScoreProjection covers the placement score end to end on
// a live Grove pathway: the operator renders the PodCliqueSet, the scheduler
// publishes a score on the PodGang, and the DGD status reports it.
func TestGroveDGDPlacementScoreProjection(t *testing.T) {
	ctx := context.Background()
	t.Log("Start an operator environment with Grove admission enabled")
	env := newGroveWorkerHashSuffixTestEnv(t)
	startGroveWorkerHashSuffixTestController(t, env)

	t.Log("Create a Grove DGD and wait for the operator to render its PodCliqueSet")
	dgd := newGroveWorkerHashSuffixTestDGD(env.Namespace(), "placement-score")
	require.NoError(t, env.Client().Create(ctx, dgd))

	pcsName := dynamo.PCSNameForDGD(dgd.Name, dgd.Spec.Components)
	pcs := waitForPodCliqueSet(t, ctx, env, dgd, pcsName)

	t.Log("With no PodGang scored yet the DGD reports an indeterminate placement")
	waitForPlacement(t, ctx, env, dgd, nvidiacomv1beta1.PlacementScoreStateUnknown, nil)

	t.Log("Create the scheduler's PodGang for the current PodCliqueSet")
	gang := placementScoreTestPodGang(env.Namespace(), pcs)
	require.NoError(t, env.Client().Create(ctx, gang))

	t.Log("Publish a placement score on the PodGang")
	require.Eventually(t, func() bool {
		current := &schedulergrovev1alpha1.PodGang{}
		if err := env.Client().Get(ctx, types.NamespacedName{Name: gang.Name, Namespace: gang.Namespace}, current); err != nil {
			return false
		}
		current.Status.PlacementScore = ptr.To(0.37)
		return env.Client().Update(ctx, current) == nil
	}, placementScoreTestTimeout, groveSuffixTestInterval, "PodGang placement score was not published")

	t.Log("The score change alone requeues the DGD and the DGD reports the score")
	waitForPlacement(t, ctx, env, dgd, nvidiacomv1beta1.PlacementScoreStateReported, ptr.To(0.37))
}

// waitForPodCliqueSet waits until the operator has rendered the PodCliqueSet for
// a DGD, which is what ties the DGD to the scheduler's PodGangs, and returns it
// so a test can build a PodGang that the PodCliqueSet owns.
func waitForPodCliqueSet(
	t *testing.T,
	ctx context.Context,
	env *operatorenv.TestEnv,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	pcsName string,
) *grovev1alpha1.PodCliqueSet {
	t.Helper()
	pcs := &grovev1alpha1.PodCliqueSet{}
	dynamotesting.Eventually(t, func() (bool, string) {
		current := &grovev1alpha1.PodCliqueSet{}
		key := types.NamespacedName{Name: pcsName, Namespace: dgd.Namespace}
		if err := env.Client().Get(ctx, key, current); err != nil {
			return false, fmt.Sprintf("get PodCliqueSet %s: %v", pcsName, err)
		}
		// The PodGang selection matches on the owner UID, which is only assigned
		// once the API server has created the object.
		if current.UID == "" {
			return false, "PodCliqueSet has no UID yet"
		}
		pcs = current
		return true, "PodCliqueSet exists"
	}, placementScoreTestTimeout, groveSuffixTestInterval, "operator did not render the Grove PodCliqueSet")
	return pcs
}

// waitForPlacement waits until the DGD status reports the expected placement.
// A nil score means the field must report no score at all.
func waitForPlacement(
	t *testing.T,
	ctx context.Context,
	env *operatorenv.TestEnv,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	wantState nvidiacomv1beta1.PlacementScoreState,
	wantScore *float64,
) {
	t.Helper()
	dynamotesting.Eventually(t, func() (bool, string) {
		current := &nvidiacomv1beta1.DynamoGraphDeployment{}
		key := types.NamespacedName{Name: dgd.Name, Namespace: dgd.Namespace}
		if err := env.Client().Get(ctx, key, current); err != nil {
			return false, fmt.Sprintf("get DynamoGraphDeployment: %v", err)
		}

		placement := current.Status.Placement
		if placement == nil {
			return false, "status.placement is not set"
		}
		if placement.State != wantState {
			return false, fmt.Sprintf("placement state = %q, want %q", placement.State, wantState)
		}
		if wantScore == nil {
			if placement.Score != nil {
				return false, fmt.Sprintf("placement score = %v, want none", *placement.Score)
			}
			return true, "placement reports no score"
		}
		if placement.Score == nil {
			return false, fmt.Sprintf("placement score is unset, want %v", *wantScore)
		}
		if *placement.Score != *wantScore {
			return false, fmt.Sprintf("placement score = %v, want %v", *placement.Score, *wantScore)
		}
		return true, "placement reports the expected score"
	}, placementScoreTestTimeout, groveSuffixTestInterval, "DGD did not report the expected placement")
}

// placementScoreTestPodGang builds a scheduler PodGang as Grove creates it for a
// PodCliqueSet, with no score published yet.
func placementScoreTestPodGang(namespace string, pcs *grovev1alpha1.PodCliqueSet) *schedulergrovev1alpha1.PodGang {
	return &schedulergrovev1alpha1.PodGang{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pcs.Name + "-anchor",
			Namespace: namespace,
			Labels: map[string]string{
				grovecommon.LabelManagedByKey: grovecommon.LabelManagedByValue,
				grovecommon.LabelPartOfKey:    pcs.Name,
				grovecommon.LabelComponentKey: grovecommon.LabelComponentNamePodGang,
			},
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: grovev1alpha1.SchemeGroupVersion.String(),
				Kind:       "PodCliqueSet",
				Name:       pcs.Name,
				UID:        pcs.UID,
				Controller: ptr.To(true),
			}},
		},
	}
}
