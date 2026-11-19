/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package grovecapacity

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestReleaseProofIsIndependentOfVictimSelection(t *testing.T) {
	for _, scenario := range []string{"victim-present", "exact-victim-removed", "count-not-converged", "retained-uid-replaced", "victim-uid-replaced"} {
		t.Run(scenario, func(t *testing.T) {
			t.Log("pin the exact victim, retained UID, operation and resulting topology")
			intent := releaseIntent{
				Fence: enginegroup.ReleaseFence{
					ReplicaID: "replica-1", SlotID: "slot-1", TransitionID: "retire", AuthorizingTopologyGeneration: 2,
					CapacityRefs: []enginegroup.CapacityRef{{Name: "victim", UID: "victim-uid"}},
				},
				Retained: []enginegroup.CapacityRef{{Name: "retained", UID: "retained-uid"}},
				Replicas: 1,
			}
			retained := corev1.Pod{Name: "retained", UID: "retained-uid"}
			victim := corev1.Pod{Name: "victim", UID: "victim-uid"}
			pods, replicas := []corev1.Pod{retained}, int32(1)
			unexpectedUID := false
			switch scenario {
			case "victim-present":
				pods = append(pods, victim)
			case "count-not-converged":
				replicas = 2
			case "retained-uid-replaced":
				pods[0].UID = "replacement-uid"
				unexpectedUID = true
			case "victim-uid-replaced":
				victim.UID = "replacement-uid"
				pods = append(pods, victim)
				unexpectedUID = true
			}

			t.Log("prove the physical outcome without consulting readiness gates or deletion-order hints")
			fences, err := observeReleaseFences(journalState{Release: &intent}, pods, replicas)
			if unexpectedUID {
				require.ErrorIs(t, err, enginegroup.ErrRecoveryRequired)
				require.Empty(t, fences)
				return
			}
			require.NoError(t, err)
			if scenario == "exact-victim-removed" {
				assert.Equal(t, []enginegroup.ReleaseFence{intent.Fence}, fences)
			} else {
				assert.Empty(t, fences)
			}
		})
	}
}
