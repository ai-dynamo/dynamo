/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package sglang

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestJoinerPodBuilderDerivesOperationSpecificBootstrap(t *testing.T) {
	primary := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "primary", Labels: map[string]string{"controller-selector": "primary"}},
		Status:     corev1.PodStatus{PodIP: "10.0.0.8"},
		Spec: corev1.PodSpec{
			NodeName: "already-scheduled",
			Containers: []corev1.Container{{
				Name:    mainContainerName,
				Command: []string{"python3", "-m", "dynamo.sglang"},
				Args: []string{
					"--tp", "4", "--dp=4", "--model-path", "model",
					"--dist-init-addr", "primary-rendezvous.test:24555",
				},
			}},
		},
	}
	target := enginegroup.CapacityReplicaTarget{
		ReplicaID: "replica-4",
		SlotID:    "slot-4",
		Bootstrap: &enginegroup.CapacityBootstrap{
			Mode:          enginegroup.BootstrapModeJoin,
			NativeMembers: []enginegroup.NativeMemberID{"dp-4"},
		},
	}

	t.Log("derive the joining process from the immutable primary profile")
	joiner, err := (JoinerPodBuilder{}).Build(primary, target)
	require.NoError(t, err)

	t.Log("preserve the runtime image contract while replacing only operation-specific geometry")
	assert.Equal(t, []string{"sglang", "serve"}, joiner.Spec.Containers[0].Command)
	assert.Contains(t, joiner.Spec.Containers[0].Args, "--elastic-ep-join-mode")
	assert.Contains(t, joiner.Spec.Containers[0].Args, "--elastic-ep-join-rank-offset")
	assert.Contains(t, joiner.Spec.Containers[0].Args, "4")
	assert.Contains(t, joiner.Spec.Containers[0].Args, "10.0.0.8:24555")
	assert.NotContains(t, joiner.Labels, "controller-selector", "a joiner must not inherit DGD workload selectors")
	assert.Empty(t, joiner.Spec.NodeName)
	assert.Equal(t, "already-scheduled", primary.Spec.NodeName, "the primary must not be mutated")
}
