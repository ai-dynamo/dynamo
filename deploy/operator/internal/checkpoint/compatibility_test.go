/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package checkpoint

import (
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/assert"
)

func TestValidateCheckpointCompatibility(t *testing.T) {
	tests := []struct {
		name         string
		experimental *nvidiacomv1beta1.ExperimentalSpec
		wantErrs     []string
	}{
		{name: "no experimental features"},
		{
			name: "checkpoint only",
			experimental: &nvidiacomv1beta1.ExperimentalSpec{
				Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true},
			},
		},
		{
			name: "failover only",
			experimental: &nvidiacomv1beta1.ExperimentalSpec{
				Failover: &nvidiacomv1beta1.FailoverSpec{},
			},
		},
		{
			name: "disabled checkpoint with failover",
			experimental: &nvidiacomv1beta1.ExperimentalSpec{
				Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{},
				Failover:   &nvidiacomv1beta1.FailoverSpec{},
			},
		},
		{
			name: "disabled checkpoint with inter-pod GMS",
			experimental: &nvidiacomv1beta1.ExperimentalSpec{
				Checkpoint:       &nvidiacomv1beta1.ComponentCheckpointConfig{},
				GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{Mode: nvidiacomv1beta1.GMSModeInterPod},
			},
		},
		{
			name: "enabled checkpoint with intra-pod GMS",
			experimental: &nvidiacomv1beta1.ExperimentalSpec{
				Checkpoint:       &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true},
				GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{Mode: nvidiacomv1beta1.GMSModeIntraPod},
			},
		},
		{
			name: "enabled checkpoint with inter-pod GMS",
			experimental: &nvidiacomv1beta1.ExperimentalSpec{
				Checkpoint:       &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true},
				GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{Mode: nvidiacomv1beta1.GMSModeInterPod},
			},
			wantErrs: []string{checkpointInterPodCompatibilityMessage},
		},
		{
			name: "enabled checkpoint with failover",
			experimental: &nvidiacomv1beta1.ExperimentalSpec{
				Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true},
				Failover:   &nvidiacomv1beta1.FailoverSpec{},
			},
			wantErrs: []string{checkpointFailoverCompatibilityMessage},
		},
		{
			name: "enabled checkpoint with inter-pod GMS and failover",
			experimental: &nvidiacomv1beta1.ExperimentalSpec{
				Checkpoint:       &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true},
				GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{Mode: nvidiacomv1beta1.GMSModeInterPod},
				Failover:         &nvidiacomv1beta1.FailoverSpec{},
			},
			wantErrs: []string{
				checkpointInterPodCompatibilityMessage,
				checkpointFailoverCompatibilityMessage,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			violations := ValidateCheckpointCompatibility(tt.experimental)
			var gotErrs []string
			for _, violation := range violations {
				gotErrs = append(gotErrs, violation.Error())
			}
			assert.Equal(t, tt.wantErrs, gotErrs)
		})
	}
}
