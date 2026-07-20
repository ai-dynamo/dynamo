/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
)

func TestDeploymentStrategyFromAnnotations(t *testing.T) {
	tests := []struct {
		name        string
		annotations map[string]string
		want        common.DeploymentStrategy
	}{
		{
			name: "missing annotation defaults to RollingUpdate",
			want: common.DeploymentStrategyRollingUpdate,
		},
		{
			name: "explicit RollingUpdate",
			annotations: map[string]string{
				KubeAnnotationDeploymentStrategy: string(common.DeploymentStrategyRollingUpdate),
			},
			want: common.DeploymentStrategyRollingUpdate,
		},
		{
			name: "explicit Recreate",
			annotations: map[string]string{
				KubeAnnotationDeploymentStrategy: string(common.DeploymentStrategyRecreate),
			},
			want: common.DeploymentStrategyRecreate,
		},
		{
			name: "unrecognized value preserves RollingUpdate fallback",
			annotations: map[string]string{
				KubeAnnotationDeploymentStrategy: "recreate",
			},
			want: common.DeploymentStrategyRollingUpdate,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, deploymentStrategyFromAnnotations(tt.annotations))
		})
	}
}
