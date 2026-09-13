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

package defaulting

import (
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"k8s.io/utils/ptr"
)

func TestDefaultMultinodeRoleReplicas(t *testing.T) {
	tests := []struct {
		name       string
		component  nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec
		wantLeader *int32
		wantWorker *int32
	}{
		{
			name: "defaults omitted replicas",
			component: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				Multinode: &nvidiacomv1beta1.MultinodeSpec{NodeCount: 4},
				Roles: []nvidiacomv1beta1.ComponentRoleSpec{
					{Name: nvidiacomv1beta1.ComponentRoleLeader},
					{Name: nvidiacomv1beta1.ComponentRoleWorker},
				},
			},
			wantLeader: ptr.To(int32(1)),
			wantWorker: ptr.To(int32(3)),
		},
		{
			name: "preserves explicit replicas",
			component: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				Multinode: &nvidiacomv1beta1.MultinodeSpec{NodeCount: 4},
				Roles: []nvidiacomv1beta1.ComponentRoleSpec{
					{Name: nvidiacomv1beta1.ComponentRoleLeader, Replicas: ptr.To(int32(2))},
					{Name: nvidiacomv1beta1.ComponentRoleWorker, Replicas: ptr.To(int32(2))},
				},
			},
			wantLeader: ptr.To(int32(2)),
			wantWorker: ptr.To(int32(2)),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defaultMultinodeRoleReplicas(&tt.component)

			if got := tt.component.Roles[0].Replicas; !ptr.Equal(got, tt.wantLeader) {
				t.Fatalf("leader replicas = %v, want %v", got, tt.wantLeader)
			}
			if got := tt.component.Roles[1].Replicas; !ptr.Equal(got, tt.wantWorker) {
				t.Fatalf("worker replicas = %v, want %v", got, tt.wantWorker)
			}
		})
	}
}
