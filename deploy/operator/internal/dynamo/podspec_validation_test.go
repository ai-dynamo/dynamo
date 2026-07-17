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

package dynamo

import (
	"testing"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestValidateRenderedPodSpecContainerNames(t *testing.T) {
	tests := []struct {
		name    string
		podSpec *corev1.PodSpec
		wantErr string
	}{
		{
			name: "unique names",
			podSpec: &corev1.PodSpec{
				Containers:          []corev1.Container{{Name: "main"}},
				InitContainers:      []corev1.Container{{Name: "prepare"}},
				EphemeralContainers: []corev1.EphemeralContainer{{EphemeralContainerCommon: corev1.EphemeralContainerCommon{Name: "debug"}}},
			},
		},
		{
			name: "regular and init container conflict",
			podSpec: &corev1.PodSpec{
				Containers:     []corev1.Container{{Name: "prepare"}},
				InitContainers: []corev1.Container{{Name: "prepare"}},
			},
			wantErr: `container name "prepare" is used by both spec.containers[0] and spec.initContainers[0]`,
		},
		{
			name: "init and ephemeral container conflict",
			podSpec: &corev1.PodSpec{
				InitContainers:      []corev1.Container{{Name: "debug"}},
				EphemeralContainers: []corev1.EphemeralContainer{{EphemeralContainerCommon: corev1.EphemeralContainerCommon{Name: "debug"}}},
			},
			wantErr: `container name "debug" is used by both spec.initContainers[0] and spec.ephemeralContainers[0]`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateRenderedPodSpecContainerNames(tt.podSpec)
			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.EqualError(t, err, tt.wantErr)
		})
	}
}
