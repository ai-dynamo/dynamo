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

package controller_common

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestAPIGroupServesVersion(t *testing.T) {
	apiGroups := &metav1.APIGroupList{
		Groups: []metav1.APIGroup{
			{
				Name: "resource.k8s.io",
				Versions: []metav1.GroupVersionForDiscovery{
					{GroupVersion: "resource.k8s.io/v1beta1", Version: "v1beta1"},
					{GroupVersion: "resource.k8s.io/v1beta2", Version: "v1beta2"},
				},
			},
			{
				Name: "apps",
				Versions: []metav1.GroupVersionForDiscovery{
					{GroupVersion: "apps/v1", Version: "v1"},
				},
			},
		},
	}

	tests := []struct {
		name      string
		groupName string
		version   string
		want      bool
	}{
		{name: "group exists when version omitted", groupName: "resource.k8s.io", want: true},
		{name: "served beta version exists", groupName: "resource.k8s.io", version: "v1beta2", want: true},
		{name: "unserved v1 version is unavailable", groupName: "resource.k8s.io", version: "v1", want: false},
		{name: "different group with v1 exists", groupName: "apps", version: "v1", want: true},
		{name: "missing group is unavailable", groupName: "missing.example.com", version: "v1", want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := apiGroupServesVersion(apiGroups, tt.groupName, tt.version); got != tt.want {
				t.Fatalf("apiGroupServesVersion() = %v, want %v", got, tt.want)
			}
		})
	}
}
