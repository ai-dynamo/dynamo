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
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
)

// stubExcludedNamespaces implements ExcludedNamespacesInterface over a fixed list.
type stubExcludedNamespaces []string

// Contains reports whether namespace is in the stubbed exclusion list.
func (s stubExcludedNamespaces) Contains(namespace string) bool {
	return slices.Contains(s, namespace)
}

func TestNamespaceAllowed(t *testing.T) {
	tests := []struct {
		name       string
		restricted string
		excluded   []string
		namespace  string
		want       bool
	}{
		{name: "restricted mode admits matching namespace", restricted: "prod", namespace: "prod", want: true},
		{name: "restricted mode drops mismatched namespace", restricted: "prod", namespace: "other", want: false},
		{name: "restricted mode drops empty namespace", restricted: "prod", namespace: "", want: false},
		{name: "cluster-wide drops excluded namespace", excluded: []string{"banned"}, namespace: "banned", want: false},
		{name: "cluster-wide drops ephemeral namespace", namespace: "ci-ephemeral-1", want: false},
		{name: "cluster-wide admits normal namespace", namespace: "prod", want: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := &configv1alpha1.OperatorConfiguration{}
			config.Namespace.Restricted = tt.restricted
			runtimeConfig := &RuntimeConfig{}
			if tt.excluded != nil {
				runtimeConfig.ExcludedNamespaces = stubExcludedNamespaces(tt.excluded)
			}
			got := NamespaceAllowed(config, runtimeConfig, &corev1.Pod{}, tt.namespace)
			assert.Equal(t, tt.want, got)
		})
	}
}
