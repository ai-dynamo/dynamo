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

package controller_common

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apivalidation "k8s.io/apimachinery/pkg/api/validation"
)

func TestServiceMetadataInventoryRespectsAnnotationSizeLimit(t *testing.T) {
	for _, test := range []struct {
		name     string
		existing bool
	}{
		{"create", false}, {"existing", true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Arrange valid annotations that leave no room for the managed-key inventory")
			annotations := map[string]string{
				"example.com/large":           strings.Repeat("x", 256*1024-200),
				NvidiaAnnotationHashKey:       strings.Repeat("a", 64),
				NvidiaAnnotationGenerationKey: "1",
			}
			require.NoError(t, apivalidation.ValidateAnnotationsSize(annotations))
			desired := &corev1.Service{Annotations: annotations, Labels: map[string]string{"example.com/team": "blue"}}
			var existing *corev1.Service
			if test.existing {
				existing = desired.DeepCopy()
			}

			t.Log("Prepare metadata and verify the inventory cannot make the Service unwritable")
			labels, got, err := managedServiceMetadataFor(existing, desired)
			require.NoError(t, err)
			require.NotContains(t, got, ManagedServiceMetadataAnnotation)
			require.NoError(t, apivalidation.ValidateAnnotationsSize(got))
			require.Equal(t, desired.Labels, labels)
			require.Equal(t, annotations["example.com/large"], got["example.com/large"])

			t.Log("Free annotation space and verify the inventory is recorded on the next reconcile")
			delete(desired.Annotations, "example.com/large")
			if existing != nil {
				delete(existing.Annotations, "example.com/large")
			}
			_, got, err = managedServiceMetadataFor(existing, desired)
			require.NoError(t, err)
			require.Contains(t, got[ManagedServiceMetadataAnnotation], "example.com/team")
		})
	}
}
