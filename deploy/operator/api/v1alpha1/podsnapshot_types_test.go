/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package v1alpha1

import (
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
)

// TestSchemeRegistersSnapshotKinds verifies the init() registrations expose all
// four new kinds through the package AddToScheme.
func TestSchemeRegistersSnapshotKinds(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := AddToScheme(scheme); err != nil {
		t.Fatalf("AddToScheme failed: %v", err)
	}
	for _, kind := range []string{"PodSnapshot", "PodSnapshotList", "PodSnapshotContent", "PodSnapshotContentList"} {
		if !scheme.Recognizes(GroupVersion.WithKind(kind)) {
			t.Errorf("scheme does not recognize kind %q in %s", kind, GroupVersion.String())
		}
	}
}

// TestSnapshotDeepCopyIsIndependent verifies the generated deepcopy produces an
// equal but independent PodSnapshot (mutating the clone must not touch the source).
func TestSnapshotDeepCopyIsIndependent(t *testing.T) {
	original := &PodSnapshot{
		ObjectMeta: metav1.ObjectMeta{Name: "snap-a", Namespace: "inference"},
		Spec: PodSnapshotSpec{
			Source: PodSnapshotSource{PodRef: PodReference{Name: "worker-0"}},
		},
		Status: PodSnapshotStatus{
			Conditions: []metav1.Condition{{Type: "Ready", Status: metav1.ConditionTrue, Reason: "Captured"}},
		},
	}

	clone := original.DeepCopy()
	if !reflect.DeepEqual(original, clone) {
		t.Fatalf("DeepCopy is not equal to original")
	}

	clone.Spec.Source.PodRef.Name = "mutated"
	clone.Status.Conditions[0].Reason = "Changed"
	if original.Spec.Source.PodRef.Name != "worker-0" {
		t.Errorf("mutating clone spec changed original: got %q", original.Spec.Source.PodRef.Name)
	}
	if original.Status.Conditions[0].Reason != "Captured" {
		t.Errorf("mutating clone condition changed original: got %q", original.Status.Conditions[0].Reason)
	}
}

// TestSnapshotContentDeepCopyIsIndependent verifies the generated deepcopy for
// the cluster-scoped PodSnapshotContent is equal but independent.
func TestSnapshotContentDeepCopyIsIndependent(t *testing.T) {
	original := &PodSnapshotContent{
		ObjectMeta: metav1.ObjectMeta{Name: "content-a"},
		Spec: PodSnapshotContentSpec{
			PodSnapshotRef: PodSnapshotReference{Namespace: "inference", Name: "snap-a", UID: types.UID("uid-1")},
			Source: PodSnapshotContentSource{
				PodRef:   PodReference{Name: "worker-0", UID: types.UID("pod-uid-1")},
				NodeName: "node-a",
			},
		},
		Status: PodSnapshotContentStatus{
			Conditions: []metav1.Condition{{Type: "Ready", Status: metav1.ConditionTrue, Reason: "Bound"}},
		},
	}

	clone := original.DeepCopy()
	if !reflect.DeepEqual(original, clone) {
		t.Fatalf("DeepCopy is not equal to original")
	}

	clone.Spec.Source.PodRef.Name = "mutated"
	clone.Status.Conditions[0].Reason = "Changed"
	if original.Spec.Source.PodRef.Name != "worker-0" {
		t.Errorf("mutating clone changed original podRef name: got %q", original.Spec.Source.PodRef.Name)
	}
	if original.Status.Conditions[0].Reason != "Bound" {
		t.Errorf("mutating clone condition changed original: got %q", original.Status.Conditions[0].Reason)
	}
}
