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

package namespace_scope

import (
	"testing"
	"time"

	"github.com/go-logr/logr"
	coordinationv1 "k8s.io/api/coordination/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8sCache "k8s.io/client-go/tools/cache"
	"k8s.io/utils/ptr"
)

func TestLeaseWatcher_HandleLeaseAdd(t *testing.T) {
	tests := []struct {
		name              string
		lease             *coordinationv1.Lease
		shouldExclude     bool
		excludedNamespace string
	}{
		{
			name: "adds namespace for valid marker lease",
			lease: &coordinationv1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name:      LeaseName,
					Namespace: "test-ns",
				},
				Spec: coordinationv1.LeaseSpec{
					HolderIdentity:       ptr.To("namespace-restricted-operator-v1.0.0"),
					LeaseDurationSeconds: ptr.To[int32](30),
					RenewTime:            &metav1.MicroTime{Time: time.Now()},
				},
			},
			shouldExclude:     true,
			excludedNamespace: "test-ns",
		},
		{
			name: "ignores lease with wrong name",
			lease: &coordinationv1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "other-lease",
					Namespace: "test-ns",
				},
			},
			shouldExclude:     false,
			excludedNamespace: "test-ns",
		},
		{
			name: "adds namespace for lease without labels",
			lease: &coordinationv1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name:      LeaseName,
					Namespace: "test-ns",
					// No labels - still identified by name
				},
				Spec: coordinationv1.LeaseSpec{
					HolderIdentity:       ptr.To("namespace-restricted-operator-v1.0.0"),
					LeaseDurationSeconds: ptr.To[int32](30),
					RenewTime:            &metav1.MicroTime{Time: time.Now()},
				},
			},
			shouldExclude:     true,
			excludedNamespace: "test-ns",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger := logr.Discard()

			lw := &LeaseWatcher{
				logger: logger,
			}

			// Handle lease add
			lw.handleLeaseAdd(tt.lease)

			// Check if namespace was excluded
			got := lw.Contains(tt.excludedNamespace)
			if got != tt.shouldExclude {
				t.Errorf("namespace exclusion = %v, want %v", got, tt.shouldExclude)
			}
		})
	}
}

func TestLeaseWatcher_HandleLeaseUpdate(t *testing.T) {
	logger := logr.Discard()

	lw := &LeaseWatcher{
		logger: logger,
	}

	lease := &coordinationv1.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Name:      LeaseName,
			Namespace: "test-ns",
		},
		Spec: coordinationv1.LeaseSpec{
			HolderIdentity:       ptr.To("namespace-restricted-operator-v1.0.0"),
			LeaseDurationSeconds: ptr.To[int32](30),
			RenewTime:            &metav1.MicroTime{Time: time.Now()},
		},
	}

	// Handle lease update (should add if not present)
	lw.handleLeaseUpdate(lease)

	// Verify namespace was added
	if !lw.Contains("test-ns") {
		t.Error("namespace should be excluded after update")
	}

	// Handle another update (should remain)
	lw.handleLeaseUpdate(lease)

	// Verify namespace is still excluded
	if !lw.Contains("test-ns") {
		t.Error("namespace should still be excluded after second update")
	}
}

func TestLeaseWatcher_HandleLeaseDelete(t *testing.T) {
	logger := logr.Discard()

	lw := &LeaseWatcher{
		logger: logger,
	}

	lease := &coordinationv1.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Name:      LeaseName,
			Namespace: "test-ns",
		},
		Spec: coordinationv1.LeaseSpec{
			HolderIdentity:       ptr.To("operator-v1.0.0"),
			LeaseDurationSeconds: ptr.To[int32](30),
			RenewTime:            &metav1.MicroTime{Time: time.Now()},
		},
	}

	// Pre-add namespace
	lw.addExcludedNamespace(lease)

	// Verify namespace is excluded before delete
	if !lw.Contains("test-ns") {
		t.Fatal("namespace should be excluded before delete")
	}

	// Handle lease delete
	lw.handleLeaseDelete(lease)

	// Verify namespace was removed
	if lw.Contains("test-ns") {
		t.Error("namespace should not be excluded after delete")
	}
}

func TestLeaseWatcher_ExtractLease(t *testing.T) {
	logger := logr.Discard()
	lw := &LeaseWatcher{
		logger: logger,
	}

	tests := []struct {
		name     string
		obj      any
		wantNil  bool
		wantName string
	}{
		{
			name: "extracts regular lease object",
			obj: &coordinationv1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-lease",
					Namespace: "test-ns",
				},
			},
			wantNil:  false,
			wantName: "test-lease",
		},
		{
			name: "extracts lease from tombstone",
			obj: k8sCache.DeletedFinalStateUnknown{
				Key: "test-ns/test-lease",
				Obj: &coordinationv1.Lease{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-lease",
						Namespace: "test-ns",
					},
				},
			},
			wantNil:  false,
			wantName: "test-lease",
		},
		{
			name:    "returns nil for non-Lease object",
			obj:     &coordinationv1.LeaseList{},
			wantNil: true,
		},
		{
			name: "returns nil for tombstone with non-Lease object",
			obj: k8sCache.DeletedFinalStateUnknown{
				Key: "test-ns/test-obj",
				Obj: &coordinationv1.LeaseList{},
			},
			wantNil: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lease := lw.extractLease(tt.obj)

			if (lease == nil) != tt.wantNil {
				t.Errorf("extractLease() returned nil = %v, want nil = %v", lease == nil, tt.wantNil)
			}

			if !tt.wantNil && lease.Name != tt.wantName {
				t.Errorf("extractLease() lease.Name = %v, want %v", lease.Name, tt.wantName)
			}
		})
	}
}

func TestLeaseWatcher_IsNamespaceScopeMarker(t *testing.T) {
	logger := logr.Discard()
	lw := &LeaseWatcher{
		logger: logger,
	}

	tests := []struct {
		name  string
		lease *coordinationv1.Lease
		want  bool
	}{
		{
			name: "returns true for lease with correct name",
			lease: &coordinationv1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name:      LeaseName,
					Namespace: "test-ns",
				},
			},
			want: true,
		},
		{
			name: "returns true for lease with correct name and no labels",
			lease: &coordinationv1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name:      LeaseName,
					Namespace: "test-ns",
				},
			},
			want: true,
		},
		{
			name: "returns false for lease with wrong name",
			lease: &coordinationv1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "other-lease",
					Namespace: "test-ns",
				},
			},
			want: false,
		},
		{
			name: "returns false for lease with wrong name even if other metadata exists",
			lease: &coordinationv1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "wrong-name",
					Namespace: "test-ns",
				},
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := lw.isNamespaceScopeMarker(tt.lease)
			if got != tt.want {
				t.Errorf("isNamespaceScopeMarker() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestLeaseWatcher_MultipleNamespaces(t *testing.T) {
	logger := logr.Discard()

	lw := &LeaseWatcher{
		logger: logger,
	}

	// Add multiple namespaces
	namespaces := []string{"test-ns", "staging", "dev"}
	for _, ns := range namespaces {
		lease := &coordinationv1.Lease{
			ObjectMeta: metav1.ObjectMeta{
				Name:      LeaseName,
				Namespace: ns,
			},
			Spec: coordinationv1.LeaseSpec{
				HolderIdentity:       ptr.To("operator-v1.0.0"),
				LeaseDurationSeconds: ptr.To[int32](30),
				RenewTime:            &metav1.MicroTime{Time: time.Now()},
			},
		}
		lw.handleLeaseAdd(lease)
	}

	// Verify all are excluded
	for _, ns := range namespaces {
		if !lw.Contains(ns) {
			t.Errorf("namespace %s should be excluded", ns)
		}
	}

	// Delete one namespace
	deleteLease := &coordinationv1.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Name:      LeaseName,
			Namespace: "staging",
		},
	}
	lw.handleLeaseDelete(deleteLease)

	// Verify staging is removed but others remain
	if lw.Contains("staging") {
		t.Error("staging should not be excluded after delete")
	}
	if !lw.Contains("test-ns") {
		t.Error("test-ns should still be excluded")
	}
	if !lw.Contains("dev") {
		t.Error("dev should still be excluded")
	}
}
