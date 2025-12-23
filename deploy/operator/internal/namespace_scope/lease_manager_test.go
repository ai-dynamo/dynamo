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
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	coordinationv1 "k8s.io/api/coordination/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/utils/ptr"
)

const (
	testNamespace       = "test-ns"
	testOperatorVersion = "v1.0.0"
)

func TestLeaseManager_CreateOrUpdateLease(t *testing.T) {
	tests := []struct {
		name            string
		namespace       string
		operatorVersion string
		existingLease   *coordinationv1.Lease
		wantRenewTime   bool // Whether RenewTime should be set
	}{
		{
			name:            "creates lease when it doesn't exist",
			namespace:       testNamespace,
			operatorVersion: testOperatorVersion,
			existingLease:   nil,
			wantRenewTime:   false, // RenewTime should be nil on creation
		},
		{
			name:            "updates existing lease",
			namespace:       testNamespace,
			operatorVersion: testOperatorVersion,
			existingLease: &coordinationv1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name:      LeaseName,
					Namespace: testNamespace,
				},
				Spec: coordinationv1.LeaseSpec{
					HolderIdentity:       ptr.To("old-holder"),
					LeaseDurationSeconds: ptr.To[int32](60),
					AcquireTime:          &metav1.MicroTime{Time: time.Now().Add(-2 * time.Minute)},
					RenewTime:            &metav1.MicroTime{Time: time.Now().Add(-1 * time.Minute)},
				},
			},
			wantRenewTime: true, // RenewTime should be set on update
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create fake client with or without existing lease
			var client *fake.Clientset
			if tt.existingLease != nil {
				client = fake.NewSimpleClientset(tt.existingLease)
			} else {
				client = fake.NewSimpleClientset()
			}

			// Create lease manager
			lm := &LeaseManager{
				client:          client,
				namespace:       tt.namespace,
				leaseDuration:   30 * time.Second,
				renewInterval:   10 * time.Second,
				holderIdentity:  "namespace-restricted-operator-" + tt.operatorVersion,
				operatorVersion: tt.operatorVersion,
				stopCh:          make(chan struct{}),
			}

			// Call createOrUpdateLease
			ctx := context.Background()
			err := lm.createOrUpdateLease(ctx)
			if err != nil {
				t.Fatalf("createOrUpdateLease() error = %v", err)
			}

			// Verify lease exists
			lease, err := client.CoordinationV1().Leases(tt.namespace).Get(ctx, LeaseName, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("failed to get lease: %v", err)
			}

			// Verify lease name
			if lease.Name != LeaseName {
				t.Errorf("lease name = %v, want %v", lease.Name, LeaseName)
			}

			// Verify lease namespace
			if lease.Namespace != tt.namespace {
				t.Errorf("lease namespace = %v, want %v", lease.Namespace, tt.namespace)
			}

			// Verify holder identity
			if lease.Spec.HolderIdentity == nil {
				t.Fatal("lease holder identity is nil")
			}
			wantIdentity := "namespace-restricted-operator-" + tt.operatorVersion
			if *lease.Spec.HolderIdentity != wantIdentity {
				t.Errorf("holder identity = %v, want %v", *lease.Spec.HolderIdentity, wantIdentity)
			}

			// Verify lease duration
			if lease.Spec.LeaseDurationSeconds == nil {
				t.Fatal("lease duration is nil")
			}
			if *lease.Spec.LeaseDurationSeconds != 30 {
				t.Errorf("lease duration = %v, want %v", *lease.Spec.LeaseDurationSeconds, 30)
			}

			// Verify renew time and acquire time based on operation
			if tt.wantRenewTime {
				// Update case: RenewTime should be set and newer than before
				if lease.Spec.RenewTime == nil {
					t.Error("lease renew time should be set on update")
				} else if tt.existingLease != nil && !lease.Spec.RenewTime.After(tt.existingLease.Spec.RenewTime.Time) {
					t.Error("renew time was not updated")
				}
				// AcquireTime should be preserved from existing lease
				if tt.existingLease != nil && tt.existingLease.Spec.AcquireTime != nil {
					if lease.Spec.AcquireTime == nil {
						t.Error("acquire time should be preserved on update")
					} else if !lease.Spec.AcquireTime.Equal(tt.existingLease.Spec.AcquireTime) {
						t.Error("acquire time should not change on update")
					}
				}
			} else {
				// Create case: RenewTime should be nil, AcquireTime should be set
				if lease.Spec.RenewTime != nil {
					t.Error("lease renew time should be nil on initial creation")
				}
				if lease.Spec.AcquireTime == nil {
					t.Error("lease acquire time should be set on initial creation")
				}
			}
		})
	}
}

func TestLeaseManager_Stop(t *testing.T) {
	namespace := testNamespace
	operatorVersion := testOperatorVersion

	// Create fake client with existing lease
	existingLease := &coordinationv1.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Name:      LeaseName,
			Namespace: namespace,
		},
		Spec: coordinationv1.LeaseSpec{
			HolderIdentity:       ptr.To("namespace-restricted-operator-v1.0.0"),
			LeaseDurationSeconds: ptr.To[int32](30),
		},
	}
	client := fake.NewSimpleClientset(existingLease)

	// Create lease manager
	lm := &LeaseManager{
		client:          client,
		namespace:       namespace,
		leaseDuration:   30 * time.Second,
		renewInterval:   10 * time.Second,
		holderIdentity:  "namespace-restricted-operator-" + operatorVersion,
		operatorVersion: operatorVersion,
		stopCh:          make(chan struct{}),
	}

	// Stop lease manager
	ctx := context.Background()
	err := lm.Stop(ctx)
	if err != nil {
		t.Fatalf("Stop() error = %v", err)
	}

	// Verify lease was deleted
	_, err = client.CoordinationV1().Leases(namespace).Get(ctx, LeaseName, metav1.GetOptions{})
	if err == nil {
		t.Error("expected lease to be deleted, but it still exists")
	}
}

func TestLeaseManager_Stop_LeaseAlreadyDeleted(t *testing.T) {
	namespace := testNamespace
	operatorVersion := testOperatorVersion

	// Create fake client WITHOUT existing lease (simulating already deleted/expired)
	client := fake.NewSimpleClientset()

	// Create lease manager
	lm := &LeaseManager{
		client:          client,
		namespace:       namespace,
		leaseDuration:   30 * time.Second,
		renewInterval:   10 * time.Second,
		holderIdentity:  "namespace-restricted-operator-" + operatorVersion,
		operatorVersion: operatorVersion,
		stopCh:          make(chan struct{}),
	}

	// Stop lease manager - should succeed even though lease doesn't exist
	ctx := context.Background()
	err := lm.Stop(ctx)
	if err != nil {
		t.Fatalf("Stop() should succeed when lease is already deleted, got error = %v", err)
	}
}

func TestLeaseManager_StartAndStop_CompleteLifecycle(t *testing.T) {
	namespace := testNamespace
	operatorVersion := testOperatorVersion

	// Create fake client
	client := fake.NewSimpleClientset()

	// Create lease manager with short intervals for testing
	lm := &LeaseManager{
		client:          client,
		namespace:       namespace,
		leaseDuration:   30 * time.Second,
		renewInterval:   50 * time.Millisecond,
		holderIdentity:  "namespace-restricted-operator-" + operatorVersion,
		operatorVersion: operatorVersion,
		stopCh:          make(chan struct{}),
	}

	// Start the lease manager
	ctx := context.Background()
	err := lm.Start(ctx)
	if err != nil {
		t.Fatalf("Start() error = %v", err)
	}

	// Verify lease was created
	lease, err := client.CoordinationV1().Leases(namespace).Get(ctx, LeaseName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get lease after Start(): %v", err)
	}
	if lease.Name != LeaseName {
		t.Errorf("lease name = %v, want %v", lease.Name, LeaseName)
	}

	// Verify initial lease has no renew time (since it was just created)
	if lease.Spec.RenewTime != nil {
		t.Error("initial lease should not have renew time set on creation")
	}

	// Poll for renewal with timeout (more robust than fixed sleep)
	renewalDetected := false
	deadline := time.Now().Add(500 * time.Millisecond)
	for time.Now().Before(deadline) {
		updatedLease, err := client.CoordinationV1().Leases(namespace).Get(ctx, LeaseName, metav1.GetOptions{})
		if err == nil && updatedLease.Spec.RenewTime != nil {
			renewalDetected = true
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if !renewalDetected {
		t.Error("lease should have renew time set after renewal")
	}

	// Stop the lease manager (should delete the lease and stop renewal loop)
	stopCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	err = lm.Stop(stopCtx)
	if err != nil {
		t.Fatalf("Stop() error = %v", err)
	}

	// Verify lease was deleted
	_, err = client.CoordinationV1().Leases(namespace).Get(ctx, LeaseName, metav1.GetOptions{})
	if err == nil {
		t.Error("lease should be deleted after Stop()")
	}
}

// TestLeaseManager_FailureTracking_SendsErrorOnMaxFailures verifies that consecutive
// lease renewal failures trigger a fatal error to prevent split-brain scenarios.
// Note: This test calls renewalLoop() directly (not Start()) to inject failures via reactor.
func TestLeaseManager_FailureTracking_SendsErrorOnMaxFailures(t *testing.T) {
	namespace := testNamespace
	operatorVersion := testOperatorVersion

	// Create existing lease
	existingLease := &coordinationv1.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Name:      LeaseName,
			Namespace: namespace,
		},
		Spec: coordinationv1.LeaseSpec{
			HolderIdentity:       ptr.To("namespace-restricted-operator-v1.0.0"),
			LeaseDurationSeconds: ptr.To[int32](30),
			RenewTime:            &metav1.MicroTime{Time: time.Now()},
		},
	}

	// Create fake client with the lease
	client := fake.NewSimpleClientset(existingLease)

	// Add reactor to make all update operations fail (simulates persistent API failure)
	client.PrependReactor("update", "leases", func(action k8stesting.Action) (handled bool, ret runtime.Object, err error) {
		return true, nil, fmt.Errorf("simulated persistent API failure for testing")
	})

	// Create lease manager with short intervals for faster test execution
	lm := &LeaseManager{
		client:          client,
		namespace:       namespace,
		leaseDuration:   30 * time.Second,
		renewInterval:   10 * time.Millisecond, // Fast for testing
		holderIdentity:  "namespace-restricted-operator-" + operatorVersion,
		operatorVersion: operatorVersion,
		stopCh:          make(chan struct{}),
		maxFailures:     3,
		errCh:           make(chan error, 1),
	}

	// Start renewal loop - all updates will fail
	ctx := context.Background()
	lm.wg.Add(1)
	go lm.renewalLoop(ctx)

	// Wait for fatal error on channel (with generous timeout)
	select {
	case err := <-lm.errCh:
		if err == nil {
			t.Fatal("expected error from error channel, got nil")
		}
		t.Logf("Received expected fatal error: %v", err)

		// Verify error message is meaningful
		if !strings.Contains(err.Error(), "split-brain") {
			t.Errorf("error should mention split-brain prevention, got: %v", err)
		}
		if !strings.Contains(err.Error(), "3") {
			t.Errorf("error should mention failure count, got: %v", err)
		}

	case <-time.After(1 * time.Second):
		t.Fatal("timeout waiting for fatal error from lease manager (expected within ~30ms)")
	}

	// Clean shutdown
	close(lm.stopCh)
	lm.wg.Wait()
}

// TestLeaseManager_FailureTracking_ResetsOnSuccess verifies that the failure counter
// is reset to 0 after a successful renewal, allowing recovery from transient failures.
// Note: This test calls renewalLoop() directly to verify internal failure counter behavior.
func TestLeaseManager_FailureTracking_ResetsOnSuccess(t *testing.T) {
	namespace := testNamespace
	operatorVersion := testOperatorVersion

	// Create fake client with existing lease
	existingLease := &coordinationv1.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Name:      LeaseName,
			Namespace: namespace,
		},
		Spec: coordinationv1.LeaseSpec{
			HolderIdentity:       ptr.To("namespace-restricted-operator-v1.0.0"),
			LeaseDurationSeconds: ptr.To[int32](30),
			RenewTime:            &metav1.MicroTime{Time: time.Now()},
		},
	}
	client := fake.NewSimpleClientset(existingLease)

	// Create lease manager with pre-existing failures
	lm := &LeaseManager{
		client:          client,
		namespace:       namespace,
		leaseDuration:   30 * time.Second,
		renewInterval:   20 * time.Millisecond, // Reasonable interval for test
		holderIdentity:  "namespace-restricted-operator-" + operatorVersion,
		operatorVersion: operatorVersion,
		stopCh:          make(chan struct{}),
		maxFailures:     3,
		errCh:           make(chan error, 1),
		failureCount:    2, // Simulates 2 previous failures
	}

	// Start renewal loop (will succeed and reset counter)
	ctx := context.Background()
	lm.wg.Add(1)
	go lm.renewalLoop(ctx)

	// Wait for at least one renewal cycle
	time.Sleep(50 * time.Millisecond)

	// Stop the loop
	close(lm.stopCh)
	lm.wg.Wait()

	// Verify failure count was reset to 0 after successful renewal
	if lm.failureCount != 0 {
		t.Errorf("failure count should be reset to 0 after success, got %d", lm.failureCount)
	}

	// Verify no fatal error was sent
	select {
	case err := <-lm.errCh:
		t.Errorf("unexpected error on channel after successful renewal: %v", err)
	default:
		// Expected: no error sent
	}
}
