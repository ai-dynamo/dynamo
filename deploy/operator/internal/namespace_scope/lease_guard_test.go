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

package namespace_scope

import (
	"context"
	"errors"
	"testing"
	"time"

	k8sErrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
)

// errStartupFailed stands in for any of the startup failures that used to call os.Exit(1)
// after the marker lease had been acquired.
var errStartupFailed = errors.New("startup failed")

// guardCleanupTimeout bounds lease deletion in these tests. The fake clientset answers
// immediately, so the value only has to be large enough not to fire spuriously.
const guardCleanupTimeout = 2 * time.Second

// newGuardTestLeaseManager builds a LeaseManager over a fake API server, which NewLeaseManager
// cannot do because it dials a *rest.Config. The failure budget is fixed at two renewals so a
// test that rejects renewals reaches the unrecoverable path within two renewal intervals.
func newGuardTestLeaseManager(client kubernetes.Interface, leaseDuration, renewInterval time.Duration) *LeaseManager {
	return &LeaseManager{
		client:          client,
		namespace:       testNamespace,
		leaseDuration:   leaseDuration,
		renewInterval:   renewInterval,
		holderIdentity:  "namespace-restricted-operator-" + testOperatorVersion,
		operatorVersion: testOperatorVersion,
		stopCh:          make(chan struct{}),
		maxFailures:     2,
	}
}

// leaseExists reports whether the marker lease is present in the fake API server.
func leaseExists(t *testing.T, client kubernetes.Interface) bool {
	t.Helper()

	_, err := client.CoordinationV1().Leases(testNamespace).Get(context.Background(), LeaseName, metav1.GetOptions{})
	if err == nil {
		return true
	}
	if k8sErrors.IsNotFound(err) {
		return false
	}

	t.Fatalf("unexpected error reading marker lease: %v", err)
	return false
}

// rejectLeaseUpdates makes every lease renewal fail, which is how the renewal loop reaches its
// failure budget and declares the lease unrecoverable.
func rejectLeaseUpdates(client *fake.Clientset) {
	client.PrependReactor("update", "leases", func(action k8stesting.Action) (bool, runtime.Object, error) {
		return true, nil, errors.New("simulated API server failure")
	})
}

// TestLeaseManager_Guard_ReleasesLeaseWhenWorkReturns is the regression test for startup
// failures downstream of lease acquisition. Before Guard, those paths called os.Exit(1), which
// skips deferred lease release and leaves the namespace excluded from cluster-wide
// reconciliation until the lease TTL expires.
func TestLeaseManager_Guard_ReleasesLeaseWhenWorkReturns(t *testing.T) {
	tests := []struct {
		name    string
		workErr error
	}{
		{
			name:    "work fails the way a startup step used to exit",
			workErr: errStartupFailed,
		},
		{
			name:    "work completes normally",
			workErr: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Given a lease manager whose renewals succeed")
			client := fake.NewSimpleClientset()
			lm := newGuardTestLeaseManager(client, 30*time.Second, 50*time.Millisecond)

			t.Log("When Guard runs work that returns")
			heldDuringWork := false
			err := lm.Guard(context.Background(), guardCleanupTimeout, func(context.Context) error {
				heldDuringWork = leaseExists(t, client)
				return tt.workErr
			})

			t.Log("Then the lease was held while work ran")
			if !heldDuringWork {
				t.Fatal("marker lease should exist while work runs; the deletion assertion below would be vacuous")
			}

			t.Log("And the lease is already gone at the moment Guard returns")
			if leaseExists(t, client) {
				t.Error("marker lease still present when Guard returned; the namespace stays excluded until TTL expiry")
			}

			t.Log("And Guard reports work's own outcome")
			if !errors.Is(err, tt.workErr) {
				t.Errorf("Guard() error = %v, want %v", err, tt.workErr)
			}
		})
	}
}

// TestLeaseManager_Guard_ReleasesLeaseWhenRenewalIsUnrecoverable is the regression test for the
// lease manager's own fatal path. The goroutine that watched Errors() used to call os.Exit(1),
// stranding the very lease whose renewal had just failed.
func TestLeaseManager_Guard_ReleasesLeaseWhenRenewalIsUnrecoverable(t *testing.T) {
	t.Log("Given a lease manager whose renewals always fail")
	client := fake.NewSimpleClientset()
	rejectLeaseUpdates(client)
	lm := newGuardTestLeaseManager(client, 30*time.Millisecond, 10*time.Millisecond)

	t.Log("When Guard runs work that only returns once its context is cancelled")
	workObservedCancel := false
	err := lm.Guard(context.Background(), guardCleanupTimeout, func(ctx context.Context) error {
		<-ctx.Done()
		workObservedCancel = true
		return ctx.Err()
	})

	t.Log("Then the unrecoverable lease cancelled work instead of ending the process")
	if !workObservedCancel {
		t.Fatal("work should have been cancelled by the unrecoverable lease error")
	}

	t.Log("And the lease is already gone at the moment Guard returns")
	if leaseExists(t, client) {
		t.Error("marker lease still present when Guard returned; the namespace stays excluded until TTL expiry")
	}

	t.Log("And Guard reports the lease failure rather than the cancellation it caused")
	if err == nil {
		t.Fatal("Guard() error = nil, want the unrecoverable lease error")
	}
	if errors.Is(err, context.Canceled) {
		t.Errorf("Guard() error = %v, want the lease failure rather than the derived cancellation", err)
	}
}

// TestLeaseManager_Guard_ReturnsWorkErrorWhenLeaseIsHealthy is a negative control for the error
// assertion above: with renewals succeeding there is no lease failure to report, so the
// "unrecoverable" wrapper must not appear.
func TestLeaseManager_Guard_ReturnsWorkErrorWhenLeaseIsHealthy(t *testing.T) {
	t.Log("Given a lease manager whose renewals succeed")
	client := fake.NewSimpleClientset()
	lm := newGuardTestLeaseManager(client, 30*time.Second, 50*time.Millisecond)

	t.Log("When Guard runs work that fails on its own")
	err := lm.Guard(context.Background(), guardCleanupTimeout, func(context.Context) error {
		return errStartupFailed
	})

	t.Log("Then Guard returns exactly that error, unwrapped by any lease diagnosis")
	if !errors.Is(err, errStartupFailed) {
		t.Fatalf("Guard() error = %v, want %v", err, errStartupFailed)
	}
	if err.Error() != errStartupFailed.Error() {
		t.Errorf("Guard() error = %q, want it reported verbatim as %q", err.Error(), errStartupFailed.Error())
	}
}

// TestLeaseManager_Guard_PrefersGenuineWorkErrorOverLeaseFailure is a negative control for the
// preference rule: the lease failure wins only over the cancellation Guard itself caused, never
// over a real diagnosis from work.
func TestLeaseManager_Guard_PrefersGenuineWorkErrorOverLeaseFailure(t *testing.T) {
	t.Log("Given a lease manager whose renewals always fail")
	client := fake.NewSimpleClientset()
	rejectLeaseUpdates(client)
	lm := newGuardTestLeaseManager(client, 30*time.Millisecond, 10*time.Millisecond)

	t.Log("When work reacts to the cancellation with a diagnosis of its own")
	err := lm.Guard(context.Background(), guardCleanupTimeout, func(ctx context.Context) error {
		<-ctx.Done()
		return errStartupFailed
	})

	t.Log("Then Guard keeps work's error rather than replacing it with the lease failure")
	if !errors.Is(err, errStartupFailed) {
		t.Errorf("Guard() error = %v, want %v", err, errStartupFailed)
	}

	t.Log("And the lease is still released")
	if leaseExists(t, client) {
		t.Error("marker lease still present when Guard returned")
	}
}

// TestLeaseManager_Guard_DoesNotRunWorkWhenLeaseCannotStart is a negative control for the
// deletion assertions: they mean something only because Guard reached the point of holding the
// lease. When acquisition itself fails there is nothing to release and nothing to run.
func TestLeaseManager_Guard_DoesNotRunWorkWhenLeaseCannotStart(t *testing.T) {
	t.Log("Given an API server that refuses to create the marker lease")
	client := fake.NewSimpleClientset()
	client.PrependReactor("create", "leases", func(action k8stesting.Action) (bool, runtime.Object, error) {
		return true, nil, errors.New("simulated API server failure")
	})
	lm := newGuardTestLeaseManager(client, 30*time.Second, 50*time.Millisecond)

	t.Log("When Guard is asked to run work under that lease")
	workRan := false
	err := lm.Guard(context.Background(), guardCleanupTimeout, func(context.Context) error {
		workRan = true
		return nil
	})

	t.Log("Then work never ran and Guard reports the acquisition failure")
	if workRan {
		t.Error("work should not run when the marker lease could not be acquired")
	}
	if err == nil {
		t.Fatal("Guard() error = nil, want the lease acquisition failure")
	}

	t.Log("And no lease was left behind")
	if leaseExists(t, client) {
		t.Error("marker lease should not exist after a failed acquisition")
	}
}
