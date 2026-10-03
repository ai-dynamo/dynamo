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
	"strings"
	"testing"
	"time"

	coordinationv1 "k8s.io/api/coordination/v1"
	k8sErrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	typedcoordinationv1 "k8s.io/client-go/kubernetes/typed/coordination/v1"
	k8stesting "k8s.io/client-go/testing"
)

// errStartupFailed stands in for any of the startup failures that used to call os.Exit(1)
// after the marker lease had been acquired.
var errStartupFailed = errors.New("startup failed")

// guardCleanupTimeout bounds lease deletion in these tests. The fake clientset answers
// immediately, so the value only has to be large enough not to fire spuriously.
const guardCleanupTimeout = 2 * time.Second

// newGuardTestLeaseManager builds a LeaseManager over a fake API server, which
// NewLeaseManager cannot do because it dials a *rest.Config.
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

// TestLeaseManager_Guard_ReleasesLeaseWhenWorkReturns covers startup failures downstream of
// lease acquisition: the lease must be gone before the failure reaches the exit boundary.
func TestLeaseManager_Guard_ReleasesLeaseWhenWorkReturns(t *testing.T) {
	tests := []struct {
		name          string
		workErr       error
		leaseDuration time.Duration
		wantTTL       int32
	}{
		{
			name:          "work fails the way a startup step used to exit",
			workErr:       errStartupFailed,
			leaseDuration: 30 * time.Second,
			wantTTL:       30,
		},
		{
			name:          "work completes normally",
			workErr:       nil,
			leaseDuration: 30 * time.Second,
			wantTTL:       30,
		},
		{
			name:          "sub-second lease completes normally",
			leaseDuration: 30 * time.Millisecond,
			wantTTL:       1,
		},
		{
			name:          "fractional lease rounds up",
			leaseDuration: 1500 * time.Millisecond,
			wantTTL:       2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Given a lease manager whose renewals succeed")
			client := fake.NewSimpleClientset()
			lm := newGuardTestLeaseManager(client, tt.leaseDuration, 10*time.Millisecond)

			t.Log("When Guard runs work that returns")
			heldDuringWork := false
			var publishedTTL int32
			err := lm.Guard(context.Background(), guardCleanupTimeout, func(context.Context) error {
				lease, err := client.CoordinationV1().Leases(testNamespace).Get(context.Background(), LeaseName, metav1.GetOptions{})
				if err != nil {
					return err
				}
				heldDuringWork = true
				publishedTTL = *lease.Spec.LeaseDurationSeconds
				return tt.workErr
			})

			t.Log("Then the lease was held while work ran")
			if !heldDuringWork {
				t.Fatal("marker lease should exist while work runs; the deletion assertion below would be vacuous")
			}
			if publishedTTL != tt.wantTTL {
				t.Errorf("published lease TTL = %d, want %d", publishedTTL, tt.wantTTL)
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

// TestLeaseManager_Guard_ReleasesLeaseWhenRenewalIsUnrecoverable covers the lease manager's
// own fatal path, which stranded the very lease whose renewal had failed.
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

// guardUnwindDeadline is the ceiling for Guard to return once the lease is unrecoverable.
// A literal, so the test still fails against a Guard that never bounds the unwind.
const guardUnwindDeadline = 2*time.Second + guardCleanupTimeout

// TestLeaseManager_Guard_BoundsUnwindWhenRenewalIsUnrecoverable pins the split-brain bound:
// an unbounded wait lets the restricted operator outlive the lease it can no longer renew.
func TestLeaseManager_Guard_BoundsUnwindWhenRenewalIsUnrecoverable(t *testing.T) {
	t.Log("Given a lease manager whose renewals always fail")
	client := fake.NewSimpleClientset()
	rejectLeaseUpdates(client)
	lm := newGuardTestLeaseManager(client, 30*time.Millisecond, 10*time.Millisecond)

	t.Log("And work that ignores cancellation, the way a wedged graceful shutdown does")
	releaseWork := make(chan struct{})
	defer close(releaseWork)

	t.Log("When Guard runs that work")
	guardDone := make(chan error, 1)
	go func() {
		guardDone <- lm.Guard(context.Background(), guardCleanupTimeout, func(context.Context) error {
			<-releaseWork
			return nil
		})
	}()

	t.Log("Then Guard gives up on work rather than waiting for it")
	var err error
	select {
	case err = <-guardDone:
	case <-time.After(guardUnwindDeadline):
		t.Fatalf("Guard did not return within %v of the unrecoverable lease error; the operator outlives the lease it can no longer renew and overlaps with the cluster-wide operator", guardUnwindDeadline)
	}

	t.Log("And the lease is released on the way out")
	if leaseExists(t, client) {
		t.Error("marker lease still present when Guard returned; the namespace stays excluded until TTL expiry")
	}

	t.Log("And Guard reports the lease failure that forced the exit")
	if err == nil {
		t.Fatal("Guard() error = nil, want the unrecoverable lease error")
	}
	if errors.Is(err, context.Canceled) {
		t.Errorf("Guard() error = %v, want the lease failure rather than the derived cancellation", err)
	}
}

// TestLeaseManager_Guard_ReturnsWorkErrorWhenLeaseIsHealthy is a negative control: with
// renewals succeeding, the "unrecoverable" wrapper must not appear.
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

// TestLeaseManager_Guard_PrefersGenuineWorkErrorOverLeaseFailure is a negative control: the
// lease failure wins only over the cancellation Guard caused, never over work's diagnosis.
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

// TestLeaseManager_Guard_DoesNotRunWorkWhenLeaseCannotStart is a negative control: when
// acquisition fails there is nothing to release and nothing to run.
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

func TestLeaseManager_Guard_ReleasesLeaseBeforeRepanicking(t *testing.T) {
	t.Log("Given a worker that panics while holding the lease")
	client := fake.NewSimpleClientset()
	lm := newGuardTestLeaseManager(client, 30*time.Second, time.Second)
	panicValue := errors.New("worker panic")
	heldDuringWork := false

	t.Log("Then the caller recovers the original panic after lease deletion")
	defer func() {
		if got := recover(); got != panicValue {
			t.Errorf("recovered %v, want original panic %v", got, panicValue)
		}
		if !heldDuringWork {
			t.Error("worker did not observe its lease before panicking")
		}
		if leaseExists(t, client) {
			t.Error("marker lease still present when panic reached caller")
		}
	}()

	t.Log("When Guard runs the panicking worker")
	_ = lm.Guard(context.Background(), guardCleanupTimeout, func(context.Context) error {
		_, err := client.CoordinationV1().Leases(testNamespace).Get(context.Background(), LeaseName, metav1.GetOptions{})
		heldDuringWork = err == nil
		panic(panicValue)
	})
	t.Error("Guard returned instead of propagating the worker panic")
}

func TestLeaseManager_Guard_CleansUpAmbiguousAcquisition(t *testing.T) {
	for _, verb := range []string{"create", "update"} {
		for _, otherOwner := range []bool{false, true} {
			name := verb + "/own-lease"
			if otherOwner {
				name = verb + "/another-holder"
			}
			t.Run(name, func(t *testing.T) {
				t.Log("Given an acquisition that commits but loses its response")
				client := fake.NewSimpleClientset()
				lm := newGuardTestLeaseManager(client, 30*time.Second, time.Second)
				transportErr := errors.New("response lost after commit")
				resource := coordinationv1.SchemeGroupVersion.WithResource("leases")
				if verb == "update" {
					lease := &coordinationv1.Lease{ObjectMeta: metav1.ObjectMeta{Name: LeaseName, Namespace: testNamespace}}
					if err := client.Tracker().Create(resource, lease, testNamespace); err != nil {
						t.Fatal(err)
					}
				}
				committed := false
				client.PrependReactor(verb, "leases", func(action k8stesting.Action) (bool, runtime.Object, error) {
					var lease *coordinationv1.Lease
					if verb == "create" {
						lease = action.(k8stesting.CreateAction).GetObject().(*coordinationv1.Lease).DeepCopy()
					} else {
						lease = action.(k8stesting.UpdateAction).GetObject().(*coordinationv1.Lease).DeepCopy()
					}

					// Give the committed object server metadata and optionally simulate takeover.
					lease.UID = "committed-lease"
					lease.ResourceVersion = "2"
					if otherOwner {
						holder := "another-manager"
						lease.Spec.HolderIdentity = &holder
					}
					var err error
					if verb == "create" {
						err = client.Tracker().Create(resource, lease, testNamespace)
					} else {
						err = client.Tracker().Update(resource, lease, testNamespace)
					}
					if err != nil {
						return true, nil, err
					}
					committed = true
					return true, nil, transportErr
				})

				t.Log("When Guard attempts acquisition")
				workRan := false
				err := lm.Guard(context.Background(), guardCleanupTimeout, func(context.Context) error {
					workRan = true
					return nil
				})

				t.Log("Then the committed acquisition error prevents work and cleanup respects ownership")
				if !committed || !errors.Is(err, transportErr) || workRan {
					t.Fatalf("committed=%v, error=%v, workRan=%v", committed, err, workRan)
				}
				if exists := leaseExists(t, client); exists != otherOwner {
					t.Errorf("lease exists=%v, want %v", exists, otherOwner)
				}

				t.Log("And deletion is conditional on the exact owned object")
				deletes := 0
				for _, action := range client.Actions() {
					if action.GetVerb() != "delete" {
						continue
					}
					deletes++
					preconditions := action.(k8stesting.DeleteAction).GetDeleteOptions().Preconditions
					if preconditions == nil || preconditions.UID == nil || preconditions.ResourceVersion == nil {
						t.Fatal("delete omitted ownership preconditions")
					}
					if *preconditions.UID != "committed-lease" || *preconditions.ResourceVersion != "2" {
						t.Errorf("unexpected delete preconditions: %+v", preconditions)
					}
				}
				if otherOwner && deletes != 0 {
					t.Errorf("attempted %d deletions of another holder's lease", deletes)
				}
			})
		}
	}
}

// blockedRenewalClient intercepts Update outside the fake client's reactor lock,
// allowing cleanup ownership reads to proceed while the RPC is blocked.
type blockedRenewalClient struct {
	kubernetes.Interface
	coordination typedcoordinationv1.CoordinationV1Interface
}

func (c *blockedRenewalClient) CoordinationV1() typedcoordinationv1.CoordinationV1Interface {
	return c.coordination
}

type blockedRenewalCoordination struct {
	typedcoordinationv1.CoordinationV1Interface
	lease *blockedRenewalLease
}

func (c *blockedRenewalCoordination) Leases(string) typedcoordinationv1.LeaseInterface {
	return c.lease
}

type blockedRenewalLease struct {
	typedcoordinationv1.LeaseInterface
	started      chan struct{}
	cancelled    chan struct{}
	release      chan struct{}
	ignoreCancel bool
}

func (l *blockedRenewalLease) Update(ctx context.Context, _ *coordinationv1.Lease, _ metav1.UpdateOptions) (*coordinationv1.Lease, error) {
	// Later attempts after cancellation must not close the notification channels again.
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	close(l.started)
	<-ctx.Done()
	close(l.cancelled)
	if l.ignoreCancel {
		<-l.release
	}
	return nil, ctx.Err()
}

func TestLeaseManager_Guard_BoundsBlockedRenewal(t *testing.T) {
	for _, watchdog := range []bool{false, true} {
		for _, ignoreCancel := range []bool{false, true} {
			name := "work-return"
			if watchdog {
				name = "expiry-watchdog"
			}
			if ignoreCancel {
				name += "/ignores-cancellation"
			} else {
				name += "/honors-cancellation"
			}
			t.Run(name, func(t *testing.T) {
				t.Log("Given a renewal RPC that cannot complete before shutdown")
				client := fake.NewSimpleClientset()
				lease := &blockedRenewalLease{
					LeaseInterface: client.CoordinationV1().Leases(testNamespace),
					started:        make(chan struct{}), cancelled: make(chan struct{}),
					release: make(chan struct{}), ignoreCancel: ignoreCancel,
				}
				defer close(lease.release)
				coordination := &blockedRenewalCoordination{CoordinationV1Interface: client.CoordinationV1(), lease: lease}
				wrapped := &blockedRenewalClient{Interface: client, coordination: coordination}
				lm := newGuardTestLeaseManager(wrapped, 5*time.Second, 10*time.Millisecond)
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()

				t.Log("When work returns or the watchdog cancels it with renewal still blocked")
				guardDone := make(chan error, 1)
				proceed := make(chan struct{})
				go func() {
					guardDone <- lm.Guard(ctx, 100*time.Millisecond, func(workCtx context.Context) error {
						select {
						case <-proceed:
						case <-workCtx.Done():
							return workCtx.Err()
						}
						if watchdog {
							<-workCtx.Done()
							return workCtx.Err()
						}
						return errStartupFailed
					})
				}()

				t.Log("Then Guard returns before the published lease expires")
				select {
				case <-lease.started:
				case <-time.After(2 * time.Second):
					t.Fatal("renewal RPC did not start")
				}
				published, err := client.CoordinationV1().Leases(testNamespace).Get(ctx, LeaseName, metav1.GetOptions{})
				if err != nil {
					t.Fatal(err)
				}
				expires := published.Spec.RenewTime.Add(time.Duration(*published.Spec.LeaseDurationSeconds) * time.Second)
				close(proceed)
				wait := time.Until(expires)
				if !watchdog {
					wait = time.Second
				}
				select {
				case err = <-guardDone:
				case <-time.After(wait):
					t.Fatal("Guard did not bound the blocked renewal")
				}
				if !time.Now().Before(expires) {
					t.Error("Guard outlived its lease")
				}
				if watchdog {
					if err == nil || !strings.Contains(err.Error(), "lease renewal did not complete before the shutdown window") {
						t.Errorf("Guard error = %v, want expiry watchdog failure", err)
					}
				} else if !errors.Is(err, errStartupFailed) {
					t.Errorf("Guard error = %v, want work error", err)
				}

				t.Log("And Stop cancelled renewal, deleting only after the RPC stopped")
				select {
				case <-lease.cancelled:
				default:
					t.Error("Stop did not cancel the renewal RPC")
				}
				if exists := leaseExists(t, client); exists != ignoreCancel {
					t.Errorf("lease exists=%v, want %v while ignoreCancel=%v", exists, ignoreCancel, ignoreCancel)
				}
			})
		}
	}
}
