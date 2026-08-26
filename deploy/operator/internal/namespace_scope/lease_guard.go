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
	"fmt"
	"time"
)

// unwindBudgetFloor keeps the post-fatal wait useful for configurations whose renew interval
// and cleanup timeout leave nothing to spend on it, where a small bound is still better than
// none.
const unwindBudgetFloor = time.Second

// unwindBudget is how long Guard waits for work after lease renewal has become unrecoverable.
//
// The maxFailures arithmetic in NewLeaseManager exists to make renewalLoop declare the lease
// unrecoverable roughly one renewInterval before the lease can expire, so that this operator is
// gone before the cluster-wide operator stops excluding the namespace. That interval is the
// whole budget for shutting down, and the deferred lease release still needs cleanupTimeout out
// of it. With the shipped defaults — 10s renew interval, 5s cleanup — the unwind gets the
// remaining 5s, and unwind plus cleanup fit inside the buffer exactly.
func unwindBudget(renewInterval, cleanupTimeout time.Duration) time.Duration {
	budget := renewInterval - cleanupTimeout
	if budget < unwindBudgetFloor {
		return unwindBudgetFloor
	}

	return budget
}

// Deprecated: Guard holds the namespace scope marker lease for the duration of work, for the
// deprecated namespace-restricted operator mode.
//
// Guard starts the lease, runs work with a context that is cancelled when lease renewal
// becomes unrecoverable, and deletes the lease before returning. Lease deletion is bounded by
// cleanupTimeout and uses a context detached from ctx, so it still runs when ctx is already
// cancelled.
//
// The point of Guard is that a fatal condition reaches the caller as a returned error rather
// than terminating the process. os.Exit does not run deferred functions, so exiting from
// inside work — or from a goroutine watching Errors — strands the marker lease in the API
// server, and LeaseWatcher.Contains keeps excluding the namespace from cluster-wide
// reconciliation until the lease TTL expires.
//
// Unwinding instead of exiting must not take longer than the lease has left to live, or the two
// operators overlap on the namespace — the split-brain the lease protocol exists to prevent. So
// once renewal is unrecoverable Guard waits for work for at most unwindBudget and then releases
// the lease and returns regardless, leaving whatever work was still doing to be terminated with
// the process. Guard puts no bound on the paths where nothing is expiring: work returning on its
// own, or shutting down because ctx was cancelled.
//
// When work fails only because Guard cancelled its context, Guard returns the lease manager's
// unrecoverable error, since that is the root cause; otherwise it returns work's own error.
// ctx and work must be non-nil.
func (lm *LeaseManager) Guard(ctx context.Context, cleanupTimeout time.Duration, work func(context.Context) error) error {
	if err := lm.Start(ctx); err != nil {
		return fmt.Errorf("failed to start namespace scope marker lease manager: %w", err)
	}

	// The lease exists from here on, so every path out of this function has to release it.
	defer func() {
		cleanupCtx, cancelCleanup := context.WithTimeout(context.Background(), cleanupTimeout)
		defer cancelCleanup()

		if err := lm.Stop(cleanupCtx); err != nil {
			lm.logger.Error(err, "Failed to stop namespace scope marker lease manager cleanly")
		}
	}()

	workCtx, cancelWork := context.WithCancel(ctx)
	defer cancelWork()

	// work runs on its own goroutine so that an unrecoverable lease error can stop waiting for
	// it. The channel is buffered so that abandoned work never blocks forever on the send.
	workDone := make(chan error, 1)
	go func() {
		workDone <- work(workCtx)
	}()

	var workErr, fatalErr error

	select {
	case workErr = <-workDone:
	case fatalErr = <-lm.Errors():
		// An unrecoverable lease error cancels work instead of ending the process, which lets
		// the deferred release above run before the error reaches the caller. Waiting for work
		// to notice is bounded, because the lease is expiring while it does.
		cancelWork()

		budget := unwindBudget(lm.renewInterval, cleanupTimeout)
		expired := time.NewTimer(budget)
		defer expired.Stop()

		select {
		case workErr = <-workDone:
		case <-expired.C:
			lm.logger.Error(nil, "Giving up on orderly shutdown after unrecoverable lease failure; releasing the lease before it expires to prevent split-brain",
				"unwindBudget", budget)
		}
	}

	if fatalErr != nil && (workErr == nil || errors.Is(workErr, context.Canceled)) {
		return fmt.Errorf("namespace scope marker lease is unrecoverable: %w", fatalErr)
	}

	return workErr
}
