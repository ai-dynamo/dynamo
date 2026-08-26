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

	// An unrecoverable lease error cancels work instead of ending the process, which lets the
	// deferred release above run before the error reaches the caller.
	var fatalErr error
	monitorDone := make(chan struct{})
	go func() {
		defer close(monitorDone)

		select {
		case err := <-lm.Errors():
			fatalErr = err
			cancelWork()
		case <-workCtx.Done():
		}
	}()

	workErr := work(workCtx)

	// Joining the monitor is what makes reading fatalErr race-free; cancelWork releases it
	// when no fatal error arrived.
	cancelWork()
	<-monitorDone

	if fatalErr != nil && (workErr == nil || errors.Is(workErr, context.Canceled)) {
		return fmt.Errorf("namespace scope marker lease is unrecoverable: %w", fatalErr)
	}

	return workErr
}
