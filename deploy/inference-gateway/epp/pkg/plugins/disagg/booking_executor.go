/*
Copyright 2026 NVIDIA Corporation.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package disagg

import (
	"context"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/go-logr/logr"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
)

const (
	maxConcurrentPrefillMarkers      = 32
	maxConcurrentBookingCleanups     = 32
	prefillMarkerQueueCapacity       = 1024
	bookingCleanupQueueCapacity      = 4096
	bookingReconcileInterval         = time.Second
	bookingCleanupRetryInterval      = 30 * time.Second
	minimumBookingCleanupRetention   = 10 * time.Minute
	bookingCleanupRetentionGrace     = 2 * time.Minute
	routerActiveRequestExpirySeconds = "DYN_ROUTER_ACTIVE_REQUEST_EXPIRY_SECS"
)

var defaultBookingExecutor = newBookingExecutor(bookingExecutorConfig{
	markerWorkers:     maxConcurrentPrefillMarkers,
	cleanupWorkers:    maxConcurrentBookingCleanups,
	markerQueueSize:   prefillMarkerQueueCapacity,
	cleanupQueueSize:  bookingCleanupQueueCapacity,
	reconcileInterval: bookingReconcileInterval,
	cleanupRetryDelay: bookingCleanupRetryInterval,
	cleanupRetention:  bookingCleanupRetention(),
	cleanupBackoff:    cleanupRetryBackoff,
})

func bookingCleanupRetention() time.Duration {
	return bookingCleanupRetentionFromLookup(os.Getenv)
}

func bookingCleanupRetentionFromLookup(getEnv func(string) string) time.Duration {
	raw := getEnv(routerActiveRequestExpirySeconds)
	seconds, err := strconv.ParseUint(raw, 10, 64)
	maxSeconds := uint64((time.Duration(1<<63-1) - bookingCleanupRetentionGrace) / time.Second)
	if err != nil || seconds == 0 || seconds > maxSeconds {
		return minimumBookingCleanupRetention
	}

	retention := time.Duration(seconds)*time.Second + bookingCleanupRetentionGrace
	if retention < minimumBookingCleanupRetention {
		return minimumBookingCleanupRetention
	}
	return retention
}

type bookingExecutorConfig struct {
	markerWorkers     int
	cleanupWorkers    int
	markerQueueSize   int
	cleanupQueueSize  int
	reconcileInterval time.Duration
	cleanupRetryDelay time.Duration
	cleanupRetention  time.Duration
	cleanupBackoff    time.Duration
}

// bookingExecutor bounds both queued and active CGO bookkeeping calls. Cleanup
// has a dedicated pool so first-token marker pressure cannot delay terminal
// request cleanup.
type bookingExecutor struct {
	markerQueue       chan prefillMarkerWork
	cleanupQueue      chan *bookingLifecycle
	reconcileInterval time.Duration
	cleanupRetryDelay time.Duration
	cleanupRetention  time.Duration
	cleanupBackoff    time.Duration
	stopCh            chan struct{}
	stopOnce          sync.Once
	wg                sync.WaitGroup
}

type prefillMarkerWork struct {
	lifecycle           *bookingLifecycle
	ctx                 context.Context
	done                chan struct{}
	markPrefillComplete func(string) error
	logger              logr.Logger
	requestID           string
}

func newBookingExecutor(cfg bookingExecutorConfig) *bookingExecutor {
	if cfg.markerWorkers <= 0 || cfg.cleanupWorkers <= 0 || cfg.markerQueueSize <= 0 || cfg.cleanupQueueSize <= 0 {
		panic("booking executor worker and queue limits must be positive")
	}
	if cfg.reconcileInterval <= 0 || cfg.cleanupRetryDelay <= 0 || cfg.cleanupRetention <= 0 || cfg.cleanupBackoff <= 0 {
		panic("booking executor durations must be positive")
	}

	executor := &bookingExecutor{
		markerQueue:       make(chan prefillMarkerWork, cfg.markerQueueSize),
		cleanupQueue:      make(chan *bookingLifecycle, cfg.cleanupQueueSize),
		reconcileInterval: cfg.reconcileInterval,
		cleanupRetryDelay: cfg.cleanupRetryDelay,
		cleanupRetention:  cfg.cleanupRetention,
		cleanupBackoff:    cfg.cleanupBackoff,
		stopCh:            make(chan struct{}),
	}
	for range cfg.markerWorkers {
		executor.wg.Add(1)
		go executor.runMarkerWorker()
	}
	for range cfg.cleanupWorkers {
		executor.wg.Add(1)
		go executor.runCleanupWorker()
	}
	executor.wg.Add(1)
	go executor.runReconciler()
	return executor
}

func (e *bookingExecutor) stop() {
	e.stopOnce.Do(func() { close(e.stopCh) })
	e.wg.Wait()
}

func (e *bookingExecutor) enqueueMarker(work prefillMarkerWork) bool {
	select {
	case <-e.stopCh:
		return false
	default:
	}
	select {
	case e.markerQueue <- work:
		return true
	case <-e.stopCh:
		return false
	default:
		return false
	}
}

func (e *bookingExecutor) runMarkerWorker() {
	defer e.wg.Done()
	for {
		select {
		case work := <-e.markerQueue:
			e.runPrefillMarker(work)
		case <-e.stopCh:
			return
		}
	}
}

func (e *bookingExecutor) runPrefillMarker(work prefillMarkerWork) {
	defer close(work.done)
	for attempt := 1; attempt <= prefillMarkMaxAttempts; attempt++ {
		select {
		case <-e.stopCh:
			return
		default:
		}
		if work.ctx.Err() != nil {
			return
		}
		if err := work.markPrefillComplete(work.lifecycle.bookingID); err == nil {
			work.logger.V(logutil.VERBOSE).Info("DynDecodeScorer ResponseBody: marked prefill complete",
				"bookingID", work.lifecycle.bookingID, "requestID", work.requestID, "attempt", attempt)
			return
		} else {
			work.logger.V(logutil.DEFAULT).Error(err, "DynDecodeScorer ResponseBody: failed to mark prefill complete",
				"bookingID", work.lifecycle.bookingID, "requestID", work.requestID, "attempt", attempt)
		}
		if attempt == prefillMarkMaxAttempts {
			return
		}

		timer := time.NewTimer(prefillMarkRetryBackoff * time.Duration(attempt))
		select {
		case <-work.ctx.Done():
			stopTimer(timer)
			return
		case <-e.stopCh:
			stopTimer(timer)
			return
		case <-timer.C:
		}
	}
}

func (e *bookingExecutor) enqueueCleanup(lifecycle *bookingLifecycle) bool {
	now := time.Now()
	lifecycle.mu.Lock()
	if !lifecycle.cleanupStarted || lifecycle.cleanupSucceeded || lifecycle.cleanupExpired ||
		lifecycle.cleanupQueued || lifecycle.cleanupRunning ||
		(lifecycle.cleanupExhausted && now.Before(lifecycle.cleanupRetryAt)) {
		lifecycle.mu.Unlock()
		return false
	}
	lifecycle.cleanupQueued = true
	lifecycle.mu.Unlock()

	select {
	case e.cleanupQueue <- lifecycle:
		return true
	case <-e.stopCh:
	default:
	}

	lifecycle.mu.Lock()
	lifecycle.cleanupQueued = false
	lifecycle.mu.Unlock()
	return false
}

func (e *bookingExecutor) runCleanupWorker() {
	defer e.wg.Done()
	for {
		select {
		case lifecycle := <-e.cleanupQueue:
			e.runCleanup(lifecycle)
		case <-e.stopCh:
			return
		}
	}
}

func (e *bookingExecutor) runCleanup(lifecycle *bookingLifecycle) {
	lifecycle.mu.Lock()
	lifecycle.cleanupQueued = false
	if lifecycle.cleanupSucceeded || lifecycle.cleanupExpired {
		lifecycle.mu.Unlock()
		return
	}
	lifecycle.cleanupRunning = true
	decodeRegistrationDone := lifecycle.decodeRegistrationDone
	logger := lifecycle.cleanupLogger
	reason := lifecycle.cleanupReason
	lifecycle.mu.Unlock()

	if !e.waitForLifecycleWork(decodeRegistrationDone) {
		lifecycle.mu.Lock()
		lifecycle.cleanupRunning = false
		lifecycle.mu.Unlock()
		return
	}

	for attempt := 1; attempt <= cleanupMaxAttempts; attempt++ {
		select {
		case <-e.stopCh:
			lifecycle.mu.Lock()
			lifecycle.cleanupRunning = false
			lifecycle.mu.Unlock()
			return
		default:
		}

		err := lifecycle.freeBooking(lifecycle.bookingID)
		if err == nil {
			lifecycle.mu.Lock()
			lifecycle.cleanupRunning = false
			lifecycle.cleanupSucceeded = true
			done := lifecycle.closeCleanupDoneLocked()
			lifecycle.mu.Unlock()
			bookingLifecycles.CompareAndDelete(lifecycle.bookingID, lifecycle)
			logger.V(logutil.VERBOSE).Info("Dynamo EPP booking cleaned up",
				"bookingID", lifecycle.bookingID, "reason", reason, "attempt", attempt)
			if done != nil {
				close(done)
			}
			return
		}

		logger.V(logutil.DEFAULT).Error(err, "Dynamo EPP booking cleanup failed",
			"bookingID", lifecycle.bookingID, "reason", reason, "attempt", attempt)
		if attempt == cleanupMaxAttempts {
			lifecycle.mu.Lock()
			lifecycle.cleanupRunning = false
			lifecycle.cleanupExhausted = true
			lifecycle.cleanupRetryAt = time.Now().Add(e.cleanupRetryDelay)
			done := lifecycle.closeCleanupDoneLocked()
			lifecycle.mu.Unlock()
			logger.V(logutil.DEFAULT).Error(err, "Dynamo EPP booking cleanup exhausted retries; retaining finite tombstone for background retry",
				"bookingID", lifecycle.bookingID, "reason", reason, "attempts", cleanupMaxAttempts,
				"retention", e.cleanupRetention)
			if done != nil {
				close(done)
			}
			return
		}

		timer := time.NewTimer(e.cleanupBackoff * time.Duration(attempt))
		select {
		case <-timer.C:
		case <-e.stopCh:
			stopTimer(timer)
			lifecycle.mu.Lock()
			lifecycle.cleanupRunning = false
			lifecycle.mu.Unlock()
			return
		}
	}
}

func (e *bookingExecutor) waitForLifecycleWork(done <-chan struct{}) bool {
	if done == nil {
		return true
	}
	select {
	case <-done:
		return true
	case <-e.stopCh:
		return false
	}
}

func (e *bookingExecutor) runReconciler() {
	defer e.wg.Done()
	ticker := time.NewTicker(e.reconcileInterval)
	defer ticker.Stop()
	for {
		select {
		case now := <-ticker.C:
			e.reconcile(now)
		case <-e.stopCh:
			return
		}
	}
}

func (e *bookingExecutor) reconcile(now time.Time) {
	bookingLifecycles.Range(func(key, value any) bool {
		lifecycle := value.(*bookingLifecycle)
		if lifecycle.executor != e {
			return true
		}

		lifecycle.mu.Lock()
		if !lifecycle.cleanupStarted || lifecycle.cleanupSucceeded || lifecycle.cleanupExpired {
			lifecycle.mu.Unlock()
			return true
		}
		if now.Sub(lifecycle.cleanupStartedAt) >= e.cleanupRetention {
			lifecycle.cleanupExpired = true
			done := lifecycle.closeCleanupDoneLocked()
			logger := lifecycle.cleanupLogger
			reason := lifecycle.cleanupReason
			lifecycle.mu.Unlock()
			bookingLifecycles.CompareAndDelete(key, lifecycle)
			logger.V(logutil.DEFAULT).Info("Dynamo EPP booking cleanup ownership expired; relying on router stale-booking reaper",
				"bookingID", lifecycle.bookingID, "reason", reason, "retention", e.cleanupRetention)
			if done != nil {
				close(done)
			}
			return true
		}
		eligible := !lifecycle.cleanupQueued && !lifecycle.cleanupRunning &&
			(!lifecycle.cleanupExhausted || !now.Before(lifecycle.cleanupRetryAt))
		lifecycle.mu.Unlock()
		if eligible {
			e.enqueueCleanup(lifecycle)
		}
		return true
	})
}

func stopTimer(timer *time.Timer) {
	if timer.Stop() {
		return
	}
	select {
	case <-timer.C:
	default:
	}
}
