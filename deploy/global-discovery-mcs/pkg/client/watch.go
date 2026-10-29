// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package client

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
)

// Event is one decoded watch event.
type Event[T any] struct {
	Type mcsv1.EventType
	// Object is set for UPSERT and DELETE.
	Object T
	// Revision is set for SNAPSHOT_END, BOOKMARK and deltas.
	Revision string
}

// WatchStream yields the events of one open watch.
type WatchStream[T any] interface {
	Recv() (Event[T], error)
}

// WatchFunc opens a watch that resumes after revision, or starts with a
// snapshot when revision is empty or unusable. The sub-clients' Watcher
// methods return one per watch kind.
type WatchFunc[T any] func(ctx context.Context, revision string) (WatchStream[T], error)

type recvFunc[T any] func() (Event[T], error)

func (f recvFunc[T]) Recv() (Event[T], error) { return f() }

// WatchHandler receives the events of a WatchLoop in order, on the loop's
// goroutine. Nil functions are skipped.
type WatchHandler[T any] struct {
	// SnapshotStart is called on SNAPSHOT_START: discard local state. The
	// following Upserts until Synced are the complete set.
	SnapshotStart func()
	// Upsert is called for a snapshot object or a changed object.
	Upsert func(obj T)
	// Delete is called with the last state of a removed object.
	Delete func(obj T)
	// Synced is called on SNAPSHOT_END: local state is complete as of
	// revision.
	Synced func(revision string)
	// Bookmark is called on BOOKMARK with the current revision.
	Bookmark func(revision string)
}

// DefaultWatchIdleTimeout is a WatchLoop IdleTimeout of six bookmark
// intervals of the server's default 5 s; the mcs-controller's import watch
// uses it.
const DefaultWatchIdleTimeout = 30 * time.Second

// WatchLoopConfig configures a WatchLoop.
type WatchLoopConfig[T any] struct {
	// Name labels the dynamo_mcs_client_watch_reconnects_total metric and the
	// logs, for example WatchServiceImports. Keep the set of names bounded.
	Name string
	// Watch opens the stream, for example c.ServiceImports().Watcher(req).
	Watch WatchFunc[T]
	// Handler receives the events.
	Handler WatchHandler[T]
	// OnResync is called when a new snapshot starts after the loop had
	// already synced, before Handler.SnapshotStart: the server could not
	// resume (another replica, a revision outside its history, or the
	// watcher fell behind), so the consumer rebuilds its state.
	OnResync func()
	// Revision is the revision to resume from on the first open, for
	// example the revision of a List; empty starts with a snapshot.
	Revision string
	// Backoff spaces reconnects; zero fields take the client's backoff.
	Backoff Backoff
	// IdleTimeout reopens a stream that delivered nothing, not even a
	// bookmark, for this long. Zero or negative disables it.
	IdleTimeout time.Duration
}

// WatchLoop keeps one watch open: it reconnects with backoff, resumes from
// the last revision it saw, and turns the event stream into Handler calls.
// It also moves its stream when the client returns to its primary URL.
type WatchLoop[T any] struct {
	c       *Client
	cfg     WatchLoopConfig[T]
	backoff Backoff
	logger  *slog.Logger

	mu         sync.Mutex
	revision   string
	synced     bool
	syncedOnce bool
	inSnapshot bool
	running    atomic.Bool
}

// NewWatchLoop returns a WatchLoop over c. Start it with Run.
func NewWatchLoop[T any](c *Client, cfg WatchLoopConfig[T]) *WatchLoop[T] {
	b := cfg.Backoff
	cb := c.cfg.Backoff
	if b.Initial <= 0 {
		b.Initial = cb.Initial
	}
	if b.Max <= 0 {
		b.Max = cb.Max
	}
	if b.Multiplier < 1 {
		b.Multiplier = cb.Multiplier
	}
	if b.Jitter == 0 {
		b.Jitter = cb.Jitter
	}
	if cfg.Name == "" {
		cfg.Name = "watch"
	}
	return &WatchLoop[T]{
		c:        c,
		cfg:      cfg,
		backoff:  b.withDefaults(),
		logger:   c.logger.With("watch", cfg.Name),
		revision: cfg.Revision,
	}
}

// Revision returns the revision to resume from: the last SNAPSHOT_END,
// BOOKMARK or delta revision, or "" while a snapshot is being received.
func (w *WatchLoop[T]) Revision() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.revision
}

// Synced reports whether the loop received a complete snapshot and has not
// started another since.
func (w *WatchLoop[T]) Synced() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.synced
}

var errWatchIdle = errors.New("watch idle timeout")

// Run runs the loop until ctx is done, then returns nil. It returns an
// error only when the loop is already running.
func (w *WatchLoop[T]) Run(ctx context.Context) error {
	if !w.running.CompareAndSwap(false, true) {
		return errors.New("mcs client: watch loop already running")
	}
	defer w.running.Store(false)
	for attempt := 0; ; {
		healthy, err := w.runOnce(ctx)
		if ctx.Err() != nil {
			return nil
		}
		if healthy {
			attempt = 0
		}
		delay := w.backoff.Delay(attempt)
		attempt++
		w.logStreamEnd(err, delay)
		w.c.metrics.watchReconnects.WithLabelValues(w.cfg.Name).Inc()
		select {
		case <-ctx.Done():
			return nil
		case <-time.After(delay):
		}
	}
}

func (w *WatchLoop[T]) logStreamEnd(err error, delay time.Duration) {
	switch status.Code(err) {
	case codes.Unavailable, codes.Canceled, codes.DeadlineExceeded:
		w.logger.Info("registry watch ended; reconnecting", "error", err, "delay", delay)
	default:
		w.logger.Warn("registry watch failed; reconnecting", "error", err, "delay", delay)
	}
}

// runOnce opens one stream and handles its events until it ends. healthy
// reports that the stream delivered at least one event.
func (w *WatchLoop[T]) runOnce(ctx context.Context) (healthy bool, err error) {
	sctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var idled, moved atomic.Bool
	restart := w.c.watchRestarted()
	go func() {
		select {
		case <-restart:
			moved.Store(true)
			cancel()
		case <-sctx.Done():
		}
	}()
	stream, err := w.cfg.Watch(sctx, w.Revision())
	if err != nil {
		return false, err
	}
	var idle *time.Timer
	if w.cfg.IdleTimeout > 0 {
		idle = time.AfterFunc(w.cfg.IdleTimeout, func() {
			idled.Store(true)
			cancel()
		})
		defer idle.Stop()
	}
	for {
		ev, err := stream.Recv()
		if err != nil {
			switch {
			case idled.Load():
				err = errWatchIdle
			case moved.Load():
				err = status.Error(codes.Unavailable, "client returned to its primary registry URL")
			}
			return healthy, err
		}
		if idle != nil {
			idle.Reset(w.cfg.IdleTimeout)
		}
		healthy = true
		w.handle(ev)
	}
}

func (w *WatchLoop[T]) handle(ev Event[T]) {
	h := w.cfg.Handler
	switch ev.Type {
	case mcsv1.EventType_EVENT_TYPE_SNAPSHOT_START:
		w.mu.Lock()
		resync := w.syncedOnce
		// A stream broken during the snapshot must not resume from the
		// revision before it: the consumer has discarded that state.
		w.revision, w.synced, w.inSnapshot = "", false, true
		w.mu.Unlock()
		if resync && w.cfg.OnResync != nil {
			w.cfg.OnResync()
		}
		if h.SnapshotStart != nil {
			h.SnapshotStart()
		}
	case mcsv1.EventType_EVENT_TYPE_UPSERT, mcsv1.EventType_EVENT_TYPE_DELETE:
		w.mu.Lock()
		if ev.Revision != "" && !w.inSnapshot {
			w.revision = ev.Revision
		}
		w.mu.Unlock()
		if ev.Type == mcsv1.EventType_EVENT_TYPE_UPSERT {
			if h.Upsert != nil {
				h.Upsert(ev.Object)
			}
		} else if h.Delete != nil {
			h.Delete(ev.Object)
		}
	case mcsv1.EventType_EVENT_TYPE_SNAPSHOT_END:
		w.mu.Lock()
		w.revision, w.synced, w.syncedOnce, w.inSnapshot = ev.Revision, true, true, false
		w.mu.Unlock()
		if h.Synced != nil {
			h.Synced(ev.Revision)
		}
	case mcsv1.EventType_EVENT_TYPE_BOOKMARK:
		w.mu.Lock()
		if ev.Revision != "" && !w.inSnapshot {
			w.revision = ev.Revision
		}
		w.mu.Unlock()
		if h.Bookmark != nil {
			h.Bookmark(ev.Revision)
		}
	default:
		w.logger.Debug("ignoring unknown watch event type", "type", ev.Type)
	}
}
