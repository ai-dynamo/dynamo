// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package client

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/pkg/fakeregistry"
)

// eventLog records WatchHandler calls as strings.
type eventLog struct {
	mu        sync.Mutex
	events    []string
	bookmarks int
	resyncs   int
}

func (l *eventLog) add(s string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.events = append(l.events, s)
}

func (l *eventLog) snapshot() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]string(nil), l.events...)
}

func (l *eventLog) handler() WatchHandler[*mcsv1.ServiceImport] {
	return WatchHandler[*mcsv1.ServiceImport]{
		SnapshotStart: func() { l.add("start") },
		Upsert:        func(r *mcsv1.ServiceImport) { l.add("upsert:" + r.GetName()) },
		Delete:        func(r *mcsv1.ServiceImport) { l.add("delete:" + r.GetName()) },
		Synced:        func(string) { l.add("synced") },
		Bookmark: func(string) {
			l.mu.Lock()
			defer l.mu.Unlock()
			l.bookmarks++
		},
	}
}

func (l *eventLog) onResync() {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.resyncs++
}

func (l *eventLog) counts() (bookmarks, resyncs int) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.bookmarks, l.resyncs
}

func (l *eventLog) waitFor(t *testing.T, want ...string) {
	t.Helper()
	require.Eventually(t, func() bool { return assert.ObjectsAreEqual(want, l.snapshot()) }, 10*time.Second, 5*time.Millisecond,
		"want %v", want)
}

func runLoop[T any](t *testing.T, loop *WatchLoop[T]) (stop func()) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- loop.Run(ctx) }()
	var once sync.Once
	stop = func() {
		once.Do(func() {
			cancel()
			assert.NoError(t, <-done)
		})
	}
	t.Cleanup(stop)
	return stop
}

func TestWatchLoopSnapshotDeltasResumeAndResync(t *testing.T) {
	f := startFake(t, "w", fakeregistry.WithBookmarkInterval(20*time.Millisecond), fakeregistry.WithHistory(2))
	f.srv.PutExport(testClusterset, "a", "llm", "r1", exportSpec(50071), nil)
	dns := newFakeResolver(map[string]string{"registry.test": "127.0.0.1"})
	reg := prometheus.NewRegistry()
	c := newTestClient(t, Config{URLs: []string{f.url("registry.test")}, Resolver: dns, Registerer: reg})

	log := &eventLog{}
	loop := NewWatchLoop(c, WatchLoopConfig[*mcsv1.ServiceImport]{
		Name:     WatchServiceImports,
		Watch:    c.ServiceImports().Watcher(&mcsv1.WatchServiceImportsRequest{Clusterset: testClusterset}),
		Handler:  log.handler(),
		OnResync: log.onResync,
	})
	runLoop(t, loop)
	log.waitFor(t, "start", "upsert:r1", "synced")
	assert.True(t, loop.Synced())
	require.Eventually(t, func() bool { b, _ := log.counts(); return b > 0 }, 5*time.Second, 5*time.Millisecond)
	require.Eventually(t, func() bool { return loop.Revision() == f.srv.Revision() }, 5*time.Second, 5*time.Millisecond)

	f.srv.PutExport(testClusterset, "a", "llm", "r2", exportSpec(50071), nil)
	log.waitFor(t, "start", "upsert:r1", "synced", "upsert:r2")
	assert.Equal(t, f.srv.Revision(), loop.Revision())

	// A closed stream resumes: no new snapshot, deltas continue.
	f.srv.CloseWatches(nil)
	require.Eventually(t, func() bool {
		v, _ := metricValue(t, reg, MetricWatchReconnectsTotal, map[string]string{"watch": WatchServiceImports})
		return v == 1 && f.srv.WatcherCount() == 1
	}, 5*time.Second, 5*time.Millisecond)
	f.srv.DeleteExport(testClusterset, "a", "llm", "r1")
	log.waitFor(t, "start", "upsert:r1", "synced", "upsert:r2", "delete:r1")
	_, resyncs := log.counts()
	assert.Zero(t, resyncs)

	// Changes beyond the history while disconnected: a new snapshot and a
	// resync.
	f.stop()
	for i := range 3 {
		f.srv.PutExport(testClusterset, "b", "llm", fmt.Sprintf("x%d", i), exportSpec(50071), nil)
	}
	f.restart(t)
	log.waitFor(t, "start", "upsert:r1", "synced", "upsert:r2", "delete:r1",
		"start", "upsert:r2", "upsert:x0", "upsert:x1", "upsert:x2", "synced")
	_, resyncs = log.counts()
	assert.Equal(t, 1, resyncs)
	assert.True(t, loop.Synced())

	assert.Error(t, loop.Run(context.Background()), "a loop runs once at a time")
}

func TestWatchLoopIdleTimeoutAndOpenFailures(t *testing.T) {
	f := startFake(t, "w", fakeregistry.WithBookmarkInterval(0))
	dns := newFakeResolver(map[string]string{"registry.test": "127.0.0.1"})
	reg := prometheus.NewRegistry()
	c := newTestClient(t, Config{URLs: []string{f.url("registry.test")}, Resolver: dns, Registerer: reg})
	f.srv.FailNext("WatchServiceImports", codes.Unavailable)
	f.srv.FailNext("WatchServiceImports", codes.Internal)

	log := &eventLog{}
	loop := NewWatchLoop(c, WatchLoopConfig[*mcsv1.ServiceImport]{
		Name:        "idle",
		Watch:       c.ServiceImports().Watcher(&mcsv1.WatchServiceImportsRequest{Clusterset: testClusterset}),
		Handler:     log.handler(),
		IdleTimeout: 50 * time.Millisecond,
		Backoff:     Backoff{Initial: time.Millisecond, Max: 5 * time.Millisecond},
	})
	runLoop(t, loop)
	log.waitFor(t, "start", "synced")
	require.Eventually(t, func() bool { return f.srv.CallCount("WatchServiceImports") >= 5 }, 5*time.Second, 5*time.Millisecond,
		"an idle stream is reopened")
	assert.Equal(t, []string{"start", "synced"}, log.snapshot(), "reopened streams resume without a snapshot")
	v, _ := metricValue(t, reg, MetricWatchReconnectsTotal, map[string]string{"watch": "idle"})
	assert.GreaterOrEqual(t, v, 4.0)
}

// scriptedStream replays events, then fails with err.
type scriptedStream struct {
	events []Event[string]
	err    error
}

func (s *scriptedStream) Recv() (Event[string], error) {
	if len(s.events) == 0 {
		return Event[string]{}, s.err
	}
	ev := s.events[0]
	s.events = s.events[1:]
	return ev, nil
}

func TestWatchLoopRevisionAcrossBrokenSnapshot(t *testing.T) {
	c := newTestClient(t, Config{URLs: []string{"http://127.0.0.1:1"}, PrimaryRetryInterval: -1})
	ev := func(typ mcsv1.EventType, obj, rev string) Event[string] {
		return Event[string]{Type: typ, Object: obj, Revision: rev}
	}
	const (
		start    = mcsv1.EventType_EVENT_TYPE_SNAPSHOT_START
		upsert   = mcsv1.EventType_EVENT_TYPE_UPSERT
		del      = mcsv1.EventType_EVENT_TYPE_DELETE
		end      = mcsv1.EventType_EVENT_TYPE_SNAPSHOT_END
		bookmark = mcsv1.EventType_EVENT_TYPE_BOOKMARK
	)
	scripts := []*scriptedStream{
		// Resume from r0 fails into a snapshot that breaks.
		{events: []Event[string]{ev(start, "", ""), ev(upsert, "a", "")}, err: io.EOF},
		// A full snapshot, a delta and a bookmark.
		{events: []Event[string]{
			ev(start, "", ""), ev(upsert, "a", ""), ev(end, "", "r5"),
			ev(del, "a", "r6"), ev(bookmark, "", "r7"), ev(mcsv1.EventType(99), "", "r8"),
		}, err: errors.New("broken")},
		// The server could not resume: a resync.
		{events: []Event[string]{ev(start, "", ""), ev(end, "", "r9")}, err: io.EOF},
	}
	var mu sync.Mutex
	var revisions, events []string
	var opens atomic.Int32
	var resyncs atomic.Int32
	loop := NewWatchLoop(c, WatchLoopConfig[string]{
		Name:     "scripted",
		Revision: "r0",
		Backoff:  Backoff{Initial: time.Millisecond, Max: time.Millisecond},
		Watch: func(ctx context.Context, revision string) (WatchStream[string], error) {
			n := int(opens.Add(1)) - 1
			mu.Lock()
			revisions = append(revisions, revision)
			mu.Unlock()
			if n >= len(scripts) {
				<-ctx.Done()
				return nil, ctx.Err()
			}
			return scripts[n], nil
		},
		Handler: WatchHandler[string]{
			SnapshotStart: func() { mu.Lock(); events = append(events, "start"); mu.Unlock() },
			Upsert:        func(s string) { mu.Lock(); events = append(events, "upsert:"+s); mu.Unlock() },
			Delete:        func(s string) { mu.Lock(); events = append(events, "delete:"+s); mu.Unlock() },
			Synced:        func(r string) { mu.Lock(); events = append(events, "synced:"+r); mu.Unlock() },
			Bookmark:      func(r string) { mu.Lock(); events = append(events, "bookmark:"+r); mu.Unlock() },
		},
		OnResync: func() { resyncs.Add(1) },
	})
	runLoop(t, loop)
	require.Eventually(t, func() bool { return opens.Load() == 4 }, 5*time.Second, time.Millisecond)
	mu.Lock()
	defer mu.Unlock()
	assert.Equal(t, []string{"r0", "", "r7", "r9"}, revisions)
	assert.Equal(t, []string{
		"start", "upsert:a",
		"start", "upsert:a", "synced:r5", "delete:a", "bookmark:r7",
		"start", "synced:r9",
	}, events)
	assert.Equal(t, int32(1), resyncs.Load(), "only a snapshot after the first sync is a resync")
}
