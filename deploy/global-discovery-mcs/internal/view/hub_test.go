// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package view

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/metrics"
)

type watcher struct {
	events chan WatchEvent
	done   chan error
	cancel context.CancelFunc
}

func startWatch(t *testing.T, h *Hub, f Filter, resume string) *watcher {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	w := &watcher{events: make(chan WatchEvent), done: make(chan error, 1), cancel: cancel}
	go func() {
		w.done <- h.Watch(ctx, cs, f, resume, func(ev WatchEvent) error {
			select {
			case w.events <- ev:
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		})
	}()
	t.Cleanup(cancel)
	return w
}

func (w *watcher) next(t *testing.T) WatchEvent {
	t.Helper()
	select {
	case ev := <-w.events:
		return ev
	case err := <-w.done:
		t.Fatalf("watch ended: %v", err)
	case <-time.After(5 * time.Second):
		t.Fatal("no watch event")
	}
	return WatchEvent{}
}

func (w *watcher) snapshot(t *testing.T) ([]WatchEvent, string) {
	t.Helper()
	ev := w.next(t)
	require.Equal(t, mcsv1.EventType_EVENT_TYPE_SNAPSHOT_START, ev.Type)
	var objs []WatchEvent
	for {
		ev = w.next(t)
		if ev.Type == mcsv1.EventType_EVENT_TYPE_SNAPSHOT_END {
			return objs, ev.Revision
		}
		require.Equal(t, mcsv1.EventType_EVENT_TYPE_UPSERT, ev.Type)
		objs = append(objs, ev)
	}
}

func (w *watcher) end(t *testing.T) error {
	t.Helper()
	select {
	case err := <-w.done:
		return err
	case <-time.After(5 * time.Second):
		t.Fatal("watch did not end")
	}
	return nil
}

func newHub(f *fixture, opts HubOptions) *Hub {
	opts.Metrics = f.metrics
	return NewHub(f.view, opts)
}

func TestWatchSnapshotThenDeltas(t *testing.T) {
	f := newFixture(t)
	f.view.Replace(baseSnapshot(), t0)
	h := newHub(f, HubOptions{})
	w := startWatch(t, h, ImportFilter{Endpoints: true}, "")
	objs, rev := w.snapshot(t)
	require.Len(t, objs, 1)
	assert.Equal(t, "e1:3", rev)
	assert.NotEmpty(t, objs[0].Object.(*mcsv1.ServiceImport).GetEndpoints()[0].GetEndpoints())

	f.view.ApplyEndpointSet(endpointSet("a", "ns", "web", 9, true))
	ev := w.next(t)
	assert.Equal(t, mcsv1.EventType_EVENT_TYPE_UPSERT, ev.Type)
	assert.Equal(t, "e1:4", ev.Revision)
	assert.Equal(t, uint64(9), ev.Object.(*mcsv1.ServiceImport).GetEndpoints()[0].GetGeneration())

	// Changes of other kinds are not delivered.
	f.view.ApplyCluster(clusterRec("c", "eu-west-1", t0))
	f.view.ApplyExportDelete(exportKey("a", "ns", "web"))
	f.view.ApplyExportDelete(exportKey("b", "ns", "web"))
	ev = w.next(t)
	assert.Equal(t, mcsv1.EventType_EVENT_TYPE_UPSERT, ev.Type, "a's export left; b remains")
	ev = w.next(t)
	assert.Equal(t, mcsv1.EventType_EVENT_TYPE_DELETE, ev.Type)
	assert.Equal(t, "web", ev.Object.(*mcsv1.ServiceImport).GetName())
	assert.Equal(t, "e1:7", ev.Revision)

	assert.Equal(t, 1, h.Watchers())
	w.cancel()
	assert.ErrorIs(t, w.end(t), context.Canceled)
	assert.Eventually(t, func() bool { return h.Watchers() == 0 }, 5*time.Second, 10*time.Millisecond)
	assert.Equal(t, 1.0, testutilValue(f.metrics.WatchEventsSent.WithLabelValues(metrics.KindServiceImport, "delete")))
}

func TestWatchResume(t *testing.T) {
	f := newFixture(t)
	f.view.Replace(baseSnapshot(), t0)
	h := newHub(f, HubOptions{})
	w := startWatch(t, h, ClusterFilter{}, "")
	_, rev := w.snapshot(t)
	w.cancel()
	_ = w.end(t)

	f.view.ApplyCluster(clusterRec("c", "eu-west-1", t0))
	f.view.ApplyExport(exportRec("c", "ns", "x", 1, t0, nil, nil, httpPort))
	f.view.ApplyCluster(clusterRec("d", "eu-west-1", t0))

	// Same epoch, in the buffer: only the missed cluster deltas.
	w = startWatch(t, h, ClusterFilter{}, rev)
	ev := w.next(t)
	assert.Equal(t, mcsv1.EventType_EVENT_TYPE_UPSERT, ev.Type)
	assert.Equal(t, "c", ev.Object.(*mcsv1.Cluster).GetCluster())
	ev = w.next(t)
	assert.Equal(t, "d", ev.Object.(*mcsv1.Cluster).GetCluster())
	w.cancel()

	// Another epoch: a fresh snapshot.
	w = startWatch(t, h, ClusterFilter{}, "other-epoch:3")
	objs, _ := w.snapshot(t)
	assert.Len(t, objs, 4)
	w.cancel()
	assert.Equal(t, 1.0, testutilValue(f.metrics.WatchResyncs.WithLabelValues(metrics.KindCluster, metrics.ResyncRevisionUnknown)))

	// A revision from the future or a malformed one: a snapshot.
	for _, bad := range []string{"e1:999", "garbage"} {
		w = startWatch(t, h, ClusterFilter{}, bad)
		_, _ = w.snapshot(t)
		w.cancel()
	}

	// The current revision resumes with nothing to replay.
	w = startWatch(t, h, ClusterFilter{}, f.view.Revision(cs))
	f.view.ApplyCluster(clusterRec("e", "eu-west-1", t0))
	ev = w.next(t)
	assert.Equal(t, "e", ev.Object.(*mcsv1.Cluster).GetCluster())
	w.cancel()
}

func TestWatchResumeExpired(t *testing.T) {
	f := newFixture(t)
	f.view = New(Options{Clock: f.clock, LeaseSkew: 2 * time.Second, WatchHistory: 2, Epoch: "e1", Metrics: f.metrics})
	f.view.Replace(baseSnapshot(), t0)
	rev := f.view.Revision(cs)
	for i := range 3 {
		f.view.ApplyCluster(clusterRec(fmt.Sprintf("c%d", i), "eu-west-1", t0))
	}
	h := newHub(f, HubOptions{})
	w := startWatch(t, h, ClusterFilter{}, rev)
	objs, _ := w.snapshot(t)
	assert.Len(t, objs, 5)
	assert.Equal(t, 1.0, testutilValue(f.metrics.WatchResyncs.WithLabelValues(metrics.KindCluster, metrics.ResyncRevisionExpired)))
}

func TestWatchLaggedResync(t *testing.T) {
	f := newFixture(t)
	f.view.Replace(baseSnapshot(), t0)
	h := newHub(f, HubOptions{QueueSize: 2})
	w := startWatch(t, h, ClusterFilter{}, "")
	_, _ = w.snapshot(t)

	// The watcher does not read while six changes are published. It takes
	// the first change and blocks sending it; the next five overflow its
	// queue. Waiting for the first take keeps the watcher from waking up,
	// and snapshotting, while the changes are still being published.
	f.view.ApplyCluster(clusterRec("c0", "eu-west-1", t0))
	require.Eventually(t, func() bool {
		set := f.view.get(cs)
		set.mu.Lock()
		defer set.mu.Unlock()
		for sub := range set.subs {
			return len(sub.queue) == 0
		}
		return false
	}, 5*time.Second, time.Millisecond)
	for i := 1; i < 6; i++ {
		f.view.ApplyCluster(clusterRec(fmt.Sprintf("c%d", i), "eu-west-1", t0))
	}
	for {
		ev := w.next(t)
		if ev.Type == mcsv1.EventType_EVENT_TYPE_SNAPSHOT_START {
			break
		}
		require.Equal(t, mcsv1.EventType_EVENT_TYPE_UPSERT, ev.Type)
	}
	var objs []WatchEvent
	for {
		ev := w.next(t)
		if ev.Type == mcsv1.EventType_EVENT_TYPE_SNAPSHOT_END {
			assert.Equal(t, f.view.Revision(cs), ev.Revision)
			break
		}
		objs = append(objs, ev)
	}
	assert.Len(t, objs, 8, "the resync snapshot holds every cluster")
	assert.Equal(t, 1.0, testutilValue(f.metrics.WatchResyncs.WithLabelValues(metrics.KindCluster, metrics.ResyncLagged)))

	// Deltas continue after the resync.
	f.view.ApplyCluster(clusterRec("z", "eu-west-1", t0))
	assert.Equal(t, "z", w.next(t).Object.(*mcsv1.Cluster).GetCluster())
}

func TestWatchBookmarks(t *testing.T) {
	f := newFixture(t)
	f.view.Replace(baseSnapshot(), t0)
	h := newHub(f, HubOptions{BookmarkInterval: 5 * time.Second})
	w := startWatch(t, h, ImportFilter{Namespace: "none"}, "")
	objs, rev := w.snapshot(t)
	assert.Empty(t, objs)
	f.clock.BlockUntil(1)

	f.clock.Advance(5 * time.Second)
	ev := w.next(t)
	assert.Equal(t, mcsv1.EventType_EVENT_TYPE_BOOKMARK, ev.Type)
	assert.Equal(t, rev, ev.Revision)

	// Filtered-out changes advance the bookmark revision.
	f.view.ApplyCluster(clusterRec("c", "eu-west-1", t0))
	f.clock.Advance(5 * time.Second)
	ev = w.next(t)
	assert.Equal(t, mcsv1.EventType_EVENT_TYPE_BOOKMARK, ev.Type)
	assert.Equal(t, f.view.Revision(cs), ev.Revision)
}

func TestWatchImportFilterExit(t *testing.T) {
	f := newFixture(t)
	f.view.ApplyCluster(clusterRec("a", "us-east-1", t0))
	rec := exportRec("a", "ns", "web", 1, t0, map[string]string{"tier": "gold"}, nil, httpPort)
	f.view.ApplyExport(rec)
	h := newHub(f, HubOptions{})
	gold := startWatch(t, h, ImportFilter{Selector: map[string]string{"tier": "gold"}}, "")
	objs, _ := gold.snapshot(t)
	require.Len(t, objs, 1)
	all := startWatch(t, h, ImportFilter{}, "")
	_, _ = all.snapshot(t)

	t.Log("A relabelled import leaves the filtered watch as a DELETE")
	rec.Spec.Labels = map[string]string{"tier": "silver"}
	rec.Generation = 2
	f.view.ApplyExport(rec)
	ev := gold.next(t)
	assert.Equal(t, mcsv1.EventType_EVENT_TYPE_DELETE, ev.Type, "the import left the filter")
	assert.Equal(t, "silver", ev.Object.(*mcsv1.ServiceImport).GetLabels()["tier"], "the DELETE carries the latest state")
	ev = all.next(t)
	assert.Equal(t, mcsv1.EventType_EVENT_TYPE_UPSERT, ev.Type)

	t.Log("A withdrawn export removes the import")
	f.view.ApplyExportDelete(rec.Key)
	ev = all.next(t)
	assert.Equal(t, mcsv1.EventType_EVENT_TYPE_DELETE, ev.Type)
	assert.Equal(t, "web", ev.Object.(*mcsv1.ServiceImport).GetName())
}

func TestWatchLimitsAndShutdown(t *testing.T) {
	f := newFixture(t)
	h := newHub(f, HubOptions{MaxWatchers: 1})
	w := startWatch(t, h, ClusterFilter{}, "")
	_, _ = w.snapshot(t)
	err := h.Watch(context.Background(), cs, ClusterFilter{}, "", func(WatchEvent) error { return nil })
	assert.ErrorIs(t, err, ErrTooManyWatchers)
	assert.Equal(t, 1.0, testutilValue(f.metrics.WatchRejected.WithLabelValues(metrics.RejectTooManyWatchers)))

	h.Close()
	h.Close()
	assert.ErrorIs(t, w.end(t), ErrShuttingDown)
	err = h.Watch(context.Background(), cs, ClusterFilter{}, "", func(WatchEvent) error { return nil })
	assert.ErrorIs(t, err, ErrShuttingDown)
}

func TestWatchSendError(t *testing.T) {
	f := newFixture(t)
	h := newHub(f, HubOptions{})
	boom := errors.New("boom")
	err := h.Watch(context.Background(), cs, ClusterFilter{}, "", func(WatchEvent) error { return boom })
	assert.ErrorIs(t, err, boom)
}

func TestRing(t *testing.T) {
	r := newRing(3)
	assert.True(t, r.covers(0, 0))
	assert.False(t, r.covers(0, 1))
	for i := uint64(1); i <= 5; i++ {
		r.push(Event{Seq: i})
	}
	assert.True(t, r.covers(2, 5))
	assert.False(t, r.covers(1, 5))
	assert.False(t, r.covers(6, 5))
	assert.Len(t, r.since(2), 3)
	assert.Len(t, r.since(4), 1)
	empty := newRing(0)
	empty.push(Event{Seq: 1})
	assert.False(t, empty.covers(0, 1))
}
