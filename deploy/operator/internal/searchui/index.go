/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package searchui

import (
	"context"
	"fmt"
	"reflect"
	"sort"
	"sync"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/searchui/history"
	"k8s.io/apimachinery/pkg/types"
)

// Index is the in-memory union of live Kubernetes objects and stored history. Live
// objects win; stored objects fill in what the cluster has already deleted.
type Index struct {
	mu sync.RWMutex

	requests   map[types.NamespacedName]map[string]any
	runs       map[string]*runEntry // by UID
	runUIDs    map[types.NamespacedName]string
	candidates map[types.NamespacedName]*candidateEntry
	samples    map[string][]history.StatusSample // by run UID

	store history.Store // optional
	now   func() time.Time
}

type runEntry struct {
	env  history.RunEnvelope
	live bool
}

type candidateEntry struct {
	env  history.CandidateEnvelope
	live bool
}

// NewIndex creates an empty index that writes through to store when it is not nil.
func NewIndex(store history.Store) *Index {
	return &Index{
		requests:   make(map[types.NamespacedName]map[string]any),
		runs:       make(map[string]*runEntry),
		runUIDs:    make(map[types.NamespacedName]string),
		candidates: make(map[types.NamespacedName]*candidateEntry),
		samples:    make(map[string][]history.StatusSample),
		store:      store,
		now:        time.Now,
	}
}

// Load fills the index from the store. Live events applied afterwards override it.
func (ix *Index) Load(ctx context.Context) error {
	if ix.store == nil {
		return nil
	}
	runs, err := ix.store.ListRuns(ctx)
	if err != nil {
		return fmt.Errorf("list stored runs: %w", err)
	}
	ix.mu.Lock()
	defer ix.mu.Unlock()
	for _, env := range runs {
		ix.runs[env.UID] = &runEntry{env: env}
		key := types.NamespacedName{Namespace: env.Namespace, Name: env.Name}
		if existing, ok := ix.runUIDs[key]; !ok || ix.runs[existing].env.FirstSeen.Before(env.FirstSeen) {
			ix.runUIDs[key] = env.UID
		}
		candidates, err := ix.store.ListCandidates(ctx, env.Namespace, env.UID)
		if err != nil {
			return fmt.Errorf("list stored candidates of %s: %w", env.Name, err)
		}
		for _, c := range candidates {
			ix.candidates[types.NamespacedName{Namespace: c.Namespace, Name: c.Name}] = &candidateEntry{env: c}
		}
		samples, err := ix.store.ListRunSamples(ctx, env.Namespace, env.UID)
		if err != nil {
			return fmt.Errorf("list stored samples of %s: %w", env.Name, err)
		}
		ix.samples[env.UID] = samples
	}
	return nil
}

func stripVolatile(obj map[string]any) map[string]any {
	if meta, ok := obj["metadata"].(map[string]any); ok {
		delete(meta, "managedFields")
	}
	return obj
}

// --- Requests ---

func (ix *Index) UpsertRequest(obj map[string]any) {
	key := types.NamespacedName{Namespace: str(obj, "metadata", "namespace"), Name: str(obj, "metadata", "name")}
	ix.mu.Lock()
	ix.requests[key] = stripVolatile(obj)
	ix.mu.Unlock()
}

func (ix *Index) DeleteRequest(namespace, name string) {
	ix.mu.Lock()
	delete(ix.requests, types.NamespacedName{Namespace: namespace, Name: name})
	ix.mu.Unlock()
}

// --- Runs ---

// UpsertRun records a live Run. It writes the latest copy and, when the status changed,
// a status sample to the store.
func (ix *Index) UpsertRun(ctx context.Context, obj map[string]any) error {
	obj = stripVolatile(obj)
	uid := str(obj, "metadata", "uid")
	key := types.NamespacedName{Namespace: str(obj, "metadata", "namespace"), Name: str(obj, "metadata", "name")}
	now := ix.now()

	ix.mu.Lock()
	entry, ok := ix.runs[uid]
	if !ok {
		entry = &runEntry{env: history.RunEnvelope{Namespace: key.Namespace, Name: key.Name, UID: uid, FirstSeen: now}}
		ix.runs[uid] = entry
	}
	statusChanged := !reflect.DeepEqual(nestedMap(entry.env.Object, "status"), nestedMap(obj, "status"))
	entry.env.Object = obj
	entry.env.LastSeen = now
	entry.env.DeletedAt = nil
	entry.live = true
	ix.runUIDs[key] = uid
	var sample *history.StatusSample
	if statusChanged {
		sample = &history.StatusSample{Time: now, Status: nestedMap(obj, "status")}
		ix.samples[uid] = append(ix.samples[uid], *sample)
	}
	env := entry.env
	ix.mu.Unlock()

	if ix.store == nil {
		return nil
	}
	if err := ix.store.PutRun(ctx, env); err != nil {
		return err
	}
	if sample != nil {
		return ix.store.PutRunSample(ctx, env.Namespace, uid, *sample)
	}
	return nil
}

// DeleteRun marks a Run as gone from the cluster. Its history stays.
func (ix *Index) DeleteRun(ctx context.Context, uid string) error {
	now := ix.now()
	ix.mu.Lock()
	entry, ok := ix.runs[uid]
	if !ok {
		ix.mu.Unlock()
		return nil
	}
	entry.live = false
	entry.env.DeletedAt = &now
	env := entry.env
	ix.mu.Unlock()
	if ix.store == nil {
		return nil
	}
	return ix.store.PutRun(ctx, env)
}

// --- Candidates ---

func (ix *Index) UpsertCandidate(ctx context.Context, obj map[string]any) error {
	obj = stripVolatile(obj)
	key := types.NamespacedName{Namespace: str(obj, "metadata", "namespace"), Name: str(obj, "metadata", "name")}
	runUID := candidateRunUID(obj)
	now := ix.now()

	ix.mu.Lock()
	entry, ok := ix.candidates[key]
	if !ok {
		entry = &candidateEntry{env: history.CandidateEnvelope{Namespace: key.Namespace, Name: key.Name, RunUID: runUID, FirstSeen: now}}
		ix.candidates[key] = entry
	}
	changed := !reflect.DeepEqual(entry.env.Object, obj)
	entry.env.Object = obj
	entry.env.LastSeen = now
	entry.env.DeletedAt = nil
	entry.live = true
	env := entry.env
	ix.mu.Unlock()

	if ix.store == nil || !changed {
		return nil
	}
	return ix.store.PutCandidate(ctx, env)
}

func (ix *Index) DeleteCandidate(ctx context.Context, namespace, name string) error {
	now := ix.now()
	ix.mu.Lock()
	entry, ok := ix.candidates[types.NamespacedName{Namespace: namespace, Name: name}]
	if !ok {
		ix.mu.Unlock()
		return nil
	}
	entry.live = false
	entry.env.DeletedAt = &now
	env := entry.env
	ix.mu.Unlock()
	if ix.store == nil {
		return nil
	}
	return ix.store.PutCandidate(ctx, env)
}

// --- Reads ---

func (ix *Index) run(namespace, name string) (*runEntry, bool) {
	uid, ok := ix.runUIDs[types.NamespacedName{Namespace: namespace, Name: name}]
	if !ok {
		return nil, false
	}
	entry, ok := ix.runs[uid]
	return entry, ok
}

func (ix *Index) candidatesOfRun(uid string) []*candidateEntry {
	var out []*candidateEntry
	for _, entry := range ix.candidates {
		if entry.env.RunUID == uid {
			out = append(out, entry)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].env.FirstSeen.Before(out[j].env.FirstSeen) })
	return out
}

// Namespaces lists every namespace with a Request or Run, live or retained.
func (ix *Index) Namespaces() []string {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	seen := map[string]struct{}{}
	for key := range ix.requests {
		seen[key.Namespace] = struct{}{}
	}
	for _, entry := range ix.runs {
		seen[entry.env.Namespace] = struct{}{}
	}
	out := make([]string, 0, len(seen))
	for ns := range seen {
		out = append(out, ns)
	}
	sort.Strings(out)
	return out
}
