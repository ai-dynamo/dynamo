/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package history is the search-ui's durable, UI-local record of everything it observed:
// the latest copy of every Run, every Run status change, every Candidate that was ever
// published (including ones the publisher later removed from the bounded projection),
// and the container logs of each Run's Job. Kubernetes object status stays authoritative
// for what exists now; this store answers what happened.
package history

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path"
	"sort"
	"strings"
	"time"
)

// ErrNotFound reports a key that no backend holds.
var ErrNotFound = errors.New("not found")

// RunEnvelope is the stored form of one DynamoGraphDeploymentRun.
type RunEnvelope struct {
	Namespace string         `json:"namespace"`
	Name      string         `json:"name"`
	UID       string         `json:"uid"`
	FirstSeen time.Time      `json:"firstSeen"`
	LastSeen  time.Time      `json:"lastSeen"`
	DeletedAt *time.Time     `json:"deletedAt,omitempty"`
	Object    map[string]any `json:"object"`
}

// CandidateEnvelope is the stored form of one DynamoGraphDeploymentCandidate.
type CandidateEnvelope struct {
	Namespace string         `json:"namespace"`
	Name      string         `json:"name"`
	RunUID    string         `json:"runUID"`
	FirstSeen time.Time      `json:"firstSeen"`
	LastSeen  time.Time      `json:"lastSeen"`
	DeletedAt *time.Time     `json:"deletedAt,omitempty"`
	Object    map[string]any `json:"object"`
}

// StatusSample is one observed Run status at a point in time.
type StatusSample struct {
	Time   time.Time      `json:"time"`
	Status map[string]any `json:"status"`
}

// Store is the durable history backend. Every method is safe for concurrent use.
type Store interface {
	PutRun(ctx context.Context, env RunEnvelope) error
	PutRunSample(ctx context.Context, namespace, uid string, sample StatusSample) error
	PutCandidate(ctx context.Context, env CandidateEnvelope) error
	PutLog(ctx context.Context, namespace, uid, container string, data []byte) error

	ListRuns(ctx context.Context) ([]RunEnvelope, error)
	ListRunSamples(ctx context.Context, namespace, uid string) ([]StatusSample, error)
	ListCandidates(ctx context.Context, namespace, uid string) ([]CandidateEnvelope, error)
	GetLog(ctx context.Context, namespace, uid, container string) ([]byte, error)
}

// kv is the minimal object interface both backends provide. Keys are '/'-separated
// paths; List returns every key with the prefix.
type kv interface {
	Put(ctx context.Context, key string, data []byte) error
	Get(ctx context.Context, key string) ([]byte, error)
	List(ctx context.Context, prefix string) ([]string, error)
}

// kvStore lays the history out as:
//
//	runs/{namespace}/{uid}/run.json
//	runs/{namespace}/{uid}/status/{timestamp}.json
//	runs/{namespace}/{uid}/candidates/{name}.json
//	runs/{namespace}/{uid}/logs/{container}.log
type kvStore struct {
	kv kv
}

const sampleTimeLayout = "20060102T150405.000000000Z"

func runKey(namespace, uid string) string {
	return path.Join("runs", namespace, uid)
}

func (s *kvStore) PutRun(ctx context.Context, env RunEnvelope) error {
	if env.Namespace == "" || env.UID == "" {
		return errors.New("run envelope needs namespace and uid")
	}
	data, err := json.Marshal(env)
	if err != nil {
		return err
	}
	return s.kv.Put(ctx, path.Join(runKey(env.Namespace, env.UID), "run.json"), data)
}

func (s *kvStore) PutRunSample(ctx context.Context, namespace, uid string, sample StatusSample) error {
	data, err := json.Marshal(sample)
	if err != nil {
		return err
	}
	name := sample.Time.UTC().Format(sampleTimeLayout) + ".json"
	return s.kv.Put(ctx, path.Join(runKey(namespace, uid), "status", name), data)
}

func (s *kvStore) PutCandidate(ctx context.Context, env CandidateEnvelope) error {
	if env.Namespace == "" || env.RunUID == "" || env.Name == "" {
		return errors.New("candidate envelope needs namespace, runUID and name")
	}
	data, err := json.Marshal(env)
	if err != nil {
		return err
	}
	return s.kv.Put(ctx, path.Join(runKey(env.Namespace, env.RunUID), "candidates", env.Name+".json"), data)
}

func (s *kvStore) PutLog(ctx context.Context, namespace, uid, container string, data []byte) error {
	if strings.ContainsAny(container, "/\\") {
		return fmt.Errorf("invalid container name %q", container)
	}
	return s.kv.Put(ctx, path.Join(runKey(namespace, uid), "logs", container+".log"), data)
}

func (s *kvStore) ListRuns(ctx context.Context) ([]RunEnvelope, error) {
	keys, err := s.kv.List(ctx, "runs/")
	if err != nil {
		return nil, err
	}
	var runs []RunEnvelope
	for _, key := range keys {
		if path.Base(key) != "run.json" {
			continue
		}
		data, err := s.kv.Get(ctx, key)
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", key, err)
		}
		var env RunEnvelope
		if err := json.Unmarshal(data, &env); err != nil {
			return nil, fmt.Errorf("decode %s: %w", key, err)
		}
		runs = append(runs, env)
	}
	sort.Slice(runs, func(i, j int) bool { return runs[i].FirstSeen.Before(runs[j].FirstSeen) })
	return runs, nil
}

func (s *kvStore) ListRunSamples(ctx context.Context, namespace, uid string) ([]StatusSample, error) {
	prefix := path.Join(runKey(namespace, uid), "status") + "/"
	keys, err := s.kv.List(ctx, prefix)
	if err != nil {
		return nil, err
	}
	sort.Strings(keys)
	samples := make([]StatusSample, 0, len(keys))
	for _, key := range keys {
		data, err := s.kv.Get(ctx, key)
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", key, err)
		}
		var sample StatusSample
		if err := json.Unmarshal(data, &sample); err != nil {
			return nil, fmt.Errorf("decode %s: %w", key, err)
		}
		samples = append(samples, sample)
	}
	return samples, nil
}

func (s *kvStore) ListCandidates(ctx context.Context, namespace, uid string) ([]CandidateEnvelope, error) {
	prefix := path.Join(runKey(namespace, uid), "candidates") + "/"
	keys, err := s.kv.List(ctx, prefix)
	if err != nil {
		return nil, err
	}
	var out []CandidateEnvelope
	for _, key := range keys {
		data, err := s.kv.Get(ctx, key)
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", key, err)
		}
		var env CandidateEnvelope
		if err := json.Unmarshal(data, &env); err != nil {
			return nil, fmt.Errorf("decode %s: %w", key, err)
		}
		out = append(out, env)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].FirstSeen.Before(out[j].FirstSeen) })
	return out, nil
}

func (s *kvStore) GetLog(ctx context.Context, namespace, uid, container string) ([]byte, error) {
	return s.kv.Get(ctx, path.Join(runKey(namespace, uid), "logs", container+".log"))
}
