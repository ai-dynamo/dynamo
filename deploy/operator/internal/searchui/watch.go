/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package searchui

import (
	"context"
	"fmt"

	"github.com/go-logr/logr"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/rest"
	toolscache "k8s.io/client-go/tools/cache"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// Watcher feeds Request, Run and Candidate events into the Index, starts log streaming
// for Runs with a Job, and notifies browser subscribers.
type Watcher struct {
	cache  cache.Cache
	index  *Index
	logs   *LogManager
	notify func()
	log    logr.Logger
}

func NewWatcher(restConfig *rest.Config, namespace string, index *Index, logs *LogManager, notify func(), log logr.Logger) (*Watcher, error) {
	opts := cache.Options{}
	if namespace != "" {
		opts.DefaultNamespaces = map[string]cache.Config{namespace: {}}
	}
	c, err := cache.New(restConfig, opts)
	if err != nil {
		return nil, fmt.Errorf("create cache: %w", err)
	}
	return &Watcher{cache: c, index: index, logs: logs, notify: notify, log: log}, nil
}

// Start registers the informers, starts the cache and blocks until ctx is done.
func (w *Watcher) Start(ctx context.Context) error {
	handlers := map[schema.GroupVersionKind]toolscache.ResourceEventHandler{
		gvkRequest:   w.requestHandler(),
		gvkRun:       w.runHandler(ctx),
		gvkCandidate: w.candidateHandler(ctx),
	}
	for gvk, handler := range handlers {
		obj := &unstructured.Unstructured{}
		obj.SetGroupVersionKind(gvk)
		informer, err := w.cache.GetInformer(ctx, obj)
		if err != nil {
			return fmt.Errorf("informer for %s: %w", gvk.Kind, err)
		}
		if _, err := informer.AddEventHandler(handler); err != nil {
			return fmt.Errorf("event handler for %s: %w", gvk.Kind, err)
		}
	}
	errCh := make(chan error, 1)
	go func() { errCh <- w.cache.Start(ctx) }()
	if !w.cache.WaitForCacheSync(ctx) {
		return fmt.Errorf("cache sync: %w", ctx.Err())
	}
	w.log.Info("informers synced")
	return <-errCh
}

// Reader exposes the cache for ad-hoc reads.
func (w *Watcher) Reader() client.Reader { return w.cache }

func asUnstructured(obj any) *unstructured.Unstructured {
	if tomb, ok := obj.(toolscache.DeletedFinalStateUnknown); ok {
		obj = tomb.Obj
	}
	u, _ := obj.(*unstructured.Unstructured)
	return u
}

func (w *Watcher) requestHandler() toolscache.ResourceEventHandler {
	upsert := func(obj any) {
		if u := asUnstructured(obj); u != nil {
			w.index.UpsertRequest(u.DeepCopy().Object)
			w.notify()
		}
	}
	return toolscache.ResourceEventHandlerFuncs{
		AddFunc:    upsert,
		UpdateFunc: func(_, obj any) { upsert(obj) },
		DeleteFunc: func(obj any) {
			if u := asUnstructured(obj); u != nil {
				w.index.DeleteRequest(u.GetNamespace(), u.GetName())
				w.notify()
			}
		},
	}
}

func (w *Watcher) runHandler(ctx context.Context) toolscache.ResourceEventHandler {
	upsert := func(obj any) {
		u := asUnstructured(obj)
		if u == nil {
			return
		}
		if err := w.index.UpsertRun(ctx, u.DeepCopy().Object); err != nil {
			w.log.Error(err, "record run", "run", u.GetName())
		}
		view, _ := w.index.RunByName(u.GetNamespace(), u.GetName())
		terminal := view.Phase == "Succeeded" || view.Phase == "Failed"
		w.logs.Ensure(ctx, u.GetNamespace(), string(u.GetUID()), view.JobName, terminal)
		w.notify()
	}
	return toolscache.ResourceEventHandlerFuncs{
		AddFunc:    upsert,
		UpdateFunc: func(_, obj any) { upsert(obj) },
		DeleteFunc: func(obj any) {
			if u := asUnstructured(obj); u != nil {
				if err := w.index.DeleteRun(ctx, string(u.GetUID())); err != nil {
					w.log.Error(err, "record run deletion", "run", u.GetName())
				}
				w.notify()
			}
		},
	}
}

func (w *Watcher) candidateHandler(ctx context.Context) toolscache.ResourceEventHandler {
	upsert := func(obj any) {
		if u := asUnstructured(obj); u != nil {
			if err := w.index.UpsertCandidate(ctx, u.DeepCopy().Object); err != nil {
				w.log.Error(err, "record candidate", "candidate", u.GetName())
			}
			w.notify()
		}
	}
	return toolscache.ResourceEventHandlerFuncs{
		AddFunc:    upsert,
		UpdateFunc: func(_, obj any) { upsert(obj) },
		DeleteFunc: func(obj any) {
			if u := asUnstructured(obj); u != nil {
				if err := w.index.DeleteCandidate(ctx, u.GetNamespace(), u.GetName()); err != nil {
					w.log.Error(err, "record candidate deletion", "candidate", u.GetName())
				}
				w.notify()
			}
		},
	}
}
