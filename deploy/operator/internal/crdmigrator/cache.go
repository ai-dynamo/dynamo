/*
SPDX-FileCopyrightText: Copyright 2024 The Kubernetes Authors.
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Derived from kubernetes-sigs/cluster-api/util/cache/cache.go at v1.13.4,
commit 27f464418c195d96ae2ef4b96f3b6a047ea89310.
*/

package crdmigrator

import (
	"time"

	kcache "k8s.io/client-go/tools/cache"
)

const expirationInterval = 10 * time.Hour

type cacheEntry interface {
	Key() string
}

type ttlCache[E cacheEntry] interface {
	Add(entry E)
	Has(key string) (E, bool)
	Len() int
	DeleteAll()
}

func newTTLCache[E cacheEntry](ttl time.Duration) ttlCache[E] {
	return newTTLCacheWithExpirationInterval[E](ttl, expirationInterval)
}

// newTTLCacheWithExpirationInterval provides a shorter cleanup interval for tests while preserving
// the upstream production default in newTTLCache.
func newTTLCacheWithExpirationInterval[E cacheEntry](ttl, interval time.Duration) ttlCache[E] {
	c := &expiringCache[E]{
		Store: kcache.NewTTLStore(func(obj any) (string, error) {
			return obj.(E).Key(), nil
		}, ttl),
	}
	go func() {
		for {
			// List clears expired entries that the TTL store otherwise expires lazily.
			c.List()
			time.Sleep(interval)
		}
	}()
	return c
}

type expiringCache[E cacheEntry] struct {
	kcache.Store
}

func (c *expiringCache[E]) Add(entry E) {
	_ = c.Store.Add(entry)
}

func (c *expiringCache[E]) Has(key string) (E, bool) {
	item, exists, _ := c.GetByKey(key)
	if exists {
		return item.(E), true
	}
	return *new(E), false
}

func (c *expiringCache[E]) Len() int {
	return len(c.ListKeys())
}

func (c *expiringCache[E]) DeleteAll() {
	_ = c.Store.Replace(nil, "")
}
