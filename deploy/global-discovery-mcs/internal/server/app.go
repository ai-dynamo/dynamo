// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package server

import (
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
	"go.uber.org/zap"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/api"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/auth"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/clock"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/config"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/metrics"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store/instrumented"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/version"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/view"
)

// AppOptions are the dependencies of an App that do not come from the
// configuration.
type AppOptions struct {
	// Store is the backend; App wraps it with metrics and spans.
	Store  store.Store
	Logger *zap.Logger
	// Registry defaults to a new registry with the Go and process
	// collectors.
	Registry *prometheus.Registry
	// Clock defaults to the system clock.
	Clock         clock.Clock
	Authenticator auth.Authenticator
}

// App is a fully wired mcs-registry: store, view, hub, refresher, compactor,
// API and listeners.
type App struct {
	Server   *Server
	Store    store.Store
	View     *view.View
	Hub      *view.Hub
	Metrics  *metrics.Metrics
	Registry *prometheus.Registry
}

// NewApp wires an mcs-registry from its configuration and binds its listeners.
func NewApp(cfg *config.Config, opts AppOptions) (*App, error) {
	if opts.Logger == nil {
		opts.Logger = zap.NewNop()
	}
	if opts.Clock == nil {
		opts.Clock = clock.Real()
	}
	reg := opts.Registry
	if reg == nil {
		reg = prometheus.NewRegistry()
		reg.MustRegister(collectors.NewGoCollector(), collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}))
	}
	m := metrics.New(reg)
	m.SetBuildInfo(version.Version, version.GitHash)
	st := instrumented.New(opts.Store, m)
	log := opts.Logger

	v := view.New(view.Options{
		Clock:            opts.Clock,
		LeaseSkew:        cfg.LeaseSkew,
		WatchHistory:     cfg.WatchHistory,
		LocalWriteWindow: cfg.RefreshInterval,
		Metrics:          m,
		Logger:           log.With(zap.String("component", "view")),
	})
	hub := view.NewHub(v, view.HubOptions{
		Clock:            opts.Clock,
		BookmarkInterval: cfg.BookmarkInterval,
		QueueSize:        cfg.WatchQueueSize,
		MaxWatchers:      cfg.MaxWatchers,
		Metrics:          m,
		Logger:           log.With(zap.String("component", "hub")),
	})
	refresher := view.NewRefresher(v, st, view.RefresherOptions{
		Clock:             opts.Clock,
		RefreshInterval:   cfg.RefreshInterval,
		LeaseEvalInterval: cfg.LeaseEvalInterval,
		DiscoveryInterval: cfg.ClusterSetDiscoveryInterval,
		Metrics:           m,
		Logger:            log.With(zap.String("component", "refresher")),
	})
	compactor := view.NewCompactor(v, st, view.CompactorOptions{
		Clock:     opts.Clock,
		Retention: cfg.Retention,
		Interval:  cfg.CompactionInterval,
		LeaseSkew: cfg.LeaseSkew,
		Metrics:   m,
		Logger:    log.With(zap.String("component", "compactor")),
	})
	endpoints := make([]*mcsv1.ServerEndpoint, 0, len(cfg.AdvertiseEndpoints))
	for _, ep := range cfg.AdvertiseEndpoints {
		endpoints = append(endpoints, &mcsv1.ServerEndpoint{Region: ep.Region, GrpcUrl: ep.GRPCURL, HttpUrl: ep.HTTPURL, Primary: ep.Primary})
	}
	apiServer := api.New(api.Options{
		Store:     st,
		View:      v,
		Hub:       hub,
		Clock:     opts.Clock,
		Region:    cfg.Region,
		Version:   version.String(),
		Endpoints: endpoints,
		Logger:    log.With(zap.String("component", "api")),
	})
	srv, err := New(Options{
		Config:        cfg,
		Store:         st,
		View:          v,
		Hub:           hub,
		Refresher:     refresher,
		Compactor:     compactor,
		API:           apiServer,
		Metrics:       m,
		Registry:      reg,
		Authenticator: opts.Authenticator,
		Clock:         opts.Clock,
		Logger:        log,
	})
	if err != nil {
		return nil, fmt.Errorf("build server: %w", err)
	}
	return &App{Server: srv, Store: st, View: v, Hub: hub, Metrics: m, Registry: reg}, nil
}
