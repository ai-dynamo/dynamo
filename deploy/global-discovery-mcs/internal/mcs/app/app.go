// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package app wires mcs-controller: the identity, the registry client, the
// cluster lease, the export, auto-export and import controllers, health
// checks and metrics on one controller-runtime manager.
package app

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"sync/atomic"

	"github.com/go-logr/logr"
	"github.com/prometheus/client_golang/prometheus"
	"go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"
	"google.golang.org/grpc"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/manager"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/config"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/export"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/identity"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/importer"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/lease"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/metrics"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/version"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/pkg/client"
)

// EventSource is the component name of the Events the controller emits.
const EventSource = "mcs-controller"

// Options are the dependencies New does not build itself.
type Options struct {
	// Registry receives the controller and registry client metrics; nil uses a
	// new registry.
	Registry prometheus.Registerer
	// Logger is the root logger; zero uses ctrl.Log.
	Logger logr.Logger
	// ManagerOptions adjusts the manager options built from the
	// configuration, for example in tests.
	ManagerOptions func(*ctrl.Options)
	// RegistryDialOptions are appended to the registry client's dial options.
	RegistryDialOptions []grpc.DialOption
	// AgentID identifies this controller instance in the registration;
	// empty uses the pod name from $HOSTNAME.
	AgentID string
	// Client reads and writes the identity at startup, before the caches
	// run; nil builds a direct client from the REST configuration.
	Client ctrlclient.Client
}

// App is a wired controller, ready to Run.
type App struct {
	Config   config.Config
	Identity identity.Identity
	Manager  manager.Manager
	Registry *client.Client
	Metrics  *metrics.Metrics
	Lease    *lease.Lease
	Exports  *export.Reconciler
	Resyncer *export.Resyncer
	Importer *importer.Importer
	// ClusterIPFamilies are the IP families of this cluster's Service
	// network, from the kubernetes Service; nil when unknown.
	ClusterIPFamilies []corev1.IPFamily

	connected atomic.Bool
}

// New resolves the identity, creates the registry client and the manager and
// registers every runnable and controller. Nothing talks to the registry before
// Run.
func New(ctx context.Context, cfg config.Config, restCfg *rest.Config, opts Options) (*App, error) {
	logger := opts.Logger
	if logger.GetSink() == nil {
		logger = ctrl.Log
	}
	reg := opts.Registry
	if reg == nil {
		reg = prometheus.NewRegistry()
	}
	scheme, err := config.NewScheme()
	if err != nil {
		return nil, err
	}
	mopts := cfg.ManagerOptions(scheme)
	mopts.Logger = logger
	if opts.ManagerOptions != nil {
		opts.ManagerOptions(&mopts)
	}
	mgr, err := ctrl.NewManager(restCfg, mopts)
	if err != nil {
		return nil, fmt.Errorf("create manager: %w", err)
	}

	direct := opts.Client
	if direct == nil {
		if direct, err = ctrlclient.New(restCfg, ctrlclient.Options{Scheme: scheme}); err != nil {
			return nil, fmt.Errorf("create API client: %w", err)
		}
	}
	id, err := identity.Resolve(ctx, direct, identity.FromConfig(&cfg))
	if err != nil {
		return nil, fmt.Errorf("resolve cluster identity: %w", err)
	}
	if cfg.WriteClusterProperties {
		if err := identity.WriteClusterProperties(ctx, direct, id); err != nil {
			return nil, err
		}
	}
	logger = logger.WithValues("clusterset", id.ClusterSet, "cluster", id.Cluster)
	families, err := importer.DetectClusterIPFamilies(ctx, direct)
	if err != nil {
		// Derived Services then list no IP family and the API server picks.
		logger.Info("Could not read the cluster's IP families; derived Services will not list any", "error", err.Error())
	}

	m := metrics.New()
	if err := m.Register(reg); err != nil {
		return nil, fmt.Errorf("register metrics: %w", err)
	}
	fakeReg, err := client.New(ctx, client.Config{
		URLs:              cfg.RegistryURLs,
		CAFile:            cfg.RegistryTLSCAFile,
		AllowInsecureHTTP: cfg.RegistryAllowInsecureHTTP,
		UserAgent:         "mcs-controller/" + version.Version,
		Registerer:        reg,
		Logger:            slog.New(logr.ToSlogHandler(logger.WithName("registry-client"))),
		DialOptions:       append([]grpc.DialOption{grpc.WithStatsHandler(otelgrpc.NewClientHandler())}, opts.RegistryDialOptions...),
	})
	if err != nil {
		return nil, fmt.Errorf("create registry client: %w", err)
	}

	a := &App{Config: cfg, Identity: id, Manager: mgr, Registry: fakeReg, Metrics: m, ClusterIPFamilies: families}
	if err := a.wire(logger, opts); err != nil {
		_ = fakeReg.Close()
		return nil, err
	}
	return a, nil
}

func (a *App) wire(logger logr.Logger, opts Options) error {
	cfg, mgr, id := a.Config, a.Manager, a.Identity
	agentID := opts.AgentID
	if agentID == "" {
		agentID = hostname()
	}
	a.Lease = lease.New(lease.Options{
		Registry:             a.Registry,
		Identity:             id,
		TTL:                  cfg.LeaseTTL,
		AgentID:              agentID,
		AgentVersion:         version.Version,
		DeregisterOnShutdown: cfg.DeregisterOnShutdown,
		Metrics:              a.Metrics,
		Logger:               logger.WithName("lease"),
		OnRegistered: func(first bool) {
			// A registration after the first means the registry lost or
			// replaced the lease: resend every export.
			if !first {
				a.Resyncer.Trigger(metrics.ResyncReregister)
			}
		},
	})
	a.Exports = export.NewReconciler(export.Options{
		Client:                  mgr.GetClient(),
		APIReader:               mgr.GetAPIReader(),
		Registry:                a.Registry,
		Identity:                id,
		Lease:                   a.Lease,
		Metrics:                 a.Metrics,
		Recorder:                mgr.GetEventRecorder(EventSource),
		MaxConcurrentReconciles: cfg.MaxConcurrentReconciles,
		WatchesNamespace:        cfg.WatchesNamespace,
		Restricted:              len(cfg.WatchNamespaces) > 0,
	})
	a.Resyncer = export.NewResyncer(export.ResyncerOptions{
		Reconciler: a.Exports, Lease: a.Lease, Logger: logger.WithName("resync"), Period: cfg.ExportResyncPeriod,
	})
	a.Registry.OnReconnect(func(url string) {
		a.Metrics.StreamReconnects.Inc()
		a.Resyncer.Trigger(metrics.ResyncReconnect)
	})
	a.Importer = importer.New(importer.Options{
		Client:                  mgr.GetClient(),
		APIReader:               mgr.GetAPIReader(),
		Registry:                a.Registry,
		Identity:                id,
		Metrics:                 a.Metrics,
		Logger:                  logger.WithName("importer"),
		Recorder:                mgr.GetEventRecorder(EventSource),
		EndpointsPerSlice:       cfg.EndpointsPerSlice,
		CreateNamespaces:        cfg.CreateNamespaces,
		AdoptUnmanaged:          cfg.AdoptUnmanagedImports,
		OmitTargetRefs:          !cfg.ImportTargetRef,
		ClusterIPFamilies:       a.ClusterIPFamilies,
		WatchNamespaces:         cfg.WatchNamespaces,
		MaxConcurrentReconciles: cfg.MaxConcurrentReconciles,
		OwnStatus:               a.Exports.SetOwnStatus,
		ResyncPeriod:            cfg.ImportResyncPeriod,
	})
	auto := export.NewAutoExporter(export.AutoExporterOptions{
		Client:                  mgr.GetClient(),
		Identity:                id,
		Metrics:                 a.Metrics,
		MaxConcurrentReconciles: cfg.MaxConcurrentReconciles,
		WatchesNamespace:        cfg.WatchesNamespace,
	})
	for _, r := range []manager.Runnable{a.Lease, a.Resyncer, a.Importer} {
		if err := mgr.Add(r); err != nil {
			return err
		}
	}
	if err := a.Exports.SetupWithManager(mgr); err != nil {
		return fmt.Errorf("set up the ServiceExport controller: %w", err)
	}
	if err := auto.SetupWithManager(mgr); err != nil {
		return fmt.Errorf("set up the auto-export controller: %w", err)
	}
	if err := a.Importer.SetupWithManager(mgr); err != nil {
		return fmt.Errorf("set up the import controller: %w", err)
	}
	if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
		return err
	}
	return mgr.AddReadyzCheck("readyz", a.ReadyCheck)
}

// hostname returns the pod name, the agent id of the registration.
func hostname() string {
	if h := os.Getenv("HOSTNAME"); h != "" {
		return h
	}
	h, _ := os.Hostname()
	return h
}

// ReadyCheck is the /readyz check: the registry client has connected at least
// once and, on the leader, the cluster is registered.
func (a *App) ReadyCheck(*http.Request) error {
	if !a.connected.Load() {
		if a.Registry.CurrentURL() == "" {
			return errors.New("not connected to the registry yet")
		}
		a.connected.Store(true)
	}
	select {
	case <-a.Manager.Elected():
		if !a.Lease.Registered() {
			return errors.New("leader without a registered cluster lease")
		}
	default:
	}
	return nil
}

// Run starts the manager and blocks until ctx is done, then closes the registry
// client.
func (a *App) Run(ctx context.Context) error {
	defer func() { _ = a.Registry.Close() }()
	ctrl.Log.WithName("setup").Info("Starting mcs-controller", append(version.KeysAndValues(),
		"clusterset", a.Identity.ClusterSet, "cluster", a.Identity.Cluster, "region", a.Identity.Region,
		"zones", a.Identity.Zones, "registryURLs", a.Config.RegistryURLs, "watchNamespaces", a.Config.WatchNamespaces,
		"leaderElect", a.Config.LeaderElect, "ipFamilies", a.ClusterIPFamilies,
		"exportResyncPeriod", a.Config.ExportResyncPeriod.String(), "importResyncPeriod", a.Config.ImportResyncPeriod.String(),
		"leaseTTL", a.Config.LeaseTTL.String(), "fieldManager", api.FieldManager)...)
	return a.Manager.Start(ctx)
}
