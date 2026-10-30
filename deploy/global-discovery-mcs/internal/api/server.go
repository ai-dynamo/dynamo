// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package api implements the dynamo.mcs.v1 gRPC services over the store and
// the materialised view: request validation against the API limits, error
// mapping to google.rpc.Status with an ErrorInfo in the mcs.dynamo.nvidia.com
// domain, lease epochs and writer generations, the write-through to the
// view after every successful store write, and watches through the hub.
//
// Reads are answered from the view of the handling replica. Writes go to
// the store and are applied to the view before the response, so a writer
// reads its own write at once.
package api

import (
	"context"
	"errors"
	"time"

	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/clock"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/registry"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/view"
)

// Options configures a Server.
type Options struct {
	Store store.Store
	View  *view.View
	Hub   *view.Hub
	Clock clock.Clock
	// Region and Version are reported by GetServerInfo.
	Region  string
	Version string
	// Endpoints are the advertised registry endpoints of every region.
	Endpoints []*mcsv1.ServerEndpoint
	Logger    *zap.Logger
}

// Server implements every dynamo.mcs.v1 service.
type Server struct {
	mcsv1.UnimplementedClusterServiceServer
	mcsv1.UnimplementedServiceExportServiceServer
	mcsv1.UnimplementedServiceImportServiceServer
	mcsv1.UnimplementedServerInfoServiceServer

	store store.Store
	view  *view.View
	hub   *view.Hub
	clock clock.Clock
	opts  Options
	log   *zap.Logger
}

// New returns a Server.
func New(opts Options) *Server {
	if opts.Clock == nil {
		opts.Clock = clock.Real()
	}
	if opts.Logger == nil {
		opts.Logger = zap.NewNop()
	}
	return &Server{
		store: opts.Store,
		view:  opts.View,
		hub:   opts.Hub,
		clock: opts.Clock,
		opts:  opts,
		log:   opts.Logger,
	}
}

// ServiceNames are the fully qualified names of the services Register adds.
var ServiceNames = []string{
	mcsv1.ClusterService_ServiceDesc.ServiceName,
	mcsv1.ServiceExportService_ServiceDesc.ServiceName,
	mcsv1.ServiceImportService_ServiceDesc.ServiceName,
	mcsv1.ServerInfoService_ServiceDesc.ServiceName,
}

// Register registers every service on r.
func (s *Server) Register(r grpc.ServiceRegistrar) {
	mcsv1.RegisterClusterServiceServer(r, s)
	mcsv1.RegisterServiceExportServiceServer(r, s)
	mcsv1.RegisterServiceImportServiceServer(r, s)
	mcsv1.RegisterServerInfoServiceServer(r, s)
}

func (s *Server) now() time.Time { return s.clock.Now().UTC() }

// requireCluster fails with FAILED_PRECONDITION CLUSTER_NOT_REGISTERED when
// the cluster has no registration.
func (s *Server) requireCluster(ctx context.Context, key store.ClusterKey) error {
	if _, ok := s.view.ClusterRecord(key); ok {
		return nil
	}
	_, err := s.store.GetCluster(ctx, key)
	switch {
	case errors.Is(err, store.ErrNotFound):
		return clusterNotRegistered(codes.FailedPrecondition, key)
	case err != nil:
		return storeError(err, "get cluster")
	}
	return nil
}

// generation returns the writer's generation, or one the server assigns
// when the writer sends 0: max(stored + 1, now in unix milliseconds).
func generation(requested, stored uint64, now time.Time) (gen uint64, assigned bool) {
	if requested != 0 {
		return requested, false
	}
	return registry.NextGeneration(stored, now), true
}
