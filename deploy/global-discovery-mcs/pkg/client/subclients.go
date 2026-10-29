// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package client

import (
	"context"

	"google.golang.org/grpc"
	"google.golang.org/protobuf/proto"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
)

// The sub-clients are thin typed wrappers over the generated clients. Unary
// calls whose context has no deadline get Config.RPCTimeout, and unary
// calls failing with UNAVAILABLE are retried up to Config.MaxAttempts
// times; both apply to Conn() users as well. Watch calls return the raw
// stream; Watcher adapts them to a WatchLoop.

// ClusterClient calls the ClusterService.
type ClusterClient struct{ c mcsv1.ClusterServiceClient }

func (x *ClusterClient) Register(ctx context.Context, req *mcsv1.RegisterClusterRequest, opts ...grpc.CallOption) (*mcsv1.RegisterClusterResponse, error) {
	return x.c.RegisterCluster(ctx, req, opts...)
}

func (x *ClusterClient) Heartbeat(ctx context.Context, req *mcsv1.HeartbeatClusterRequest, opts ...grpc.CallOption) (*mcsv1.HeartbeatClusterResponse, error) {
	return x.c.HeartbeatCluster(ctx, req, opts...)
}

func (x *ClusterClient) Deregister(ctx context.Context, req *mcsv1.DeregisterClusterRequest, opts ...grpc.CallOption) (*mcsv1.DeregisterClusterResponse, error) {
	return x.c.DeregisterCluster(ctx, req, opts...)
}

func (x *ClusterClient) Get(ctx context.Context, req *mcsv1.GetClusterRequest, opts ...grpc.CallOption) (*mcsv1.GetClusterResponse, error) {
	return x.c.GetCluster(ctx, req, opts...)
}

func (x *ClusterClient) List(ctx context.Context, req *mcsv1.ListClustersRequest, opts ...grpc.CallOption) (*mcsv1.ListClustersResponse, error) {
	return x.c.ListClusters(ctx, req, opts...)
}

func (x *ClusterClient) Watch(ctx context.Context, req *mcsv1.WatchClustersRequest, opts ...grpc.CallOption) (grpc.ServerStreamingClient[mcsv1.WatchClustersResponse], error) {
	return x.c.WatchClusters(ctx, req, opts...)
}

func (x *ClusterClient) ListClusterSets(ctx context.Context, req *mcsv1.ListClusterSetsRequest, opts ...grpc.CallOption) (*mcsv1.ListClusterSetsResponse, error) {
	return x.c.ListClusterSets(ctx, req, opts...)
}

// Watcher returns a WatchFunc for WatchClusters with req's filters.
func (x *ClusterClient) Watcher(req *mcsv1.WatchClustersRequest) WatchFunc[*mcsv1.Cluster] {
	return streamWatcher(req, func(r *mcsv1.WatchClustersRequest, revision string) { r.ResumeRevision = revision },
		x.c.WatchClusters, (*mcsv1.WatchClustersResponse).GetCluster)
}

// ServiceExportClient calls the ServiceExportService.
type ServiceExportClient struct {
	c mcsv1.ServiceExportServiceClient
}

func (x *ServiceExportClient) Put(ctx context.Context, req *mcsv1.PutServiceExportRequest, opts ...grpc.CallOption) (*mcsv1.PutServiceExportResponse, error) {
	return x.c.PutServiceExport(ctx, req, opts...)
}

func (x *ServiceExportClient) Delete(ctx context.Context, req *mcsv1.DeleteServiceExportRequest, opts ...grpc.CallOption) (*mcsv1.DeleteServiceExportResponse, error) {
	return x.c.DeleteServiceExport(ctx, req, opts...)
}

func (x *ServiceExportClient) List(ctx context.Context, req *mcsv1.ListServiceExportsRequest, opts ...grpc.CallOption) (*mcsv1.ListServiceExportsResponse, error) {
	return x.c.ListServiceExports(ctx, req, opts...)
}

func (x *ServiceExportClient) ReplaceEndpoints(ctx context.Context, req *mcsv1.ReplaceEndpointsRequest, opts ...grpc.CallOption) (*mcsv1.ReplaceEndpointsResponse, error) {
	return x.c.ReplaceEndpoints(ctx, req, opts...)
}

// ServiceImportClient calls the ServiceImportService.
type ServiceImportClient struct {
	c mcsv1.ServiceImportServiceClient
}

func (x *ServiceImportClient) Get(ctx context.Context, req *mcsv1.GetServiceImportRequest, opts ...grpc.CallOption) (*mcsv1.GetServiceImportResponse, error) {
	return x.c.GetServiceImport(ctx, req, opts...)
}

func (x *ServiceImportClient) List(ctx context.Context, req *mcsv1.ListServiceImportsRequest, opts ...grpc.CallOption) (*mcsv1.ListServiceImportsResponse, error) {
	return x.c.ListServiceImports(ctx, req, opts...)
}

func (x *ServiceImportClient) Watch(ctx context.Context, req *mcsv1.WatchServiceImportsRequest, opts ...grpc.CallOption) (grpc.ServerStreamingClient[mcsv1.WatchServiceImportsResponse], error) {
	return x.c.WatchServiceImports(ctx, req, opts...)
}

// Watcher returns a WatchFunc for WatchServiceImports with req's filters.
func (x *ServiceImportClient) Watcher(req *mcsv1.WatchServiceImportsRequest) WatchFunc[*mcsv1.ServiceImport] {
	return streamWatcher(req, func(r *mcsv1.WatchServiceImportsRequest, revision string) { r.ResumeRevision = revision },
		x.c.WatchServiceImports, (*mcsv1.WatchServiceImportsResponse).GetServiceImport)
}

// watchMessage is the envelope every watch response shares.
type watchMessage interface {
	GetType() mcsv1.EventType
	GetRevision() string
}

// streamWatcher returns a WatchFunc that opens a stream with open for a copy
// of req resumed at the revision the loop asks for, and turns each response
// into an Event carrying object(response).
func streamWatcher[Req proto.Message, Resp any, PResp interface {
	*Resp
	watchMessage
}, T any](
	req Req, resume func(Req, string),
	open func(context.Context, Req, ...grpc.CallOption) (grpc.ServerStreamingClient[Resp], error),
	object func(PResp) T,
) WatchFunc[T] {
	req = proto.CloneOf(req)
	return func(ctx context.Context, revision string) (WatchStream[T], error) {
		r := proto.CloneOf(req)
		resume(r, revision)
		s, err := open(ctx, r)
		if err != nil {
			return nil, err
		}
		return recvFunc[T](func() (Event[T], error) {
			m, err := s.Recv()
			if err != nil {
				return Event[T]{}, err
			}
			msg := PResp(m)
			return Event[T]{Type: msg.GetType(), Object: object(msg), Revision: msg.GetRevision()}, nil
		}), nil
	}
}

// ServerInfoClient calls the ServerInfoService.
type ServerInfoClient struct {
	c mcsv1.ServerInfoServiceClient
}

func (x *ServerInfoClient) Get(ctx context.Context, opts ...grpc.CallOption) (*mcsv1.GetServerInfoResponse, error) {
	return x.c.GetServerInfo(ctx, &mcsv1.GetServerInfoRequest{}, opts...)
}
