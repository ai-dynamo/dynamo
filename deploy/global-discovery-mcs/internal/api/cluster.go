// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package api

import (
	"context"
	"errors"

	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/convert"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/registry"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/view"
)

func validateClusterKey(v *validator, clusterSet, cluster string) store.ClusterKey {
	v.dnsLabel("clusterset", clusterSet)
	v.dnsLabel("cluster", cluster)
	return store.ClusterKey{ClusterSet: clusterSet, Cluster: cluster}
}

// RegisterCluster implements mcsv1.ClusterServiceServer.
func (s *Server) RegisterCluster(ctx context.Context, req *mcsv1.RegisterClusterRequest) (*mcsv1.RegisterClusterResponse, error) {
	var v validator
	key := validateClusterKey(&v, req.GetClusterset(), req.GetCluster())
	// The region becomes the topology.kubernetes.io/region label of the
	// EndpointSlices every importing cluster writes.
	v.labelValue("region", req.GetRegion())
	v.maxLen("provider", req.GetProvider(), 63)
	v.maxLen("agent_id", req.GetAgentId(), maxNameBytes)
	v.maxLen("agent_version", req.GetAgentVersion(), maxNameBytes)
	if len(req.GetZones()) > maxZones {
		v.add("zones", "has more than %d entries", maxZones)
	}
	v.labels("labels", req.GetLabels())
	ttl := v.leaseTTL("lease_ttl", req.GetLeaseTtl(), registry.DefaultClusterLeaseTTL)
	if err := v.err(); err != nil {
		return nil, err
	}
	now := s.now()
	rec, err := s.store.PutCluster(ctx, store.ClusterRecord{
		Key:          key,
		Region:       req.GetRegion(),
		Zones:        req.GetZones(),
		Provider:     req.GetProvider(),
		Labels:       req.GetLabels(),
		AgentID:      req.GetAgentId(),
		AgentVersion: req.GetAgentVersion(),
		Lease:        store.Lease{TTL: ttl},
	}, now)
	if err != nil {
		return nil, storeError(err, "register cluster")
	}
	s.view.ApplyCluster(rec)
	s.log.Info("cluster registered", zap.String("clusterset", key.ClusterSet), zap.String("cluster", key.Cluster),
		zap.String("region", rec.Region), zap.Uint64("epoch", rec.Lease.Epoch), zap.Duration("lease_ttl", ttl))
	return &mcsv1.RegisterClusterResponse{
		Cluster:    convert.ClusterToProto(rec, registry.LeaseStateActive),
		ServerTime: convert.Timestamp(now),
	}, nil
}

// HeartbeatCluster implements mcsv1.ClusterServiceServer.
func (s *Server) HeartbeatCluster(ctx context.Context, req *mcsv1.HeartbeatClusterRequest) (*mcsv1.HeartbeatClusterResponse, error) {
	var v validator
	key := validateClusterKey(&v, req.GetClusterset(), req.GetCluster())
	if req.GetEpoch() == 0 {
		v.add("epoch", "is required")
	}
	if err := v.err(); err != nil {
		return nil, err
	}
	now := s.now()
	rec, err := s.store.RenewClusterLease(ctx, key, req.GetEpoch(), now)
	switch {
	case errors.Is(err, store.ErrNotFound):
		return nil, clusterNotRegistered(codes.NotFound, key)
	case err != nil:
		return nil, storeError(err, "heartbeat cluster")
	}
	s.view.ApplyCluster(rec)
	return &mcsv1.HeartbeatClusterResponse{
		Lease:      convert.LeaseToProto(rec.Lease, registry.LeaseStateActive),
		ServerTime: convert.Timestamp(now),
	}, nil
}

// DeregisterCluster implements mcsv1.ClusterServiceServer.
func (s *Server) DeregisterCluster(ctx context.Context, req *mcsv1.DeregisterClusterRequest) (*mcsv1.DeregisterClusterResponse, error) {
	var v validator
	key := validateClusterKey(&v, req.GetClusterset(), req.GetCluster())
	if err := v.err(); err != nil {
		return nil, err
	}
	if err := s.store.DeleteCluster(ctx, key); err != nil {
		return nil, storeError(err, "deregister cluster")
	}
	s.view.ApplyClusterDelete(key)
	s.log.Info("cluster deregistered", zap.String("clusterset", key.ClusterSet), zap.String("cluster", key.Cluster))
	return &mcsv1.DeregisterClusterResponse{}, nil
}

// GetCluster implements mcsv1.ClusterServiceServer.
func (s *Server) GetCluster(_ context.Context, req *mcsv1.GetClusterRequest) (*mcsv1.GetClusterResponse, error) {
	var v validator
	key := validateClusterKey(&v, req.GetClusterset(), req.GetCluster())
	if err := v.err(); err != nil {
		return nil, err
	}
	o, ok := s.view.Get(key.ClusterSet, view.KindCluster, key.Cluster)
	if !ok {
		return nil, clusterNotRegistered(codes.NotFound, key)
	}
	return &mcsv1.GetClusterResponse{Cluster: o.Full.(*mcsv1.Cluster)}, nil
}

// ListClusters implements mcsv1.ClusterServiceServer.
func (s *Server) ListClusters(_ context.Context, req *mcsv1.ListClustersRequest) (*mcsv1.ListClustersResponse, error) {
	var v validator
	v.dnsLabel("clusterset", req.GetClusterset())
	if err := v.err(); err != nil {
		return nil, err
	}
	msgs, rev := s.view.List(req.GetClusterset(), view.ClusterFilter{IncludeExpired: req.GetIncludeExpired()})
	resp := &mcsv1.ListClustersResponse{Revision: rev}
	for _, m := range msgs {
		resp.Clusters = append(resp.Clusters, m.(*mcsv1.Cluster))
	}
	return resp, nil
}

// WatchClusters implements mcsv1.ClusterServiceServer.
func (s *Server) WatchClusters(req *mcsv1.WatchClustersRequest, stream grpc.ServerStreamingServer[mcsv1.WatchClustersResponse]) error {
	var v validator
	v.dnsLabel("clusterset", req.GetClusterset())
	if err := v.err(); err != nil {
		return err
	}
	f := view.ClusterFilter{IncludeExpired: req.GetIncludeExpired()}
	return watchError(s.hub.Watch(stream.Context(), req.GetClusterset(), f, req.GetResumeRevision(), func(ev view.WatchEvent) error {
		resp := &mcsv1.WatchClustersResponse{Type: ev.Type, Revision: ev.Revision}
		if ev.Object != nil {
			resp.Cluster = ev.Object.(*mcsv1.Cluster)
		}
		return stream.Send(resp)
	}))
}

// ListClusterSets implements mcsv1.ClusterServiceServer.
func (s *Server) ListClusterSets(context.Context, *mcsv1.ListClusterSetsRequest) (*mcsv1.ListClusterSetsResponse, error) {
	resp := &mcsv1.ListClusterSetsResponse{}
	for _, sum := range s.view.Summaries() {
		resp.Clustersets = append(resp.Clustersets, &mcsv1.ClusterSetSummary{
			Clusterset:         sum.ClusterSet,
			ClusterCount:       int32(sum.Clusters),
			ActiveClusterCount: int32(sum.ActiveClusters),
			ExportCount:        int32(sum.Exports),
		})
	}
	return resp, nil
}
