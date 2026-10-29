// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package fakeregistry

import (
	"context"
	"maps"
	"slices"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/durationpb"
	"google.golang.org/protobuf/types/known/timestamppb"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/registry"
)

type clusterService struct {
	mcsv1.UnimplementedClusterServiceServer
	s *Server
}

func (c *clusterService) RegisterCluster(_ context.Context, req *mcsv1.RegisterClusterRequest) (*mcsv1.RegisterClusterResponse, error) {
	if err := require("clusterset", req.GetClusterset(), "cluster", req.GetCluster()); err != nil {
		return nil, err
	}
	s := c.s
	s.mu.Lock()
	defer s.mu.Unlock()
	rec := s.registerClusterLocked(req)
	return &mcsv1.RegisterClusterResponse{
		Cluster:    clusterView(rec, s.now()),
		ServerTime: timestamppb.New(s.now()),
	}, nil
}

func (s *Server) registerClusterLocked(req *mcsv1.RegisterClusterRequest) *clusterRec {
	now := s.now()
	cs := s.setLocked(req.GetClusterset())
	ttl := registry.ClampLeaseTTL(req.GetLeaseTtl().AsDuration(), registry.DefaultClusterLeaseTTL)
	var prevEpoch, prevGen uint64
	registered := now
	if old := cs.clusters[req.GetCluster()]; old != nil {
		prevEpoch, prevGen = old.c.GetLease().GetEpoch(), old.c.GetGeneration()
		registered = old.c.GetRegisteredAt().AsTime()
	}
	rec := &clusterRec{
		ttl: ttl,
		c: &mcsv1.Cluster{
			Clusterset:   req.GetClusterset(),
			Cluster:      req.GetCluster(),
			Region:       req.GetRegion(),
			Zones:        slices.Clone(req.GetZones()),
			Provider:     req.GetProvider(),
			Labels:       maps.Clone(req.GetLabels()),
			AgentId:      req.GetAgentId(),
			AgentVersion: req.GetAgentVersion(),
			Lease: &mcsv1.Lease{
				Epoch:     registry.NextGeneration(prevEpoch, now),
				Ttl:       durationpb.New(ttl),
				RenewedAt: timestamppb.New(now),
			},
			Generation:   prevGen + 1,
			RegisteredAt: timestamppb.New(registered),
			UpdatedAt:    timestamppb.New(now),
		},
	}
	cs.clusters[req.GetCluster()] = rec
	s.commitLocked(cs)
	return rec
}

func (c *clusterService) HeartbeatCluster(_ context.Context, req *mcsv1.HeartbeatClusterRequest) (*mcsv1.HeartbeatClusterResponse, error) {
	if err := require("clusterset", req.GetClusterset(), "cluster", req.GetCluster()); err != nil {
		return nil, err
	}
	s := c.s
	s.mu.Lock()
	defer s.mu.Unlock()
	now := s.now()
	cs := s.setLocked(req.GetClusterset())
	rec := cs.clusters[req.GetCluster()]
	if rec == nil {
		return nil, errClusterNotRegistered(codes.NotFound, req.GetClusterset(), req.GetCluster())
	}
	if cur := rec.c.GetLease().GetEpoch(); req.GetEpoch() != cur {
		return nil, errEpochMismatch("cluster", req.GetEpoch(), cur)
	}
	rec.c = proto.CloneOf(rec.c)
	rec.c.Lease.RenewedAt = timestamppb.New(now)
	s.commitLocked(cs)
	return &mcsv1.HeartbeatClusterResponse{
		Lease:      leaseView(rec.c.GetLease(), rec.ttl, now),
		ServerTime: timestamppb.New(now),
	}, nil
}

func (c *clusterService) DeregisterCluster(_ context.Context, req *mcsv1.DeregisterClusterRequest) (*mcsv1.DeregisterClusterResponse, error) {
	if err := require("clusterset", req.GetClusterset(), "cluster", req.GetCluster()); err != nil {
		return nil, err
	}
	c.s.removeCluster(req.GetClusterset(), req.GetCluster())
	return &mcsv1.DeregisterClusterResponse{}, nil
}

// removeCluster deletes a cluster with its exports and endpoint sets.
func (s *Server) removeCluster(clusterset, cluster string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	cs := s.setLocked(clusterset)
	delete(cs.clusters, cluster)
	for k := range cs.exports {
		if k.cluster == cluster {
			delete(cs.exports, k)
		}
	}
	for k := range cs.endpoints {
		if k.cluster == cluster {
			delete(cs.endpoints, k)
		}
	}
	s.commitLocked(cs)
}

func (c *clusterService) GetCluster(_ context.Context, req *mcsv1.GetClusterRequest) (*mcsv1.GetClusterResponse, error) {
	if err := require("clusterset", req.GetClusterset(), "cluster", req.GetCluster()); err != nil {
		return nil, err
	}
	s := c.s
	s.mu.Lock()
	defer s.mu.Unlock()
	cs := s.refreshedLocked(req.GetClusterset())
	obj, ok := cs.views[kindClusters][req.GetCluster()]
	if !ok {
		return nil, errClusterNotRegistered(codes.NotFound, req.GetClusterset(), req.GetCluster())
	}
	return &mcsv1.GetClusterResponse{Cluster: obj.(*mcsv1.Cluster)}, nil
}

func clusterMatcher(includeExpired bool) func(*mcsv1.Cluster) bool {
	return func(c *mcsv1.Cluster) bool {
		return includeExpired || c.GetLease().GetState() != mcsv1.LeaseState_LEASE_STATE_EXPIRED
	}
}

func (c *clusterService) ListClusters(_ context.Context, req *mcsv1.ListClustersRequest) (*mcsv1.ListClustersResponse, error) {
	if err := require("clusterset", req.GetClusterset()); err != nil {
		return nil, err
	}
	s := c.s
	s.mu.Lock()
	defer s.mu.Unlock()
	cs := s.refreshedLocked(req.GetClusterset())
	return &mcsv1.ListClustersResponse{
		Clusters: sortedViews(cs, kindClusters, clusterMatcher(req.GetIncludeExpired())),
		Revision: s.revision(s.seq),
	}, nil
}

func (c *clusterService) WatchClusters(req *mcsv1.WatchClustersRequest, stream grpc.ServerStreamingServer[mcsv1.WatchClustersResponse]) error {
	match := clusterMatcher(req.GetIncludeExpired())
	return c.s.serveWatch(stream.Context(), &watchSpec{
		kind:   kindClusters,
		cs:     req.GetClusterset(),
		resume: req.GetResumeRevision(),
		match:  func(m proto.Message) bool { return match(m.(*mcsv1.Cluster)) },
		view:   func(m proto.Message) proto.Message { return m },
		send: func(typ mcsv1.EventType, obj proto.Message, rev string) error {
			resp := &mcsv1.WatchClustersResponse{Type: typ, Revision: rev}
			if obj != nil {
				resp.Cluster = obj.(*mcsv1.Cluster)
			}
			return stream.Send(resp)
		},
	})
}

func (c *clusterService) ListClusterSets(context.Context, *mcsv1.ListClusterSetsRequest) (*mcsv1.ListClusterSetsResponse, error) {
	s := c.s
	s.mu.Lock()
	defer s.mu.Unlock()
	resp := &mcsv1.ListClusterSetsResponse{}
	for _, name := range slices.Sorted(maps.Keys(s.sets)) {
		cs := s.refreshedLocked(name)
		sum := &mcsv1.ClusterSetSummary{
			Clusterset:   name,
			ClusterCount: int32(len(cs.views[kindClusters])),
			ExportCount:  int32(len(cs.exports)),
		}
		for _, obj := range cs.views[kindClusters] {
			if obj.(*mcsv1.Cluster).GetLease().GetState() == mcsv1.LeaseState_LEASE_STATE_ACTIVE {
				sum.ActiveClusterCount++
			}
		}
		resp.Clustersets = append(resp.Clustersets, sum)
	}
	return resp, nil
}
