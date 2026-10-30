// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package fakeregistry

import (
	"cmp"
	"context"
	"slices"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/registry"
)

type exportService struct {
	mcsv1.UnimplementedServiceExportServiceServer
	s *Server
}

func exportIDs(clusterset, cluster, namespace, name string) error {
	return require("clusterset", clusterset, "cluster", cluster, "namespace", namespace, "name", name)
}

// exportViewLocked returns a stored export with its endpoint counts.
func exportViewLocked(cs *clusterSet, key exportKey) *mcsv1.ServiceExport {
	e := proto.CloneOf(cs.exports[key])
	if rec := cs.endpoints[key]; rec != nil {
		e.EndpointsGeneration = rec.generation
		e.EndpointCount, e.ReadyEndpointCount = countEndpoints(rec.endpoints)
	}
	return e
}

func (x *exportService) PutServiceExport(_ context.Context, req *mcsv1.PutServiceExportRequest) (*mcsv1.PutServiceExportResponse, error) {
	if err := exportIDs(req.GetClusterset(), req.GetCluster(), req.GetNamespace(), req.GetName()); err != nil {
		return nil, err
	}
	s := x.s
	s.mu.Lock()
	defer s.mu.Unlock()
	cs := s.setLocked(req.GetClusterset())
	if cs.clusters[req.GetCluster()] == nil {
		return nil, errClusterNotRegistered(codes.FailedPrecondition, req.GetClusterset(), req.GetCluster())
	}
	key := exportKey{req.GetCluster(), req.GetNamespace(), req.GetName()}
	if _, err := s.putExportLocked(cs, key, req.GetSpec(), req.GetGeneration()); err != nil {
		return nil, err
	}
	return &mcsv1.PutServiceExportResponse{Export: exportViewLocked(cs, key)}, nil
}

// generation returns the writer's generation, or one the server assigns
// when the writer sends 0: max(stored + 1, now in unix milliseconds), as
// mcs-registry does. A writer's own generation must exceed the stored one.
func generationOf(what string, requested, stored uint64, now time.Time) (uint64, error) {
	switch {
	case requested == 0:
		return registry.NextGeneration(stored, now), nil
	case requested <= stored:
		return 0, errStaleGeneration(what, requested, stored)
	}
	return requested, nil
}

// putExportLocked stores an export under generation, or under a
// server-assigned one when it is 0, and returns the stored generation. A
// spec without export_creation_timestamp keeps the stored one, or gets the
// server time for a new export.
func (s *Server) putExportLocked(cs *clusterSet, key exportKey, spec *mcsv1.ServiceExportSpec, generation uint64) (uint64, error) {
	now := s.now()
	old := cs.exports[key]
	generation, err := generationOf("export", generation, old.GetGeneration(), now)
	if err != nil {
		return 0, err
	}
	spec = proto.CloneOf(spec)
	if spec == nil {
		spec = &mcsv1.ServiceExportSpec{}
	}
	if spec.ExportCreationTimestamp == nil {
		if old != nil && old.GetSpec().GetExportCreationTimestamp() != nil {
			spec.ExportCreationTimestamp = old.GetSpec().GetExportCreationTimestamp()
		} else {
			spec.ExportCreationTimestamp = timestamppb.New(now)
		}
	}
	cs.exports[key] = &mcsv1.ServiceExport{
		Clusterset: cs.name,
		Cluster:    key.cluster,
		Namespace:  key.namespace,
		Name:       key.name,
		Spec:       spec,
		Generation: generation,
		UpdatedAt:  timestamppb.New(now),
	}
	s.commitLocked(cs)
	return generation, nil
}

func (x *exportService) DeleteServiceExport(_ context.Context, req *mcsv1.DeleteServiceExportRequest) (*mcsv1.DeleteServiceExportResponse, error) {
	if err := exportIDs(req.GetClusterset(), req.GetCluster(), req.GetNamespace(), req.GetName()); err != nil {
		return nil, err
	}
	x.s.DeleteExport(req.GetClusterset(), req.GetCluster(), req.GetNamespace(), req.GetName())
	return &mcsv1.DeleteServiceExportResponse{}, nil
}

func (x *exportService) ListServiceExports(_ context.Context, req *mcsv1.ListServiceExportsRequest) (*mcsv1.ListServiceExportsResponse, error) {
	if err := require("clusterset", req.GetClusterset(), "cluster", req.GetCluster()); err != nil {
		return nil, err
	}
	s := x.s
	s.mu.Lock()
	defer s.mu.Unlock()
	cs := s.setLocked(req.GetClusterset())
	var keys []exportKey
	for k := range cs.exports {
		if k.cluster == req.GetCluster() {
			keys = append(keys, k)
		}
	}
	slices.SortFunc(keys, func(a, b exportKey) int {
		return cmp.Or(cmp.Compare(a.namespace, b.namespace), cmp.Compare(a.name, b.name))
	})
	resp := &mcsv1.ListServiceExportsResponse{}
	for _, k := range keys {
		resp.Exports = append(resp.Exports, exportViewLocked(cs, k))
	}
	return resp, nil
}

func (x *exportService) ReplaceEndpoints(_ context.Context, req *mcsv1.ReplaceEndpointsRequest) (*mcsv1.ReplaceEndpointsResponse, error) {
	if err := exportIDs(req.GetClusterset(), req.GetCluster(), req.GetNamespace(), req.GetName()); err != nil {
		return nil, err
	}
	if n := len(req.GetEndpoints()); n > api.MaxEndpointsPerExport {
		return nil, errTooManyEndpoints(n)
	}
	s := x.s
	s.mu.Lock()
	defer s.mu.Unlock()
	cs := s.setLocked(req.GetClusterset())
	if cs.clusters[req.GetCluster()] == nil {
		return nil, errClusterNotRegistered(codes.FailedPrecondition, req.GetClusterset(), req.GetCluster())
	}
	key := exportKey{req.GetCluster(), req.GetNamespace(), req.GetName()}
	gen, err := s.putEndpointsLocked(cs, key, req.GetEndpoints(), req.GetGeneration())
	if err != nil {
		return nil, err
	}
	total, ready := countEndpoints(req.GetEndpoints())
	return &mcsv1.ReplaceEndpointsResponse{Generation: gen, EndpointCount: total, ReadyEndpointCount: ready}, nil
}

// putEndpointsLocked replaces an endpoint set under generation, or under a
// server-assigned one when it is 0, and returns the stored generation. An
// endpoint set may exist without its export.
func (s *Server) putEndpointsLocked(cs *clusterSet, key exportKey, endpoints []*mcsv1.Endpoint, generation uint64) (uint64, error) {
	var stored uint64
	if old := cs.endpoints[key]; old != nil {
		stored = old.generation
	}
	generation, err := generationOf("endpoint set", generation, stored, s.now())
	if err != nil {
		return 0, err
	}
	rec := &endpointRec{generation: generation}
	for _, ep := range endpoints {
		rec.endpoints = append(rec.endpoints, proto.CloneOf(ep))
	}
	cs.endpoints[key] = rec
	s.commitLocked(cs)
	return generation, nil
}

type importService struct {
	mcsv1.UnimplementedServiceImportServiceServer
	s *Server
}

// selects reports whether labels contain every entry of the selector.
func selects(sel *mcsv1.LabelSelector, labels map[string]string) bool {
	for k, v := range sel.GetMatchLabels() {
		if got, ok := labels[k]; !ok || got != v {
			return false
		}
	}
	return true
}

func importMatcher(namespace string, sel *mcsv1.LabelSelector) func(*mcsv1.ServiceImport) bool {
	return func(imp *mcsv1.ServiceImport) bool {
		return (namespace == "" || imp.GetNamespace() == namespace) && selects(sel, imp.GetLabels())
	}
}

// importView returns imp without endpoint lists unless includeEndpoints;
// counts stay.
func importView(imp *mcsv1.ServiceImport, includeEndpoints bool) *mcsv1.ServiceImport {
	if includeEndpoints {
		return imp
	}
	imp = proto.CloneOf(imp)
	for _, ce := range imp.Endpoints {
		ce.Endpoints = nil
	}
	return imp
}

func (x *importService) GetServiceImport(_ context.Context, req *mcsv1.GetServiceImportRequest) (*mcsv1.GetServiceImportResponse, error) {
	if err := require("clusterset", req.GetClusterset(), "namespace", req.GetNamespace(), "name", req.GetName()); err != nil {
		return nil, err
	}
	s := x.s
	s.mu.Lock()
	defer s.mu.Unlock()
	cs := s.refreshedLocked(req.GetClusterset())
	obj, ok := cs.views[kindImports][req.GetNamespace()+"/"+req.GetName()]
	if !ok {
		return nil, errNotFound("service import %s/%s not found in clusterset %q", req.GetNamespace(), req.GetName(), req.GetClusterset())
	}
	return &mcsv1.GetServiceImportResponse{ServiceImport: importView(obj.(*mcsv1.ServiceImport), req.GetIncludeEndpoints())}, nil
}

func (x *importService) ListServiceImports(_ context.Context, req *mcsv1.ListServiceImportsRequest) (*mcsv1.ListServiceImportsResponse, error) {
	if err := require("clusterset", req.GetClusterset()); err != nil {
		return nil, err
	}
	s := x.s
	s.mu.Lock()
	defer s.mu.Unlock()
	cs := s.refreshedLocked(req.GetClusterset())
	resp := &mcsv1.ListServiceImportsResponse{Revision: s.revision(s.seq)}
	for _, imp := range sortedViews(cs, kindImports, importMatcher(req.GetNamespace(), req.GetSelector())) {
		resp.ServiceImports = append(resp.ServiceImports, importView(imp, req.GetIncludeEndpoints()))
	}
	return resp, nil
}

func (x *importService) WatchServiceImports(req *mcsv1.WatchServiceImportsRequest, stream grpc.ServerStreamingServer[mcsv1.WatchServiceImportsResponse]) error {
	match := importMatcher(req.GetNamespace(), req.GetSelector())
	return x.s.serveWatch(stream.Context(), &watchSpec{
		kind:   kindImports,
		cs:     req.GetClusterset(),
		resume: req.GetResumeRevision(),
		match:  func(m proto.Message) bool { return match(m.(*mcsv1.ServiceImport)) },
		view: func(m proto.Message) proto.Message {
			return importView(m.(*mcsv1.ServiceImport), req.GetIncludeEndpoints())
		},
		send: func(typ mcsv1.EventType, obj proto.Message, rev string) error {
			resp := &mcsv1.WatchServiceImportsResponse{Type: typ, Revision: rev}
			if obj != nil {
				resp.ServiceImport = obj.(*mcsv1.ServiceImport)
			}
			return stream.Send(resp)
		},
	})
}
