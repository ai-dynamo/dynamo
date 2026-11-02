// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package api

import (
	"context"
	"fmt"

	"go.uber.org/zap"
	"google.golang.org/grpc"

	mcsapi "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/convert"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/registry"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/view"
)

func validateExportKey(v *validator, clusterSet, cluster, namespace, name string) store.ExportKey {
	v.dnsLabel("clusterset", clusterSet)
	v.dnsLabel("cluster", cluster)
	v.dnsLabel("namespace", namespace)
	v.dnsLabel("name", name)
	return store.ExportKey{ClusterSet: clusterSet, Cluster: cluster, Namespace: namespace, Name: name}
}

// normalizeSpec applies the Kubernetes defaults: TCP ports and no session
// affinity.
func normalizeSpec(s *store.ExportSpec) {
	for i := range s.Ports {
		if s.Ports[i].Protocol == "" {
			s.Ports[i].Protocol = store.ProtocolTCP
		}
	}
	if s.SessionAffinity == "" {
		s.SessionAffinity = store.SessionAffinityNone
	}
}

func exportToProto(st view.ExportStatus) *mcsv1.ServiceExport {
	r := st.Record
	return &mcsv1.ServiceExport{
		Clusterset:          r.Key.ClusterSet,
		Cluster:             r.Key.Cluster,
		Namespace:           r.Key.Namespace,
		Name:                r.Key.Name,
		Spec:                convert.ExportSpecToProto(r.Spec),
		Generation:          r.Generation,
		UpdatedAt:           convert.Timestamp(r.UpdatedAt),
		EndpointsGeneration: st.EndpointsGeneration,
		EndpointCount:       st.EndpointCount,
		ReadyEndpointCount:  st.ReadyEndpointCount,
	}
}

// PutServiceExport implements mcsv1.ServiceExportServiceServer.
func (s *Server) PutServiceExport(ctx context.Context, req *mcsv1.PutServiceExportRequest) (*mcsv1.PutServiceExportResponse, error) {
	var v validator
	key := validateExportKey(&v, req.GetClusterset(), req.GetCluster(), req.GetNamespace(), req.GetName())
	v.exportSpec("spec", req.GetSpec())
	if err := v.err(); err != nil {
		return nil, err
	}
	if err := s.requireCluster(ctx, key.ClusterKey()); err != nil {
		return nil, err
	}
	now := s.now()
	spec := convert.ExportSpecFromProto(req.GetSpec())
	normalizeSpec(&spec)
	stored, hasStored := s.view.ExportRecord(key)
	if spec.ExportCreationTimestamp.IsZero() {
		spec.ExportCreationTimestamp = now
		if hasStored {
			spec.ExportCreationTimestamp = stored.Spec.ExportCreationTimestamp
		}
	}
	gen, assigned := generation(req.GetGeneration(), stored.Generation, now)
	rec := store.ExportRecord{Key: key, Spec: spec, Generation: gen, UpdatedAt: now}
	err := s.store.PutExport(ctx, rec)
	if cur, stale := store.StaleGeneration(err); stale && assigned {
		rec.Generation = registry.NextGeneration(cur, now)
		err = s.store.PutExport(ctx, rec)
	}
	if cur, stale := store.StaleGeneration(err); stale {
		return nil, staleGeneration("export", rec.Generation, cur)
	}
	if err != nil {
		return nil, storeError(err, "put export")
	}
	s.view.ApplyExport(rec)
	s.log.Debug("export stored", zap.String("clusterset", key.ClusterSet), zap.String("cluster", key.Cluster),
		zap.String("namespace", key.Namespace), zap.String("name", key.Name), zap.Uint64("generation", rec.Generation))
	st, _ := s.view.ExportStatusOf(key)
	return &mcsv1.PutServiceExportResponse{Export: exportToProto(st)}, nil
}

// DeleteServiceExport implements mcsv1.ServiceExportServiceServer.
func (s *Server) DeleteServiceExport(ctx context.Context, req *mcsv1.DeleteServiceExportRequest) (*mcsv1.DeleteServiceExportResponse, error) {
	var v validator
	key := validateExportKey(&v, req.GetClusterset(), req.GetCluster(), req.GetNamespace(), req.GetName())
	if err := v.err(); err != nil {
		return nil, err
	}
	if err := s.store.DeleteExport(ctx, key); err != nil {
		return nil, storeError(err, "delete export")
	}
	s.view.ApplyExportDelete(key)
	s.log.Debug("export deleted", zap.String("clusterset", key.ClusterSet), zap.String("cluster", key.Cluster),
		zap.String("namespace", key.Namespace), zap.String("name", key.Name))
	return &mcsv1.DeleteServiceExportResponse{}, nil
}

// ListServiceExports implements mcsv1.ServiceExportServiceServer.
func (s *Server) ListServiceExports(_ context.Context, req *mcsv1.ListServiceExportsRequest) (*mcsv1.ListServiceExportsResponse, error) {
	var v validator
	key := validateClusterKey(&v, req.GetClusterset(), req.GetCluster())
	if err := v.err(); err != nil {
		return nil, err
	}
	resp := &mcsv1.ListServiceExportsResponse{}
	for _, st := range s.view.Exports(key) {
		resp.Exports = append(resp.Exports, exportToProto(st))
	}
	return resp, nil
}

// ReplaceEndpoints implements mcsv1.ServiceExportServiceServer.
func (s *Server) ReplaceEndpoints(ctx context.Context, req *mcsv1.ReplaceEndpointsRequest) (*mcsv1.ReplaceEndpointsResponse, error) {
	var v validator
	key := validateExportKey(&v, req.GetClusterset(), req.GetCluster(), req.GetNamespace(), req.GetName())
	if err := v.err(); err != nil {
		return nil, err
	}
	if n := len(req.GetEndpoints()); n > mcsapi.MaxEndpointsPerExport {
		return nil, tooManyEndpoints(n)
	}
	eps := make([]store.Endpoint, 0, len(req.GetEndpoints()))
	var ready int32
	for i, e := range req.GetEndpoints() {
		v.endpoint(fmt.Sprintf("endpoints[%d]", i), e)
		ep := convert.EndpointFromProto(e)
		if ep.AddressType == "" {
			ep.AddressType = convert.AddressTypeFromProto(addressType(ep.Addresses))
		}
		for j := range ep.Ports {
			if ep.Ports[j].Protocol == "" {
				ep.Ports[j].Protocol = store.ProtocolTCP
			}
		}
		if ep.Conditions.Ready {
			ready++
		}
		eps = append(eps, ep)
	}
	if err := v.err(); err != nil {
		return nil, err
	}
	if err := s.requireCluster(ctx, key.ClusterKey()); err != nil {
		return nil, err
	}
	now := s.now()
	storedGen, _ := s.view.EndpointSetGeneration(key)
	gen, assigned := generation(req.GetGeneration(), storedGen, now)
	rec := store.EndpointSetRecord{Key: key, Endpoints: eps, Generation: gen, UpdatedAt: now}
	err := s.store.PutEndpointSet(ctx, rec)
	if cur, stale := store.StaleGeneration(err); stale && assigned {
		rec.Generation = registry.NextGeneration(cur, now)
		err = s.store.PutEndpointSet(ctx, rec)
	}
	if cur, stale := store.StaleGeneration(err); stale {
		return nil, staleGeneration("endpoint set", rec.Generation, cur)
	}
	if err != nil {
		return nil, storeError(err, "replace endpoints")
	}
	s.view.ApplyEndpointSet(rec)
	return &mcsv1.ReplaceEndpointsResponse{
		Generation:         rec.Generation,
		EndpointCount:      int32(len(eps)),
		ReadyEndpointCount: ready,
	}, nil
}

// GetServiceImport implements mcsv1.ServiceImportServiceServer.
func (s *Server) GetServiceImport(_ context.Context, req *mcsv1.GetServiceImportRequest) (*mcsv1.GetServiceImportResponse, error) {
	var v validator
	v.dnsLabel("clusterset", req.GetClusterset())
	v.dnsLabel("namespace", req.GetNamespace())
	v.dnsLabel("name", req.GetName())
	if err := v.err(); err != nil {
		return nil, err
	}
	o, ok := s.view.Get(req.GetClusterset(), view.KindServiceImport, view.ImportKey(req.GetNamespace(), req.GetName()))
	if !ok {
		return nil, notFound("no ServiceImport %s/%s in clusterset %q", req.GetNamespace(), req.GetName(), req.GetClusterset())
	}
	return &mcsv1.GetServiceImportResponse{ServiceImport: o.Message(req.GetIncludeEndpoints()).(*mcsv1.ServiceImport)}, nil
}

func importFilter(v *validator, namespace string, selector *mcsv1.LabelSelector, endpoints bool) view.ImportFilter {
	v.optionalDNSLabel("namespace", namespace)
	v.selector("selector", selector)
	return view.ImportFilter{Namespace: namespace, Selector: selector.GetMatchLabels(), Endpoints: endpoints}
}

// ListServiceImports implements mcsv1.ServiceImportServiceServer.
func (s *Server) ListServiceImports(_ context.Context, req *mcsv1.ListServiceImportsRequest) (*mcsv1.ListServiceImportsResponse, error) {
	var v validator
	v.dnsLabel("clusterset", req.GetClusterset())
	f := importFilter(&v, req.GetNamespace(), req.GetSelector(), req.GetIncludeEndpoints())
	if err := v.err(); err != nil {
		return nil, err
	}
	msgs, rev := s.view.List(req.GetClusterset(), f)
	resp := &mcsv1.ListServiceImportsResponse{Revision: rev}
	for _, m := range msgs {
		resp.ServiceImports = append(resp.ServiceImports, m.(*mcsv1.ServiceImport))
	}
	return resp, nil
}

// WatchServiceImports implements mcsv1.ServiceImportServiceServer.
func (s *Server) WatchServiceImports(req *mcsv1.WatchServiceImportsRequest, stream grpc.ServerStreamingServer[mcsv1.WatchServiceImportsResponse]) error {
	var v validator
	v.dnsLabel("clusterset", req.GetClusterset())
	f := importFilter(&v, req.GetNamespace(), req.GetSelector(), req.GetIncludeEndpoints())
	if err := v.err(); err != nil {
		return err
	}
	return watchError(s.hub.Watch(stream.Context(), req.GetClusterset(), f, req.GetResumeRevision(), func(ev view.WatchEvent) error {
		resp := &mcsv1.WatchServiceImportsResponse{Type: ev.Type, Revision: ev.Revision}
		if ev.Object != nil {
			resp.ServiceImport = ev.Object.(*mcsv1.ServiceImport)
		}
		return stream.Send(resp)
	}))
}
