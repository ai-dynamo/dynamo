// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package api

import (
	"strings"
)

// Prefix is the prefix of every registry label, annotation and finalizer key.
// The controller forwards a Service's labels and annotations with this prefix
// to the registry as exported labels and annotations.
const Prefix = "dynamo.nvidia.com/mcs-"

// Service labels that drive exports.
const (
	// LabelExport set to ExportEnabled on a Service makes the controller
	// create and own a ServiceExport of the same name.
	LabelExport = Prefix + "export"
	// ExportEnabled is the value of LabelExport that enables the export.
	ExportEnabled = "true"
)

// Endpoint attributes carrying the target reference of an exported
// endpoint. The mcs-controller sets kind, name and uid from the
// EndpointSlice endpoint's targetRef. When it imports an endpoint with a
// name, it sets the imported endpoint's targetRef to that name, the kind
// (Pod when absent) and the namespace when present; the uid stays behind,
// since it names an object of another cluster. Consumers such as the Global
// Discovery Service name a replica after targetRef.name.
const (
	EndpointAttrTargetKind      = "targetRef.kind"
	EndpointAttrTargetName      = "targetRef.name"
	EndpointAttrTargetNamespace = "targetRef.namespace"
	EndpointAttrTargetUID       = "targetRef.uid"
)

// Labels and finalizers on objects the mcs-controller creates.
const (
	// LabelManagedBy marks objects the controller created; the value is
	// ManagedByController.
	LabelManagedBy = Prefix + "managed-by"
	// ManagedByController is the value of LabelManagedBy.
	ManagedByController = "mcs-controller"
	// LabelClusterSet scopes controller-created objects to a clusterset.
	LabelClusterSet = Prefix + "clusterset"
	// FinalizerServiceExport lets the controller deregister an export from
	// the registry before the ServiceExport is deleted.
	FinalizerServiceExport = Prefix + "service-export"
	// FieldManager is the server-side apply field manager of the controller.
	FieldManager = "mcs-controller"
	// LeaderElectionID is the name of the controller's leader election Lease.
	LeaderElectionID = "mcs-controller.dynamo.nvidia.com"
)

// Kubernetes and KEP-1645 keys the controller writes.
const (
	// LabelEndpointSliceManagedBy is the EndpointSlice managed-by label;
	// imported slices carry EndpointSliceManagedBy.
	LabelEndpointSliceManagedBy = "endpointslice.kubernetes.io/managed-by"
	// EndpointSliceManagedBy is the value of LabelEndpointSliceManagedBy on
	// imported slices.
	EndpointSliceManagedBy = "mcs-controller.dynamo.nvidia.com"
	// LabelKubernetesServiceName ties an EndpointSlice to its Service (the
	// derived Service for imported slices).
	LabelKubernetesServiceName = "kubernetes.io/service-name"
	// LabelMCSServiceName names the multi-cluster service of an
	// EndpointSlice.
	LabelMCSServiceName = "multicluster.kubernetes.io/service-name"
	// LabelMCSSourceCluster names the cluster an imported EndpointSlice
	// comes from.
	LabelMCSSourceCluster = "multicluster.kubernetes.io/source-cluster"
	// LabelTopologyRegion is the well-known region label. Imported slices
	// carry the exporting cluster's region in it when the cluster has one,
	// so consumers such as the Global Discovery Service can place a pool
	// without asking the registry.
	LabelTopologyRegion = "topology.kubernetes.io/region"
	// AnnotationMCSDerivedService on a ServiceImport names its derived
	// Service.
	AnnotationMCSDerivedService = "multicluster.kubernetes.io/derived-service"
	// ClusterPropertyClusterSet is the ClusterProperty holding the clusterset
	// name.
	ClusterPropertyClusterSet = "clusterset.k8s.io"
	// ClusterPropertyClusterID is the ClusterProperty holding the cluster id.
	ClusterPropertyClusterID = "cluster.clusterset.k8s.io"
)

// IsExported reports whether Service labels request a label-driven export.
func IsExported(labels map[string]string) bool {
	return labels[LabelExport] == ExportEnabled
}

// IsMCSKey reports whether a label or annotation key has the registry prefix.
func IsMCSKey(key string) bool {
	return strings.HasPrefix(key, Prefix) && len(key) > len(Prefix)
}
