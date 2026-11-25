/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package multicluster renders the ServiceExports that announce a Global Router
// pool to other clusters through the Kubernetes Multi-Cluster Services (MCS) API,
// lists them, and reads the cluster's ID in its ClusterSet.
package multicluster

import (
	"context"
	"fmt"
	"slices"
	"strings"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// ServiceExportGroupKind is the SIG Multicluster ServiceExport. Every supported
// implementation reads it in the cluster that runs the Service, Karmada included:
// its ServiceExport controller watches this kind in member clusters. Karmada
// watches a member only while its control plane has propagated at least one
// ServiceExport there, so keep one in place in each member. That setup is outside
// the operator. The operator reads and writes it at the multicluster.x-k8s.io
// version that the multiClusterExport feature gate resolved.
var ServiceExportGroupKind = schema.GroupKind{
	Group: "multicluster.x-k8s.io",
	Kind:  "ServiceExport",
}

// ClusterPropertyGVK is the About API's ClusterProperty (KEP-2149), which holds
// facts about the cluster for its ClusterSet.
var ClusterPropertyGVK = schema.GroupVersionKind{
	Group:   "about.k8s.io",
	Version: "v1alpha1",
	Kind:    "ClusterProperty",
}

const (
	// ClusterIDProperty names the ClusterProperty that holds the cluster's ID in its
	// ClusterSet. A standard MCS implementation labels the EndpointSlices it imports
	// with that ID, and the Global Discovery Service takes a pool's cluster from the
	// label.
	ClusterIDProperty = "cluster.clusterset.k8s.io"

	// poolRelayServiceSuffix names a pool's PoolRelay Service, as the Global
	// Discovery Service expects by default.
	poolRelayServiceSuffix = "-pool-relay"

	// maxKarmadaServiceNameLength is the longest Service name Karmada imports.
	// Karmada labels each collected slice kubernetes.io/service-name:
	// derived-<service>, and a label value has at most 63 characters.
	maxKarmadaServiceNameLength = 55
)

// PoolRelayServiceName returns the name of the PoolRelay Service of the DGD named
// dgdName, <DGD name>-pool-relay, as the Global Discovery Service expects by
// default. The operator creates this Service for a DynamoPoolRelay of the same
// name.
func PoolRelayServiceName(dgdName string) string {
	return dgdName + poolRelayServiceSuffix
}

// FrontendServiceNames returns the names of the Services the operator creates for
// the DGD's frontend components, in component order: <DGD name>-<component name>,
// lowercased, with dots replaced by dashes.
func FrontendServiceNames(dgd *v1beta1.DynamoGraphDeployment) []string {
	var names []string
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if component.ComponentType != v1beta1.ComponentTypeFrontend {
			continue
		}
		name := dynamo.NormalizeKubeResourceName(dynamo.GetDCDResourceName(dgd, component.ComponentName, ""))
		if !slices.Contains(names, name) {
			names = append(names, name)
		}
	}
	return names
}

// ServiceNames returns the Services the operator exports for the DGD as a Global
// Router pool: its PoolRelay Service, then its frontend Services.
func ServiceNames(dgd *v1beta1.DynamoGraphDeployment) []string {
	names := []string{PoolRelayServiceName(dgd.Name)}
	for _, name := range FrontendServiceNames(dgd) {
		if !slices.Contains(names, name) {
			names = append(names, name)
		}
	}
	return names
}

// MaxServiceNameLength returns the longest Service name that implementation can
// export with the cluster it came from. Standard implementations label imported
// EndpointSlices with their source cluster, so any Service name works.
func MaxServiceNameLength(implementation configv1alpha1.MCSImplementation) int {
	if implementation == configv1alpha1.MCSImplementationKarmada {
		return maxKarmadaServiceNameLength
	}
	return validation.DNS1035LabelMaxLength
}

// NameProblem is why a Service name cannot be exported.
type NameProblem int

const (
	// NameExportable means that the implementation can export a Service of the name.
	NameExportable NameProblem = iota
	// NameInvalid means that no Service can have the name.
	NameInvalid
	// NameTooLong means that the implementation imports no Service name that long.
	NameTooLong
)

// CheckServiceName returns whether implementation can export a Service named name
// with the cluster it came from, and describes the problem when it cannot.
func CheckServiceName(implementation configv1alpha1.MCSImplementation, name string) (NameProblem, string) {
	if problems := validation.IsDNS1035Label(name); len(problems) > 0 {
		return NameInvalid, strings.Join(problems, "; ")
	}
	if maxLength := MaxServiceNameLength(implementation); len(name) > maxLength {
		return NameTooLong, fmt.Sprintf("%s imports no Service name longer than %d characters", implementation, maxLength)
	}
	return NameExportable, ""
}

// UsesClusterSetID reports whether implementation labels the EndpointSlices it
// imports with the cluster's ID in its ClusterSet, the ClusterIDProperty. Karmada
// names its member clusters itself.
func UsesClusterSetID(implementation configv1alpha1.MCSImplementation) bool {
	return implementation == configv1alpha1.MCSImplementationStandard
}

// NewServiceExport returns an empty ServiceExport at version, to watch exports or
// to name one. version must be a multicluster.x-k8s.io version that the cluster
// serves.
func NewServiceExport(version string) *unstructured.Unstructured {
	serviceExport := &unstructured.Unstructured{}
	serviceExport.SetGroupVersionKind(ServiceExportGroupKind.WithVersion(version))
	return serviceExport
}

// RenderServiceExport returns the ServiceExport at version for the Service named
// name in namespace, labeled with the DGD that exports it. It sets no owner
// reference. version must be a multicluster.x-k8s.io version that the cluster
// serves.
func RenderServiceExport(version, namespace, name, dgdName string) *unstructured.Unstructured {
	serviceExport := NewServiceExport(version)
	serviceExport.SetNamespace(namespace)
	serviceExport.SetName(name)
	serviceExport.SetLabels(map[string]string{
		commonconsts.KubeLabelDynamoGraphDeploymentName: dgdName,
	})
	return serviceExport
}

// Export is a ServiceExport as the operator sees it.
type Export struct {
	Name string
	UID  types.UID
	// ResourceVersion is the version of the export that was read. A write
	// conditioned on it fails once the export has changed.
	ResourceVersion string
	// Labels are the export's labels, and Controller its controller reference,
	// nil without one: the operator writes both on the exports of a pool.
	Labels     map[string]string
	Controller *metav1.OwnerReference
	// Deleting is true once the export's deletion has started.
	Deleting bool
}

// ControlledBy reports whether the object with uid controls the export.
func (e Export) ControlledBy(uid types.UID) bool {
	return e.Controller != nil && e.Controller.UID == uid
}

// ListExports returns the ServiceExports in namespace by name, read at version.
// reader should serve unstructured objects from a cache: the manager's client
// reads them from the API server.
func ListExports(
	ctx context.Context,
	reader client.Reader,
	version, namespace string,
) (map[string]Export, error) {
	list := &unstructured.UnstructuredList{}
	list.SetGroupVersionKind(schema.GroupVersionKind{
		Group: ServiceExportGroupKind.Group, Version: version, Kind: ServiceExportGroupKind.Kind + "List",
	})
	if err := reader.List(ctx, list, client.InNamespace(namespace)); err != nil {
		return nil, fmt.Errorf("failed to list ServiceExports: %w", err)
	}
	exports := make(map[string]Export, len(list.Items))
	for i := range list.Items {
		item := &list.Items[i]
		export := Export{
			Name:            item.GetName(),
			UID:             item.GetUID(),
			ResourceVersion: item.GetResourceVersion(),
			Labels:          item.GetLabels(),
			Controller:      metav1.GetControllerOf(item),
			Deleting:        item.GetDeletionTimestamp() != nil,
		}
		exports[export.Name] = export
	}
	return exports, nil
}

// NewClusterProperty returns an empty ClusterProperty, to watch properties or to
// read one.
func NewClusterProperty() *unstructured.Unstructured {
	property := &unstructured.Unstructured{}
	property.SetGroupVersionKind(ClusterPropertyGVK)
	return property
}

// ClusterPropertyValue returns the value of property, or "" without one.
func ClusterPropertyValue(property *unstructured.Unstructured) string {
	value, _, _ := unstructured.NestedString(property.Object, "spec", "value")
	return value
}

// ReadClusterIDProperty returns the About API's ClusterProperty named
// ClusterIDProperty, whose value is the cluster's ID in its ClusterSet, or nil
// when the cluster has no such property or does not serve the About API. Any
// other failure is an error. reader should be the manager's cache, which a watch
// of ClusterProperties fills; without permission to list and watch them, that
// watch never syncs, and the controller does not start.
func ReadClusterIDProperty(ctx context.Context, reader client.Reader) (*unstructured.Unstructured, error) {
	property := NewClusterProperty()
	err := reader.Get(ctx, client.ObjectKey{Name: ClusterIDProperty}, property)
	if apierrors.IsNotFound(err) || meta.IsNoMatchError(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("failed to read ClusterProperty %s: %w", ClusterIDProperty, err)
	}
	return property, nil
}
