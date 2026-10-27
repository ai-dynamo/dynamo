/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package multicluster renders the ServiceExports that announce a Global Router
// pool to other clusters through the Kubernetes Multi-Cluster Services (MCS) API.
package multicluster

import (
	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation"
)

// ServiceExportGVK is the SIG Multicluster ServiceExport. Every supported
// implementation reads it in the cluster that runs the Service, Karmada included:
// its ServiceExport controller watches this kind in member clusters. Karmada
// watches a member only while its control plane has propagated at least one
// ServiceExport there, so keep one in place in each member. That setup is outside
// the operator.
var ServiceExportGVK = schema.GroupVersionKind{
	Group:   "multicluster.x-k8s.io",
	Version: "v1alpha1",
	Kind:    "ServiceExport",
}

const (
	// PoolRelayServiceSuffix names a pool's PoolRelay Service, <DGD name>-pool-relay,
	// as the Global Discovery Service expects by default.
	PoolRelayServiceSuffix = "-pool-relay"
	// FrontendServiceSuffix names a pool's frontend Service, <DGD name>-frontend, as
	// the Global Discovery Service expects by default. The operator creates this
	// Service for a frontend component named "Frontend".
	FrontendServiceSuffix = "-frontend"

	// maxKarmadaServiceNameLength is the longest Service name Karmada imports.
	// Karmada labels each collected slice kubernetes.io/service-name:
	// derived-<service>, and a label value has at most 63 characters.
	maxKarmadaServiceNameLength = 55
)

// ServiceNames returns the names of a pool's PoolRelay and frontend Services, the
// Services the operator exports for a Global Router pool. Like the operator's own
// Service names, they replace the dots in a DGD name with dashes.
func ServiceNames(dgdName string) []string {
	return []string{
		dynamo.NormalizeKubeResourceName(dgdName + PoolRelayServiceSuffix),
		dynamo.NormalizeKubeResourceName(dgdName + FrontendServiceSuffix),
	}
}

// IsPool reports whether a DGD with these annotations is marked as a Global Router
// pool. A nil map is supported and is not a pool.
func IsPool(annotations map[string]string) bool {
	return annotations[commonconsts.KubeAnnotationGlobalRouterPool] == "true"
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

// ServiceExportMetadata returns an empty metadata-only ServiceExport, to read,
// delete or watch exports through the cache without their content.
func ServiceExportMetadata() *metav1.PartialObjectMetadata {
	serviceExport := &metav1.PartialObjectMetadata{}
	serviceExport.SetGroupVersionKind(ServiceExportGVK)
	return serviceExport
}

// RenderServiceExport returns the ServiceExport for the Service named name in
// namespace, labeled with the DGD that exports it. It sets no owner reference.
func RenderServiceExport(namespace, name, dgdName string) *unstructured.Unstructured {
	serviceExport := &unstructured.Unstructured{}
	serviceExport.SetGroupVersionKind(ServiceExportGVK)
	serviceExport.SetNamespace(namespace)
	serviceExport.SetName(name)
	serviceExport.SetLabels(map[string]string{
		commonconsts.KubeLabelDynamoGraphDeploymentName: dgdName,
	})
	return serviceExport
}
