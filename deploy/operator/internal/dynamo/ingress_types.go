/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

// IngressTLSSpec is the internal TLS subset used when preserving v1alpha1
// ingress behavior while the controllers reconcile v1beta1 objects.
type IngressTLSSpec struct {
	SecretName string `json:"secretName,omitempty"`
}

// IngressSpec is an internal controller compatibility shape for ingress config.
// It is intentionally not part of the v1beta1 API; it lets the migrated
// controllers keep reconciling ingress and virtual service resources from
// preserved v1alpha1 payloads and operator-level defaults.
type IngressSpec struct {
	Enabled                    bool              `json:"enabled,omitempty"`
	Host                       string            `json:"host,omitempty"`
	UseVirtualService          bool              `json:"useVirtualService,omitempty"`
	VirtualServiceGateway      *string           `json:"virtualServiceGateway,omitempty"`
	HostPrefix                 *string           `json:"hostPrefix,omitempty"`
	Annotations                map[string]string `json:"annotations,omitempty"`
	Labels                     map[string]string `json:"labels,omitempty"`
	TLS                        *IngressTLSSpec   `json:"tls,omitempty"`
	HostSuffix                 *string           `json:"hostSuffix,omitempty"`
	IngressControllerClassName *string           `json:"ingressControllerClassName,omitempty"`
}

// IsVirtualServiceEnabled reports whether this ingress config should reconcile
// an Istio VirtualService in addition to, or instead of, a Kubernetes Ingress.
func (i *IngressSpec) IsVirtualServiceEnabled() bool {
	if i == nil {
		return false
	}
	return i.Enabled && i.UseVirtualService && i.VirtualServiceGateway != nil
}
