/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

type IngressTLSSpec struct {
	SecretName string `json:"secretName,omitempty"`
}

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

func (i *IngressSpec) IsVirtualServiceEnabled() bool {
	if i == nil {
		return false
	}
	return i.Enabled && i.UseVirtualService && i.VirtualServiceGateway != nil
}
