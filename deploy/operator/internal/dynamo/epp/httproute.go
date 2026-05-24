/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package epp

import (
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	gatewayv1 "sigs.k8s.io/gateway-api/apis/v1"
)

const (
	// gatewayAPIGroup is the core Gateway API group (HTTPRoute, Gateway).
	gatewayAPIGroup = "gateway.networking.k8s.io"
	// defaultHTTPRoutePathPrefix matches all paths so the gateway forwards the
	// full OpenAI-compatible surface (/v1/chat/completions, /v1/models, ...).
	defaultHTTPRoutePathPrefix = "/"
)

// GenerateHTTPRoute generates the HTTPRoute that binds the public model to the
// EPP-fronted InferencePool, closing the one GAIE resource the operator did not
// previously emit (see INFERENCE_GATEWAY_FEATURE_SPIKE.md, Fix-4).
//
// It mirrors deploy/inference-gateway/standalone/helm/dynamo-gaie/templates/http-router.yaml:
// parentRef -> the Gateway, single rule with a PathPrefix("/") match and a
// backendRef to the InferencePool (inference.networking.k8s.io).
//
// The route is only generated when the inference-gateway annotation carries a
// gateway name; the caller (reconcileEPPResources) gates on that.
func GenerateHTTPRoute(
	dgd *v1beta1.DynamoGraphDeployment,
	eppConfig *v1beta1.EPPConfig,
	gatewayName string,
) *gatewayv1.HTTPRoute {
	poolName := GetPoolName(dgd.Name, eppConfig)
	poolNamespace := GetPoolNamespace(dgd.Namespace, eppConfig)

	parent := gatewayv1.ParentReference{
		Group: ptr.To(gatewayv1.Group(gatewayAPIGroup)),
		Kind:  ptr.To(gatewayv1.Kind("Gateway")),
		Name:  gatewayv1.ObjectName(gatewayName),
	}
	// Default the Gateway's namespace to the pool's when not cross-namespace.
	parent.Namespace = ptr.To(gatewayv1.Namespace(poolNamespace))

	return &gatewayv1.HTTPRoute{
		ObjectMeta: metav1.ObjectMeta{
			Name:      dgd.Name + "-route",
			Namespace: poolNamespace,
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
			},
		},
		Spec: gatewayv1.HTTPRouteSpec{
			CommonRouteSpec: gatewayv1.CommonRouteSpec{
				ParentRefs: []gatewayv1.ParentReference{parent},
			},
			Rules: []gatewayv1.HTTPRouteRule{
				{
					Matches: []gatewayv1.HTTPRouteMatch{
						{
							Path: &gatewayv1.HTTPPathMatch{
								Type:  ptr.To(gatewayv1.PathMatchPathPrefix),
								Value: ptr.To(defaultHTTPRoutePathPrefix),
							},
						},
					},
					BackendRefs: []gatewayv1.HTTPBackendRef{
						{
							BackendRef: gatewayv1.BackendRef{
								BackendObjectReference: gatewayv1.BackendObjectReference{
									Group: ptr.To(gatewayv1.Group(InferencePoolGroup)),
									Kind:  ptr.To(gatewayv1.Kind("InferencePool")),
									Name:  gatewayv1.ObjectName(poolName),
									Port:  ptr.To(gatewayv1.PortNumber(consts.DynamoServicePort)),
								},
								Weight: ptr.To(int32(1)),
							},
						},
					},
				},
			},
		},
	}
}
