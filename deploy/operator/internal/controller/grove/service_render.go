// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package grove

import (
	"fmt"
	"hash/fnv"
	"maps"
	"strings"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	componentdeployment "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/component"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

// maxServiceNameLength is the Kubernetes limit on a Service name, which must be a
// DNS-1035 label.
const maxServiceNameLength = 63

// ElasticEPLeaderServiceName returns the name a single-pod elastic-EP leader is
// reachable at: its component service name plus a "-ray" suffix.
//
// That suffix can push a long name past the DNS-1035 limit, which the API server
// rejects, failing the whole stable-resources reconcile. Truncate with a hash suffix
// instead, as PCSNameForDGD does. The hash must be deterministic because the follower
// derives this name independently rather than reading it back from the Service.
func ElasticEPLeaderServiceName(componentServiceName string) string {
	name := strings.ToLower(strings.ReplaceAll(componentServiceName+"-ray", ".", "-"))
	if len(name) <= maxServiceNameLength {
		return name
	}

	hash := fnv.New32a()
	hash.Write([]byte(name))
	suffix := fmt.Sprintf("%04x", hash.Sum32()&0xFFFF)
	return name[:maxServiceNameLength-len(suffix)-1] + "-" + suffix
}

// RenderElasticEPLeaderService returns a headless Service that gives a single-pod
// elastic-EP leader a stable address its followers join with
// `ray start --address=<service>:6379` (see injectElasticEPRayLaunchFlags).
//
// Two spec choices are deliberate:
//
//   - clusterIP: None, because a load-balanced ClusterIP cannot carry Ray's multi-port
//     head<->worker traffic.
//   - PublishNotReadyAddresses, because the leader's engine only starts once its
//     data-parallel ranks join, so gating the address on readiness would deadlock.
//
// It returns nil unless the component is a single-pod elastic-EP Ray leader.
// component must be non-nil. Inputs are not modified; elasticRayLaunch is the resolved launch mode.
func RenderElasticEPLeaderService(component *v1beta1.DynamoComponentDeploymentSharedSpec, params componentdeployment.ServiceParams, elasticRayLaunch bool) *corev1.Service {
	if !elasticRayLaunch || component.GetNumberOfNodes() > 1 || (component.Replicas != nil && *component.Replicas != 1) {
		return nil
	}

	// Copy the caller's metadata so the Service carries the component's labels and
	// annotations without aliasing the caller's maps.
	labels := make(map[string]string)
	maps.Copy(labels, params.Labels)

	annotations := make(map[string]string)
	maps.Copy(annotations, params.Annotations)

	// Select the elastic-EP component itself; the caller's single-pod gate is what makes
	// this resolve to the one leader pod.
	selector := map[string]string{
		commonconsts.KubeLabelDynamoComponentType: params.ComponentType,
		commonconsts.KubeLabelDynamoNamespace:     params.DynamoNamespace,
		commonconsts.KubeLabelDynamoComponent:     params.ComponentName,
	}

	return &corev1.Service{
		Name:        ElasticEPLeaderServiceName(params.ServiceName),
		Namespace:   params.Namespace,
		Labels:      labels,
		Annotations: annotations,
		Spec: corev1.ServiceSpec{
			ClusterIP:                corev1.ClusterIPNone,
			PublishNotReadyAddresses: true,
			Selector:                 selector,
			Ports: []corev1.ServicePort{
				{
					// Ray GCS head port (VLLMPort). Followers connect their raylet
					// here via `ray start --address=<svc>:6379`.
					Name:       "ray-gcs",
					Port:       6379,
					TargetPort: intstr.FromInt(6379),
					Protocol:   corev1.ProtocolTCP,
				},
				{
					// Leader system/health port; the follower's /live gate polls it.
					Name:       commonconsts.DynamoSystemPortName,
					Port:       commonconsts.DynamoSystemPort,
					TargetPort: intstr.FromString(commonconsts.DynamoSystemPortName),
					Protocol:   corev1.ProtocolTCP,
				},
			},
		},
	}
}
