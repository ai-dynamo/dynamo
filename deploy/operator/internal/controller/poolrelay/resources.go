/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package poolrelay

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"strings"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// Use the legacy runtime entry point, flag and gRPC service until the runtime
// rename lands. DEP #15742 retains these names as compatibility aliases.
const relayGRPCService = "dynamo.kvrelay.v1.KvEventRelay"

// relayResourceName also supports CR names longer than the Service name limit.
func relayResourceName(name string) string {
	digest := sha256.Sum256([]byte(name))
	return fmt.Sprintf("relay-%s-%x", strings.ReplaceAll(name[:min(len(name), 40)], ".", "-"), digest[:5])
}

// renderRelayResources constructs the namespace-local runtime in dependency order.
// relay must be non-nil. The returned objects do not share mutable data with it.
func renderRelayResources(relay *v1alpha1.DynamoPoolRelay, connection relayConnection, sources relaySources) ([]client.Object, error) {
	main, err := relayMainContainer(relay)
	if err != nil {
		return nil, err
	}
	// Use the CR UID to keep replacement relays from selecting their predecessor's pods.
	name := relayResourceName(relay.Name)
	labels := map[string]string{"app.kubernetes.io/name": "dynamo-relay", "nvidia.com/relay-uid": string(relay.UID)}
	metadata := metav1.ObjectMeta{Name: name, Namespace: relay.Namespace}
	serviceAccount := &corev1.ServiceAccount{ObjectMeta: *metadata.DeepCopy()}
	role := &rbacv1.Role{ObjectMeta: *metadata.DeepCopy(), Rules: []rbacv1.PolicyRule{
		{APIGroups: []string{"nvidia.com"}, Resources: []string{"dynamoworkermetadatas"}, Verbs: []string{"get", "list", "watch", "create", "patch", "delete"}},
		{APIGroups: []string{"discovery.k8s.io"}, Resources: []string{"endpointslices"}, Verbs: []string{"get", "list", "watch"}},
	}}
	binding := &rbacv1.RoleBinding{ObjectMeta: *metadata.DeepCopy(),
		RoleRef:  rbacv1.RoleRef{APIGroup: rbacv1.GroupName, Kind: "Role", Name: name},
		Subjects: []rbacv1.Subject{{Kind: "ServiceAccount", Name: name, Namespace: relay.Namespace}},
	}

	// Publish only the cluster-internal listener; discovery labels are required by pod-mode registration.
	service := &corev1.Service{ObjectMeta: *metadata.DeepCopy(), Spec: corev1.ServiceSpec{
		Type: corev1.ServiceTypeClusterIP, Selector: labels,
		Ports: []corev1.ServicePort{{Name: "grpc", Port: 5561, TargetPort: intstr.FromString("grpc"), Protocol: corev1.ProtocolTCP}},
	}}
	service.Labels = map[string]string{"nvidia.com/dynamo-discovery-backend": "kubernetes", "nvidia.com/dynamo-discovery-enabled": "true"}

	env := []corev1.EnvVar{
		{Name: "DYN_NAMESPACE", Value: name},
		{Name: "DYN_DISCOVERY_BACKEND", Value: "kubernetes"},
		{Name: "DYN_KUBE_DISCOVERY_MODE", Value: "pod"},
		{Name: "DYN_RELAY_CONNECTION_REVISION", Value: sources.ConnectionRevision},
		{Name: "DYN_SYSTEM_PORT", Value: "8081"},
		{Name: "DYN_TCP_RESPONSE_STREAM_PORT", Value: "5562"},
	}
	for _, field := range []struct{ name, path string }{
		{"POD_NAME", "metadata.name"}, {"POD_NAMESPACE", "metadata.namespace"},
		{"POD_UID", "metadata.uid"}, {"DYN_TCP_RESPONSE_STREAM_HOST", "status.podIP"},
	} {
		env = append(env, corev1.EnvVar{Name: field.name, ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{APIVersion: "v1", FieldPath: field.path}}})
	}
	// Merge the compatible transport env; pod identity/return addresses stay Relay-local.
	for _, entry := range connection.Env {
		if entry.Name == "DYN_DISCOVERY_BACKEND" || entry.Name == "DYN_KUBE_DISCOVERY_MODE" {
			continue
		}
		env = append(env, *entry.DeepCopy())
	}
	sourcesData, _ := json.Marshal(sources)
	configMap := &corev1.ConfigMap{ObjectMeta: *metadata.DeepCopy(), Data: map[string]string{"sources.json": string(sourcesData)}}
	mounts := append([]corev1.VolumeMount{}, connection.Mounts...)
	mounts = append(mounts, corev1.VolumeMount{Name: relaySourcesVolume, MountPath: "/var/run/dynamo-relay/sources", ReadOnly: true})
	volumes := append([]corev1.Volume{}, connection.Volumes...)
	volumes = append(volumes, corev1.Volume{Name: relaySourcesVolume, VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{LocalObjectReference: corev1.LocalObjectReference{Name: name}, DefaultMode: ptr.To(int32(420))}}})

	// Keep modest starting allocations overridable as a complete Kubernetes resource block.
	resources := corev1.ResourceRequirements{
		Requests: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("1"), corev1.ResourceMemory: resource.MustParse("1Gi")},
		Limits:   corev1.ResourceList{corev1.ResourceMemory: resource.MustParse("4Gi")},
	}
	if main.Resources.Limits != nil || main.Resources.Requests != nil {
		resources = *main.Resources.DeepCopy()
	}
	// Match Kubernetes defaulting when a limit is given without its corresponding request.
	for name, limit := range resources.Limits {
		if _, exists := resources.Requests[name]; !exists {
			if resources.Requests == nil {
				resources.Requests = corev1.ResourceList{}
			}
			resources.Requests[name] = limit.DeepCopy()
		}
	}

	container := corev1.Container{
		Name: "main", Image: main.Image, ImagePullPolicy: corev1.PullIfNotPresent,
		Command: []string{"python3", "-m", "dynamo.kv_dc_relay"},
		Args:    []string{"--dc-id", relay.Spec.ClusterID, "--sources-file", relaySourcesPath, "--bind", "0.0.0.0:5561"},
		Env:     env, Resources: resources, VolumeMounts: mounts,
		Ports:                  []corev1.ContainerPort{{Name: "grpc", ContainerPort: 5561, Protocol: corev1.ProtocolTCP}, {Name: "system", ContainerPort: 8081, Protocol: corev1.ProtocolTCP}, {Name: "recovery", ContainerPort: 5562, Protocol: corev1.ProtocolTCP}},
		StartupProbe:           &corev1.Probe{ProbeHandler: corev1.ProbeHandler{GRPC: &corev1.GRPCAction{Port: 5561, Service: ptr.To(relayGRPCService)}}, PeriodSeconds: 5, TimeoutSeconds: 1, SuccessThreshold: 1, FailureThreshold: 60},
		ReadinessProbe:         &corev1.Probe{ProbeHandler: corev1.ProbeHandler{GRPC: &corev1.GRPCAction{Port: 5561, Service: ptr.To(relayGRPCService)}}, PeriodSeconds: 10, TimeoutSeconds: 1, SuccessThreshold: 1, FailureThreshold: 3},
		TerminationMessagePath: corev1.TerminationMessagePathDefault, TerminationMessagePolicy: corev1.TerminationMessageReadFile,
	}

	// Active replicas independently observe every source; keep available Pods during updates.
	replicas := ptr.Deref(relay.Spec.Replicas, 2)
	if replicas < 1 {
		return nil, fmt.Errorf("replicas must be at least one")
	}
	deployment := &appsv1.Deployment{ObjectMeta: *metadata.DeepCopy(), Spec: appsv1.DeploymentSpec{
		Replicas: ptr.To(replicas), Selector: &metav1.LabelSelector{MatchLabels: labels},
		Strategy: appsv1.DeploymentStrategy{Type: appsv1.RollingUpdateDeploymentStrategyType,
			RollingUpdate: &appsv1.RollingUpdateDeployment{MaxUnavailable: ptr.To(intstr.FromInt32(0)), MaxSurge: ptr.To(intstr.FromInt32(1))}},
		Template: corev1.PodTemplateSpec{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{
			"app.kubernetes.io/name": "dynamo-relay", "nvidia.com/relay-uid": string(relay.UID),
			"nvidia.com/dynamo-discovery-backend": "kubernetes", "nvidia.com/dynamo-discovery-enabled": "true",
		}}, Spec: corev1.PodSpec{
			ServiceAccountName: name, TerminationGracePeriodSeconds: ptr.To(int64(60)),
			Volumes: volumes, Containers: []corev1.Container{container},
			RestartPolicy: corev1.RestartPolicyAlways, DNSPolicy: corev1.DNSClusterFirst,
			SchedulerName: corev1.DefaultSchedulerName,
		}},
	}}
	if err := mergeRelayPodTemplate(&deployment.Spec.Template, relay); err != nil {
		return nil, err
	}
	ports, err := relayServicePorts(relay, &deployment.Spec.Template)
	if err != nil {
		return nil, err
	}
	service.Spec.Ports = ports
	return []client.Object{serviceAccount, role, binding, service, configMap, deployment}, nil
}
