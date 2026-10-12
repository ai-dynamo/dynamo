/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package poolrelay

import (
	"testing"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"
)

func TestDynamoPoolRelayPodTemplateRendering(t *testing.T) {
	t.Log("Declare Envoy before main, with independent env, init container, TLS volume and probes")
	probe := &corev1.Probe{ProbeHandler: corev1.ProbeHandler{Exec: &corev1.ExecAction{Command: []string{"python3", "-m", "dynamo.kv_dc_relay.healthcheck"}}}, TimeoutSeconds: 3}
	envoy := corev1.Container{Name: "envoy", Image: "example/envoy:1", Env: []corev1.EnvVar{{Name: "ENVOY_SETTING", Value: "yes"}}, Ports: []corev1.ContainerPort{{Name: "grpc-mtls", ContainerPort: 5560}}, VolumeMounts: []corev1.VolumeMount{{Name: "sds", MountPath: "/etc/sds", ReadOnly: true}}}
	relay := &v1alpha1.DynamoPoolRelay{ObjectMeta: metav1.ObjectMeta{Name: "dc", Namespace: "workers", UID: "relay"}, Spec: v1alpha1.DynamoPoolRelaySpec{ClusterID: "dc", PodTemplate: corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"app": "relay"}, Annotations: map[string]string{"example.com/reload": "a"}},
		Spec:       corev1.PodSpec{SecurityContext: &corev1.PodSecurityContext{FSGroup: ptr.To(int64(101))}, NodeSelector: map[string]string{"zone": "a"}, ImagePullSecrets: []corev1.LocalObjectReference{{Name: "registry"}}, InitContainers: []corev1.Container{{Name: "prepare", Image: "example/init:1", Command: []string{"true"}}}, Containers: []corev1.Container{envoy, {Name: "main", Image: "example/relay:1", Args: []string{"--bind=127.0.0.1:5561"}, Env: []corev1.EnvVar{{Name: "DYN_LOG", Value: "info"}}, StartupProbe: probe, ReadinessProbe: probe}}, Volumes: []corev1.Volume{{Name: "sds", VolumeSource: corev1.VolumeSource{Projected: &corev1.ProjectedVolumeSource{Sources: []corev1.VolumeProjection{{Secret: &corev1.SecretProjection{LocalObjectReference: corev1.LocalObjectReference{Name: "tls"}}}}}}}}},
	}, Service: &v1alpha1.DynamoPoolRelayServiceSpec{Ports: []corev1.ServicePort{{Name: "grpc", Port: 5560, TargetPort: intstr.FromString("grpc-mtls")}}}}}
	before := relay.DeepCopy()
	connection := relayConnection{Env: []corev1.EnvVar{{Name: "DYN_EVENT_PLANE", Value: "nats"}, {Name: "NATS_SERVER", Value: "nats://test:4222"}}}
	objects, err := renderRelayResources(relay, connection, newRelaySources("connection", nil))
	require.NoError(t, err)
	pod := objects[5].(*appsv1.Deployment).Spec.Template

	t.Log("Find main by name, preserve sidecar settings and inject managed configuration only into main")
	require.Equal(t, "main", pod.Spec.Containers[0].Name)
	require.Equal(t, envoy, pod.Spec.Containers[1])
	require.Equal(t, probe, pod.Spec.Containers[0].ReadinessProbe)
	require.Nil(t, pod.Spec.Containers[0].ReadinessProbe.GRPC)
	require.Contains(t, pod.Spec.Containers[0].Args, "--sources-file")
	require.Contains(t, pod.Spec.Containers[0].Args, "--bind=127.0.0.1:5561")
	require.Contains(t, pod.Spec.Containers[0].Env, corev1.EnvVar{Name: "DYN_EVENT_PLANE", Value: "nats"})
	require.Equal(t, int64(101), *pod.Spec.SecurityContext.FSGroup)
	require.Equal(t, "a", pod.Spec.NodeSelector["zone"])
	require.Len(t, pod.Spec.InitContainers, 1)
	require.Len(t, pod.Spec.Volumes, 2)
	require.Equal(t, "registry", pod.Spec.ImagePullSecrets[0].Name)
	service := objects[3].(*corev1.Service)
	require.Equal(t, int32(5560), service.Spec.Ports[0].Port)
	require.Equal(t, intstr.FromString("grpc-mtls"), service.Spec.Ports[0].TargetPort)
	require.Equal(t, before, relay)

	t.Log("Adding a source cannot change any field of the sidecar-enabled PodTemplate")
	next, err := renderRelayResources(relay, connection, newRelaySources("connection", []relaySource{{Namespace: "workers-worker"}}))
	require.NoError(t, err)
	require.Equal(t, pod, next[5].(*appsv1.Deployment).Spec.Template)
}

func TestDynamoPoolRelayPodTemplateRejectsConflicts(t *testing.T) {
	for _, tc := range []struct {
		name     string
		main     corev1.Container
		pod      corev1.PodSpec
		metadata metav1.ObjectMeta
		service  *v1alpha1.DynamoPoolRelayServiceSpec
		want     string
	}{
		{name: "command", main: corev1.Container{Command: []string{"sh"}}, want: "command"},
		{name: "envFrom", main: corev1.Container{EnvFrom: []corev1.EnvFromSource{{Prefix: "IGNORED"}}}, want: "envFrom"},
		{name: "transport env", main: corev1.Container{Env: []corev1.EnvVar{{Name: "NATS_AUTH_TOKEN", Value: "private"}}}, want: "operator-managed env"},
		{name: "cluster identity env", main: corev1.Container{Env: []corev1.EnvVar{{Name: "DYN_CLUSTER_ID", Value: "other"}}}, want: "operator-managed env"},
		{name: "legacy cluster identity env", main: corev1.Container{Env: []corev1.EnvVar{{Name: "DYN_DC_ID", Value: "other"}}}, want: "operator-managed env"},
		{name: "sources env", main: corev1.Container{Env: []corev1.EnvVar{{Name: "DYN_RELAY_NAMESPACES", Value: "other"}}}, want: "operator-managed env"},
		{name: "sources flag", main: corev1.Container{Args: []string{"--sources-file=/other"}}, want: "args allow only"},
		{name: "namespace flag", main: corev1.Container{Args: []string{"--namespaces", "other"}}, want: "args allow only"},
		{name: "loopback probe", main: corev1.Container{Args: []string{"--bind", "127.0.0.1:5561"}}, want: "exec startup and readiness"},
		{name: "mixed probe", main: corev1.Container{ReadinessProbe: &corev1.Probe{ProbeHandler: corev1.ProbeHandler{HTTPGet: &corev1.HTTPGetAction{}, Exec: &corev1.ExecAction{Command: []string{"true"}}}}}, want: "exactly one handler"},
		{name: "reserved volume", pod: corev1.PodSpec{Volumes: []corev1.Volume{{Name: relaySourcesVolume}}}, want: "reserved volume"},
		{name: "sources parent mount", main: corev1.Container{VolumeMounts: []corev1.VolumeMount{{Name: "user", MountPath: "/var/run"}}}, pod: corev1.PodSpec{Volumes: []corev1.Volume{{Name: "user", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}}}}, want: "shadows"},
		{name: "account", pod: corev1.PodSpec{ServiceAccountName: "other"}, want: "service account"},
		{name: "host network", pod: corev1.PodSpec{HostNetwork: true}, want: "host networking"},
		{name: "identity label", metadata: metav1.ObjectMeta{Labels: map[string]string{"nvidia.com/relay-uid": "other"}}, want: "reserved label"},
		{name: "metadata identity", metadata: metav1.ObjectMeta{Name: "other"}, want: "only labels and annotations"},
		{name: "missing target", service: &v1alpha1.DynamoPoolRelayServiceSpec{Ports: []corev1.ServicePort{{Name: "grpc", Port: 5560, TargetPort: intstr.FromString("absent")}}}, want: "not declared"},
		{name: "ack target", service: &v1alpha1.DynamoPoolRelayServiceSpec{Ports: []corev1.ServicePort{{Name: "grpc", Port: 5560, TargetPort: intstr.FromInt32(8081)}}}, want: "reserved sources"},
		{name: "no publication", service: &v1alpha1.DynamoPoolRelayServiceSpec{Ports: []corev1.ServicePort{{Name: "other", Port: 5560}}}, want: "named grpc"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Validate a conflicting user template before rendering resources")
			tc.main.Name, tc.main.Image = "main", "example/relay:1"
			tc.pod.Containers = []corev1.Container{tc.main}
			relay := &v1alpha1.DynamoPoolRelay{ObjectMeta: metav1.ObjectMeta{Name: "relay", UID: "relay"}, Spec: v1alpha1.DynamoPoolRelaySpec{ClusterID: "dc", PodTemplate: corev1.PodTemplateSpec{ObjectMeta: tc.metadata, Spec: tc.pod}, Service: tc.service}}
			objects, err := renderRelayResources(relay, relayConnection{}, newRelaySources("connection", nil))
			require.ErrorContains(t, err, tc.want)
			require.NotContains(t, err.Error(), "private")
			require.Nil(t, objects)
		})
	}
}
