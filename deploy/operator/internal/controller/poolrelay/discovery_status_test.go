/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package poolrelay

import (
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func podDiscoveryComponentStatus(namespace string) v1beta1.ComponentReplicaStatus {
	return v1beta1.ComponentReplicaStatus{
		ComponentKind:    v1beta1.ComponentKindDeployment,
		RuntimeNamespace: namespace,
		Roles: []v1beta1.ComponentRoleStatus{{Name: "main", Containers: []v1beta1.ComponentContainerStatus{{
			Name: "main", Image: "engine:test",
			Env: map[string]v1beta1.EnvironmentValueStatus{
				commonconsts.DynamoDiscoveryBackendEnvVar: {Value: new("kubernetes")},
			},
		}}}},
	}
}

func TestPodDiscoveryUsesPublishedEnvironment(t *testing.T) {
	for _, test := range []struct {
		name string
		env  map[string]v1beta1.EnvironmentValueStatus
		want metav1.ConditionStatus
	}{
		{name: "missing mode uses runtime pod default", env: map[string]v1beta1.EnvironmentValueStatus{
			"DYN_DISCOVERY_BACKEND": {Value: new("kubernetes")},
		}, want: metav1.ConditionTrue},
		{name: "explicit pod mode", env: map[string]v1beta1.EnvironmentValueStatus{
			"DYN_DISCOVERY_BACKEND": {Value: new("kubernetes")}, "DYN_KUBE_DISCOVERY_MODE": {Value: new("pod")},
		}, want: metav1.ConditionTrue},
		{name: "missing backend uses runtime etcd default", want: metav1.ConditionFalse},
		{name: "explicit etcd backend", env: map[string]v1beta1.EnvironmentValueStatus{
			"DYN_DISCOVERY_BACKEND": {Value: new("etcd")},
		}, want: metav1.ConditionFalse},
		{name: "container mode", env: map[string]v1beta1.EnvironmentValueStatus{
			"DYN_DISCOVERY_BACKEND": {Value: new("kubernetes")}, "DYN_KUBE_DISCOVERY_MODE": {Value: new("container")},
		}, want: metav1.ConditionFalse},
		{name: "known empty is not absent", env: map[string]v1beta1.EnvironmentValueStatus{
			"DYN_DISCOVERY_BACKEND": {Value: new("kubernetes")}, "DYN_KUBE_DISCOVERY_MODE": {Value: new("")},
		}, want: metav1.ConditionFalse},
		{name: "unknown backend is pending", env: map[string]v1beta1.EnvironmentValueStatus{
			"DYN_DISCOVERY_BACKEND": {Unresolved: true},
		}, want: metav1.ConditionUnknown},
		{name: "unknown mode is pending", env: map[string]v1beta1.EnvironmentValueStatus{
			"DYN_DISCOVERY_BACKEND": {Value: new("kubernetes")}, "DYN_KUBE_DISCOVERY_MODE": {Unresolved: true},
		}, want: metav1.ConditionUnknown},
		{name: "published literals are not expanded again", env: map[string]v1beta1.EnvironmentValueStatus{
			"DYN_DISCOVERY_BACKEND": {Value: new("kubernetes")}, "DYN_KUBE_DISCOVERY_MODE": {Value: new("$(MODE)")}, "MODE": {Value: new("pod")},
		}, want: metav1.ConditionFalse},
		{name: "known incompatibility wins over unknown setting", env: map[string]v1beta1.EnvironmentValueStatus{
			"DYN_DISCOVERY_BACKEND": {Value: new("etcd")}, "DYN_KUBE_DISCOVERY_MODE": {Unresolved: true},
		}, want: metav1.ConditionFalse},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Use independent published facts even when authored discovery settings disagree")
			component := podDiscoveryComponentStatus("")
			component.Roles[0].Containers[0].Env = test.env
			dgd := &v1beta1.DynamoGraphDeployment{
				Spec: v1beta1.DynamoGraphDeploymentSpec{
					Env:         []corev1.EnvVar{{Name: "DYN_KUBE_DISCOVERY_MODE", Value: "container"}},
					Annotations: map[string]string{commonconsts.KubeAnnotationDynamoKubeDiscoveryMode: "container"},
					Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
						ComponentName: "decode", ComponentType: v1beta1.ComponentTypeWorker,
						PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
							Name: "main", Env: []corev1.EnvVar{{Name: "DYN_DISCOVERY_BACKEND", Value: "etcd"}},
							EnvFrom: []corev1.EnvFromSource{{ConfigMapRef: &corev1.ConfigMapEnvSource{Name: "opaque"}}},
						}}}},
					}},
				},
				Status: v1beta1.DynamoGraphDeploymentStatus{Components: map[string]v1beta1.ComponentReplicaStatus{"decode": component}},
			}
			if got := podDiscoveryStatus(dgd, configv1alpha1.DiscoveryBackendKubernetes); got != test.want {
				t.Fatalf("discovery=%s, want %s", got, test.want)
			}
		})
	}
}

func TestPodDiscoveryRequiresCurrentRoles(t *testing.T) {
	for _, test := range []struct {
		name       string
		generation int64
		roles      []v1beta1.ComponentRoleStatus
		want       metav1.ConditionStatus
	}{
		{name: "missing component publication", want: metav1.ConditionUnknown},
		{name: "stale publication", generation: 1, roles: podDiscoveryComponentStatus("").Roles, want: metav1.ConditionUnknown},
		{name: "missing runtime container", roles: []v1beta1.ComponentRoleStatus{{Name: "main"}}, want: metav1.ConditionUnknown},
		{name: "unrelated container is not a runtime", roles: []v1beta1.ComponentRoleStatus{{Name: "main", Containers: []v1beta1.ComponentContainerStatus{{Name: "metrics"}}}}, want: metav1.ConditionUnknown},
		{name: "complete target", roles: podDiscoveryComponentStatus("").Roles, want: metav1.ConditionTrue},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Require a current component target instead of falling back to authored configuration")
			dgd := &v1beta1.DynamoGraphDeployment{
				Generation: test.generation,
				Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
					{ComponentName: "decode", ComponentType: v1beta1.ComponentTypeWorker},
				}},
				Status: v1beta1.DynamoGraphDeploymentStatus{Components: map[string]v1beta1.ComponentReplicaStatus{
					"decode": {Roles: test.roles},
				}},
			}
			if got := podDiscoveryStatus(dgd, configv1alpha1.DiscoveryBackendKubernetes); got != test.want {
				t.Fatalf("discovery=%s, want %s", got, test.want)
			}
		})
	}
}

func TestPodDiscoveryChecksRoleRuntimes(t *testing.T) {
	t.Log("Native runtimes own discovery, not engine-only main containers or helper containers")
	leader := podDiscoveryComponentStatus("").Roles[0]
	leader.Name = "leader"
	leader.Containers[0].Name = "runtime"
	leader.InitContainers = leader.Containers
	leader.Containers = []v1beta1.ComponentContainerStatus{{Name: "main", Env: map[string]v1beta1.EnvironmentValueStatus{
		"DYN_DISCOVERY_BACKEND": {Value: new("etcd")},
	}}}
	worker := v1beta1.ComponentRoleStatus{Name: "worker", Containers: []v1beta1.ComponentContainerStatus{{Name: "main"}, {Name: "metrics"}}}
	dgd := &v1beta1.DynamoGraphDeployment{
		Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
			{ComponentName: "decode", ComponentType: v1beta1.ComponentTypeWorker},
			{ComponentName: "planner", ComponentType: v1beta1.ComponentTypePlanner},
		}},
		Status: v1beta1.DynamoGraphDeploymentStatus{Components: map[string]v1beta1.ComponentReplicaStatus{
			"decode": {Roles: []v1beta1.ComponentRoleStatus{leader, worker}},
		}},
	}
	if got := podDiscoveryStatus(dgd, configv1alpha1.DiscoveryBackendKubernetes); got != metav1.ConditionTrue {
		t.Fatalf("native runtime with a compute-only worker was rejected: %s", got)
	}

	t.Log("Every role with a runtime must support pod discovery")
	worker.InitContainers = []v1beta1.ComponentContainerStatus{{Name: "runtime", Env: map[string]v1beta1.EnvironmentValueStatus{
		"DYN_DISCOVERY_BACKEND": {Value: new("kubernetes")}, "DYN_KUBE_DISCOVERY_MODE": {Value: new("container")},
	}}}
	dgd.Status.Components["decode"] = v1beta1.ComponentReplicaStatus{Roles: []v1beta1.ComponentRoleStatus{leader, worker}}
	if got := podDiscoveryStatus(dgd, configv1alpha1.DiscoveryBackendKubernetes); got != metav1.ConditionFalse {
		t.Fatalf("incompatible worker runtime was accepted: %s", got)
	}

	t.Log("Grove's graph observer remains an independent discovery prerequisite")
	dgd.Annotations = map[string]string{commonconsts.KubeAnnotationDynamoDiscoveryBackend: "etcd"}
	if got := podDiscoveryStatus(dgd, configv1alpha1.DiscoveryBackendKubernetes); got != metav1.ConditionFalse {
		t.Fatalf("non-Kubernetes graph observer was accepted: %s", got)
	}

	t.Log("A regular helper named runtime must not hide incompatible main discovery")
	dgd.Annotations = nil
	leader.InitContainers = nil
	leader.Containers = append(leader.Containers, v1beta1.ComponentContainerStatus{Name: "runtime", Env: map[string]v1beta1.EnvironmentValueStatus{
		"DYN_DISCOVERY_BACKEND": {Value: new("kubernetes")},
	}})
	dgd.Status.Components["decode"] = v1beta1.ComponentReplicaStatus{Roles: []v1beta1.ComponentRoleStatus{leader}}
	if got := podDiscoveryStatus(dgd, configv1alpha1.DiscoveryBackendKubernetes); got != metav1.ConditionFalse {
		t.Fatalf("a regular helper hid incompatible main discovery: %s", got)
	}
}
