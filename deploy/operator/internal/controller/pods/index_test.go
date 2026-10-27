/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package pods

import (
	"context"
	"errors"
	"sort"
	"testing"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	poddiagnostics "github.com/ai-dynamo/dynamo/deploy/operator/internal/pods"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

// testPod describes a running Pod with labels whose main container restarted
// restarts times, last exiting with code 1, and waits to restart when restarts is positive.
func testPod(namespace, name string, labels map[string]string, restarts int32) *corev1.Pod {
	pod := &corev1.Pod{
		Name:      name,
		Namespace: namespace,
		Labels:    labels,
		Status: corev1.PodStatus{
			Phase:             corev1.PodRunning,
			Conditions:        []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionFalse}},
			ContainerStatuses: []corev1.ContainerStatus{{Name: "main", RestartCount: restarts}},
		},
	}
	if restarts > 0 {
		pod.Status.ContainerStatuses[0].LastTerminationState.Terminated = &corev1.ContainerStateTerminated{ExitCode: 1, Reason: "Error"}
		pod.Status.ContainerStatuses[0].State.Waiting = &corev1.ContainerStateWaiting{Reason: "CrashLoopBackOff"}
	}
	return pod
}

// componentLabels labels a Pod for component of graph.
func componentLabels(graph, component string) map[string]string {
	return map[string]string{
		commonconsts.KubeLabelDynamoGraphDeploymentName: graph,
		commonconsts.KubeLabelDynamoComponent:           component,
	}
}

func podNames(pods []corev1.Pod) []string {
	names := make([]string, 0, len(pods))
	for i := range pods {
		names = append(names, pods[i].Name)
	}
	sort.Strings(names)
	return names
}

func TestComponentIndexValues(t *testing.T) {
	tests := []struct {
		name string
		obj  client.Object
		want []string
	}{
		{name: "graph component Pod", obj: &corev1.Pod{Labels: componentLabels("graph", "decode")}, want: []string{"graph/decode"}},
		{name: "missing graph label", obj: &corev1.Pod{Labels: map[string]string{commonconsts.KubeLabelDynamoComponent: "decode"}}},
		{name: "missing component label", obj: &corev1.Pod{Labels: map[string]string{commonconsts.KubeLabelDynamoGraphDeploymentName: "graph"}}},
		{name: "non-Pod object", obj: &corev1.ConfigMap{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Verify the index value of the case's object")
			assert.Equal(t, tt.want, ComponentIndexValues(tt.obj))
		})
	}
}

func TestSelectorIndexValues(t *testing.T) {
	tests := []struct {
		name string
		obj  client.Object
		want []string
	}{
		{name: "Pod with the selector label", obj: &corev1.Pod{Labels: map[string]string{commonconsts.KubeLabelDynamoSelector: "graph-frontend"}}, want: []string{"graph-frontend"}},
		{name: "Pod without the selector label", obj: &corev1.Pod{}},
		{name: "non-Pod object", obj: &corev1.ConfigMap{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Verify the index value of the case's object")
			assert.Equal(t, tt.want, SelectorIndexValues(tt.obj))
		})
	}
}

func TestListComponentPods(t *testing.T) {
	t.Log("Label Pods for the queried component and for other components, graphs and namespaces")
	reader := fake.NewClientBuilder().
		WithScheme(scheme.Scheme).
		WithIndex(&corev1.Pod{}, ComponentIndex, ComponentIndexValues).
		WithObjects(
			testPod("inference", "matching-a", componentLabels("graph", "decode"), 0),
			testPod("inference", "matching-b", componentLabels("graph", "decode"), 0),
			testPod("inference", "other-component", componentLabels("graph", "prefill"), 0),
			testPod("inference", "other-graph", componentLabels("other", "decode"), 0),
			testPod("other-namespace", "other-namespace", componentLabels("graph", "decode"), 0),
		).
		Build()

	t.Log("List the component's Pods through the index")
	pods, err := ListComponentPods(t.Context(), reader, "inference", "graph", "decode")

	t.Log("Verify only the component's Pods in its namespace are returned")
	require.NoError(t, err)
	assert.Equal(t, []string{"matching-a", "matching-b"}, podNames(pods))
}

func TestListComponentPodsReturnsListErrors(t *testing.T) {
	t.Log("Fail every list through the cache")
	reader := fake.NewClientBuilder().
		WithScheme(scheme.Scheme).
		WithIndex(&corev1.Pod{}, ComponentIndex, ComponentIndexValues).
		WithInterceptorFuncs(interceptor.Funcs{
			List: func(context.Context, client.WithWatch, client.ObjectList, ...client.ListOption) error {
				return errors.New("cache unavailable")
			},
		}).
		Build()

	t.Log("List the component's Pods and verify the error is returned")
	_, err := ListComponentPods(t.Context(), reader, "inference", "graph", "decode")
	require.ErrorContains(t, err, "cache unavailable")
}

func TestComponentCrashLoops(t *testing.T) {
	t.Log("Label a crash-looping frontend, a recovering worker and a crash-looping Pod of another graph")
	reader := fake.NewClientBuilder().
		WithScheme(scheme.Scheme).
		WithIndex(&corev1.Pod{}, ComponentIndex, ComponentIndexValues).
		WithObjects(
			testPod("inference", "graph-frontend-a", componentLabels("graph", "frontend"), 3),
			testPod("inference", "graph-worker-a", componentLabels("graph", "worker"), 1),
			testPod("inference", "other-worker-a", componentLabels("other", "worker"), 4),
		).
		Build()

	t.Log("Diagnose each component from the Pods labeled for it")
	crashLoops, err := ComponentCrashLoops(t.Context(), reader, "inference", "graph", []string{"frontend", "worker"})

	t.Log("Verify only the crash-looping component of graph is reported")
	require.NoError(t, err)
	assert.Equal(t, map[string]poddiagnostics.CrashLoop{
		"frontend": {Pod: "graph-frontend-a", Container: "main", RestartCount: 3, ExitCode: 1, Reason: "Error"},
	}, crashLoops)
}

func TestFindDeploymentCrashLoop(t *testing.T) {
	selected := map[string]string{commonconsts.KubeLabelDynamoSelector: "graph-frontend", "tier": "frontend"}
	terminating := testPod("inference", "terminating", selected, 3)
	terminating.DeletionTimestamp = new(metav1.Now())
	terminating.Finalizers = []string{"test.nvidia.com/hold"}

	tests := []struct {
		name  string
		pods  []client.Object
		want  poddiagnostics.CrashLoop
		found bool
	}{
		{
			name:  "selected Pod without ReplicaSet metadata",
			pods:  []client.Object{testPod("inference", "selected", selected, 3)},
			want:  poddiagnostics.CrashLoop{Pod: "selected", Container: "main", RestartCount: 3, ExitCode: 1, Reason: "Error"},
			found: true,
		},
		{
			name: "Pod with the selector label that the full selector excludes",
			pods: []client.Object{testPod("inference", "excluded", map[string]string{commonconsts.KubeLabelDynamoSelector: "graph-frontend"}, 3)},
		},
		{
			name: "selected labels in another namespace",
			pods: []client.Object{testPod("other-namespace", "elsewhere", selected, 3)},
		},
		{
			name: "terminating Pod during a rollout",
			pods: []client.Object{terminating},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Index the Pods and describe the Deployment that selects on two labels")
			reader := fake.NewClientBuilder().
				WithScheme(scheme.Scheme).
				WithIndex(&corev1.Pod{}, SelectorIndex, SelectorIndexValues).
				WithObjects(tt.pods...).
				Build()
			deployment := &appsv1.Deployment{
				Name:      "graph-frontend",
				Namespace: "inference",
				Spec:      appsv1.DeploymentSpec{Selector: &metav1.LabelSelector{MatchLabels: selected}},
			}

			t.Log("Look up the Deployment's crash loop")
			got, found, err := FindDeploymentCrashLoop(t.Context(), reader, deployment)

			t.Log("Verify the reported container")
			require.NoError(t, err)
			assert.Equal(t, tt.found, found)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestFindDeploymentCrashLoopRequiresTheSelectorLabel(t *testing.T) {
	t.Log("Describe a Deployment that selects without the Dynamo selector label")
	reader := fake.NewClientBuilder().WithScheme(scheme.Scheme).WithIndex(&corev1.Pod{}, SelectorIndex, SelectorIndexValues).Build()
	deployment := &appsv1.Deployment{
		Name:      "graph-frontend",
		Namespace: "inference",
		Spec:      appsv1.DeploymentSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "frontend"}}},
	}

	t.Log("Look up its crash loop and verify the missing label is reported")
	_, _, err := FindDeploymentCrashLoop(t.Context(), reader, deployment)
	require.ErrorContains(t, err, commonconsts.KubeLabelDynamoSelector)
}
