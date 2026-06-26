/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"strings"
	"testing"
	"time"

	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/event"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

func dynamoComponentPodLabels(labels map[string]string) map[string]string {
	result := map[string]string{
		consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
		consts.KubeLabelDynamoComponent:           "worker",
	}
	for k, v := range labels {
		result[k] = v
	}
	return result
}

func TestTopologyLabelReconciler_CopiesToPod(t *testing.T) {
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "node-1",
			Labels: map[string]string{"topology.kubernetes.io/zone": "us-east-1a"},
		},
	}
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "worker-abc",
			Namespace: "default",
			Annotations: map[string]string{
				consts.KubeAnnotationTopologyLabelKey: "topology.kubernetes.io/zone",
			},
			Labels: dynamoComponentPodLabels(nil),
		},
		Spec: corev1.PodSpec{NodeName: "node-1"},
	}

	cl := fake.NewClientBuilder().WithObjects(pod).Build()
	nodeReader := fake.NewClientBuilder().WithObjects(node).Build()
	r := &TopologyLabelReconciler{Client: cl, NodeReader: nodeReader}

	result, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: "worker-abc", Namespace: "default"},
	})
	require.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)

	var patched corev1.Pod
	require.NoError(t, cl.Get(context.Background(), types.NamespacedName{Name: "worker-abc", Namespace: "default"}, &patched))
	assert.Equal(t, "us-east-1a", patched.Labels["topology.kubernetes.io/zone"])
}

func TestTopologyLabelReconciler_CopiesClusterTopologyLevelsToDynamoPodLabels(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	require.NoError(t, grovev1alpha1.AddToScheme(scheme))

	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-1",
			Labels: map[string]string{
				"topology.kubernetes.io/zone": "us-east-1a",
				"nvidia.com/rack":             "rack-22",
			},
		},
	}
	ct := &grovev1alpha1.ClusterTopology{
		ObjectMeta: metav1.ObjectMeta{Name: "grove-topology"},
		Spec: grovev1alpha1.ClusterTopologySpec{
			Levels: []grovev1alpha1.TopologyLevel{
				{Domain: grovev1alpha1.TopologyDomainZone, Key: "topology.kubernetes.io/zone"},
				{Domain: grovev1alpha1.TopologyDomainRack, Key: "nvidia.com/rack"},
			},
		},
	}
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "worker-abc",
			Namespace: "default",
			Annotations: map[string]string{
				consts.KubeAnnotationTopologyClusterTopologyName: "grove-topology",
			},
			Labels: dynamoComponentPodLabels(nil),
		},
		Spec: corev1.PodSpec{NodeName: "node-1"},
	}

	cl := fake.NewClientBuilder().WithScheme(scheme).WithObjects(pod, ct).Build()
	nodeReader := fake.NewClientBuilder().WithScheme(scheme).WithObjects(node).Build()
	r := &TopologyLabelReconciler{Client: cl, NodeReader: nodeReader}

	result, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: "worker-abc", Namespace: "default"},
	})
	require.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)

	var patched corev1.Pod
	require.NoError(t, cl.Get(context.Background(), types.NamespacedName{Name: "worker-abc", Namespace: "default"}, &patched))
	assert.Equal(t, "us-east-1a", patched.Labels[consts.DynamoTopologyLabelKey("zone")])
	assert.Equal(t, "rack-22", patched.Labels[consts.DynamoTopologyLabelKey("rack")])
	assert.NotContains(t, patched.Labels, "topology.kubernetes.io/zone")
	assert.NotContains(t, patched.Labels, "nvidia.com/rack")
}

func TestTopologyLabelReconciler_SkipsIfLabelExists(t *testing.T) {
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "node-1",
			Labels: map[string]string{"topology.kubernetes.io/zone": "us-east-1b"},
		},
	}
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "worker-abc",
			Namespace: "default",
			Annotations: map[string]string{
				consts.KubeAnnotationTopologyLabelKey: "topology.kubernetes.io/zone",
			},
			Labels: dynamoComponentPodLabels(map[string]string{
				"topology.kubernetes.io/zone": "us-east-1a",
			}),
		},
		Spec: corev1.PodSpec{NodeName: "node-1"},
	}

	cl := fake.NewClientBuilder().WithObjects(node, pod).Build()
	r := &TopologyLabelReconciler{Client: cl, NodeReader: cl}

	result, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: "worker-abc", Namespace: "default"},
	})
	require.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)

	var unchanged corev1.Pod
	require.NoError(t, cl.Get(context.Background(), types.NamespacedName{Name: "worker-abc", Namespace: "default"}, &unchanged))
	assert.Equal(t, "us-east-1a", unchanged.Labels["topology.kubernetes.io/zone"])
}

func TestTopologyLabelReconciler_SkipsUnscheduledPod(t *testing.T) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "worker-abc",
			Namespace: "default",
			Annotations: map[string]string{
				consts.KubeAnnotationTopologyLabelKey: "topology.kubernetes.io/zone",
			},
			Labels: dynamoComponentPodLabels(nil),
		},
		Spec: corev1.PodSpec{}, // No NodeName yet
	}

	cl := fake.NewClientBuilder().WithObjects(pod).Build()
	r := &TopologyLabelReconciler{Client: cl, NodeReader: cl}

	result, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: "worker-abc", Namespace: "default"},
	})
	require.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)

	var unchanged corev1.Pod
	require.NoError(t, cl.Get(context.Background(), types.NamespacedName{Name: "worker-abc", Namespace: "default"}, &unchanged))
	assert.NotContains(t, unchanged.Labels, "topology.kubernetes.io/zone")
}

func TestTopologyLabelReconciler_SkipsIfNodeMissingLabel(t *testing.T) {
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "node-1",
			Labels: map[string]string{}, // Missing the topology label
		},
	}
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "worker-abc",
			Namespace: "default",
			Annotations: map[string]string{
				consts.KubeAnnotationTopologyLabelKey: "topology.kubernetes.io/zone",
			},
			Labels: dynamoComponentPodLabels(nil),
		},
		Spec: corev1.PodSpec{NodeName: "node-1"},
	}

	cl := fake.NewClientBuilder().WithObjects(node, pod).Build()
	recorder := events.NewFakeRecorder(1)
	r := &TopologyLabelReconciler{Client: cl, NodeReader: cl, Recorder: recorder}

	result, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: "worker-abc", Namespace: "default"},
	})
	require.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)

	// Pod should NOT have the label
	var patched corev1.Pod
	require.NoError(t, cl.Get(context.Background(), types.NamespacedName{Name: "worker-abc", Namespace: "default"}, &patched))
	assert.NotContains(t, patched.Labels, "topology.kubernetes.io/zone")

	select {
	case event := <-recorder.Events:
		assert.True(t, strings.Contains(event, corev1.EventTypeWarning), event)
		assert.True(t, strings.Contains(event, topologyLabelMissingReason), event)
		assert.True(t, strings.Contains(event, "node-1"), event)
		assert.True(t, strings.Contains(event, "topology.kubernetes.io/zone"), event)
	case <-time.After(time.Second):
		t.Fatal("expected topology missing warning event")
	}
}

func TestTopologyLabelReconciler_SkipsPodWithoutAnnotation(t *testing.T) {
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "node-1",
			Labels: map[string]string{"topology.kubernetes.io/zone": "us-east-1a"},
		},
	}
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "worker-abc",
			Namespace: "default",
			// No topology-label-key annotation
			Labels: dynamoComponentPodLabels(nil),
		},
		Spec: corev1.PodSpec{NodeName: "node-1"},
	}

	cl := fake.NewClientBuilder().WithObjects(node, pod).Build()
	r := &TopologyLabelReconciler{Client: cl, NodeReader: cl}

	result, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: "worker-abc", Namespace: "default"},
	})
	require.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)

	var unchanged corev1.Pod
	require.NoError(t, cl.Get(context.Background(), types.NamespacedName{Name: "worker-abc", Namespace: "default"}, &unchanged))
	assert.NotContains(t, unchanged.Labels, "topology.kubernetes.io/zone")
}

func TestTopologyLabelReconciler_SkipsNonDynamoComponentPod(t *testing.T) {
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "node-1",
			Labels: map[string]string{"topology.kubernetes.io/zone": "us-east-1a"},
		},
	}
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "worker-abc",
			Namespace: "default",
			Annotations: map[string]string{
				consts.KubeAnnotationTopologyLabelKey: "topology.kubernetes.io/zone",
			},
		},
		Spec: corev1.PodSpec{NodeName: "node-1"},
	}

	cl := fake.NewClientBuilder().WithObjects(node, pod).Build()
	r := &TopologyLabelReconciler{Client: cl, NodeReader: cl}

	result, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: "worker-abc", Namespace: "default"},
	})
	require.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)

	var unchanged corev1.Pod
	require.NoError(t, cl.Get(context.Background(), types.NamespacedName{Name: "worker-abc", Namespace: "default"}, &unchanged))
	assert.NotContains(t, unchanged.Labels, "topology.kubernetes.io/zone")
}

const (
	topologyPredicateLabelKey      = "topology.kubernetes.io/zone"
	topologyPredicateOtherLabelKey = "topology.kubernetes.io/rack"
)

func topologyPredicatePod(nodeName string, annotations map[string]string, labels map[string]string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "worker-abc",
			Namespace:   "default",
			Annotations: annotations,
			Labels:      dynamoComponentPodLabels(labels),
		},
		Spec: corev1.PodSpec{NodeName: nodeName},
	}
}

func topologyPredicateClusterPod(nodeName string, labels map[string]string) *corev1.Pod {
	p := topologyPredicatePod(nodeName, map[string]string{
		consts.KubeAnnotationTopologyClusterTopologyName: "grove-topology",
	}, labels)
	p.Spec.Volumes = []corev1.Volume{
		{
			Name: "topology-labels",
			VolumeSource: corev1.VolumeSource{
				DownwardAPI: &corev1.DownwardAPIVolumeSource{
					Items: []corev1.DownwardAPIVolumeFile{
						{
							Path: "zone",
							FieldRef: &corev1.ObjectFieldSelector{
								FieldPath: "metadata.labels['" + consts.DynamoTopologyLabelKey("zone") + "']",
							},
						},
						{
							Path: "rack",
							FieldRef: &corev1.ObjectFieldSelector{
								FieldPath: "metadata.labels['" + consts.DynamoTopologyLabelKey("rack") + "']",
							},
						},
					},
				},
			},
		},
	}
	return p
}

func topologyPredicateNonDynamoPod(labels map[string]string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "non-dynamo",
			Namespace: "default",
			Annotations: map[string]string{
				consts.KubeAnnotationTopologyLabelKey: topologyPredicateLabelKey,
			},
			Labels: labels,
		},
		Spec: corev1.PodSpec{NodeName: "node-1"},
	}
}

func topologyPredicateLabelFieldPath(labelKey string) string {
	return "metadata.labels['" + labelKey + "']"
}

func TestExpectedDynamoTopologyLabelKeys(t *testing.T) {
	tests := []struct {
		name string
		pod  *corev1.Pod
		want []string
	}{
		{
			name: "extracts Dynamo topology labels from Downward API field refs",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Volumes: []corev1.Volume{
						{
							Name: "config",
							VolumeSource: corev1.VolumeSource{
								ConfigMap: &corev1.ConfigMapVolumeSource{},
							},
						},
						{
							Name: "topology-labels",
							VolumeSource: corev1.VolumeSource{
								DownwardAPI: &corev1.DownwardAPIVolumeSource{
									Items: []corev1.DownwardAPIVolumeFile{
										{Path: "nil-field-ref"},
										{
											Path: "metadata-name",
											FieldRef: &corev1.ObjectFieldSelector{
												FieldPath: "metadata.name",
											},
										},
										{
											Path: "app-label",
											FieldRef: &corev1.ObjectFieldSelector{
												FieldPath: topologyPredicateLabelFieldPath("app.kubernetes.io/name"),
											},
										},
										{
											Path: "zone",
											FieldRef: &corev1.ObjectFieldSelector{
												FieldPath: topologyPredicateLabelFieldPath(consts.DynamoTopologyLabelKey("zone")),
											},
										},
										{
											Path: "double-quotes",
											FieldRef: &corev1.ObjectFieldSelector{
												FieldPath: "metadata.labels[\"" + consts.DynamoTopologyLabelKey("node") + "\"]",
											},
										},
										{
											Path: "rack",
											FieldRef: &corev1.ObjectFieldSelector{
												FieldPath: topologyPredicateLabelFieldPath(consts.DynamoTopologyLabelKey("rack")),
											},
										},
										{
											Path: "malformed",
											FieldRef: &corev1.ObjectFieldSelector{
												FieldPath: "metadata.labels['" + consts.DynamoTopologyLabelKey("host"),
											},
										},
									},
								},
							},
						},
					},
				},
			},
			want: []string{
				consts.DynamoTopologyLabelKey("zone"),
				consts.DynamoTopologyLabelKey("rack"),
			},
		},
		{
			name: "returns no keys without matching Downward API label refs",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Volumes: []corev1.Volume{
						{
							Name: "topology-labels",
							VolumeSource: corev1.VolumeSource{
								DownwardAPI: &corev1.DownwardAPIVolumeSource{
									Items: []corev1.DownwardAPIVolumeFile{
										{
											Path: "metadata-name",
											FieldRef: &corev1.ObjectFieldSelector{
												FieldPath: "metadata.name",
											},
										},
										{
											Path: "app-label",
											FieldRef: &corev1.ObjectFieldSelector{
												FieldPath: topologyPredicateLabelFieldPath("app.kubernetes.io/name"),
											},
										},
									},
								},
							},
						},
					},
				},
			},
			want: nil,
		},
		{
			name: "nil pod",
			pod:  nil,
			want: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, expectedDynamoTopologyLabelKeys(tt.pod))
		})
	}
}

func TestMissingDynamoTopologyLabel(t *testing.T) {
	tests := []struct {
		name string
		pod  *corev1.Pod
		want bool
	}{
		{
			name: "expected labels all present",
			pod: topologyPredicateClusterPod("node-1", map[string]string{
				consts.DynamoTopologyLabelKey("zone"): "",
				consts.DynamoTopologyLabelKey("rack"): "rack-22",
			}),
			want: false,
		},
		{
			name: "expected label partially missing",
			pod: topologyPredicateClusterPod("node-1", map[string]string{
				consts.DynamoTopologyLabelKey("zone"): "us-east-1a",
			}),
			want: true,
		},
		{
			name: "no expected labels and no Dynamo topology labels",
			pod:  topologyPredicatePod("node-1", nil, nil),
			want: true,
		},
		{
			name: "no expected labels falls back to any Dynamo topology label",
			pod: topologyPredicatePod("node-1", nil, map[string]string{
				consts.DynamoTopologyLabelKey("zone"): "us-east-1a",
			}),
			want: false,
		},
		{
			name: "nil pod",
			pod:  nil,
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, missingDynamoTopologyLabel(tt.pod))
		})
	}
}

func TestTopologyLabelPredicateCreate(t *testing.T) {
	predicate := topologyLabelPredicate()

	tests := []struct {
		name string
		pod  *corev1.Pod
		want bool
	}{
		{
			name: "labelKey scheduled missing label",
			pod: topologyPredicatePod("node-1", map[string]string{
				consts.KubeAnnotationTopologyLabelKey: topologyPredicateLabelKey,
			}, nil),
			want: true,
		},
		{
			name: "cluster topology scheduled missing labels",
			pod:  topologyPredicateClusterPod("node-1", nil),
			want: true,
		},
		{
			name: "cluster topology scheduled partially labeled",
			pod: topologyPredicateClusterPod("node-1", map[string]string{
				consts.DynamoTopologyLabelKey("zone"): "us-east-1a",
			}),
			want: true,
		},
		{
			name: "unscheduled annotated pod",
			pod: topologyPredicatePod("", map[string]string{
				consts.KubeAnnotationTopologyLabelKey: topologyPredicateLabelKey,
			}, nil),
			want: false,
		},
		{
			name: "scheduled pod without topology source",
			pod:  topologyPredicatePod("node-1", nil, nil),
			want: false,
		},
		{
			name: "labelKey already present",
			pod: topologyPredicatePod("node-1", map[string]string{
				consts.KubeAnnotationTopologyLabelKey: topologyPredicateLabelKey,
			}, map[string]string{
				topologyPredicateLabelKey: "us-east-1a",
			}),
			want: false,
		},
		{
			name: "cluster topology fully labeled",
			pod: topologyPredicateClusterPod("node-1", map[string]string{
				consts.DynamoTopologyLabelKey("zone"): "us-east-1a",
				consts.DynamoTopologyLabelKey("rack"): "rack-22",
			}),
			want: false,
		},
		{
			name: "missing Dynamo ownership labels",
			pod:  topologyPredicateNonDynamoPod(nil),
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, predicate.Create(event.CreateEvent{Object: tt.pod}))
		})
	}
}

func TestTopologyLabelPredicateUpdate(t *testing.T) {
	predicate := topologyLabelPredicate()

	needsLabelCopy := topologyPredicatePod("node-1", map[string]string{
		consts.KubeAnnotationTopologyLabelKey: topologyPredicateLabelKey,
	}, nil)
	labelCopyComplete := topologyPredicatePod("node-1", map[string]string{
		consts.KubeAnnotationTopologyLabelKey: topologyPredicateLabelKey,
	}, map[string]string{
		topologyPredicateLabelKey: "us-east-1a",
	})
	needsClusterTopologyCopy := topologyPredicateClusterPod("node-1", nil)
	clusterTopologyCopyComplete := topologyPredicateClusterPod("node-1", map[string]string{
		consts.DynamoTopologyLabelKey("zone"): "us-east-1a",
		consts.DynamoTopologyLabelKey("rack"): "rack-22",
	})

	tests := []struct {
		name string
		old  client.Object
		new  client.Object
		want bool
	}{
		{
			name: "labelKey pod becomes scheduled",
			old: topologyPredicatePod("", map[string]string{
				consts.KubeAnnotationTopologyLabelKey: topologyPredicateLabelKey,
			}, nil),
			new:  needsLabelCopy,
			want: true,
		},
		{
			name: "labelKey topology source changes",
			old: topologyPredicatePod("node-1", map[string]string{
				consts.KubeAnnotationTopologyLabelKey: topologyPredicateOtherLabelKey,
			}, nil),
			new:  needsLabelCopy,
			want: true,
		},
		{
			name: "labelKey label removed",
			old:  labelCopyComplete,
			new:  needsLabelCopy,
			want: true,
		},
		{
			name: "cluster topology pod becomes scheduled",
			old:  topologyPredicateClusterPod("", nil),
			new:  needsClusterTopologyCopy,
			want: true,
		},
		{
			name: "cluster topology source changes",
			old: topologyPredicatePod("node-1", map[string]string{
				consts.KubeAnnotationTopologyClusterTopologyName: "old-topology",
			}, nil),
			new:  needsClusterTopologyCopy,
			want: true,
		},
		{
			name: "cluster topology label removed",
			old:  clusterTopologyCopyComplete,
			new: topologyPredicateClusterPod("node-1", map[string]string{
				consts.DynamoTopologyLabelKey("zone"): "us-east-1a",
			}),
			want: true,
		},
		{
			name: "labelKey pod still waiting",
			old:  needsLabelCopy,
			new:  needsLabelCopy,
			want: false,
		},
		{
			name: "cluster topology pod still waiting",
			old:  needsClusterTopologyCopy,
			new:  needsClusterTopologyCopy,
			want: false,
		},
		{
			name: "new pod no longer needs copy",
			old:  needsLabelCopy,
			new:  labelCopyComplete,
			want: false,
		},
		{
			name: "non-pod update",
			old:  &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}},
			new:  &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, predicate.Update(event.UpdateEvent{
				ObjectOld: tt.old,
				ObjectNew: tt.new,
			}))
		})
	}
}
