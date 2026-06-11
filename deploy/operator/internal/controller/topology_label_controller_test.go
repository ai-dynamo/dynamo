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

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
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
	recorder := record.NewFakeRecorder(1)
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

func TestTopologyLabelPredicate(t *testing.T) {
	const labelKey = "topology.kubernetes.io/zone"
	const otherLabelKey = "topology.kubernetes.io/rack"

	pod := func(nodeName string, annotations map[string]string, labels map[string]string) *corev1.Pod {
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

	predicate := topologyLabelPredicate()
	needsLabelCopy := pod("node-1", map[string]string{
		consts.KubeAnnotationTopologyLabelKey: labelKey,
	}, nil)

	assert.True(t, predicate.Create(event.CreateEvent{Object: needsLabelCopy}))
	assert.True(t, predicate.Update(event.UpdateEvent{
		ObjectOld: pod("", map[string]string{consts.KubeAnnotationTopologyLabelKey: labelKey}, nil),
		ObjectNew: needsLabelCopy,
	}))
	assert.True(t, predicate.Update(event.UpdateEvent{
		ObjectOld: pod("node-1", nil, nil),
		ObjectNew: needsLabelCopy,
	}))
	assert.True(t, predicate.Update(event.UpdateEvent{
		ObjectOld: pod("node-1", map[string]string{consts.KubeAnnotationTopologyLabelKey: otherLabelKey}, nil),
		ObjectNew: needsLabelCopy,
	}))
	assert.True(t, predicate.Update(event.UpdateEvent{
		ObjectOld: pod("node-1", map[string]string{consts.KubeAnnotationTopologyLabelKey: labelKey}, map[string]string{
			labelKey: "us-east-1a",
		}),
		ObjectNew: needsLabelCopy,
	}))
	assert.False(t, predicate.Create(event.CreateEvent{
		Object: pod("", map[string]string{consts.KubeAnnotationTopologyLabelKey: labelKey}, nil),
	}))
	assert.False(t, predicate.Create(event.CreateEvent{
		Object: pod("node-1", nil, nil),
	}))
	assert.False(t, predicate.Create(event.CreateEvent{
		Object: pod("node-1", map[string]string{consts.KubeAnnotationTopologyLabelKey: labelKey}, map[string]string{
			labelKey: "us-east-1a",
		}),
	}))
	assert.False(t, predicate.Update(event.UpdateEvent{
		ObjectOld: needsLabelCopy,
		ObjectNew: needsLabelCopy,
	}))
	assert.False(t, predicate.Update(event.UpdateEvent{
		ObjectOld: needsLabelCopy,
		ObjectNew: pod("node-1", map[string]string{consts.KubeAnnotationTopologyLabelKey: labelKey}, map[string]string{
			labelKey: "us-east-1a",
		}),
	}))
	nonDynamoLabelCases := map[string]map[string]string{
		"missing ownership labels": nil,
		"missing DGD label": {
			consts.KubeLabelDynamoComponent: "worker",
		},
		"missing component label": {
			consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
		},
	}
	for name, labels := range nonDynamoLabelCases {
		t.Run(name, func(t *testing.T) {
			assert.False(t, predicate.Create(event.CreateEvent{
				Object: &corev1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "non-dynamo",
						Namespace: "default",
						Annotations: map[string]string{
							consts.KubeAnnotationTopologyLabelKey: labelKey,
						},
						Labels: labels,
					},
					Spec: corev1.PodSpec{NodeName: "node-1"},
				},
			}))
		})
	}
	assert.False(t, predicate.Delete(event.DeleteEvent{Object: needsLabelCopy}))
	assert.False(t, predicate.Generic(event.GenericEvent{Object: needsLabelCopy}))
}
