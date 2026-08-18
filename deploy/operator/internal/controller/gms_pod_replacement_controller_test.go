/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/controller-runtime/pkg/event"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

func TestGMSPodReplacementPredicate(t *testing.T) {
	pred := gmsPodReplacementPredicate()

	t.Run("admits restart transition", func(t *testing.T) {
		oldPod := gmsPodReplacementTestPod("pod-uid", 0)
		newPod := gmsPodReplacementTestPod("pod-uid", 1)
		assert.True(t, pred.Update(event.UpdateEvent{ObjectOld: oldPod, ObjectNew: newPod}))
	})

	t.Run("admits already-eligible create", func(t *testing.T) {
		assert.True(t, pred.Create(event.CreateEvent{Object: gmsPodReplacementTestPod("pod-uid", 1)}))
	})

	t.Run("admits same-name replacement with a new UID", func(t *testing.T) {
		oldPod := gmsPodReplacementTestPod("old-uid", 1)
		newPod := gmsPodReplacementTestPod("new-uid", 1)
		assert.True(t, pred.Update(event.UpdateEvent{ObjectOld: oldPod, ObjectNew: newPod}))
	})

	t.Run("rejects ineligible Pods", func(t *testing.T) {
		tests := map[string]func(*corev1.Pod){
			"not a restore target": func(pod *corev1.Pod) {
				delete(pod.Labels, snapshotprotocol.RestoreTargetLabel)
			},
			"ownerless": func(pod *corev1.Pod) {
				pod.OwnerReferences = nil
			},
			"ordinary init container": func(pod *corev1.Pod) {
				pod.Spec.InitContainers[0].RestartPolicy = nil
			},
			"native sidecar has not restarted": func(pod *corev1.Pod) {
				pod.Status.InitContainerStatuses[0].RestartCount = 0
			},
		}
		for name, mutate := range tests {
			t.Run(name, func(t *testing.T) {
				pod := gmsPodReplacementTestPod("pod-uid", 1)
				mutate(pod)
				assert.False(t, pred.Create(event.CreateEvent{Object: pod}))
			})
		}
	})

	t.Run("rejects unchanged eligible and deleting Pods", func(t *testing.T) {
		oldPod := gmsPodReplacementTestPod("pod-uid", 1)
		newPod := gmsPodReplacementTestPod("pod-uid", 1)
		assert.False(t, pred.Update(event.UpdateEvent{ObjectOld: oldPod, ObjectNew: newPod}))

		now := metav1.Now()
		newPod.DeletionTimestamp = &now
		assert.False(t, pred.Update(event.UpdateEvent{ObjectOld: oldPod, ObjectNew: newPod}))
	})

	t.Run("ignores deletion and generic events", func(t *testing.T) {
		pod := gmsPodReplacementTestPod("pod-uid", 1)
		assert.False(t, pred.Delete(event.DeleteEvent{Object: pod}))
		assert.False(t, pred.Generic(event.GenericEvent{Object: pod}))
	})
}

func TestGMSPodReplacementReconcile(t *testing.T) {
	tests := []struct {
		name        string
		podExists   bool
		mutate      func(*corev1.Pod)
		wantDeleted bool
	}{
		{name: "eligible Pod is deleted", podExists: true, wantDeleted: true},
		{
			name:      "ineligible Pod is retained",
			podExists: true,
			mutate: func(pod *corev1.Pod) {
				pod.Labels[snapshotprotocol.RestoreTargetLabel] = "false"
			},
		},
		{
			name:      "deleting Pod is retained",
			podExists: true,
			mutate: func(pod *corev1.Pod) {
				now := metav1.Now()
				pod.DeletionTimestamp = &now
				pod.Finalizers = []string{"test.example/finalizer"}
			},
		},
		{name: "missing Pod is harmless"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			pod := gmsPodReplacementTestPod("observed-uid", 1)
			if test.mutate != nil {
				test.mutate(pod)
			}

			var deleteUID *types.UID
			builder := fake.NewClientBuilder().
				WithScheme(gmsPodReplacementTestScheme(t)).
				WithInterceptorFuncs(interceptor.Funcs{
					Delete: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.DeleteOption) error {
						deleteUID = (&client.DeleteOptions{}).ApplyOptions(opts).Preconditions.UID
						return c.Delete(ctx, obj, opts...)
					},
				})
			if test.podExists {
				builder = builder.WithObjects(pod)
			}
			c := builder.Build()
			reconciler := &gmsPodReplacementReconciler{Client: c}
			key := client.ObjectKeyFromObject(pod)

			result, err := reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: key})
			require.NoError(t, err)
			assert.Equal(t, ctrl.Result{}, result)

			err = c.Get(context.Background(), key, &corev1.Pod{})
			if test.wantDeleted || !test.podExists {
				assert.True(t, apierrors.IsNotFound(err), "Get() error = %v", err)
			} else {
				require.NoError(t, err)
			}

			if test.wantDeleted {
				require.NotNil(t, deleteUID)
				assert.Equal(t, pod.UID, *deleteUID)
			} else {
				assert.Nil(t, deleteUID)
			}
		})
	}
}

func TestGMSPodReplacementErrors(t *testing.T) {
	retryErr := errors.New("client failed")

	t.Run("get error is returned for retry", func(t *testing.T) {
		c := fake.NewClientBuilder().
			WithScheme(gmsPodReplacementTestScheme(t)).
			WithInterceptorFuncs(interceptor.Funcs{
				Get: func(context.Context, client.WithWatch, client.ObjectKey, client.Object, ...client.GetOption) error {
					return retryErr
				},
			}).
			Build()
		reconciler := &gmsPodReplacementReconciler{Client: c}

		_, err := reconciler.Reconcile(context.Background(), ctrl.Request{
			NamespacedName: types.NamespacedName{Namespace: "inference", Name: "restore-worker"},
		})
		require.ErrorIs(t, err, retryErr)
		assert.ErrorContains(t, err, "get GMS Pod replacement candidate")
	})

	t.Run("delete NotFound is harmless", func(t *testing.T) {
		pod := gmsPodReplacementTestPod("observed-uid", 1)
		c := gmsPodReplacementErrorClient(t, pod,
			apierrors.NewNotFound(schema.GroupResource{Resource: "pods"}, pod.Name))
		reconciler := &gmsPodReplacementReconciler{Client: c}

		_, err := reconciler.Reconcile(context.Background(), ctrl.Request{
			NamespacedName: client.ObjectKeyFromObject(pod),
		})
		require.NoError(t, err)
	})

	t.Run("delete error is returned for retry", func(t *testing.T) {
		pod := gmsPodReplacementTestPod("observed-uid", 1)
		c := gmsPodReplacementErrorClient(t, pod, retryErr)
		reconciler := &gmsPodReplacementReconciler{Client: c}

		_, err := reconciler.Reconcile(context.Background(), ctrl.Request{
			NamespacedName: client.ObjectKeyFromObject(pod),
		})
		require.ErrorIs(t, err, retryErr)
		assert.ErrorContains(t, err, "delete GMS Pod replacement candidate inference/restore-worker")
	})
}

func gmsPodReplacementErrorClient(t *testing.T, pod *corev1.Pod, deleteErr error) client.Client {
	t.Helper()
	return fake.NewClientBuilder().
		WithScheme(gmsPodReplacementTestScheme(t)).
		WithObjects(pod).
		WithInterceptorFuncs(interceptor.Funcs{
			Delete: func(context.Context, client.WithWatch, client.Object, ...client.DeleteOption) error {
				return deleteErr
			},
		}).
		Build()
}

func gmsPodReplacementTestPod(uid types.UID, restartCount int32) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "restore-worker",
			Namespace: "inference",
			UID:       uid,
			Labels: map[string]string{
				snapshotprotocol.RestoreTargetLabel: consts.KubeLabelValueTrue,
			},
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: "apps/v1",
				Kind:       "Deployment",
				Name:       "restore-worker",
				UID:        "owner-uid",
				Controller: ptr.To(true),
			}},
		},
		Spec: corev1.PodSpec{
			InitContainers: []corev1.Container{{
				Name:          gms.ServerContainerName,
				RestartPolicy: ptr.To(corev1.ContainerRestartPolicyAlways),
			}},
		},
		Status: corev1.PodStatus{
			InitContainerStatuses: []corev1.ContainerStatus{{
				Name:         gms.ServerContainerName,
				RestartCount: restartCount,
			}},
		},
	}
}

func gmsPodReplacementTestScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	return scheme
}
