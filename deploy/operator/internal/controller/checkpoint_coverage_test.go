/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package controller

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

// makeCheckpointReconcilerWithInterceptor mirrors makeCheckpointReconciler but threads
// interceptor.Funcs so a test can inject API errors on specific code paths.
func makeCheckpointReconcilerWithInterceptor(s *runtime.Scheme, funcs interceptor.Funcs, objs ...client.Object) *CheckpointReconciler {
	return &CheckpointReconciler{
		Client: fake.NewClientBuilder().WithScheme(s).WithObjects(objs...).
			WithStatusSubresource(&nvidiacomv1alpha1.DynamoCheckpoint{}).
			WithInterceptorFuncs(funcs).Build(),
		Config:   checkpointTestConfig(),
		Recorder: record.NewFakeRecorder(10),
	}
}

// ownedCheckpointSnapshot builds a PodSnapshot carrying the owner search label AND a controller
// owner ref to ckpt, so findOwnedPodSnapshot matches it.
func ownedCheckpointSnapshot(ckpt *nvidiacomv1alpha1.DynamoCheckpoint, name string) *nvidiacomv1alpha1.PodSnapshot {
	snap := &nvidiacomv1alpha1.PodSnapshot{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: testNamespace,
			Labels:    map[string]string{consts.SnapshotOwnerLabel: ckpt.Name},
		},
	}
	setCheckpointOwner(ckpt, snap)
	return snap
}

// healthyCheckpointJob is a Job with a deadline that has not passed and no terminal condition.
func healthyCheckpointJob(name string) *batchv1.Job {
	job := newCheckpointJob(name)
	job.CreationTimestamp = metav1.Now()
	job.Spec.ActiveDeadlineSeconds = ptr.To(int64(3600))
	return job
}

// drainEvent reports whether the FakeRecorder emitted an event containing want.
func drainEvent(t *testing.T, r *CheckpointReconciler, want string) bool {
	t.Helper()
	rec, ok := r.Recorder.(*record.FakeRecorder)
	require.True(t, ok)
	select {
	case ev := <-rec.Events:
		return assert.Contains(t, ev, want)
	default:
		return false
	}
}

func TestCheckpointCreatePodSnapshot_CreateErrorEmitsEvent(t *testing.T) {
	ckpt := newOwnedCheckpoint()
	funcs := interceptor.Funcs{
		Create: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.CreateOption) error {
			if _, ok := obj.(*nvidiacomv1alpha1.PodSnapshot); ok {
				return errors.New("apiserver unavailable")
			}
			return c.Create(ctx, obj, opts...)
		},
	}
	r := makeCheckpointReconcilerWithInterceptor(checkpointTestScheme(), funcs, ckpt)

	_, err := r.createPodSnapshot(context.Background(), ckpt, testHash, podNamed("worker-xyz"))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "create PodSnapshot")
	assert.True(t, drainEvent(t, r, "PodSnapshotCreateFailed"), "expected a PodSnapshotCreateFailed event")
}

func TestCheckpointFindSourcePod_ListErrorPropagates(t *testing.T) {
	job := newCheckpointJob(defaultCheckpointJobName)
	funcs := interceptor.Funcs{
		List: func(ctx context.Context, c client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
			if _, ok := list.(*corev1.PodList); ok {
				return errors.New("list pods failed")
			}
			return c.List(ctx, list, opts...)
		},
	}
	r := makeCheckpointReconcilerWithInterceptor(checkpointTestScheme(), funcs, job)

	got, err := r.findSourcePod(context.Background(), job)
	require.Error(t, err)
	assert.Nil(t, got)
	assert.Contains(t, err.Error(), "list pods failed")
}

func TestCheckpointUpdateFailedStatus_StatusErrorDoesNotPanic(t *testing.T) {
	ckpt := newOwnedCheckpoint()
	funcs := interceptor.Funcs{
		SubResourceUpdate: func(ctx context.Context, c client.Client, sub string, obj client.Object, opts ...client.SubResourceUpdateOption) error {
			return errors.New("status update failed")
		},
	}
	r := makeCheckpointReconcilerWithInterceptor(checkpointTestScheme(), funcs, ckpt)

	// updateFailedStatus swallows the status-write error (log-only) but still sets the phase.
	r.updateFailedStatus(context.Background(), ckpt, errors.New("snapshot boom"))
	assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseFailed, ckpt.Status.Phase)
	assert.Contains(t, ckpt.Status.Message, "snapshot boom")
}

func TestCheckpointHandleCreating_MultipleOwnedRequeues(t *testing.T) {
	ckpt := newOwnedCheckpoint()
	ckpt.Status.JobName = defaultCheckpointJobName
	job := healthyCheckpointJob(defaultCheckpointJobName)
	snapA := ownedCheckpointSnapshot(ckpt, "owned-a")
	snapB := ownedCheckpointSnapshot(ckpt, "owned-b")
	r := makeCheckpointReconciler(checkpointTestScheme(), ckpt, job, snapA, snapB)

	_, err := r.handleCreating(context.Background(), ckpt)
	require.Error(t, err, "more than one owned PodSnapshot is a non-terminal invariant violation")

	got := &nvidiacomv1alpha1.DynamoCheckpoint{}
	require.NoError(t, r.Get(context.Background(), client.ObjectKey{Namespace: testNamespace, Name: ckpt.Name}, got))
	assert.NotEqual(t, nvidiacomv1alpha1.DynamoCheckpointPhaseFailed, got.Status.Phase, "ambiguous lookup must requeue, not fail")
}

// Note: the success / job-deleted / missing-snapshot+failed/healthy-job / source-pod-absent
// handleCreating outcomes are covered by the table-driven TestCheckpointReconciler_HandleCreating
// subtests; this file adds only the error-injection and pure-helper cases not expressible there.

func TestCheckpointHandleCreating_NoJobNameResetsToPending(t *testing.T) {
	ckpt := newOwnedCheckpoint()
	ckpt.Status.JobName = ""
	r := makeCheckpointReconciler(checkpointTestScheme(), ckpt)

	_, err := r.handleCreating(context.Background(), ckpt)
	require.NoError(t, err)

	got := &nvidiacomv1alpha1.DynamoCheckpoint{}
	require.NoError(t, r.Get(context.Background(), client.ObjectKey{Namespace: testNamespace, Name: ckpt.Name}, got))
	assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhasePending, got.Status.Phase)
}
