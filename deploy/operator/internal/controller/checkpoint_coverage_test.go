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
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commonController "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
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

func setCheckpointJobOwner(ckpt *nvidiacomv1alpha1.DynamoCheckpoint, job *batchv1.Job) {
	job.OwnerReferences = []metav1.OwnerReference{{
		APIVersion: nvidiacomv1alpha1.GroupVersion.String(),
		Kind:       "DynamoCheckpoint",
		Name:       ckpt.Name,
		UID:        ckpt.UID,
		Controller: ptr.To(true),
	}}
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

func TestCheckpointHandleCreating_RemovesLegacyTTLAndRetriesUpdate(t *testing.T) {
	ckpt := newOwnedCheckpoint()
	ckpt.Status.JobName = defaultCheckpointJobName
	job := healthyCheckpointJob(defaultCheckpointJobName)
	job.Spec.TTLSecondsAfterFinished = ptr.To(snapshotprotocol.DefaultCheckpointJobTTLSeconds)
	setCheckpointJobOwner(ckpt, job)

	updateCalls := 0
	updateErr := errors.New("update job failed")
	funcs := interceptor.Funcs{
		Update: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.UpdateOption) error {
			if _, ok := obj.(*batchv1.Job); ok {
				updateCalls++
				if updateCalls == 1 {
					return updateErr
				}
			}
			return c.Update(ctx, obj, opts...)
		},
	}
	r := makeCheckpointReconcilerWithInterceptor(checkpointTestScheme(), funcs, ckpt, job)
	r.RuntimeConfig = &commonController.RuntimeConfig{Gate: features.Gates{Checkpoint: true}}

	_, err := r.handleCreating(context.Background(), ckpt)
	require.ErrorIs(t, err, updateErr)
	stored := &batchv1.Job{}
	require.NoError(t, r.Get(context.Background(), client.ObjectKeyFromObject(job), stored))
	require.NotNil(t, stored.Spec.TTLSecondsAfterFinished)

	_, err = r.handleCreating(context.Background(), ckpt)
	require.NoError(t, err)
	require.NoError(t, r.Get(context.Background(), client.ObjectKeyFromObject(job), stored))
	assert.Nil(t, stored.Spec.TTLSecondsAfterFinished)
	assert.Equal(t, 2, updateCalls)
}

func TestCheckpointHandlePendingRecoversOwnedLegacyJob(t *testing.T) {
	ctx := context.Background()
	ckpt := newOwnedCheckpoint()
	ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhasePending
	job := healthyCheckpointJob(defaultCheckpointJobName)
	job.Spec.TTLSecondsAfterFinished = ptr.To(snapshotprotocol.DefaultCheckpointJobTTLSeconds)
	serverLabels := map[string]string{
		batchv1.ControllerUidLabel: string(job.UID),
		batchv1.JobNameLabel:       job.Name,
	}
	job.Spec.Selector = &metav1.LabelSelector{MatchLabels: serverLabels}
	job.Spec.Template.Labels = serverLabels
	setCheckpointJobOwner(ckpt, job)

	updateErr := errors.New("update job failed")
	statusErr := errors.New("status update failed")
	jobUpdateCalls := 0
	statusUpdateCalls := 0
	funcs := interceptor.Funcs{
		Update: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.UpdateOption) error {
			if _, ok := obj.(*batchv1.Job); ok {
				jobUpdateCalls++
				if jobUpdateCalls == 1 {
					return updateErr
				}
			}
			return c.Update(ctx, obj, opts...)
		},
		SubResourceUpdate: func(ctx context.Context, c client.Client, subResourceName string, obj client.Object, opts ...client.SubResourceUpdateOption) error {
			statusUpdateCalls++
			if statusUpdateCalls == 1 {
				return statusErr
			}
			return c.SubResource(subResourceName).Update(ctx, obj, opts...)
		},
	}
	r := makeCheckpointReconcilerWithInterceptor(checkpointTestScheme(), funcs, ckpt, job)
	r.RuntimeConfig = &commonController.RuntimeConfig{Gate: features.Gates{Checkpoint: true}}

	_, err := r.handlePending(ctx, ckpt)
	require.ErrorIs(t, err, updateErr)

	storedCheckpoint := &nvidiacomv1alpha1.DynamoCheckpoint{}
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(ckpt), storedCheckpoint))
	_, err = r.handlePending(ctx, storedCheckpoint)
	require.ErrorIs(t, err, statusErr)

	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(ckpt), storedCheckpoint))
	_, err = r.handlePending(ctx, storedCheckpoint)
	require.NoError(t, err)

	storedJob := &batchv1.Job{}
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(job), storedJob))
	assert.Nil(t, storedJob.Spec.TTLSecondsAfterFinished)
	assert.Equal(t, serverLabels, storedJob.Spec.Selector.MatchLabels)
	assert.Equal(t, serverLabels, storedJob.Spec.Template.Labels)
	assert.Equal(t, 2, jobUpdateCalls, "owned existing Job must only receive the narrow TTL update")

	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(ckpt), storedCheckpoint))
	assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseCreating, storedCheckpoint.Status.Phase)
	assert.Equal(t, job.Name, storedCheckpoint.Status.JobName)
}

func TestCheckpointHandlePendingRecoversOwnedJobWhenRenderingFails(t *testing.T) {
	ctx := context.Background()
	ckpt := newOwnedCheckpoint()
	ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhasePending
	ckpt.Spec.Job.PodTemplateSpec.Spec.ResourceClaims = []corev1.PodResourceClaim{{
		Name:                      "gpu",
		ResourceClaimTemplateName: ptr.To("missing"),
	}}
	ckpt.Spec.Job.PodTemplateSpec.Spec.Containers[0].Resources.Claims = []corev1.ResourceClaim{{Name: "gpu"}}
	job := healthyCheckpointJob(defaultCheckpointJobName)
	job.Spec.TTLSecondsAfterFinished = ptr.To(snapshotprotocol.DefaultCheckpointJobTTLSeconds)
	setCheckpointJobOwner(ckpt, job)
	r := makeCheckpointReconciler(checkpointTestScheme(), ckpt, job)
	r.RuntimeConfig = &commonController.RuntimeConfig{Gate: features.Gates{Checkpoint: true}}

	_, err := buildCheckpointJob(ctx, r.Client, r.Config, ckpt, job.Name)
	require.ErrorContains(t, err, "failed to get ResourceClaimTemplate default/missing")
	_, err = r.handlePending(ctx, ckpt)
	require.NoError(t, err)

	storedJob := &batchv1.Job{}
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(job), storedJob))
	assert.Nil(t, storedJob.Spec.TTLSecondsAfterFinished)
	storedCheckpoint := &nvidiacomv1alpha1.DynamoCheckpoint{}
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(ckpt), storedCheckpoint))
	assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseCreating, storedCheckpoint.Status.Phase)
	assert.Equal(t, job.Name, storedCheckpoint.Status.JobName)
}

func TestCheckpointReconcilePreservesLegacyReadyJob(t *testing.T) {
	tests := map[string]batchv1.JobStatus{
		"Active": {Active: 1},
		"JobComplete": {Conditions: []batchv1.JobCondition{{
			Type:   batchv1.JobComplete,
			Status: corev1.ConditionTrue,
		}}},
		"JobFailed": {Conditions: []batchv1.JobCondition{{
			Type:   batchv1.JobFailed,
			Status: corev1.ConditionTrue,
		}}},
	}

	for name, status := range tests {
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			ckpt := newOwnedCheckpoint()
			ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhaseReady
			ckpt.Status.JobName = defaultCheckpointJobName
			ckpt.Status.CreatedAt = ptr.To(metav1.NewTime(time.Unix(1, 0)))
			createdAt := ckpt.Status.CreatedAt.DeepCopy()

			job := healthyCheckpointJob(defaultCheckpointJobName)
			job.Spec.TTLSecondsAfterFinished = ptr.To(snapshotprotocol.DefaultCheckpointJobTTLSeconds)
			job.Status = status
			setCheckpointJobOwner(ckpt, job)
			r := makeCheckpointReconciler(checkpointTestScheme(), ckpt, job)

			_, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(ckpt)})
			require.NoError(t, err)

			storedCheckpoint := &nvidiacomv1alpha1.DynamoCheckpoint{}
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(ckpt), storedCheckpoint))
			assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseReady, storedCheckpoint.Status.Phase)
			assert.Equal(t, createdAt, storedCheckpoint.Status.CreatedAt)
			assert.False(t, checkpointReadyForJobCleanup(storedCheckpoint))

			storedJob := &batchv1.Job{}
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(job), storedJob))
			require.NotNil(t, storedJob.Spec.TTLSecondsAfterFinished)
			assert.Equal(t, snapshotprotocol.DefaultCheckpointJobTTLSeconds, *storedJob.Spec.TTLSecondsAfterFinished)
		})
	}
}

func TestCheckpointTerminalStatusIsDurableBeforeJobCleanup(t *testing.T) {
	tests := []struct {
		name       string
		phase      nvidiacomv1alpha1.DynamoCheckpointPhase
		complete   bool
		transition func(context.Context, *CheckpointReconciler, *nvidiacomv1alpha1.DynamoCheckpoint) (ctrl.Result, error)
	}{
		{
			name:     "Ready",
			phase:    nvidiacomv1alpha1.DynamoCheckpointPhaseReady,
			complete: true,
			transition: func(ctx context.Context, r *CheckpointReconciler, ckpt *nvidiacomv1alpha1.DynamoCheckpoint) (ctrl.Result, error) {
				return r.markCheckpointReady(ctx, ckpt, testHash, "captured")
			},
		},
		{
			name:  "Failed",
			phase: nvidiacomv1alpha1.DynamoCheckpointPhaseFailed,
			transition: func(ctx context.Context, r *CheckpointReconciler, ckpt *nvidiacomv1alpha1.DynamoCheckpoint) (ctrl.Result, error) {
				return r.failCreating(ctx, ckpt, "PodSnapshotFailed", "capture failed")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			ckpt := newOwnedCheckpoint()
			ckpt.Status.JobName = defaultCheckpointJobName
			job := healthyCheckpointJob(defaultCheckpointJobName)
			if tt.complete {
				markCheckpointJobComplete(job)
			}
			setCheckpointJobOwner(ckpt, job)
			funcs := interceptor.Funcs{
				Delete: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.DeleteOption) error {
					stored := &nvidiacomv1alpha1.DynamoCheckpoint{}
					require.NoError(t, c.Get(ctx, client.ObjectKeyFromObject(ckpt), stored))
					assert.Equal(t, tt.phase, stored.Status.Phase)
					assert.Equal(t, job.Name, stored.Status.JobName)
					if tt.phase == nvidiacomv1alpha1.DynamoCheckpointPhaseReady {
						assert.True(t, checkpointReadyForJobCleanup(stored))
					}
					return c.Delete(ctx, obj, opts...)
				},
			}
			r := makeCheckpointReconcilerWithInterceptor(checkpointTestScheme(), funcs, ckpt, job)

			_, err := tt.transition(ctx, r, ckpt)
			require.NoError(t, err)
			err = r.Get(ctx, client.ObjectKeyFromObject(job), &batchv1.Job{})
			assert.True(t, apierrors.IsNotFound(err), "terminal checkpoint Job was not deleted: %v", err)
		})
	}
}

func TestCheckpointStatusUpdateFailureRetainsJob(t *testing.T) {
	for _, transition := range []func(context.Context, *CheckpointReconciler, *nvidiacomv1alpha1.DynamoCheckpoint) (ctrl.Result, error){
		func(ctx context.Context, r *CheckpointReconciler, ckpt *nvidiacomv1alpha1.DynamoCheckpoint) (ctrl.Result, error) {
			return r.markCheckpointReady(ctx, ckpt, testHash, "captured")
		},
		func(ctx context.Context, r *CheckpointReconciler, ckpt *nvidiacomv1alpha1.DynamoCheckpoint) (ctrl.Result, error) {
			return r.failCreating(ctx, ckpt, "PodSnapshotFailed", "capture failed")
		},
	} {
		ckpt := newOwnedCheckpoint()
		ckpt.Status.JobName = defaultCheckpointJobName
		job := healthyCheckpointJob(defaultCheckpointJobName)
		setCheckpointJobOwner(ckpt, job)
		statusErr := errors.New("status update failed")
		funcs := interceptor.Funcs{
			SubResourceUpdate: func(context.Context, client.Client, string, client.Object, ...client.SubResourceUpdateOption) error {
				return statusErr
			},
		}
		r := makeCheckpointReconcilerWithInterceptor(checkpointTestScheme(), funcs, ckpt, job)

		_, err := transition(context.Background(), r, ckpt)
		require.ErrorIs(t, err, statusErr)
		require.NoError(t, r.Get(context.Background(), client.ObjectKeyFromObject(job), &batchv1.Job{}))
	}
}

func TestCheckpointDeleteFailureRetriesBeforeArtifactNormalization(t *testing.T) {
	ctx := context.Background()
	ckpt := newOwnedCheckpoint()
	ckpt.Status.JobName = defaultCheckpointJobName
	ckpt.Annotations = map[string]string{snapshotprotocol.CheckpointArtifactVersionAnnotation: "2"}
	job := markCheckpointJobComplete(newCheckpointJob(defaultCheckpointJobName))
	setCheckpointJobOwner(ckpt, job)
	deleteCalls := 0
	funcs := interceptor.Funcs{
		Delete: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.DeleteOption) error {
			deleteCalls++
			if deleteCalls == 1 {
				return errors.New("delete failed")
			}
			return c.Delete(ctx, obj, opts...)
		},
	}
	r := makeCheckpointReconcilerWithInterceptor(checkpointTestScheme(), funcs, ckpt, job)
	r.RuntimeConfig = &commonController.RuntimeConfig{Gate: features.Gates{Checkpoint: true}}

	_, err := r.markCheckpointReady(ctx, ckpt, testHash, "captured")
	require.ErrorContains(t, err, "delete terminal checkpoint job")
	stored := &nvidiacomv1alpha1.DynamoCheckpoint{}
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(ckpt), stored))
	assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseReady, stored.Status.Phase)
	assert.Equal(t, job.Name, stored.Status.JobName)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(job), &batchv1.Job{}))

	_, err = r.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(ckpt)})
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(ckpt), stored))
	assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseCreating, stored.Status.Phase)
	assert.Equal(t, "checkpoint-job-"+testHash+"-2", stored.Status.JobName)
	assert.Equal(t, 2, deleteCalls)
	err = r.Get(ctx, client.ObjectKeyFromObject(job), &batchv1.Job{})
	assert.True(t, apierrors.IsNotFound(err), "old Job was not deleted before normalization: %v", err)
}

func TestCheckpointTerminalCleanupRequiresOwnershipAndUID(t *testing.T) {
	ctx := context.Background()
	ckpt := newOwnedCheckpoint()
	ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhaseFailed
	ckpt.Status.JobName = defaultCheckpointJobName
	job := healthyCheckpointJob(defaultCheckpointJobName)
	setCheckpointJobOwner(ckpt, job)
	var deleteUID *types.UID
	var propagationPolicy *metav1.DeletionPropagation
	funcs := interceptor.Funcs{
		Delete: func(_ context.Context, _ client.WithWatch, _ client.Object, opts ...client.DeleteOption) error {
			deleteOpts := (&client.DeleteOptions{}).ApplyOptions(opts)
			deleteUID = deleteOpts.Preconditions.UID
			propagationPolicy = deleteOpts.PropagationPolicy
			return nil
		},
	}
	r := makeCheckpointReconcilerWithInterceptor(checkpointTestScheme(), funcs, ckpt, job)
	require.NoError(t, r.cleanupTerminalCheckpointJob(ctx, ckpt))
	require.NotNil(t, deleteUID)
	assert.Equal(t, job.UID, *deleteUID)
	require.NotNil(t, propagationPolicy)
	assert.Equal(t, metav1.DeletePropagationBackground, *propagationPolicy)

	job.OwnerReferences = nil
	deleteUID = nil
	propagationPolicy = nil
	r = makeCheckpointReconcilerWithInterceptor(checkpointTestScheme(), funcs, ckpt, job)
	require.NoError(t, r.cleanupTerminalCheckpointJob(ctx, ckpt))
	assert.Nil(t, deleteUID, "foreign Job must not be deleted")
	assert.Nil(t, propagationPolicy, "foreign Job must not be deleted")
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(job), &batchv1.Job{}))
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
