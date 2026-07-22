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

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

// setCheckpointOwner marks snap as controller-owned by ckpt (manual owner ref, scheme-free).
func setCheckpointOwner(ckpt *nvidiacomv1alpha1.DynamoCheckpoint, snap *nvidiacomv1alpha1.PodSnapshot) {
	snap.OwnerReferences = []metav1.OwnerReference{{
		APIVersion: nvidiacomv1alpha1.GroupVersion.String(),
		Kind:       "DynamoCheckpoint",
		Name:       ckpt.Name,
		UID:        ckpt.UID,
		Controller: ptr.To(true),
	}}
}

func newCheckpointJob(name string) *batchv1.Job {
	return &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: testNamespace, UID: types.UID("job-uid")},
	}
}

// markCheckpointJobComplete stamps JobComplete=True for Ready promotion tests.
func markCheckpointJobComplete(job *batchv1.Job) *batchv1.Job {
	job.Status.Conditions = append(job.Status.Conditions, batchv1.JobCondition{
		Type:   batchv1.JobComplete,
		Status: corev1.ConditionTrue,
	})
	return job
}

// podNameFromJob derives the test source-pod name for a checkpoint Job.
func podNameFromJob(jobName string) string {
	return jobName + "-pod"
}

// podNamed builds a minimal source pod with the test UID convention "<name>-uid".
func podNamed(name string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: testNamespace, UID: types.UID(name + "-uid")},
	}
}

func newOwnedPod(podName string, job *batchv1.Job) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: testNamespace,
			UID:       types.UID(podName + "-uid"),
			Labels:    map[string]string{batchv1.JobNameLabel: job.Name},
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: "batch/v1",
				Kind:       "Job",
				Name:       job.Name,
				UID:        job.UID,
				Controller: ptr.To(true),
			}},
		},
	}
}

func newOwnedCheckpoint() *nvidiacomv1alpha1.DynamoCheckpoint {
	ckpt := makeTestCheckpoint(nvidiacomv1alpha1.DynamoCheckpointPhaseCreating)
	ckpt.UID = types.UID("ckpt-uid")
	return ckpt
}

func TestFindSourcePod_ReturnsJobOwnedPod(t *testing.T) {
	job := newCheckpointJob(defaultCheckpointJobName)
	pod := newOwnedPod("worker-xyz", job)
	r := makeCheckpointReconciler(checkpointTestScheme(), job, pod)

	got, err := r.findSourcePod(context.Background(), job)
	require.NoError(t, err)
	require.NotNil(t, got)
	assert.Equal(t, "worker-xyz", got.Name)
}

func TestFindSourcePod_NotCreatedReturnsNotFound(t *testing.T) {
	job := newCheckpointJob(defaultCheckpointJobName)
	r := makeCheckpointReconciler(checkpointTestScheme(), job)

	got, err := r.findSourcePod(context.Background(), job)
	require.Error(t, err)
	assert.True(t, apierrors.IsNotFound(err))
	assert.Nil(t, got)
	assert.NoError(t, client.IgnoreNotFound(err))
}

func TestFindSourcePod_IgnoresPodNotOwnedByJob(t *testing.T) {
	job := newCheckpointJob(defaultCheckpointJobName)
	other := newOwnedPod("stray", &batchv1.Job{ObjectMeta: metav1.ObjectMeta{Name: job.Name, UID: types.UID("different-uid")}})
	r := makeCheckpointReconciler(checkpointTestScheme(), job, other)

	_, err := r.findSourcePod(context.Background(), job)
	assert.True(t, apierrors.IsNotFound(err))
}

// foreignPodSnapshot builds a PodSnapshot at the checkpoint's name carrying the owner search label
// but NOT controlled by ckpt (a name/label collision from another owner).
func foreignPodSnapshot(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) *nvidiacomv1alpha1.PodSnapshot {
	return &nvidiacomv1alpha1.PodSnapshot{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podSnapshotName(ckpt),
			Namespace: testNamespace,
			Labels:    map[string]string{consts.SnapshotOwnerLabel: ckpt.Name},
		},
		Spec: nvidiacomv1alpha1.PodSnapshotSpec{
			Source: nvidiacomv1alpha1.PodSnapshotSource{PodRef: nvidiacomv1alpha1.PodReference{Name: "someone-else"}},
		},
	}
}

func TestCreatePodSnapshot_CreatesWhenAbsent(t *testing.T) {
	ckpt := newOwnedCheckpoint()
	r := makeCheckpointReconciler(checkpointTestScheme(), ckpt)

	created, err := r.createPodSnapshot(context.Background(), ckpt, testHash, podNamed("worker-xyz"))
	require.NoError(t, err)
	require.NotNil(t, created)
	assert.Equal(t, ckpt.Name, created.Name, "PodSnapshot name is the checkpoint name")

	snap := &nvidiacomv1alpha1.PodSnapshot{}
	require.NoError(t, r.Get(context.Background(),
		client.ObjectKey{Namespace: testNamespace, Name: podSnapshotName(ckpt)}, snap))
	assert.Equal(t, ckpt.Name, snap.Labels[consts.SnapshotOwnerLabel])
	assert.Equal(t, testHash, snap.Labels[snapshotprotocol.CheckpointIDLabel])
	assert.Equal(t, "worker-xyz", snap.Spec.Source.PodRef.Name)
	assert.Equal(t, "worker-xyz-uid", string(snap.Spec.Source.PodRef.UID),
		"source pod UID is pinned so a same-named recreation is rejected")
	assert.True(t, metav1.IsControlledBy(snap, ckpt), "snapshot must be controlled by the checkpoint")
}

func TestCreatePodSnapshot_AlreadyExistsOwnedReturnsExisting(t *testing.T) {
	ckpt := newOwnedCheckpoint()
	r := makeCheckpointReconciler(checkpointTestScheme(), ckpt)
	// First create succeeds; second create hits AlreadyExists but classifies the object as ours
	// (cache lag) and returns it — no error, no duplicate.
	first, err := r.createPodSnapshot(context.Background(), ckpt, testHash, podNamed("worker-xyz"))
	require.NoError(t, err)

	second, err := r.createPodSnapshot(context.Background(), ckpt, testHash, podNamed("worker-xyz"))
	require.NoError(t, err)
	require.NotNil(t, second)
	assert.Equal(t, first.Name, second.Name)

	var snaps nvidiacomv1alpha1.PodSnapshotList
	require.NoError(t, r.List(context.Background(), &snaps, client.InNamespace(testNamespace)))
	assert.Len(t, snaps.Items, 1, "no duplicate snapshot created")
}

func TestCreatePodSnapshot_AlreadyExistsForeignFailsTerminally(t *testing.T) {
	ckpt := newOwnedCheckpoint()
	r := makeCheckpointReconciler(checkpointTestScheme(), ckpt, foreignPodSnapshot(ckpt))

	_, err := r.createPodSnapshot(context.Background(), ckpt, testHash, podNamed("worker-xyz"))
	require.Error(t, err)
	assert.ErrorIs(t, err, errPodSnapshotNameConflict)
	assert.Contains(t, err.Error(), "is not controlled by checkpoint")
}

func TestCreatePodSnapshot_AlreadyExistsNotYetVisibleRequeues(t *testing.T) {
	ckpt := newOwnedCheckpoint()
	// The interceptor makes every PodSnapshot Create return AlreadyExists, but no PodSnapshot is
	// seeded — so classifyExistingPodSnapshot's re-read Get returns NotFound, meaning the cache
	// hasn't caught up. The result must requeue (non-nil error, IsAlreadyExists) but must NOT be
	// the terminal sentinel.
	name := podSnapshotName(ckpt)
	funcs := interceptor.Funcs{
		Create: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.CreateOption) error {
			if _, ok := obj.(*nvidiacomv1alpha1.PodSnapshot); ok {
				return apierrors.NewAlreadyExists(
					schema.GroupResource{Group: "nvidia.com", Resource: "podsnapshots"}, name)
			}
			return c.Create(ctx, obj, opts...)
		},
	}
	r := makeCheckpointReconcilerWithInterceptor(checkpointTestScheme(), funcs, ckpt)

	_, err := r.createPodSnapshot(context.Background(), ckpt, testHash, podNamed("worker-xyz"))
	require.Error(t, err)
	assert.True(t, apierrors.IsAlreadyExists(err), "should requeue via the original AlreadyExists")
	assert.False(t, errors.Is(err, errPodSnapshotNameConflict), "cache-lag requeue is not a terminal conflict")
}

func TestFindOwnedPodSnapshot_FindsOwnedIgnoresForeign(t *testing.T) {
	ckpt := newOwnedCheckpoint()
	owned := buildPodSnapshot(ckpt, testHash, podNamed("worker-xyz"))
	setCheckpointOwner(ckpt, owned)
	foreign := foreignPodSnapshot(ckpt)
	foreign.Name = "foreign-snap" // different name so both can coexist, same owner label
	r := makeCheckpointReconciler(checkpointTestScheme(), ckpt, owned, foreign)

	got, err := r.findOwnedPodSnapshot(context.Background(), ckpt)
	require.NoError(t, err)
	require.NotNil(t, got)
	assert.Equal(t, ckpt.Name, got.Name)
}

func TestFindOwnedPodSnapshot_NoneReturnsNotFound(t *testing.T) {
	ckpt := newOwnedCheckpoint()
	r := makeCheckpointReconciler(checkpointTestScheme(), ckpt)

	got, err := r.findOwnedPodSnapshot(context.Background(), ckpt)
	require.Error(t, err)
	assert.True(t, apierrors.IsNotFound(err))
	assert.Nil(t, got)
}

func TestFindOwnedPodSnapshot_MultipleOwnedErrors(t *testing.T) {
	ckpt := newOwnedCheckpoint()
	first := buildPodSnapshot(ckpt, testHash, podNamed("worker-0"))
	setCheckpointOwner(ckpt, first)
	second := buildPodSnapshot(ckpt, testHash, podNamed("worker-1"))
	second.Name = "second-owned"
	setCheckpointOwner(ckpt, second)
	r := makeCheckpointReconciler(checkpointTestScheme(), ckpt, first, second)

	_, err := r.findOwnedPodSnapshot(context.Background(), ckpt)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "multiple PodSnapshots owned")
	// Non-terminal: a transient invariant report, not a Forbidden.
	assert.False(t, apierrors.IsForbidden(err))
}

func TestMapSourcePodToCheckpoint(t *testing.T) {
	withLabel := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{
		Name: "worker-0", Namespace: testNamespace,
		Labels: map[string]string{consts.SnapshotOwnerLabel: "my-checkpoint"},
	}}
	reqs := mapSourcePodToCheckpoint(context.Background(), withLabel)
	require.Len(t, reqs, 1)
	assert.Equal(t, testNamespace, reqs[0].Namespace)
	assert.Equal(t, "my-checkpoint", reqs[0].Name)

	noLabel := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "worker-0", Namespace: testNamespace}}
	assert.Nil(t, mapSourcePodToCheckpoint(context.Background(), noLabel))
}

func TestIsCheckpointSourcePod(t *testing.T) {
	source := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{
		Labels: map[string]string{snapshotprotocol.CheckpointSourceLabel: "true"},
	}}
	assert.True(t, isCheckpointSourcePod(source))

	plain := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{}}
	assert.False(t, isCheckpointSourcePod(plain))
}

func TestUpdateFailedStatus_MarksCheckpointFailed(t *testing.T) {
	ckpt := newOwnedCheckpoint()
	r := makeCheckpointReconciler(checkpointTestScheme(), ckpt)

	r.updateFailedStatus(context.Background(), ckpt, assert.AnError)
	assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseFailed, ckpt.Status.Phase)
	assert.Contains(t, ckpt.Status.Message, "snapshot creation failed")
}
