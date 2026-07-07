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
	"fmt"
	"slices"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// podSnapshotName returns the PodSnapshot name for a checkpoint: the DynamoCheckpoint's own name.
// The name may change in a future naming scheme, so the controller never reconstructs it for
// lookups — it searches by the SnapshotOwnerLabel instead (see findOwnedPodSnapshot) and records the
// created name in status.podSnapshotName.
func podSnapshotName(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) string {
	return ckpt.Name
}

// findSourcePod returns the checkpoint Job's pod, or a NotFound error if the Job has not
// created it yet (callers use client.IgnoreNotFound to requeue).
func (r *CheckpointReconciler) findSourcePod(ctx context.Context, job *batchv1.Job) (*corev1.Pod, error) {
	var pods corev1.PodList
	if err := r.List(ctx, &pods,
		client.InNamespace(job.Namespace),
		client.MatchingLabels{batchv1.JobNameLabel: job.Name},
	); err != nil {
		return nil, err
	}
	for i := range pods.Items {
		if metav1.IsControlledBy(&pods.Items[i], job) {
			return &pods.Items[i], nil
		}
	}
	return nil, apierrors.NewNotFound(corev1.Resource("pods"), job.Name)
}

// findOwnedPodSnapshot returns this checkpoint's PodSnapshot, located by the SnapshotOwnerLabel and
// confirmed via IsControlledBy. It returns a NotFound error when none exists. List and Get share the
// informer cache, so this is foreign-object isolation (never act on a snapshot that is not ours), not
// a staleness fix — the authoritative existence signal for a just-created object is the Create
// AlreadyExists in createPodSnapshot. More than one owned match is a controller invariant violation
// (deterministic naming + owner refs make it impossible), so it emits a Warning and returns a
// non-terminal error to requeue rather than silently picking one.
func (r *CheckpointReconciler) findOwnedPodSnapshot(ctx context.Context, ckpt *nvidiacomv1alpha1.DynamoCheckpoint) (*nvidiacomv1alpha1.PodSnapshot, error) {
	var snaps nvidiacomv1alpha1.PodSnapshotList
	if err := r.List(ctx, &snaps,
		client.InNamespace(ckpt.Namespace),
		client.MatchingLabels{consts.SnapshotOwnerLabel: ckpt.Name},
	); err != nil {
		return nil, err
	}
	owned := slices.DeleteFunc(snaps.Items, func(snap nvidiacomv1alpha1.PodSnapshot) bool {
		return !metav1.IsControlledBy(&snap, ckpt)
	})
	switch len(owned) {
	case 0:
		return nil, apierrors.NewNotFound(
			schema.GroupResource{Group: nvidiacomv1alpha1.GroupVersion.Group, Resource: "podsnapshots"},
			ckpt.Name,
		)
	case 1:
		return &owned[0], nil
	default:
		err := fmt.Errorf("multiple PodSnapshots owned by checkpoint %q (e.g. %q and %q); expected at most one",
			ckpt.Name, owned[0].Name, owned[1].Name)
		r.Recorder.Event(ckpt, corev1.EventTypeWarning, "PodSnapshotLookupAmbiguous", err.Error())
		return nil, err
	}
}

// createPodSnapshot creates this checkpoint's PodSnapshot. The caller has confirmed (via
// findOwnedPodSnapshot) that none exists, so this is a pure create: a plain Create, not an SSA
// upsert. On AlreadyExists — a stale informer cache that missed an existing object — it returns a
// non-terminal error so the reconcile requeues and the next owner-filtered lookup picks the object up.
func (r *CheckpointReconciler) createPodSnapshot(ctx context.Context, ckpt *nvidiacomv1alpha1.DynamoCheckpoint, checkpointID, sourcePodName string) (*nvidiacomv1alpha1.PodSnapshot, error) {
	snap := buildPodSnapshot(ckpt, checkpointID, sourcePodName)
	if err := ctrl.SetControllerReference(ckpt, snap, r.Scheme()); err != nil {
		return nil, err
	}
	if err := r.Create(ctx, snap); err != nil {
		if apierrors.IsAlreadyExists(err) {
			// A PodSnapshot with this name already exists but the owner-filtered List in
			// findOwnedPodSnapshot missed it (cache lag). AlreadyExists is non-terminal
			// (IgnoreIntermediateError), so just requeue: the next reconcile's lookup finds and
			// observes it. No adopt/ownership probe here — keep the create path simple.
			return nil, fmt.Errorf("PodSnapshot %q already exists, requeueing: %w", snap.Name, err)
		}
		r.Recorder.Event(ckpt, corev1.EventTypeWarning, "PodSnapshotCreateFailed", err.Error())
		return nil, fmt.Errorf("create PodSnapshot %q: %w", snap.Name, err)
	}
	r.Recorder.Eventf(ckpt, corev1.EventTypeNormal, "PodSnapshotCreated", "Created PodSnapshot %s", snap.Name)
	return snap, nil
}

// buildPodSnapshot constructs the desired PodSnapshot for a checkpoint. The name is the checkpoint's
// own name; the SnapshotOwnerLabel is the stable lookup/search key and CheckpointIDLabel is retained
// for observability.
func buildPodSnapshot(ckpt *nvidiacomv1alpha1.DynamoCheckpoint, checkpointID, sourcePodName string) *nvidiacomv1alpha1.PodSnapshot {
	return &nvidiacomv1alpha1.PodSnapshot{
		TypeMeta: metav1.TypeMeta{
			APIVersion: nvidiacomv1alpha1.GroupVersion.String(),
			Kind:       "PodSnapshot",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      podSnapshotName(ckpt),
			Namespace: ckpt.Namespace,
			Labels: map[string]string{
				consts.SnapshotOwnerLabel:          ckpt.Name,
				snapshotprotocol.CheckpointIDLabel: checkpointID,
			},
		},
		Spec: nvidiacomv1alpha1.PodSnapshotSpec{
			Source: nvidiacomv1alpha1.PodSnapshotSource{
				PodRef: nvidiacomv1alpha1.PodReference{Name: sourcePodName},
			},
		},
	}
}

// updateFailedStatus marks the checkpoint Failed after a terminal PodSnapshot error. The failure
// event is emitted at the point of failure in createPodSnapshot; this records status only and does
// not stomp the JobCreated condition (the Job was created; only the PodSnapshot failed).
func (r *CheckpointReconciler) updateFailedStatus(ctx context.Context, ckpt *nvidiacomv1alpha1.DynamoCheckpoint, err error) {
	ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhaseFailed
	ckpt.Status.Message = fmt.Sprintf("snapshot creation failed: %v", err)
	if uerr := r.Status().Update(ctx, ckpt); uerr != nil {
		log.FromContext(ctx).Error(uerr, "failed to update DynamoCheckpoint status after snapshot failure")
	}
}
