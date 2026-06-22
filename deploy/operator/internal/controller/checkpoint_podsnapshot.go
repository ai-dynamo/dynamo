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

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// checkpointSnapshotFieldManager is the Server-Side Apply field owner for Snapshots.
const checkpointSnapshotFieldManager = "dynamo-checkpoint-controller"

// podSnapshotName returns the deterministic PodSnapshot name for a checkpoint ID.
func podSnapshotName(checkpointID string) string {
	return "podsnapshot-" + checkpointID
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

// ensurePodSnapshot creates this checkpoint's PodSnapshot (owned by ckpt) via Server-Side Apply when
// absent, and is a no-op when it already exists and is ours.
func (r *CheckpointReconciler) ensurePodSnapshot(ctx context.Context, ckpt *nvidiacomv1alpha1.DynamoCheckpoint, checkpointID, sourcePodName string) error {
	owned, err := r.findOwnedPodSnapshot(ctx, ckpt, podSnapshotName(checkpointID))
	if err != nil {
		return err
	}
	if owned {
		return nil
	}
	return r.applyPodSnapshot(ctx, ckpt, buildPodSnapshot(ckpt, checkpointID, sourcePodName))
}

// findOwnedPodSnapshot reports whether this checkpoint's PodSnapshot already exists and is owned by
// ckpt. It returns a terminal Forbidden error (and emits an event) when a PodSnapshot with the same
// name exists but is owned by another controller; (false, nil) means none exists yet.
func (r *CheckpointReconciler) findOwnedPodSnapshot(ctx context.Context, ckpt *nvidiacomv1alpha1.DynamoCheckpoint, name string) (bool, error) {
	existing := &nvidiacomv1alpha1.PodSnapshot{}
	if err := r.Get(ctx, client.ObjectKey{Namespace: ckpt.Namespace, Name: name}, existing); err != nil {
		return false, client.IgnoreNotFound(err)
	}
	if metav1.IsControlledBy(existing, ckpt) {
		return true, nil
	}
	// Forbidden is terminal (see controller_common.IgnoreIntermediateError): a foreign-owned
	// name collision will not resolve on retry.
	conflict := apierrors.NewForbidden(
		nvidiacomv1alpha1.GroupVersion.WithResource("podsnapshots").GroupResource(),
		name,
		fmt.Errorf("exists but is not owned by checkpoint %q", ckpt.Name),
	)
	r.Recorder.Event(ckpt, corev1.EventTypeWarning, "PodSnapshotCreateFailed", conflict.Error())
	return false, conflict
}

// buildPodSnapshot constructs the desired PodSnapshot for a checkpoint.
func buildPodSnapshot(ckpt *nvidiacomv1alpha1.DynamoCheckpoint, checkpointID, sourcePodName string) *nvidiacomv1alpha1.PodSnapshot {
	return &nvidiacomv1alpha1.PodSnapshot{
		TypeMeta: metav1.TypeMeta{
			APIVersion: nvidiacomv1alpha1.GroupVersion.String(),
			Kind:       "PodSnapshot",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      podSnapshotName(checkpointID),
			Namespace: ckpt.Namespace,
			Labels:    map[string]string{snapshotprotocol.CheckpointIDLabel: checkpointID},
		},
		Spec: nvidiacomv1alpha1.PodSnapshotSpec{
			Source: nvidiacomv1alpha1.PodSnapshotSource{
				PodRef: nvidiacomv1alpha1.PodReference{Name: sourcePodName},
			},
		},
	}
}

// applyPodSnapshot sets ckpt as controller owner and applies the PodSnapshot via Server-Side Apply,
// emitting an event on success or failure.
func (r *CheckpointReconciler) applyPodSnapshot(ctx context.Context, ckpt *nvidiacomv1alpha1.DynamoCheckpoint, snap *nvidiacomv1alpha1.PodSnapshot) error {
	if err := ctrl.SetControllerReference(ckpt, snap, r.Scheme()); err != nil {
		return err
	}
	if err := r.Patch(ctx, snap, client.Apply,
		client.FieldOwner(checkpointSnapshotFieldManager), client.ForceOwnership); err != nil {
		r.Recorder.Event(ckpt, corev1.EventTypeWarning, "PodSnapshotCreateFailed", err.Error())
		return err
	}
	r.Recorder.Eventf(ckpt, corev1.EventTypeNormal, "PodSnapshotCreated", "Created PodSnapshot %s", snap.Name)
	return nil
}

// updateFailedStatus marks the checkpoint Failed after a terminal PodSnapshot error. The failure
// event is emitted at the point of failure in ensurePodSnapshot; this records status only and does
// not stomp the JobCreated condition (the Job was created; only the PodSnapshot failed).
func (r *CheckpointReconciler) updateFailedStatus(ctx context.Context, ckpt *nvidiacomv1alpha1.DynamoCheckpoint, err error) {
	ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhaseFailed
	ckpt.Status.Message = fmt.Sprintf("snapshot creation failed: %v", err)
	if uerr := r.Status().Update(ctx, ckpt); uerr != nil {
		log.FromContext(ctx).Error(uerr, "failed to update DynamoCheckpoint status after snapshot failure")
	}
}
