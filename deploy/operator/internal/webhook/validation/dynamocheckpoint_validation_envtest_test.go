//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation_test

import (
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

func TestDynamoCheckpointAdmissionDefaultEquivalentAutomaticUpdate(t *testing.T) {
	t.Log("Build an automatic checkpoint with API-defaultable capture fields")
	checkpoint := &nvidiacomv1alpha1.DynamoCheckpoint{
		TypeMeta: metav1.TypeMeta{
			APIVersion: nvidiacomv1alpha1.GroupVersion.String(),
			Kind:       "DynamoCheckpoint",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "automatic-defaulting",
			Labels: map[string]string{
				snapshotprotocol.CheckpointIDLabel: "automatic-defaulting",
			},
			Annotations: map[string]string{
				consts.CheckpointAutoAnnotation:                      consts.KubeLabelValueTrue,
				snapshotprotocol.CheckpointArtifactVersionAnnotation: snapshotprotocol.DefaultCheckpointArtifactVersion,
			},
		},
		Spec: nvidiacomv1alpha1.DynamoCheckpointSpec{
			Job: nvidiacomv1alpha1.DynamoCheckpointJobConfig{
				PodTemplateSpec: corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{Name: consts.MainContainerName}},
					},
				},
			},
		},
	}

	t.Log("Create and update the object through API-server admission")
	result := runAdmissionTest(t, admissionTestCase{
		object:    checkpoint.DeepCopy(),
		oldObject: checkpoint.DeepCopy(),
		gates:     features.Gates{Checkpoint: true},
	})
	if result == nil {
		t.Fatal("default-equivalent automatic checkpoint update was rejected")
	}
}

func TestDynamoCheckpointAdmissionRetainDetach(t *testing.T) {
	oldCheckpoint := &nvidiacomv1alpha1.DynamoCheckpoint{
		TypeMeta: metav1.TypeMeta{
			APIVersion: nvidiacomv1alpha1.GroupVersion.String(),
			Kind:       "DynamoCheckpoint",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "automatic-retain-detach",
			Labels: map[string]string{
				snapshotprotocol.CheckpointIDLabel:        "automatic-retain-detach",
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoComponent:           "worker",
				consts.KubeLabelDynamoWorkerHash:          "worker-hash",
			},
			Annotations: map[string]string{
				consts.CheckpointAutoAnnotation:                      consts.KubeLabelValueTrue,
				snapshotprotocol.CheckpointArtifactVersionAnnotation: snapshotprotocol.DefaultCheckpointArtifactVersion,
				consts.CheckpointDeletionPolicyAnnotation:            string(nvidiacomv1alpha1.CheckpointDeletionPolicyRetain),
			},
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: "nvidia.com/v1beta1",
				Kind:       "DynamoGraphDeployment",
				Name:       "test-dgd",
				UID:        "dgd-uid",
				Controller: ptr.To(true),
			}},
		},
		Spec: nvidiacomv1alpha1.DynamoCheckpointSpec{
			Job: nvidiacomv1alpha1.DynamoCheckpointJobConfig{
				TargetContainerName: consts.MainContainerName,
				PodTemplateSpec: corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{
							Name:  consts.MainContainerName,
							Image: "worker:expected",
						}},
					},
				},
			},
		},
	}

	for _, tt := range []struct {
		name      string
		newPolicy nvidiacomv1alpha1.CheckpointDeletionPolicy
		wantErrs  []string
	}{
		{
			name:      "atomic owner and DGD label detach",
			newPolicy: nvidiacomv1alpha1.CheckpointDeletionPolicyRetain,
		},
		{
			name:      "policy flip cannot authorize detach",
			newPolicy: nvidiacomv1alpha1.CheckpointDeletionPolicyDelete,
			wantErrs: []string{
				`metadata.labels[nvidia.com/dynamo-graph-deployment-name]: Invalid value: "": field is immutable`,
				"metadata.ownerReferences: Invalid value: null: field is immutable",
			},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			current := oldCheckpoint.DeepCopy()
			current.Annotations[consts.CheckpointDeletionPolicyAnnotation] =
				string(tt.newPolicy)
			current.OwnerReferences = nil
			delete(current.Labels, consts.KubeLabelDynamoGraphDeploymentName)

			result := runAdmissionTest(t, admissionTestCase{
				object:            current,
				oldObject:         oldCheckpoint.DeepCopy(),
				gates:             features.Gates{Checkpoint: true},
				wantWebhookErrors: tt.wantErrs,
			})
			if len(tt.wantErrs) == 0 && result == nil {
				t.Fatal("Retain detach was rejected")
			}
		})
	}
}
