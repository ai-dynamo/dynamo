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

package checkpoint

import (
	"context"
	"fmt"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commonController "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

func CheckpointID(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) (string, error) {
	if ckpt == nil {
		return "", fmt.Errorf("checkpoint is nil")
	}
	if ckpt.Status.CheckpointID != "" {
		return ckpt.Status.CheckpointID, nil
	}
	if ckpt.Status.IdentityHash != "" {
		return ckpt.Status.IdentityHash, nil
	}
	if ckpt.Labels != nil && ckpt.Labels[snapshotprotocol.CheckpointIDLabel] != "" {
		return ckpt.Labels[snapshotprotocol.CheckpointIDLabel], nil
	}

	hash, err := ComputeIdentityHash(ckpt.Spec.Identity)
	if err != nil {
		return "", fmt.Errorf("failed to compute checkpoint hash for %s: %w", ckpt.Name, err)
	}

	return hash, nil
}

func FindCheckpointByCheckpointID(
	ctx context.Context,
	c client.Reader,
	namespace string,
	checkpointID string,
	excludeName string,
) (*nvidiacomv1alpha1.DynamoCheckpoint, error) {
	checkpoints := &nvidiacomv1alpha1.DynamoCheckpointList{}
	if err := c.List(
		ctx,
		checkpoints,
		client.InNamespace(namespace),
		client.MatchingLabels{snapshotprotocol.CheckpointIDLabel: checkpointID},
	); err != nil {
		return nil, fmt.Errorf("failed to list checkpoints by checkpoint ID label: %w", err)
	}

	var existing *nvidiacomv1alpha1.DynamoCheckpoint
	for i := range checkpoints.Items {
		ckpt := &checkpoints.Items[i]
		if ckpt.Name == excludeName {
			continue
		}
		existingCheckpointID, err := CheckpointID(ckpt)
		if err != nil {
			return nil, err
		}
		if existingCheckpointID != checkpointID {
			continue
		}
		if existing != nil {
			return nil, fmt.Errorf("multiple checkpoints found for checkpoint ID %s", checkpointID)
		}
		existing = ckpt.DeepCopy()
	}
	if existing != nil {
		return existing, nil
	}

	// Fall back to a full scan so legacy checkpoints without the hash label still resolve.
	checkpoints = &nvidiacomv1alpha1.DynamoCheckpointList{}
	if err := c.List(ctx, checkpoints, client.InNamespace(namespace)); err != nil {
		return nil, fmt.Errorf("failed to list checkpoints: %w", err)
	}

	for i := range checkpoints.Items {
		ckpt := &checkpoints.Items[i]
		if ckpt.Name == excludeName {
			continue
		}
		existingCheckpointID, err := CheckpointID(ckpt)
		if err != nil {
			return nil, err
		}
		if existingCheckpointID != checkpointID {
			continue
		}
		if existing != nil {
			return nil, fmt.Errorf("multiple checkpoints found for checkpoint ID %s", checkpointID)
		}
		existing = ckpt.DeepCopy()
	}

	return existing, nil
}

func FindCheckpointByIdentityHash(
	ctx context.Context,
	c client.Reader,
	namespace string,
	hash string,
	excludeName string,
) (*nvidiacomv1alpha1.DynamoCheckpoint, error) {
	return FindCheckpointByCheckpointID(ctx, c, namespace, hash, excludeName)
}

// CreateOrGetAutoCheckpoint creates the expected automatic checkpoint or
// verifies and adopts a same-name object with identical capture provenance.
func CreateOrGetAutoCheckpoint(
	ctx context.Context,
	c client.Client,
	expected *nvidiacomv1alpha1.DynamoCheckpoint,
) (*nvidiacomv1alpha1.DynamoCheckpoint, error) {
	if expected == nil {
		return nil, fmt.Errorf("expected automatic checkpoint is nil")
	}
	ckpt := expected.DeepCopy()
	checkpointID := ckpt.Labels[snapshotprotocol.CheckpointIDLabel]
	if checkpointID == "" {
		return nil, fmt.Errorf("expected automatic checkpoint ID is missing")
	}
	namespace := ckpt.Namespace
	deletionPolicy := nvidiacomv1alpha1.CheckpointDeletionPolicy(
		ckpt.Annotations[commonconsts.CheckpointDeletionPolicyAnnotation],
	)
	if deletionPolicy == "" {
		deletionPolicy = nvidiacomv1alpha1.CheckpointDeletionPolicyDelete
	}
	expectedOwnerReferences := append([]metav1.OwnerReference(nil), ckpt.OwnerReferences...)
	expectedController := metav1.GetControllerOf(ckpt)
	if deletionPolicy == nvidiacomv1alpha1.CheckpointDeletionPolicyRetain {
		ckpt.OwnerReferences = nil
	}

	if err := c.Create(ctx, ckpt); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return nil, fmt.Errorf("failed to create checkpoint %s: %w", ckpt.Name, err)
		}
		existing := &nvidiacomv1alpha1.DynamoCheckpoint{}
		key := types.NamespacedName{Name: ckpt.Name, Namespace: namespace}
		if err := c.Get(ctx, key, existing); err != nil {
			return nil, fmt.Errorf("failed to get checkpoint %s after already exists: %w", ckpt.Name, err)
		}

		existingCheckpointID, err := CheckpointID(existing)
		if err != nil {
			return nil, err
		}
		if existingCheckpointID != checkpointID {
			return nil, fmt.Errorf("checkpoint %s already exists with checkpoint ID %s", ckpt.Name, existingCheckpointID)
		}
		verificationExpected := ckpt.DeepCopy()
		// Deletion policy and ownership are lifecycle fields synchronized
		// below, not capture provenance. Still verify any existing controller
		// against the expected controller before adoption.
		verificationExpected.Annotations[commonconsts.CheckpointDeletionPolicyAnnotation] =
			existing.Annotations[commonconsts.CheckpointDeletionPolicyAnnotation]
		existingController := metav1.GetControllerOf(existing)
		if existingController == nil {
			verificationExpected.OwnerReferences = nil
		} else {
			if expectedController == nil ||
				existingController.Name != expectedController.Name ||
				existingController.UID != expectedController.UID {
				return nil, fmt.Errorf(
					"checkpoint %s automatic checkpoint mismatch: checkpoint owner differs",
					ckpt.Name,
				)
			}
			verificationExpected.OwnerReferences = existing.OwnerReferences
		}
		if err := VerifyExpectedAutoCheckpoint(existing, verificationExpected); err != nil {
			return nil, fmt.Errorf("checkpoint %s automatic checkpoint mismatch: %w", ckpt.Name, err)
		}
		original := existing.DeepCopy()
		desiredDeletionPolicy := string(deletionPolicy)
		desired := existing.DeepCopy()
		if desired.Annotations == nil {
			desired.Annotations = map[string]string{}
		}
		desired.Annotations[commonconsts.CheckpointDeletionPolicyAnnotation] = desiredDeletionPolicy
		commonController.AddFinalizer(desired)
		if deletionPolicy == nvidiacomv1alpha1.CheckpointDeletionPolicyRetain {
			desired.OwnerReferences = nil
		} else {
			desired.OwnerReferences = append(
				[]metav1.OwnerReference(nil),
				expectedOwnerReferences...,
			)
		}
		if !equality.Semantic.DeepEqual(original.Annotations, desired.Annotations) ||
			!equality.Semantic.DeepEqual(original.OwnerReferences, desired.OwnerReferences) ||
			!equality.Semantic.DeepEqual(original.Finalizers, desired.Finalizers) {
			patch := client.MergeFromWithOptions(
				original,
				client.MergeFromWithOptimisticLock{},
			)
			if err := c.Patch(ctx, desired, patch); err != nil {
				return nil, fmt.Errorf("failed to update checkpoint %s deletion policy: %w", ckpt.Name, err)
			}
			existing = desired
		}

		return existing, nil
	}

	return ckpt, nil
}

// ExpectedAutoCheckpoint returns the defaulted operator-owned checkpoint
// object without reading or writing Kubernetes resources.
func ExpectedAutoCheckpoint(
	scheme *runtime.Scheme,
	namespace string,
	checkpointID string,
	identity nvidiacomv1alpha1.DynamoCheckpointIdentity,
	podTemplate corev1.PodTemplateSpec,
	targetContainerName string,
	deletionPolicy nvidiacomv1alpha1.CheckpointDeletionPolicy,
	gpuMemoryService *nvidiacomv1alpha1.GPUMemoryServiceSpec,
	owner client.Object,
) (*nvidiacomv1alpha1.DynamoCheckpoint, error) {
	if deletionPolicy == "" {
		deletionPolicy = nvidiacomv1alpha1.CheckpointDeletionPolicyDelete
	}

	labels := map[string]string{
		snapshotprotocol.CheckpointIDLabel: checkpointID,
	}
	for _, key := range []string{
		commonconsts.KubeLabelDynamoGraphDeploymentName,
		commonconsts.KubeLabelDynamoComponent,
		commonconsts.KubeLabelDynamoWorkerHash,
	} {
		if value := podTemplate.Labels[key]; value != "" {
			labels[key] = value
		}
	}

	ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("checkpoint-%s", checkpointID),
			Namespace: namespace,
			Labels:    labels,
			Annotations: map[string]string{
				snapshotprotocol.CheckpointArtifactVersionAnnotation: snapshotprotocol.DefaultCheckpointArtifactVersion,
				commonconsts.CheckpointAutoAnnotation:                commonconsts.KubeLabelValueTrue,
				commonconsts.CheckpointDeletionPolicyAnnotation:      string(deletionPolicy),
			},
		},
		Spec: nvidiacomv1alpha1.DynamoCheckpointSpec{
			Identity:         identity,
			GPUMemoryService: gpuMemoryService,
			Job: nvidiacomv1alpha1.DynamoCheckpointJobConfig{
				PodTemplateSpec:     podTemplate,
				TargetContainerName: targetContainerName,
			},
		},
	}
	ckpt.Spec = defaultAutoCheckpointSpec(ckpt.Spec)
	if owner != nil {
		if err := controllerutil.SetControllerReference(owner, ckpt, scheme); err != nil {
			return nil, fmt.Errorf("failed to set checkpoint owner reference: %w", err)
		}
	}
	commonController.AddFinalizer(ckpt)
	return ckpt, nil
}

type automaticCheckpointProvenance struct {
	Name            string
	Namespace       string
	CheckpointID    string
	AutomaticMarker string
	ArtifactVersion string
	DeletionPolicy  string
	Controller      *metav1.OwnerReference
	CaptureSpec     nvidiacomv1alpha1.DynamoCheckpointSpec
}

func automaticCheckpointProvenanceProjection(
	ckpt *nvidiacomv1alpha1.DynamoCheckpoint,
) (automaticCheckpointProvenance, error) {
	if ckpt == nil {
		return automaticCheckpointProvenance{}, fmt.Errorf("automatic checkpoint is nil")
	}
	checkpointID, err := CheckpointID(ckpt)
	if err != nil {
		return automaticCheckpointProvenance{}, err
	}
	return automaticCheckpointProvenance{
		Name:            ckpt.Name,
		Namespace:       ckpt.Namespace,
		CheckpointID:    checkpointID,
		AutomaticMarker: ckpt.Annotations[commonconsts.CheckpointAutoAnnotation],
		ArtifactVersion: ckpt.Annotations[snapshotprotocol.CheckpointArtifactVersionAnnotation],
		DeletionPolicy:  ckpt.Annotations[commonconsts.CheckpointDeletionPolicyAnnotation],
		Controller:      metav1.GetControllerOf(ckpt),
		CaptureSpec:     autoCheckpointCaptureProvenance(ckpt.Spec),
	}, nil
}

// VerifyExpectedAutoCheckpoint compares the canonical provenance that an
// existing automatic checkpoint must retain.
func VerifyExpectedAutoCheckpoint(
	actual, expected *nvidiacomv1alpha1.DynamoCheckpoint,
) error {
	if actual == nil || expected == nil {
		return fmt.Errorf("automatic checkpoint verification requires actual and expected objects")
	}
	actualProvenance, err := automaticCheckpointProvenanceProjection(actual)
	if err != nil {
		return err
	}
	expectedProvenance, err := automaticCheckpointProvenanceProjection(expected)
	if err != nil {
		return err
	}
	switch {
	case actualProvenance.Name != expectedProvenance.Name ||
		actualProvenance.Namespace != expectedProvenance.Namespace:
		return fmt.Errorf("checkpoint identity differs")
	case actualProvenance.CheckpointID != expectedProvenance.CheckpointID:
		return fmt.Errorf("checkpoint ID differs")
	case actualProvenance.AutomaticMarker != expectedProvenance.AutomaticMarker:
		return fmt.Errorf("automatic checkpoint marker differs")
	case actualProvenance.ArtifactVersion != expectedProvenance.ArtifactVersion:
		return fmt.Errorf("checkpoint artifact version differs")
	case actualProvenance.DeletionPolicy != expectedProvenance.DeletionPolicy:
		return fmt.Errorf("checkpoint deletion policy differs")
	case !equality.Semantic.DeepEqual(actualProvenance.Controller, expectedProvenance.Controller):
		return fmt.Errorf("checkpoint owner differs")
	case !equality.Semantic.DeepEqual(actualProvenance.CaptureSpec, expectedProvenance.CaptureSpec):
		return fmt.Errorf("checkpoint capture spec differs")
	default:
		return nil
	}
}

func normalizeCaptureProbe(probe *corev1.Probe) {
	if probe == nil || probe.GRPC == nil || probe.GRPC.Service == nil {
		return
	}
	if *probe.GRPC.Service == "" {
		probe.GRPC.Service = nil
	}
}

// defaultAutoCheckpointSpec applies defaults owned by the DynamoCheckpoint
// CRD. Core PodTemplate fields are normalized by the capture-provenance
// projection below.
func defaultAutoCheckpointSpec(
	spec nvidiacomv1alpha1.DynamoCheckpointSpec,
) nvidiacomv1alpha1.DynamoCheckpointSpec {
	spec = *spec.DeepCopy()
	if spec.Identity.TensorParallelSize == 0 {
		spec.Identity.TensorParallelSize = 1
	}
	if spec.Identity.PipelineParallelSize == 0 {
		spec.Identity.PipelineParallelSize = 1
	}
	if spec.Job.TargetContainerName == "" {
		spec.Job.TargetContainerName = commonconsts.MainContainerName
	}
	if spec.Job.ActiveDeadlineSeconds == nil {
		defaultDeadline := int64(3600)
		spec.Job.ActiveDeadlineSeconds = &defaultDeadline
	}
	defaultAutoCheckpointGMS(spec.GPUMemoryService)
	return spec
}

// autoCheckpointCaptureProvenance returns the immutable inputs that define the
// process and filesystem state captured by an automatic checkpoint. It keeps
// pod metadata, scheduling, images, commands, args, env, resources, mounts,
// devices, security context, probes, and volumes.
func autoCheckpointCaptureProvenance(
	spec nvidiacomv1alpha1.DynamoCheckpointSpec,
) nvidiacomv1alpha1.DynamoCheckpointSpec {
	spec = defaultAutoCheckpointSpec(spec)
	pod := &spec.Job.PodTemplateSpec.Spec
	for _, containers := range [][]corev1.Container{
		pod.InitContainers,
		pod.Containers,
	} {
		for i := range containers {
			normalizeCaptureContainer(&containers[i])
		}
	}
	for i := range pod.EphemeralContainers {
		container := (*corev1.Container)(&pod.EphemeralContainers[i].EphemeralContainerCommon)
		normalizeCaptureContainer(container)
	}
	normalizeCaptureVolumes(pod.Volumes)
	return spec
}

func normalizeCaptureContainer(container *corev1.Container) {
	normalizeCaptureProbe(container.LivenessProbe)
	normalizeCaptureProbe(container.ReadinessProbe)
	normalizeCaptureProbe(container.StartupProbe)
	for i := range container.Ports {
		if container.Ports[i].Protocol == "" {
			container.Ports[i].Protocol = corev1.ProtocolTCP
		}
	}
	for i := range container.Env {
		if source := container.Env[i].ValueFrom; source != nil {
			if ref := source.ConfigMapKeyRef; ref != nil {
				normalizeFalsePointer(&ref.Optional)
			}
			if ref := source.SecretKeyRef; ref != nil {
				normalizeFalsePointer(&ref.Optional)
			}
			if ref := source.FileKeyRef; ref != nil {
				normalizeFalsePointer(&ref.Optional)
			}
		}
	}
	for i := range container.EnvFrom {
		source := &container.EnvFrom[i]
		if ref := source.ConfigMapRef; ref != nil {
			normalizeFalsePointer(&ref.Optional)
		}
		if ref := source.SecretRef; ref != nil {
			normalizeFalsePointer(&ref.Optional)
		}
	}
}

func normalizeCaptureVolumes(volumes []corev1.Volume) {
	for i := range volumes {
		source := &volumes[i].VolumeSource
		if ref := source.ConfigMap; ref != nil {
			normalizeFalsePointer(&ref.Optional)
		}
		if ref := source.Secret; ref != nil {
			normalizeFalsePointer(&ref.Optional)
		}
		if projected := source.Projected; projected != nil {
			for j := range projected.Sources {
				projection := &projected.Sources[j]
				if ref := projection.ConfigMap; ref != nil {
					normalizeFalsePointer(&ref.Optional)
				}
				if ref := projection.Secret; ref != nil {
					normalizeFalsePointer(&ref.Optional)
				}
			}
		}
		if rbd := source.RBD; rbd != nil {
			if rbd.RBDPool == "" {
				rbd.RBDPool = "rbd"
			}
			if rbd.RadosUser == "" {
				rbd.RadosUser = "admin"
			}
			if rbd.Keyring == "" {
				rbd.Keyring = "/etc/ceph/keyring"
			}
		}
	}
}

func normalizeFalsePointer(value **bool) {
	if *value != nil && !**value {
		*value = nil
	}
}

func defaultAutoCheckpointGMS(spec *nvidiacomv1alpha1.GPUMemoryServiceSpec) {
	if spec == nil {
		return
	}
	if spec.Mode == "" {
		spec.Mode = nvidiacomv1alpha1.GMSModeIntraPod
	}
	if spec.DeviceClassName == "" {
		spec.DeviceClassName = dra.DefaultDeviceClassName
	}
}
