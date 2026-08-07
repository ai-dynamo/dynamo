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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

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
	if IsAutomaticCheckpoint(ckpt) {
		return AutomaticCheckpointID(ckpt)
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

	// Standalone checkpoints retain semantic identity for upgrade compatibility.
	//nolint:staticcheck // SA1019: Standalone checkpoints still require the deprecated compatibility identity.
	identity := ckpt.Spec.Identity
	if identity == nil {
		return "", fmt.Errorf("checkpoint %s identity is missing", ckpt.Name)
	}
	hash, err := ComputeIdentityHash(*identity)
	if err != nil {
		return "", fmt.Errorf("failed to compute checkpoint hash for %s: %w", ckpt.Name, err)
	}

	return hash, nil
}

func IsAutomaticCheckpoint(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) bool {
	return ckpt != nil &&
		ckpt.Annotations != nil &&
		ckpt.Annotations[commonconsts.CheckpointAutoAnnotation] == commonconsts.KubeLabelValueTrue
}

// ClaimsAutomaticCheckpointAuthority reports whether the automatic checkpoint
// annotation key is present, regardless of its value.
func ClaimsAutomaticCheckpointAuthority(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) bool {
	if ckpt == nil || ckpt.Annotations == nil {
		return false
	}
	_, claimed := ckpt.Annotations[commonconsts.CheckpointAutoAnnotation]
	return claimed
}

// AutomaticCheckpointID returns the authoritative ID assigned directly to a
// DGD-managed checkpoint. Status and legacy identity fields are never used.
func AutomaticCheckpointID(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) (string, error) {
	if ckpt == nil {
		return "", fmt.Errorf("automatic checkpoint is nil")
	}
	if !IsAutomaticCheckpoint(ckpt) {
		return "", fmt.Errorf("checkpoint %s is not marked automatic", ckpt.Name)
	}
	if ckpt.Labels == nil || ckpt.Labels[snapshotprotocol.CheckpointIDLabel] == "" {
		return "", fmt.Errorf("automatic checkpoint %s ID label is missing", ckpt.Name)
	}
	return ckpt.Labels[snapshotprotocol.CheckpointIDLabel], nil
}

// ValidateAutomaticCheckpointClaim rejects objects that claim automatic
// authority without the metadata needed to bind capture and artifact handling.
func ValidateAutomaticCheckpointClaim(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) error {
	if ckpt == nil {
		return fmt.Errorf("checkpoint is nil")
	}

	// Treat any marker value as an automatic-authority claim.
	if !ClaimsAutomaticCheckpointAuthority(ckpt) {
		return nil
	}
	marker := ckpt.Annotations[commonconsts.CheckpointAutoAnnotation]
	if marker != commonconsts.KubeLabelValueTrue {
		return fmt.Errorf(
			"checkpoint %s automatic checkpoint marker must be %q",
			ckpt.Name,
			commonconsts.KubeLabelValueTrue,
		)
	}

	// Require the direct artifact ID and format version before controller use.
	if _, err := AutomaticCheckpointID(ckpt); err != nil {
		return err
	}
	if ckpt.Annotations[snapshotprotocol.CheckpointArtifactVersionAnnotation] == "" {
		return fmt.Errorf("automatic checkpoint %s artifact version annotation is missing", ckpt.Name)
	}
	return nil
}

type checkpointLookupDomain uint8

const (
	checkpointArtifactIDLookup checkpointLookupDomain = iota
	checkpointSemanticIdentityLookup
)

func FindCheckpointByCheckpointID(
	ctx context.Context,
	c client.Reader,
	namespace string,
	checkpointID string,
	excludeName string,
) (*nvidiacomv1alpha1.DynamoCheckpoint, error) {
	return findCheckpointByID(
		ctx,
		c,
		namespace,
		checkpointID,
		excludeName,
		checkpointArtifactIDLookup,
	)
}

func findCheckpointByID(
	ctx context.Context,
	c client.Reader,
	namespace string,
	checkpointID string,
	excludeName string,
	domain checkpointLookupDomain,
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
		if domain == checkpointSemanticIdentityLookup && ClaimsAutomaticCheckpointAuthority(ckpt) {
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
		// Automatic checkpoints resolve only through their direct ID label.
		// Marker-bearing malformed claims are also excluded from semantic lookup.
		if IsAutomaticCheckpoint(ckpt) ||
			(domain == checkpointSemanticIdentityLookup &&
				ClaimsAutomaticCheckpointAuthority(ckpt)) {
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
	return findCheckpointByID(
		ctx,
		c,
		namespace,
		hash,
		excludeName,
		checkpointSemanticIdentityLookup,
	)
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
	checkpointID, err := AutomaticCheckpointID(ckpt)
	if err != nil {
		return nil, fmt.Errorf("invalid expected automatic checkpoint: %w", err)
	}
	namespace := ckpt.Namespace
	deletionPolicy := nvidiacomv1alpha1.CheckpointDeletionPolicy(
		ckpt.Annotations[commonconsts.CheckpointDeletionPolicyAnnotation],
	)
	if deletionPolicy == "" {
		deletionPolicy = nvidiacomv1alpha1.CheckpointDeletionPolicyDelete
	}
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

		existingCheckpointID, err := AutomaticCheckpointID(existing)
		if err != nil {
			return nil, fmt.Errorf("checkpoint %s automatic checkpoint mismatch: %w", ckpt.Name, err)
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
			if !sameControllerSource(existingController, expectedController) {
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
		if deletionPolicy == nvidiacomv1alpha1.CheckpointDeletionPolicyRetain &&
			existing.Annotations[commonconsts.CheckpointDeletionPolicyAnnotation] ==
				string(nvidiacomv1alpha1.CheckpointDeletionPolicyRetain) {
			desired.OwnerReferences = nil
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

func sameControllerSource(actual, expected *metav1.OwnerReference) bool {
	if actual == nil || expected == nil {
		return actual == nil && expected == nil
	}
	return actual.APIVersion == expected.APIVersion &&
		actual.Kind == expected.Kind &&
		actual.Name == expected.Name &&
		actual.UID == expected.UID &&
		actual.Controller != nil &&
		expected.Controller != nil &&
		*actual.Controller &&
		*expected.Controller
}

// ExpectedAutoCheckpoint returns the defaulted operator-owned checkpoint
// object without reading or writing Kubernetes resources.
func ExpectedAutoCheckpoint(
	scheme *runtime.Scheme,
	namespace string,
	checkpointID string,
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
			GPUMemoryService: gpuMemoryService,
			Job: nvidiacomv1alpha1.DynamoCheckpointJobConfig{
				PodTemplateSpec:     podTemplate,
				TargetContainerName: targetContainerName,
			},
		},
	}
	captureInputs := projectAutoCheckpointCaptureInputs(ckpt.Spec)
	ckpt.Spec.Job = captureInputs.Job
	ckpt.Spec.GPUMemoryService = captureInputs.GPUMemoryService
	if owner != nil {
		if err := controllerutil.SetControllerReference(owner, ckpt, scheme); err != nil {
			return nil, fmt.Errorf("failed to set checkpoint owner reference: %w", err)
		}
	}
	commonController.AddFinalizer(ckpt)
	return ckpt, nil
}

type automaticCheckpointProvenance struct {
	Name            string                           `json:"name"`
	Namespace       string                           `json:"namespace"`
	CheckpointID    string                           `json:"checkpointID"`
	AutomaticMarker string                           `json:"automaticMarker"`
	ArtifactVersion string                           `json:"artifactVersion"`
	Controller      *metav1.OwnerReference           `json:"controller,omitempty"`
	DGDName         string                           `json:"dgdName"`
	ComponentName   string                           `json:"componentName"`
	WorkerHash      string                           `json:"workerHash"`
	CaptureInputs   automaticCheckpointCaptureInputs `json:"captureInputs"`
}

// automaticCheckpointCaptureInputs contains only the rendered inputs that
// determine an automatic checkpoint capture. Deprecated identity metadata is
// not part of automatic provenance, including on objects created by older
// operator versions.
type automaticCheckpointCaptureInputs struct {
	Job              nvidiacomv1alpha1.DynamoCheckpointJobConfig `json:"job"`
	GPUMemoryService *nvidiacomv1alpha1.GPUMemoryServiceSpec     `json:"gpuMemoryService,omitempty"`
}

func automaticCheckpointProvenanceProjection(
	ckpt *nvidiacomv1alpha1.DynamoCheckpoint,
) (automaticCheckpointProvenance, error) {
	if ckpt == nil {
		return automaticCheckpointProvenance{}, fmt.Errorf("automatic checkpoint is nil")
	}
	checkpointID, err := AutomaticCheckpointID(ckpt)
	if err != nil {
		return automaticCheckpointProvenance{}, err
	}
	return automaticCheckpointProvenance{
		Name:            ckpt.Name,
		Namespace:       ckpt.Namespace,
		CheckpointID:    checkpointID,
		AutomaticMarker: ckpt.Annotations[commonconsts.CheckpointAutoAnnotation],
		ArtifactVersion: ckpt.Annotations[snapshotprotocol.CheckpointArtifactVersionAnnotation],
		Controller:      metav1.GetControllerOf(ckpt),
		DGDName:         ckpt.Labels[commonconsts.KubeLabelDynamoGraphDeploymentName],
		ComponentName:   ckpt.Labels[commonconsts.KubeLabelDynamoComponent],
		WorkerHash:      ckpt.Labels[commonconsts.KubeLabelDynamoWorkerHash],
		CaptureInputs:   projectAutoCheckpointCaptureInputs(ckpt.Spec),
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
		return fmt.Errorf("checkpoint object differs")
	case actualProvenance.CheckpointID != expectedProvenance.CheckpointID:
		return fmt.Errorf("checkpoint ID differs")
	case actualProvenance.AutomaticMarker != expectedProvenance.AutomaticMarker:
		return fmt.Errorf("automatic checkpoint marker differs")
	case actualProvenance.ArtifactVersion != expectedProvenance.ArtifactVersion:
		return fmt.Errorf("checkpoint artifact version differs")
	case !equality.Semantic.DeepEqual(actualProvenance.Controller, expectedProvenance.Controller):
		return fmt.Errorf("checkpoint owner differs")
	case actualProvenance.DGDName != expectedProvenance.DGDName ||
		actualProvenance.ComponentName != expectedProvenance.ComponentName ||
		actualProvenance.WorkerHash != expectedProvenance.WorkerHash:
		return fmt.Errorf("checkpoint DGD source differs")
	case !equality.Semantic.DeepEqual(actualProvenance.CaptureInputs, expectedProvenance.CaptureInputs):
		return fmt.Errorf("checkpoint capture inputs differ")
	default:
		return nil
	}
}

const automaticCheckpointBindingVersion = "v1"

type automaticCheckpointBinding struct {
	UID        types.UID
	Generation int64
	Digest     [sha256.Size]byte
}

type automaticCheckpointBindingProjection struct {
	Name            string                           `json:"name"`
	Namespace       string                           `json:"namespace"`
	CheckpointID    string                           `json:"checkpointID"`
	AutomaticMarker string                           `json:"automaticMarker"`
	ArtifactVersion string                           `json:"artifactVersion"`
	ComponentName   string                           `json:"componentName"`
	WorkerHash      string                           `json:"workerHash"`
	CaptureInputs   automaticCheckpointCaptureInputs `json:"captureInputs"`
}

func automaticCheckpointBindingFor(
	ckpt *nvidiacomv1alpha1.DynamoCheckpoint,
) (automaticCheckpointBinding, error) {
	if ckpt == nil {
		return automaticCheckpointBinding{}, fmt.Errorf("automatic checkpoint binding requires an object")
	}
	if ckpt.UID == "" {
		return automaticCheckpointBinding{}, fmt.Errorf("automatic checkpoint UID is missing")
	}
	if ckpt.Generation < 1 {
		return automaticCheckpointBinding{}, fmt.Errorf("automatic checkpoint generation is missing")
	}
	provenance, err := automaticCheckpointProvenanceProjection(ckpt)
	if err != nil {
		return automaticCheckpointBinding{}, fmt.Errorf("automatic checkpoint provenance is invalid: %w", err)
	}
	bindingProjection := automaticCheckpointBindingProjection{
		Name:            provenance.Name,
		Namespace:       provenance.Namespace,
		CheckpointID:    provenance.CheckpointID,
		AutomaticMarker: provenance.AutomaticMarker,
		ArtifactVersion: provenance.ArtifactVersion,
		ComponentName:   provenance.ComponentName,
		WorkerHash:      provenance.WorkerHash,
		CaptureInputs:   provenance.CaptureInputs,
	}
	canonical, err := json.Marshal(bindingProjection)
	if err != nil {
		return automaticCheckpointBinding{}, fmt.Errorf("marshal automatic checkpoint binding projection: %w", err)
	}
	return automaticCheckpointBinding{
		UID:        ckpt.UID,
		Generation: ckpt.Generation,
		Digest:     sha256.Sum256(canonical),
	}, nil
}

// AutomaticCheckpointBinding binds a workload to an automatic checkpoint's
// UID, generation, capture inputs, and stable source identity. Retain-detach
// lifecycle metadata is intentionally excluded.
func AutomaticCheckpointBinding(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) (string, error) {
	binding, err := automaticCheckpointBindingFor(ckpt)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf(
		"%s/%s/%d/%x",
		automaticCheckpointBindingVersion,
		binding.UID,
		binding.Generation,
		binding.Digest,
	), nil
}

func parseAutomaticCheckpointBinding(value string) (automaticCheckpointBinding, error) {
	const expectedFormat = `v1/<uid>/<generation>/<sha256>`
	parts := strings.Split(value, "/")
	if len(parts) != 4 {
		return automaticCheckpointBinding{}, fmt.Errorf(
			"automatic checkpoint binding has invalid format %q; expected %s",
			value,
			expectedFormat,
		)
	}
	if parts[0] != automaticCheckpointBindingVersion {
		return automaticCheckpointBinding{}, fmt.Errorf(
			"automatic checkpoint binding version %q is unsupported",
			parts[0],
		)
	}
	if parts[1] == "" {
		return automaticCheckpointBinding{}, fmt.Errorf("automatic checkpoint binding UID is missing")
	}
	generation, err := strconv.ParseInt(parts[2], 10, 64)
	if err != nil || generation < 1 {
		return automaticCheckpointBinding{}, fmt.Errorf(
			"automatic checkpoint binding generation %q is invalid",
			parts[2],
		)
	}
	decodedDigest, err := hex.DecodeString(parts[3])
	if err != nil || len(decodedDigest) != sha256.Size {
		return automaticCheckpointBinding{}, fmt.Errorf(
			"automatic checkpoint binding SHA-256 digest %q is invalid",
			parts[3],
		)
	}
	var digest [sha256.Size]byte
	copy(digest[:], decodedDigest)
	return automaticCheckpointBinding{
		UID:        types.UID(parts[1]),
		Generation: generation,
		Digest:     digest,
	}, nil
}

func VerifyAutomaticCheckpointBinding(
	ckpt *nvidiacomv1alpha1.DynamoCheckpoint,
	expected string,
) error {
	if expected == "" {
		return fmt.Errorf("automatic checkpoint binding is missing")
	}
	expectedBinding, err := parseAutomaticCheckpointBinding(expected)
	if err != nil {
		return err
	}
	actualBinding, err := automaticCheckpointBindingFor(ckpt)
	if err != nil {
		return err
	}
	switch {
	case actualBinding.UID != expectedBinding.UID:
		return fmt.Errorf("automatic checkpoint UID differs")
	case actualBinding.Generation != expectedBinding.Generation:
		return fmt.Errorf("automatic checkpoint generation differs")
	case actualBinding.Digest != expectedBinding.Digest:
		return fmt.Errorf("automatic checkpoint provenance differs")
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

// projectAutoCheckpointCaptureInputs returns the immutable Job and GMS inputs that
// define the process and filesystem state captured by an automatic checkpoint.
// It keeps pod metadata, scheduling, images, commands, args, env, resources,
// mounts, devices, security context, probes, and volumes.
func projectAutoCheckpointCaptureInputs(
	spec nvidiacomv1alpha1.DynamoCheckpointSpec,
) automaticCheckpointCaptureInputs {
	inputs := automaticCheckpointCaptureInputs{
		Job: *spec.Job.DeepCopy(),
	}
	if spec.GPUMemoryService != nil {
		inputs.GPUMemoryService = spec.GPUMemoryService.DeepCopy()
	}
	if inputs.Job.TargetContainerName == "" {
		inputs.Job.TargetContainerName = commonconsts.MainContainerName
	}
	if inputs.Job.ActiveDeadlineSeconds == nil {
		defaultDeadline := int64(3600)
		inputs.Job.ActiveDeadlineSeconds = &defaultDeadline
	}
	defaultAutoCheckpointGMS(inputs.GPUMemoryService)
	pod := &inputs.Job.PodTemplateSpec.Spec
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
	return inputs
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
		if azureDisk := source.AzureDisk; azureDisk != nil {
			if azureDisk.FSType == nil {
				fsType := "ext4"
				azureDisk.FSType = &fsType
			}
			normalizeFalsePointer(&azureDisk.ReadOnly)
		}
		if iscsi := source.ISCSI; iscsi != nil && iscsi.ISCSIInterface == "" {
			iscsi.ISCSIInterface = "default"
		}
		if scaleIO := source.ScaleIO; scaleIO != nil {
			if scaleIO.FSType == "" {
				scaleIO.FSType = "xfs"
			}
			if scaleIO.StorageMode == "" {
				scaleIO.StorageMode = "ThinProvisioned"
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
