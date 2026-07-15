/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package mutation

import (
	"context"
	"encoding/json"
	"net/http"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

const (
	podCheckpointRestoreWebhookName = "pod-checkpoint-restore-mutating-webhook"
	podCheckpointRestoreWebhookPath = "/mutate-core-v1-pod-checkpoint-restore"
)

type PodCheckpointRestoreMutator struct {
	client ctrlclient.Client
	config *configv1alpha1.OperatorConfiguration
	scheme *runtime.Scheme
}

func NewPodCheckpointRestoreMutator(client ctrlclient.Client, config *configv1alpha1.OperatorConfiguration) *PodCheckpointRestoreMutator {
	return &PodCheckpointRestoreMutator{client: client, config: config}
}

func (h *PodCheckpointRestoreMutator) RegisterWithManager(mgr manager.Manager, gate features.Gate) error {
	h.scheme = mgr.GetScheme()
	webhook := internalwebhook.WithGate((&admission.Webhook{Handler: h}).WithRecoverPanic(true), gate)
	mgr.GetWebhookServer().Register(podCheckpointRestoreWebhookPath, webhook)
	return nil
}

func (h *PodCheckpointRestoreMutator) Handle(ctx context.Context, req admission.Request) admission.Response {
	logger := log.FromContext(ctx).WithName(podCheckpointRestoreWebhookName)

	// Restore injection changes pod spec fields that are only meaningful before
	// the pod is created; UPDATE requests are admitted unchanged.
	if req.Operation != admissionv1.Create {
		return admission.Allowed("not a pod create")
	}
	if !features.GateFromContext(ctx).Enabled(features.Checkpoint) {
		return admission.Allowed("checkpoint disabled")
	}
	if excluded := internalwebhook.GetExcludedNamespaces(); excluded != nil && excluded.Contains(req.Namespace) {
		return admission.Allowed("namespace excluded")
	}
	if h.client == nil {
		logger.Info("checkpoint restore mutator is unavailable because client is nil; allowing pod unchanged")
		return admission.Allowed("checkpoint restore mutator unavailable")
	}
	if h.scheme == nil {
		logger.Info("checkpoint restore mutator is unavailable because scheme is nil; allowing pod unchanged")
		return admission.Allowed("checkpoint restore mutator unavailable")
	}

	pod := &corev1.Pod{}
	decoder := admission.NewDecoder(h.scheme)
	if err := decoder.Decode(req, pod); err != nil {
		return admission.Errored(http.StatusBadRequest, err)
	}
	original := req.Object.Raw
	podNamespace := pod.Namespace
	if podNamespace == "" {
		podNamespace = req.Namespace
	}

	if pod.Labels != nil &&
		(pod.Labels[snapshotprotocol.CheckpointIDLabel] != "" ||
			pod.Labels[snapshotprotocol.CheckpointSourceLabel] != "") {
		return admission.Allowed("pod is already checkpoint-shaped")
	}
	if pod.Annotations == nil ||
		pod.Annotations[consts.CheckpointRestoreCandidateAnnotation] != consts.KubeLabelValueTrue {
		return admission.Allowed("pod is not a checkpoint restore candidate")
	}
	checkpointName := pod.Annotations[consts.CheckpointNameAnnotation]
	if checkpointName == "" {
		return admission.Allowed("restore candidate has no checkpoint name")
	}
	if pod.Labels == nil ||
		pod.Labels[consts.KubeLabelDynamoComponent] == "" ||
		pod.Labels[consts.KubeLabelDynamoNamespace] == "" ||
		pod.Labels[consts.KubeLabelDynamoSelector] == "" {
		return admission.Allowed("restore candidate is not operator-stamped")
	}

	ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{}
	if err := h.client.Get(ctx, types.NamespacedName{Namespace: podNamespace, Name: checkpointName}, ckpt); err != nil {
		logger.V(1).Info("checkpoint restore candidate not mutated because checkpoint could not be read",
			"namespace", podNamespace, "checkpoint", checkpointName, "error", err.Error())
		return admission.Allowed("checkpoint not available")
	}
	if ckpt.Status.Phase != nvidiacomv1alpha1.DynamoCheckpointPhaseReady {
		return admission.Allowed("checkpoint not ready")
	}

	checkpointID, err := checkpoint.CheckpointID(ckpt)
	if err != nil {
		logger.Error(err, "checkpoint restore candidate not mutated because checkpoint ID could not be resolved",
			"namespace", podNamespace, "checkpoint", checkpointName)
		return admission.Allowed("checkpoint ID unavailable")
	}
	targets, err := snapshotprotocol.TargetContainersFromAnnotations(pod.Annotations, 1, 0)
	if err != nil {
		logger.Error(err, "checkpoint restore candidate not mutated because target containers annotation is invalid",
			"namespace", podNamespace, "pod", pod.Name, "checkpoint", checkpointName)
		return admission.Allowed("checkpoint target containers invalid")
	}
	artifactVersion := snapshotprotocol.ArtifactVersion(ckpt.Annotations[snapshotprotocol.CheckpointArtifactVersionAnnotation])
	if artifactVersion == "" {
		artifactVersion = snapshotprotocol.DefaultCheckpointArtifactVersion
	}

	info := &checkpoint.CheckpointInfo{
		Enabled:                 true,
		Exists:                  true,
		GPUMemoryService:        ckpt.Spec.GPUMemoryService,
		Hash:                    checkpointID,
		ArtifactVersion:         artifactVersion,
		CheckpointName:          ckpt.Name,
		Ready:                   true,
		StartupPolicy:           nvidiacomv1alpha1.CheckpointStartupPolicyImmediate,
		RestoreTargetContainers: targets,
	}
	if pod.Labels == nil {
		pod.Labels = map[string]string{}
	}
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}
	if err := checkpoint.ApplyRestorePodMetadataWithStorageConfig(pod.Labels, pod.Annotations, info, h.config.Checkpoint.Storage); err != nil {
		logger.Error(err, "checkpoint restore candidate not mutated because restore metadata could not be applied",
			"namespace", podNamespace, "pod", pod.Name, "checkpoint", checkpointName)
		return admission.Allowed("checkpoint restore metadata unavailable")
	}
	if err := checkpoint.InjectCheckpointIntoPodSpecWithStorageConfig(
		ctx,
		h.client,
		podNamespace,
		&pod.Spec,
		info,
		h.config.Checkpoint.Storage,
		h.config.Checkpoint.EffectiveSeccompProfile(),
	); err != nil {
		logger.Error(err, "checkpoint restore candidate not mutated because restore pod spec injection failed",
			"namespace", podNamespace, "pod", pod.Name, "checkpoint", checkpointName)
		return admission.Allowed("checkpoint restore injection unavailable")
	}

	mutated, err := json.Marshal(pod)
	if err != nil {
		logger.Error(err, "checkpoint restore candidate not mutated because mutated pod could not be marshaled",
			"namespace", podNamespace, "pod", pod.Name, "checkpoint", checkpointName)
		return admission.Allowed("checkpoint restore mutation unavailable")
	}
	return admission.PatchResponseFromRaw(original, mutated)
}
