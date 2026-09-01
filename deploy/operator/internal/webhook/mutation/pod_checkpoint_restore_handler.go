/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package mutation

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"path"
	"strings"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	podcontract "github.com/ai-dynamo/snapshot/api/podcontract"
	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpointjob"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook"
)

const (
	podCheckpointRestoreWebhookName = "pod-checkpoint-restore-mutating-webhook"
	podCheckpointRestoreWebhookPath = "/mutate-core-v1-pod-checkpoint-restore"
)

type PodCheckpointRestoreMutator struct {
	client    ctrlclient.Client
	apiReader ctrlclient.Reader
	config    *configv1alpha1.OperatorConfiguration
	scheme    *runtime.Scheme
}

// NewPodCheckpointRestoreMutator creates a mutator with cached legacy access
// and direct API-server reads for native snapshot incarnation validation.
func NewPodCheckpointRestoreMutator(
	client ctrlclient.Client,
	apiReader ctrlclient.Reader,
	config *configv1alpha1.OperatorConfiguration,
) *PodCheckpointRestoreMutator {
	return &PodCheckpointRestoreMutator{client: client, apiReader: apiReader, config: config}
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
	if !features.MustGateFrom(ctx).Enabled(features.Checkpoint) {
		return admission.Allowed("checkpoint disabled")
	}
	if excluded := internalwebhook.GetExcludedNamespaces(); excluded != nil && excluded.Contains(req.Namespace) {
		return admission.Allowed("namespace excluded")
	}
	if h.client == nil {
		return admission.Errored(http.StatusInternalServerError, fmt.Errorf("checkpoint restore client is unavailable"))
	}
	if h.apiReader == nil {
		return admission.Errored(http.StatusInternalServerError, fmt.Errorf("checkpoint restore API reader is unavailable"))
	}
	if h.scheme == nil {
		return admission.Errored(http.StatusInternalServerError, fmt.Errorf("checkpoint restore scheme is unavailable"))
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

	isCandidate := pod.Annotations != nil &&
		pod.Annotations[consts.CheckpointRestoreCandidateAnnotation] == consts.KubeLabelValueTrue
	sourceKind := pod.Annotations[consts.CheckpointSourceKindAnnotation]
	isLegacyShaped := pod.Labels != nil &&
		(pod.Labels[snapshotprotocol.CheckpointIDLabel] != "" ||
			pod.Labels[snapshotprotocol.CheckpointSourceLabel] != "")
	if isLegacyShaped {
		if isCandidate && sourceKind == consts.CheckpointSourceKindSnapshot {
			return admission.Denied("native restore candidate conflicts with legacy checkpoint metadata")
		}
		return admission.Allowed("pod is already checkpoint-shaped")
	}
	if !isCandidate {
		return admission.Allowed("pod is not a checkpoint restore candidate")
	}
	checkpointName := pod.Annotations[consts.CheckpointNameAnnotation]
	if checkpointName == "" {
		if sourceKind == consts.CheckpointSourceKindSnapshot {
			return admission.Denied("native restore candidate has no PodSnapshot name")
		}
		return admission.Allowed("restore candidate has no checkpoint name")
	}
	if pod.Labels == nil ||
		pod.Labels[consts.KubeLabelDynamoComponent] == "" ||
		pod.Labels[consts.KubeLabelDynamoNamespace] == "" ||
		pod.Labels[consts.KubeLabelDynamoSelector] == "" {
		if sourceKind == consts.CheckpointSourceKindSnapshot {
			return admission.Denied("native restore candidate is not operator-stamped")
		}
		return admission.Allowed("restore candidate is not operator-stamped")
	}

	// Native candidates are rebuilt from the public Snapshot contract and fail
	// closed. Unmarked candidates remain legacy for in-flight automatic capture.
	if sourceKind == consts.CheckpointSourceKindSnapshot {
		shaped, err := h.buildNativeRestorePod(ctx, pod, podNamespace)
		if err != nil {
			logger.Error(err, "native restore candidate rejected",
				"namespace", podNamespace, "pod", pod.Name, "snapshot", checkpointName)
			return admission.Denied(err.Error())
		}
		pod = shaped
	} else if sourceKind != "" && sourceKind != consts.CheckpointSourceKindLegacy {
		return admission.Denied(fmt.Sprintf("unsupported checkpoint source kind %q", sourceKind))
	} else {
		response, stop := h.mutateLegacyRestoreCandidate(ctx, pod, podNamespace, checkpointName)
		if stop {
			return response
		}
	}

	mutated, err := json.Marshal(pod)
	if err != nil {
		logger.Error(err, "checkpoint restore candidate not mutated because mutated pod could not be marshaled",
			"namespace", podNamespace, "pod", pod.Name, "checkpoint", checkpointName)
		return admission.Allowed("checkpoint restore mutation unavailable")
	}
	return admission.PatchResponseFromRaw(original, mutated)
}

func (h *PodCheckpointRestoreMutator) mutateLegacyRestoreCandidate(
	ctx context.Context,
	pod *corev1.Pod,
	podNamespace string,
	checkpointName string,
) (admission.Response, bool) {
	logger := log.FromContext(ctx).WithName(podCheckpointRestoreWebhookName)

	ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{}
	if err := h.client.Get(ctx, types.NamespacedName{Namespace: podNamespace, Name: checkpointName}, ckpt); err != nil {
		logger.V(1).Info("checkpoint restore candidate not mutated because checkpoint could not be read",
			"namespace", podNamespace, "checkpoint", checkpointName, "error", err.Error())
		return admission.Allowed("checkpoint not available"), true
	}
	if ckpt.Status.Phase != nvidiacomv1alpha1.DynamoCheckpointPhaseReady {
		return admission.Allowed("checkpoint not ready"), true
	}

	checkpointID, err := checkpoint.CheckpointID(ckpt)
	if err != nil {
		logger.Error(err, "checkpoint restore candidate not mutated because checkpoint ID could not be resolved",
			"namespace", podNamespace, "checkpoint", checkpointName)
		return admission.Allowed("checkpoint ID unavailable"), true
	}
	targets, err := snapshotprotocol.TargetContainersFromAnnotations(pod.Annotations, 1, 0)
	if err != nil {
		logger.Error(err, "checkpoint restore candidate not mutated because target containers annotation is invalid",
			"namespace", podNamespace, "pod", pod.Name, "checkpoint", checkpointName)
		return admission.Allowed("checkpoint target containers invalid"), true
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
		return admission.Allowed("checkpoint restore metadata unavailable"), true
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
		return admission.Allowed("checkpoint restore injection unavailable"), true
	}

	return admission.Response{}, false
}

func (h *PodCheckpointRestoreMutator) buildNativeRestorePod(
	ctx context.Context,
	pod *corev1.Pod,
	podNamespace string,
) (*corev1.Pod, error) {
	snapshotName := pod.Annotations[consts.CheckpointNameAnnotation]
	expectedWorkerHash := pod.Labels[consts.KubeLabelDynamoWorkerHash]
	config := &nvidiacomv1alpha1.ServiceCheckpointConfig{
		Enabled:       true,
		CheckpointRef: &snapshotName,
	}

	// Admission bypasses the informer cache and repeats compatibility validation
	// so a deleted and recreated PodSnapshot cannot satisfy old incarnation pins.
	info, err := checkpoint.ResolvePodSnapshotForService(
		ctx,
		h.apiReader,
		podNamespace,
		config,
		expectedWorkerHash,
	)
	if err != nil {
		return nil, err
	}
	if !info.Ready {
		return nil, fmt.Errorf("referenced PodSnapshot %s/%s is not Ready", podNamespace, snapshotName)
	}
	if err := validateNativeSnapshotCandidate(pod.Annotations, info.NativeSnapshot); err != nil {
		return nil, err
	}

	// Dynamo chooses restore destinations from its rendered topology while the
	// immutable PodSnapshot spec remains authoritative for the captured source.
	targets, err := checkpoint.RestoreCandidateTargetContainers(pod.Annotations)
	if err != nil {
		return nil, fmt.Errorf("resolve native restore destinations: %w", err)
	}
	mappings := make([]podcontract.ContainerMapping, 0, len(targets))
	for _, target := range targets {
		mappings = append(mappings, podcontract.ContainerMapping{
			Source:      info.NativeSnapshot.SourceContainer,
			Destination: target,
		})
	}

	request := podcontract.Request{
		SnapshotName:    snapshotName,
		SourceContainer: info.NativeSnapshot.SourceContainer,
		Mappings:        mappings,
	}
	options := podcontract.Options{
		SeccompProfile: h.config.Checkpoint.EffectiveSeccompProfile(),
	}
	shaped, err := podcontract.Build(pod, request, options)
	if err != nil {
		return nil, fmt.Errorf("shape native restore Pod: %w", err)
	}
	if err := applyDynamoRestorePolicy(shaped, mappings); err != nil {
		return nil, err
	}

	// Candidate-only annotations must not survive onto the restore target. The
	// standalone Snapshot annotations emitted by the builder are the wire API.
	removeRestoreCandidateAnnotations(shaped.Annotations)
	if err := podcontract.Validate(shaped, request); err != nil {
		return nil, fmt.Errorf("validate native restore Pod: %w", err)
	}
	return shaped, nil
}

func validateNativeSnapshotCandidate(annotations map[string]string, resolved *checkpoint.ResolvedPodSnapshot) error {
	if resolved == nil {
		return fmt.Errorf("resolved PodSnapshot metadata is required")
	}
	if annotations[consts.SnapshotCandidateUIDAnnotation] != string(resolved.UID) {
		return fmt.Errorf("PodSnapshot UID changed after workload reconciliation")
	}
	if annotations[consts.SnapshotCandidateContentAnnotation] != resolved.BoundContentName {
		return fmt.Errorf("PodSnapshot content binding changed after workload reconciliation")
	}
	if annotations[consts.SnapshotCandidateVersionAnnotation] != resolved.CompatibilityVersion {
		return fmt.Errorf("PodSnapshot compatibility version changed after workload reconciliation")
	}
	if annotations[consts.SnapshotCandidateGMSModeAnnotation] != resolved.GMSMode {
		return fmt.Errorf("PodSnapshot GMS mode changed after workload reconciliation")
	}
	return nil
}

func applyDynamoRestorePolicy(pod *corev1.Pod, mappings []podcontract.ContainerMapping) error {
	containers := make(map[string]*corev1.Container, len(pod.Spec.Containers))
	for i := range pod.Spec.Containers {
		container := &pod.Spec.Containers[i]
		containers[container.Name] = container
	}

	// Validate every destination first so unsupported or conflicting workload
	// entrypoints cannot leave a partially modified Pod even in caller tests.
	for _, mapping := range mappings {
		container := containers[mapping.Destination]
		if container == nil {
			return fmt.Errorf("restore destination container %q not found", mapping.Destination)
		}
		if !usesSupportedDynamoRestoreEntrypoint(container) {
			return fmt.Errorf(
				"restore destination container %q must directly invoke python -m dynamo.vllm, python -m dynamo.sglang, or python -m dynamo.trtllm; command=%q args=%q",
				mapping.Destination,
				container.Command,
				container.Args,
			)
		}
		for _, env := range container.Env {
			if env.Name == podcontract.RestoreStandbyModeEnv && (env.Value != "1" || env.ValueFrom != nil) {
				return fmt.Errorf("restore destination container %q has conflicting %s", mapping.Destination, podcontract.RestoreStandbyModeEnv)
			}
		}
	}

	// Apply Dynamo's standby and startup policy only after the complete
	// destination set has passed validation, preserving all-or-nothing mutation.
	for _, mapping := range mappings {
		container := containers[mapping.Destination]
		found := false
		for _, env := range container.Env {
			if env.Name == podcontract.RestoreStandbyModeEnv {
				found = true
				break
			}
		}
		if !found {
			container.Env = append(container.Env, corev1.EnvVar{
				Name:  podcontract.RestoreStandbyModeEnv,
				Value: "1",
			})
		}
		snapshotprotocol.EnsureRestoreStartupProbe(container)
	}
	return nil
}

// usesSupportedDynamoRestoreEntrypoint recognizes only direct Python module
// invocations that are known to consume SNAPSHOT_RESTORE_STANDBY. Shell and
// custom wrappers are rejected because admission cannot prove they honor it.
func usesSupportedDynamoRestoreEntrypoint(container *corev1.Container) bool {
	if len(container.Command) == 0 {
		return false
	}
	python := path.Base(container.Command[0])
	if python != "python" && python != "python3" && !strings.HasPrefix(python, "python3.") {
		return false
	}

	arguments := make([]string, 0, len(container.Command)+len(container.Args))
	arguments = append(arguments, container.Command...)
	arguments = append(arguments, container.Args...)

	// Skip only operand-free interpreter options so -m remains unambiguous.
	moduleFlagIndex := 1
	for moduleFlagIndex < len(arguments) && isOperandFreePythonInterpreterFlag(arguments[moduleFlagIndex]) {
		moduleFlagIndex++
	}
	if moduleFlagIndex+1 >= len(arguments) || arguments[moduleFlagIndex] != "-m" {
		return false
	}

	switch arguments[moduleFlagIndex+1] {
	case "dynamo.vllm", "dynamo.sglang", "dynamo.trtllm":
		return true
	default:
		return false
	}
}

// isOperandFreePythonInterpreterFlag recognizes options that cannot consume
// the following -m argument. Operand-taking and execution-selector options
// remain fail-closed.
func isOperandFreePythonInterpreterFlag(argument string) bool {
	switch argument {
	case "-b", "-bb", "-B", "-d", "-E", "-i", "-I", "-O", "-OO", "-P", "-q", "-s", "-S", "-u", "-v", "-vv", "-x":
		return true
	default:
		return false
	}
}

func removeRestoreCandidateAnnotations(annotations map[string]string) {
	delete(annotations, consts.CheckpointRestoreCandidateAnnotation)
	delete(annotations, consts.CheckpointNameAnnotation)
	delete(annotations, consts.CheckpointStartupPolicyAnnotation)
	delete(annotations, consts.CheckpointSourceKindAnnotation)
	delete(annotations, consts.SnapshotCandidateUIDAnnotation)
	delete(annotations, consts.SnapshotCandidateContentAnnotation)
	delete(annotations, consts.SnapshotCandidateGMSModeAnnotation)
	delete(annotations, consts.SnapshotCandidateVersionAnnotation)
	delete(annotations, consts.RestoreCandidateTargetContainersAnnotation)
	delete(annotations, snapshotprotocol.TargetContainersAnnotation)
}
