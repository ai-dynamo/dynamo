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
	snapshotv1alpha1 "github.com/ai-dynamo/snapshot/api/v1alpha1"
	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook"
)

const (
	podCheckpointRestoreWebhookName = "pod-checkpoint-restore-mutating-webhook"
	podCheckpointRestoreWebhookPath = "/mutate-core-v1-pod-checkpoint-restore"
)

type PodCheckpointRestoreMutator struct {
	apiReader ctrlclient.Reader
	config    *configv1alpha1.OperatorConfiguration
	scheme    *runtime.Scheme
}

// NewPodCheckpointRestoreMutator creates a mutator with direct API-server
// reads for snapshot incarnation validation.
func NewPodCheckpointRestoreMutator(
	apiReader ctrlclient.Reader,
	config *configv1alpha1.OperatorConfiguration,
) *PodCheckpointRestoreMutator {
	return &PodCheckpointRestoreMutator{apiReader: apiReader, config: config}
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
	if !isCandidate {
		return admission.Allowed("pod is not a checkpoint restore candidate")
	}
	checkpointName := pod.Annotations[consts.CheckpointNameAnnotation]
	if checkpointName == "" {
		return admission.Denied("restore candidate has no source name")
	}
	if pod.Labels == nil ||
		pod.Labels[consts.KubeLabelDynamoComponent] == "" ||
		pod.Labels[consts.KubeLabelDynamoComponentType] == "" ||
		pod.Labels[consts.KubeLabelDynamoNamespace] == "" ||
		pod.Labels[consts.KubeLabelDynamoSelector] == "" {
		return admission.Denied("restore candidate is not operator-stamped")
	}

	// Candidates are rebuilt from the public Snapshot contract. An automatic
	// Immediate candidate remains a cold-start Pod until its job completes.
	shaped, restore, err := h.buildRestoreCandidatePod(ctx, pod, podNamespace)
	if err != nil {
		logger.Error(err, "restore candidate rejected",
			"namespace", podNamespace, "pod", pod.Name, "snapshot", checkpointName)
		return admission.Denied(err.Error())
	}
	if !restore {
		return admission.Allowed("automatic snapshot is not ready; Immediate policy permits cold start")
	}
	pod = shaped

	mutated, err := json.Marshal(pod)
	if err != nil {
		logger.Error(err, "checkpoint restore candidate not mutated because mutated pod could not be marshaled",
			"namespace", podNamespace, "pod", pod.Name, "checkpoint", checkpointName)
		return admission.Allowed("checkpoint restore mutation unavailable")
	}
	return admission.PatchResponseFromRaw(original, mutated)
}

func (h *PodCheckpointRestoreMutator) buildRestoreCandidatePod(
	ctx context.Context,
	pod *corev1.Pod,
	podNamespace string,
) (*corev1.Pod, bool, error) {
	switch pod.Annotations[consts.RestoreCandidateSourceKindAnnotation] {
	case consts.RestoreCandidateSourcePodSnapshot:
		shaped, err := h.buildPinnedPodSnapshotRestorePod(ctx, pod, podNamespace)
		return shaped, err == nil, err
	case consts.RestoreCandidateSourceSnapshotJob:
		return h.buildAutomaticSnapshotJobRestorePod(ctx, pod, podNamespace)
	default:
		return nil, false, fmt.Errorf("restore candidate has unsupported source kind %q", pod.Annotations[consts.RestoreCandidateSourceKindAnnotation])
	}
}

func (h *PodCheckpointRestoreMutator) buildPinnedPodSnapshotRestorePod(
	ctx context.Context,
	pod *corev1.Pod,
	podNamespace string,
) (*corev1.Pod, error) {
	snapshotName := pod.Annotations[consts.CheckpointNameAnnotation]
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
		expectedWorkerHashForPod(pod),
		checkpoint.ExplicitPodSnapshotUse(),
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
	return h.shapeNativeRestorePod(pod, snapshotName, info.NativeSnapshot)
}

func (h *PodCheckpointRestoreMutator) buildAutomaticSnapshotJobRestorePod(
	ctx context.Context,
	pod *corev1.Pod,
	podNamespace string,
) (*corev1.Pod, bool, error) {
	policy := nvidiacomv1alpha1.CheckpointStartupPolicy(pod.Annotations[consts.CheckpointStartupPolicyAnnotation])
	if policy == "" {
		policy = nvidiacomv1alpha1.CheckpointStartupPolicyImmediate
	}
	if policy != nvidiacomv1alpha1.CheckpointStartupPolicyImmediate &&
		policy != nvidiacomv1alpha1.CheckpointStartupPolicyWaitForCheckpoint {
		return nil, false, fmt.Errorf("restore candidate has unsupported checkpoint startup policy %q", policy)
	}

	jobName := pod.Annotations[consts.CheckpointNameAnnotation]
	expectedJobUID := types.UID(pod.Annotations[consts.SnapshotJobCandidateUIDAnnotation])
	if expectedJobUID == "" {
		return nil, false, fmt.Errorf("automatic restore candidate has no SnapshotJob UID")
	}
	job := &snapshotv1alpha1.SnapshotJob{}
	err := h.apiReader.Get(ctx, types.NamespacedName{Namespace: podNamespace, Name: jobName}, job)
	if apierrors.IsNotFound(err) {
		return automaticCandidateUnavailable(policy, "SnapshotJob no longer exists")
	}
	if err != nil {
		return nil, false, fmt.Errorf("get automatic SnapshotJob %s/%s: %w", podNamespace, jobName, err)
	}
	if job.UID != expectedJobUID {
		return automaticCandidateUnavailable(policy, "SnapshotJob UID changed after workload reconciliation")
	}
	// Only a controller-stamped automatic job may authorize the managed path.
	if job.Annotations[consts.CheckpointAutoAnnotation] != consts.KubeLabelValueTrue {
		return nil, false, fmt.Errorf("automatic SnapshotJob %s/%s is not marked as a Dynamo automatic capture", podNamespace, jobName)
	}
	ownerUID := types.UID(job.Annotations[consts.CheckpointOwnerUIDAnnotation])
	if ownerUID == "" {
		return nil, false, fmt.Errorf("automatic SnapshotJob %s/%s has no owning DGD UID", podNamespace, jobName)
	}
	if snapshotv1alpha1.IsSnapshotJobFailed(job) {
		return automaticCandidateUnavailable(policy, "SnapshotJob failed")
	}
	if !snapshotv1alpha1.IsSnapshotJobCompleted(job) {
		return automaticCandidateUnavailable(policy, "SnapshotJob has not completed")
	}
	if job.Status.PodSnapshotName == "" || job.Status.PodSnapshotUID == "" {
		return nil, false, fmt.Errorf("completed SnapshotJob %s/%s has no PodSnapshot identity", podNamespace, jobName)
	}

	snapshotName := job.Status.PodSnapshotName
	config := &nvidiacomv1alpha1.ServiceCheckpointConfig{
		Enabled:       true,
		CheckpointRef: &snapshotName,
	}
	info, err := checkpoint.ResolvePodSnapshotForService(
		ctx,
		h.apiReader,
		podNamespace,
		config,
		expectedWorkerHashForPod(pod),
		checkpoint.ManagedPodSnapshotUse(ownerUID),
	)
	if apierrors.IsNotFound(err) {
		return automaticCandidateUnavailable(policy, "SnapshotJob PodSnapshot no longer exists")
	}
	if err != nil {
		return nil, false, err
	}
	if info.NativeSnapshot.UID != job.Status.PodSnapshotUID {
		return automaticCandidateUnavailable(policy, "SnapshotJob PodSnapshot UID changed")
	}
	if !info.Ready {
		return automaticCandidateUnavailable(policy, "SnapshotJob PodSnapshot is not Ready")
	}

	shaped, err := h.shapeNativeRestorePod(pod, snapshotName, info.NativeSnapshot)
	return shaped, err == nil, err
}

func automaticCandidateUnavailable(
	policy nvidiacomv1alpha1.CheckpointStartupPolicy,
	reason string,
) (*corev1.Pod, bool, error) {
	if policy == nvidiacomv1alpha1.CheckpointStartupPolicyImmediate {
		return nil, false, nil
	}
	return nil, false, fmt.Errorf("automatic restore candidate unavailable: %s", reason)
}

func expectedWorkerHashForPod(pod *corev1.Pod) *string {
	if !dynamo.IsWorkerComponent(pod.Labels[consts.KubeLabelDynamoComponentType]) {
		return nil
	}
	workerHash := pod.Labels[consts.KubeLabelDynamoWorkerHash]
	return &workerHash
}

func (h *PodCheckpointRestoreMutator) shapeNativeRestorePod(
	pod *corev1.Pod,
	snapshotName string,
	resolved *checkpoint.ResolvedPodSnapshot,
) (*corev1.Pod, error) {

	// Dynamo chooses restore destinations from its rendered topology while the
	// immutable PodSnapshot spec remains authoritative for the captured source.
	targets, err := checkpoint.RestoreCandidateTargetContainers(pod.Annotations)
	if err != nil {
		return nil, fmt.Errorf("resolve native restore destinations: %w", err)
	}
	mappings := make([]podcontract.ContainerMapping, 0, len(targets))
	for _, target := range targets {
		mappings = append(mappings, podcontract.ContainerMapping{
			Source:      resolved.SourceContainer,
			Destination: target,
		})
	}

	request := podcontract.Request{
		SnapshotName:    snapshotName,
		SourceContainer: resolved.SourceContainer,
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
		checkpoint.EnsureRestoreStartupProbe(container)
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
	delete(annotations, consts.RestoreCandidateSourceKindAnnotation)
	delete(annotations, consts.SnapshotJobCandidateUIDAnnotation)
	delete(annotations, consts.CheckpointStartupPolicyAnnotation)
	delete(annotations, consts.SnapshotCandidateUIDAnnotation)
	delete(annotations, consts.SnapshotCandidateContentAnnotation)
	delete(annotations, consts.SnapshotCandidateGMSModeAnnotation)
	delete(annotations, consts.SnapshotCandidateVersionAnnotation)
	delete(annotations, consts.RestoreCandidateTargetContainersAnnotation)
}
