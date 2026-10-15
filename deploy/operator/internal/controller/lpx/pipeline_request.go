// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"hash"
	"slices"
	"strconv"
	"strings"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

const (
	pipelineRequestModelAnnotation = "scheduling.lpu.nvidia.com/dynamo-model"
)

// getPipelineRequests returns only the non-nil PCS's requests, indexed by name.
// The UID index establishes ownership; callers do not recheck it. The supported
// foreground replacement path needs no discovery of LPRs after their PCS is gone.
func (r *graphReconciler) getPipelineRequests(ctx context.Context, pcs *grovev1alpha1.PodCliqueSet) (map[string]*lpxv1alpha1.LPUPipelineRequest, error) {
	requestList := &lpxv1alpha1.LPUPipelineRequestList{}
	if err := r.List(ctx, requestList, client.InNamespace(pcs.Namespace), client.MatchingFields{pipelineRequestPCSOwnerUIDIndex: string(pcs.UID)}); err != nil {
		return nil, err
	}
	requests := make(map[string]*lpxv1alpha1.LPUPipelineRequest)
	for i := range requestList.Items {
		request := &requestList.Items[i]
		requests[request.Name] = request
	}
	return requests, nil
}

// resolveNamedPipelineRequests collects missing or changed intent in replica/model order.
// Pointer inputs are non-nil; deployment has a validated DGD controller owner.
// currentRequests contains owned observations indexed by name and may be nil.
// Inputs are not mutated; owner references are added only on publication.
func resolveNamedPipelineRequests(
	deployment *v1alpha1.LPXGraphDeployment,
	currentRequests map[string]*lpxv1alpha1.LPUPipelineRequest,
	workload *lpx.Workload,
	replicas int32,
) (map[string]*lpxv1alpha1.LPUPipelineRequest, []*lpxv1alpha1.LPUPipelineRequest) {
	// The reconcile boundary already validated the DGD owner; rendering uses that identity.
	dgdOwner := metav1.GetControllerOf(deployment)

	// Model projections are already ordered (default, or draft0..draft7 then target).
	projections := workload.Models()
	requests := make(map[string]*lpxv1alpha1.LPUPipelineRequest, len(projections)*int(replicas))
	var missing []*lpxv1alpha1.LPUPipelineRequest

	for replica := range replicas {
		for index, projection := range projections {
			if projection.AgentReplicas() == 0 {
				continue
			}
			digest := pipelineRequestIdentityDigest(deployment.Namespace, deployment.Name, deployment.UID, workload.GroupName(), projection.Name(), replica)

			request := &lpxv1alpha1.LPUPipelineRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name: pipelineRequestName(deployment.Name, digest), Namespace: deployment.Namespace,
					Annotations: map[string]string{
						lpx.DeploymentNameAnnotation:                 deployment.Name,
						pipelineRequestModelAnnotation:               projection.Name(),
						lpx.WorkloadDigestAnnotation:                 projection.Digest().String(),
						lpxv1alpha1.CompilerSnapshotDigestAnnotation: projection.CompilerSnapshotDigest(),
					},
				},
				Spec: workload.RequestSpec(index, replica),
			}

			if currentRequest := currentRequests[request.Name]; currentRequest != nil && pipelineRequestMatches(currentRequest, request) {
				requests[request.Name] = currentRequest
				if !currentRequest.DeletionTimestamp.IsZero() {
					missing = append(missing, request)
				}
			} else {
				request.Labels = map[string]string{
					consts.KubeLabelDynamoGraphDeploymentName: dgdOwner.Name,
					deploymentUIDLabel:                        string(deployment.UID),
				}
				requests[request.Name] = request
				missing = append(missing, request)
			}
		}
	}

	return requests, missing
}

// pipelineRequestIdentityDigest identifies a model and replica within an optional group.
// An empty group preserves the established single-workload identity.
// InputRevision and generation are deliberately absent: there is no publication batch.
func pipelineRequestIdentityDigest(
	namespace string,
	name string,
	uid types.UID,
	group string,
	model string,
	replicaIndex int32,
) string {
	h := sha256.New()
	writeIdentityHashField(h, "namespace", namespace)
	writeIdentityHashField(h, "name", name)
	writeIdentityHashField(h, "uid", string(uid))
	if group != "" {
		writeIdentityHashField(h, "group", group)
	}
	writeIdentityHashField(h, "model", model)
	if replicaIndex > 0 {
		writeIdentityHashField(h, "group-replica", strconv.FormatInt(int64(replicaIndex), 10))
	}
	return fmt.Sprintf("sha256:%x", h.Sum(nil))
}

func writeIdentityHashField(h hash.Hash, tag, value string) {
	var size [8]byte
	for _, field := range []string{tag, value} {
		binary.BigEndian.PutUint64(size[:], uint64(len(field)))
		_, _ = h.Write(size[:])
		_, _ = h.Write([]byte(field))
	}
}

func pipelineRequestName(deploymentName, digest string) string {
	prefix := strings.ReplaceAll(deploymentName, ".", "-")
	if len(prefix) > 23 {
		prefix = strings.TrimRight(prefix[:23], "-")
	}
	hexDigest := strings.TrimPrefix(digest, "sha256:")
	return fmt.Sprintf("lpx-%s-%s", prefix, hexDigest[:32])
}

// pipelineRequestMatches compares immutable intent after ownership was established
// by the LPR index. Both requests are non-nil.
func pipelineRequestMatches(observed, desired *lpxv1alpha1.LPUPipelineRequest) bool {
	for key, value := range desired.Annotations {
		if observed.Annotations[key] != value {
			return false
		}
	}
	return apiequality.Semantic.DeepEqual(observed.Spec, desired.Spec)
}

// reconcilePipelineRequests hands each complete Grove engine to LPX after its old requests finalize.
func (r *graphReconciler) reconcilePipelineRequests(
	ctx context.Context,
	deployment *v1alpha1.LPXGraphDeployment,
	pcs *grovev1alpha1.PodCliqueSet,
	pcsgs map[string]*grovev1alpha1.PodCliqueScalingGroup,
	pclqs map[string]*grovev1alpha1.PodClique,
	observed map[string]*lpxv1alpha1.LPUPipelineRequest,
	pending []*lpxv1alpha1.LPUPipelineRequest,
) error {
	for len(pending) > 0 {
		target := *pending[0].Spec.MaterializationTarget.PodCliqueScalingGroupRef
		end := 1
		for end < len(pending) && *pending[end].Spec.MaterializationTarget.PodCliqueScalingGroupRef == target {
			end++
		}
		engine := pending[:end]
		pending = pending[end:]
		if !replicaCliquesMatch(pcs, pcsgs[target.Name], pclqs, target.ReplicaIndex) {
			continue
		}
		var retiring []*lpxv1alpha1.LPUPipelineRequest
		for _, desired := range engine {
			if request := observed[desired.Name]; request != nil {
				retiring = append(retiring, request)
			}
		}
		if slices.ContainsFunc(retiring, pipelineRequestCommitted) {
			continue
		}
		if len(retiring) > 0 {
			if err := r.deletePipelineRequests(ctx, retiring); err != nil {
				return err
			}
			continue
		}
		for _, request := range engine {
			if err := controllerutil.SetControllerReference(pcs, request, r.Scheme()); err != nil {
				return fmt.Errorf("create LPX request %q: %w", request.Name, err)
			}
			// AlreadyExists is an observation delay, not permission to adopt an unseen request.
			if err := r.Create(ctx, request); err != nil && !apierrors.IsAlreadyExists(err) {
				return fmt.Errorf("create LPX request %q: %w", request.Name, err)
			}
		}
	}
	setReadyCondition(deployment, v1beta1.DGDStatePending, "Waiting for Grove replicas and their LPX requests to reach the desired build")
	return nil
}

// pipelineRequestsPendingDeletion selects removed requests and current terminating
// names that must disappear before publication. All requests belong to the observed PCS.
func pipelineRequestsPendingDeletion(
	requests map[string]*lpxv1alpha1.LPUPipelineRequest,
	desiredRequests map[string]*lpxv1alpha1.LPUPipelineRequest,
) []*lpxv1alpha1.LPUPipelineRequest {
	var pending []*lpxv1alpha1.LPUPipelineRequest
	for _, request := range requests {
		if _, desired := desiredRequests[request.Name]; !desired || !request.DeletionTimestamp.IsZero() {
			pending = append(pending, request)
		}
	}
	return pending
}

// pipelineRequestCommitted blocks retirement until LPX releases the plan;
// delete preconditions reject a newer commitment missing from the cache.
func pipelineRequestCommitted(request *lpxv1alpha1.LPUPipelineRequest) bool {
	return request.Status != nil && request.Status.Committed != nil
}

// removeReleasedAgentPods removes journaled Agent Pods for matching, still-desired
// requests. The Dynamo operator's LPX controller owns this cleanup; Grove recreates
// the Pods, and the LPX scheduler completes release and replanning.
// pcsgs contains owned observations. Requests are owned observations, or unpublished
// desired objects with nil status.
func (r *graphReconciler) removeReleasedAgentPods(ctx context.Context, pcsgs map[string]*grovev1alpha1.PodCliqueScalingGroup, requests map[string]*lpxv1alpha1.LPUPipelineRequest) error {
	var cleanupErr error
	for _, request := range requests {
		status := request.Status
		if !request.DeletionTimestamp.IsZero() || status == nil || status.Phase != lpxv1alpha1.RequestPhaseReleasing || status.Committed == nil {
			continue
		}
		execution := &status.Committed.Execution
		if execution.NodeLocal == nil || execution.Release == nil {
			continue
		}
		switch execution.Release.Reason {
		case lpxv1alpha1.ReleaseReasonRequestChanged, lpxv1alpha1.ReleaseReasonDependencyChanged, lpxv1alpha1.ReleaseReasonBindingFailed:
		default:
			continue
		}

		// Include unbound targets and predecessors retained by plan replacement.
		var pods []lpxv1alpha1.ObjectReference
		for _, partition := range execution.NodeLocal.PartitionSelections {
			for _, row := range partition.SelectedRows {
				if row.Current != nil {
					pods = append(pods, row.Current.PodRef)
				}
				if row.Target != nil {
					pods = append(pods, row.Target.PodRef)
				}
			}
		}
		if replacement := execution.NodeLocal.PlanReplacement; replacement != nil {
			for _, member := range replacement.RealizeMembers {
				if old := member.Predecessor.OldOnly; old != nil {
					pods = append(pods, old.Current.PodRef)
				}
			}
		}

		// Never turn incomplete journal evidence into an unfenced or cross-namespace delete.
		for _, ref := range pods {
			if ref.Namespace != request.Namespace || ref.Name == "" || ref.UID == "" {
				return fmt.Errorf("LPUPipelineRequest %q has an invalid released Agent Pod reference", request.Name)
			}
		}
		for _, ref := range pods {
			// A stale entry must not prevent cleanup of the remaining journaled Pods.
			if err := r.removeReleasedAgentPod(ctx, pcsgs, ref); err != nil {
				cleanupErr = errors.Join(cleanupErr, err)
			}
		}
	}
	return cleanupErr
}

// removeReleasedAgentPod requires a valid journal reference and owned scaling groups.
func (r *graphReconciler) removeReleasedAgentPod(ctx context.Context, pcsgs map[string]*grovev1alpha1.PodCliqueScalingGroup, ref lpxv1alpha1.ObjectReference) error {
	// Cache misses must retry: this controller does not watch Pod cache arrivals.
	pod := &corev1.Pod{}
	if err := r.Get(ctx, client.ObjectKey{Namespace: ref.Namespace, Name: ref.Name}, pod); err != nil {
		return fmt.Errorf("observe released Agent Pod %q: %w", ref.Name, err)
	}
	if pod.UID != types.UID(ref.UID) {
		return nil
	}

	cliqueOwner := metav1.GetControllerOf(pod)
	if cliqueOwner == nil || cliqueOwner.APIVersion != grovev1alpha1.SchemeGroupVersion.String() || cliqueOwner.Kind != "PodClique" {
		return fmt.Errorf("released Agent Pod %q has no PodClique controller", pod.Name)
	}
	clique := &grovev1alpha1.PodClique{}
	if err := r.Get(ctx, client.ObjectKey{Namespace: pod.Namespace, Name: cliqueOwner.Name}, clique); err != nil {
		return fmt.Errorf("observe released Agent Pod %q's PodClique: %w", pod.Name, err)
	}
	if clique.UID != cliqueOwner.UID {
		return fmt.Errorf("released Agent Pod %q's PodClique has a different UID", pod.Name)
	}

	groupOwner := metav1.GetControllerOf(clique)
	if groupOwner == nil || groupOwner.APIVersion != grovev1alpha1.SchemeGroupVersion.String() || groupOwner.Kind != "PodCliqueScalingGroup" {
		return fmt.Errorf("released Agent Pod %q's PodClique has no PodCliqueScalingGroup controller", pod.Name)
	}
	if group := pcsgs[groupOwner.Name]; group == nil || group.UID != groupOwner.UID {
		return fmt.Errorf("released Agent Pod %q is not controlled by an observed LPX PodCliqueScalingGroup", pod.Name)
	}

	// Fence both the recorded identity and the ownership observation.
	if err := client.IgnoreNotFound(r.Delete(ctx, pod, client.Preconditions{UID: &pod.UID, ResourceVersion: &pod.ResourceVersion})); err != nil {
		return fmt.Errorf("delete released Agent Pod %q: %w", pod.Name, err)
	}
	return nil
}

// deletePipelineRequests accepts already-validated LPRs. Terminating requests
// remain pending; Grove and garbage collection own Pod cleanup.
func (r *graphReconciler) deletePipelineRequests(ctx context.Context, requests []*lpxv1alpha1.LPUPipelineRequest) error {
	for _, request := range requests {
		if !request.DeletionTimestamp.IsZero() {
			continue
		}

		if err := client.IgnoreNotFound(r.Delete(ctx, request, client.Preconditions{UID: &request.UID, ResourceVersion: &request.ResourceVersion})); err != nil {
			return fmt.Errorf("delete LPUPipelineRequest %q: %w", request.Name, err)
		}
	}

	return nil
}

// replicaCliquesMatch requires all roles at the desired revision, before Pods can schedule.
func replicaCliquesMatch(pcs *grovev1alpha1.PodCliqueSet, pcsg *grovev1alpha1.PodCliqueScalingGroup, pclqs map[string]*grovev1alpha1.PodClique, replica int64) bool {
	if replica >= int64(pcsg.Spec.Replicas) {
		return false
	}
	for _, templateName := range pcsg.Spec.CliqueNames {
		index := slices.IndexFunc(pcs.Spec.Template.Cliques, func(clique *grovev1alpha1.PodCliqueTemplateSpec) bool { return clique.Name == templateName })
		template := pcs.Spec.Template.Cliques[index]
		name := grovecommon.GeneratePodCliqueName(grovecommon.ResourceNameReplica{Name: pcsg.Name, Replica: int(replica)}, templateName)
		clique := pclqs[name]
		if clique == nil || clique.Labels[consts.KubeLabelDynamoWorkerHash] != template.Labels[consts.KubeLabelDynamoWorkerHash] ||
			clique.Annotations[lpx.WorkloadDigestAnnotation] != template.Annotations[lpx.WorkloadDigestAnnotation] {
			return false
		}
	}
	return true
}
