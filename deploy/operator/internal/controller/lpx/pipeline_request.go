// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"hash"
	"strconv"
	"strings"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
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
	for _, request := range requestList.Items {
		requests[request.Name] = &request
	}
	return requests, nil
}

// resolvePipelineRequests combines desired intent with matching observed LPRs and
// collects missing requests in replica/model order in the same pass. An immutable
// mismatch returns nil collections and true: the caller must replace the PCS.
// Pointer inputs are non-nil; deployment has a validated DGD controller owner.
// currentRequests contains owned observations indexed by name and may be nil.
// Inputs are not mutated; owner references are added only on publication.
func resolvePipelineRequests(
	deployment *v1alpha1.LPXGraphDeployment,
	currentRequests map[string]*lpxv1alpha1.LPUPipelineRequest,
	workload *lpx.Workload,
	plan *lpx.MaterializationPlan,
) (map[string]*lpxv1alpha1.LPUPipelineRequest, []*lpxv1alpha1.LPUPipelineRequest, bool) {
	// The reconcile boundary already validated the DGD owner; rendering uses that identity.
	dgdOwner := metav1.GetControllerOf(deployment)

	// Only multiple workloads need a group identity; sole workloads retain existing LPR names.
	groupName := ""
	if plan.ResourcePrefix != plan.PodCliqueSetName {
		groupName = workload.ServingComponentName()
	}

	// Model projections are already ordered (default, or draft0..draft7 then target).
	projections := workload.ModelProjections()
	requests := make(map[string]*lpxv1alpha1.LPUPipelineRequest, len(projections)*int(plan.Replicas))
	var missing []*lpxv1alpha1.LPUPipelineRequest

	for replica := range plan.Replicas {
		replicaPlan := plan.ForReplica(replica)

		for index, projection := range projections {
			digest := pipelineRequestIdentityDigest(deployment.Namespace, deployment.Name, deployment.UID, groupName, projection.Model(), replica)

			request := &lpxv1alpha1.LPUPipelineRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name: pipelineRequestName(deployment.Name, digest), Namespace: deployment.Namespace,
					Labels: map[string]string{
						consts.KubeLabelDynamoGraphDeploymentName: dgdOwner.Name,
						deploymentUIDLabel:                        string(deployment.UID),
					},
					Annotations: map[string]string{
						lpx.DeploymentNameAnnotation:                 deployment.Name,
						pipelineRequestModelAnnotation:               projection.Model(),
						lpx.WorkloadDigestAnnotation:                 projection.Digest().String(),
						lpxv1alpha1.CompilerSnapshotDigestAnnotation: projection.CompilerSnapshotDigest(),
					},
				},
				Spec: projection.RequestSpec(replicaPlan, replicaPlan.Agents[index].CliqueName),
			}

			if currentRequest, exists := currentRequests[request.Name]; exists {
				if !pipelineRequestMatches(currentRequest, request) {
					return nil, nil, true
				}

				requests[request.Name] = currentRequest
			} else {
				requests[request.Name] = request
				missing = append(missing, request)
			}
		}
	}

	return requests, missing, false
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

// reconcilePipelineRequests publishes the non-empty list of missing requests and
// waits for their watch events. deployment and pcs are non-nil; requests are in
// ordinal/model order. Each Create is independent: partial publication is safe to retry.
func (r *graphReconciler) reconcilePipelineRequests(
	ctx context.Context,
	deployment *v1alpha1.LPXGraphDeployment,
	pcs *grovev1alpha1.PodCliqueSet,
	requests []*lpxv1alpha1.LPUPipelineRequest,
) error {
	// Publish independently without adopting requests absent from the cache observation.
	for _, request := range requests {
		if err := controllerutil.SetControllerReference(pcs, request, r.Scheme()); err != nil {
			return fmt.Errorf("create LPX request %q: %w", request.Name, err)
		}

		// AlreadyExists is an observation delay, not permission to adopt an unseen request.
		if err := r.Create(ctx, request); err != nil && !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("create LPX request %q: %w", request.Name, err)
		}
	}

	setReadyCondition(deployment, v1beta1.DGDStatePending, "Waiting for published LPX requests to appear in the cache")
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
