/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lws

import (
	"context"
	"strconv"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
)

// resolvedPod is a cache-validated LWS Pod identity for current-evidence classification.
type resolvedPod struct {
	groupIndex int
	revision   string
	leaderUID  types.UID
}

func resolvePod(
	ctx context.Context,
	reader client.Reader,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	lws *leaderworkersetv1.LeaderWorkerSet,
	pod *corev1.Pod,
) (resolvedPod, string, error) {
	if reason := validateResolveRoots(dcd, lws, pod); reason != "" {
		return resolvedPod{}, reason, nil
	}

	groupLabel, groupIndex, workerIndex, podRevision, reason := parsePodIdentityLabels(pod, lws)
	if reason != "" {
		return resolvedPod{}, reason, nil
	}

	owningSTS, reason, err := getControllerStatefulSet(ctx, reader, pod)
	if err != nil || reason != "" {
		return resolvedPod{}, reason, err
	}
	if owningSTS.DeletionTimestamp != nil {
		return resolvedPod{}, ReasonDependencyDeleting, nil
	}

	owningRevision := owningSTS.Spec.Template.Labels[leaderworkersetv1.RevisionKey]
	if owningRevision == "" || owningRevision != podRevision {
		return resolvedPod{}, ReasonOwningSTSTemplateRevisionMismatch, nil
	}

	leaderUID := pod.UID
	if workerIndex == 0 {
		// Leader: Pod → leader STS → LWS → DCD.
		if reason := validateSTSChainToDCD(owningSTS, lws, dcd); reason != "" {
			return resolvedPod{}, reason, nil
		}
	} else {
		// Follower: Pod → worker STS → leader Pod → leader STS → LWS → DCD.
		leaderUID, reason, err = resolveFollowerLeaderChain(ctx, reader, owningSTS, lws, dcd, groupLabel, podRevision)
		if err != nil || reason != "" {
			return resolvedPod{}, reason, err
		}
	}

	return resolvedPod{
		groupIndex: groupIndex,
		revision:   podRevision,
		leaderUID:  leaderUID,
	}, "", nil
}

// validateResolveRoots checks frozen DCD/LWS identity before any dependency reads.
func validateResolveRoots(
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	lws *leaderworkersetv1.LeaderWorkerSet,
	pod *corev1.Pod,
) string {
	if dcd.UID == "" || lws.UID == "" || dcd.Generation == 0 || lws.Generation == 0 {
		return ReasonMissingRootIdentity
	}
	if dcd.DeletionTimestamp != nil || lws.DeletionTimestamp != nil {
		return ReasonRootDeleting
	}
	if lws.Status.ObservedGeneration != lws.Generation {
		return ReasonLWSObservedGenerationLag
	}
	if pod.Namespace != dcd.Namespace || pod.Namespace != lws.Namespace {
		return ReasonDCDIdentityMismatch
	}
	return ""
}

// parsePodIdentityLabels validates set-name, group/worker indices, ranges, and revision labels.
// groupLabel is returned raw so follower comparison preserves zero-padding differences.
func parsePodIdentityLabels(
	pod *corev1.Pod,
	lws *leaderworkersetv1.LeaderWorkerSet,
) (groupLabel string, groupIndex, workerIndex int, podRevision string, reason string) {
	setName := pod.Labels[leaderworkersetv1.SetNameLabelKey]
	if setName == "" {
		return "", 0, 0, "", ReasonMissingSetNameLabel
	}
	if setName != lws.Name {
		return "", 0, 0, "", ReasonLWSIdentityMismatch
	}

	groupLabel = pod.Labels[leaderworkersetv1.GroupIndexLabelKey]
	if groupLabel == "" {
		return "", 0, 0, "", ReasonMissingGroupIndexLabel
	}
	groupIndex, err := strconv.Atoi(groupLabel)
	if err != nil || groupIndex < 0 {
		return "", 0, 0, "", ReasonGroupIndexInvalid
	}

	workerLabel := pod.Labels[leaderworkersetv1.WorkerIndexLabelKey]
	if workerLabel == "" {
		return "", 0, 0, "", ReasonMissingWorkerIndexLabel
	}
	workerIndex, err = strconv.Atoi(workerLabel)
	if err != nil || workerIndex < 0 {
		return "", 0, 0, "", ReasonWorkerIndexInvalid
	}

	// Compare in int space so large label values cannot wrap through int32.
	replicas := 1
	if lws.Spec.Replicas != nil {
		replicas = int(*lws.Spec.Replicas)
	}
	if groupIndex >= replicas {
		return "", 0, 0, "", ReasonGroupIndexOutOfRange
	}

	size := 1
	if lws.Spec.LeaderWorkerTemplate.Size != nil {
		size = int(*lws.Spec.LeaderWorkerTemplate.Size)
	}
	if size < 1 || workerIndex >= size {
		return "", 0, 0, "", ReasonWorkerIndexOutOfRange
	}

	podRevision = pod.Labels[leaderworkersetv1.RevisionKey]
	if podRevision == "" {
		return "", 0, 0, "", ReasonMissingRevision
	}
	return groupLabel, groupIndex, workerIndex, podRevision, ""
}

// resolveFollowerLeaderChain walks worker STS → leader Pod → leader STS → LWS → DCD.
func resolveFollowerLeaderChain(
	ctx context.Context,
	reader client.Reader,
	workerSTS *appsv1.StatefulSet,
	lws *leaderworkersetv1.LeaderWorkerSet,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	groupLabel, podRevision string,
) (types.UID, string, error) {
	leaderPod, reason, err := getControllerPod(ctx, reader, workerSTS)
	if err != nil || reason != "" {
		return "", reason, err
	}
	if leaderPod.DeletionTimestamp != nil {
		return "", ReasonDependencyDeleting, nil
	}
	if leaderPod.Labels[leaderworkersetv1.WorkerIndexLabelKey] != "0" {
		return "", ReasonLeaderWorkerIndexMismatch, nil
	}
	if leaderPod.Labels[leaderworkersetv1.SetNameLabelKey] != lws.Name {
		return "", ReasonLWSIdentityMismatch, nil
	}
	if leaderPod.Labels[leaderworkersetv1.GroupIndexLabelKey] != groupLabel {
		return "", ReasonGroupIndexInvalid, nil
	}
	leaderRevision := leaderPod.Labels[leaderworkersetv1.RevisionKey]
	if leaderRevision == "" {
		return "", ReasonMissingRevision, nil
	}
	if leaderRevision != podRevision {
		return "", ReasonRevisionMismatch, nil
	}

	leaderSTS, reason, err := getControllerStatefulSet(ctx, reader, leaderPod)
	if err != nil || reason != "" {
		return "", reason, err
	}
	if leaderSTS.DeletionTimestamp != nil {
		return "", ReasonDependencyDeleting, nil
	}
	leaderSTSRevision := leaderSTS.Spec.Template.Labels[leaderworkersetv1.RevisionKey]
	if leaderSTSRevision == "" || leaderSTSRevision != podRevision {
		return "", ReasonLeaderSTSTemplateRevisionMismatch, nil
	}
	if reason := validateSTSChainToDCD(leaderSTS, lws, dcd); reason != "" {
		return "", reason, nil
	}
	return leaderPod.UID, "", nil
}

// validateSTSChainToDCD checks STS → LWS ownership then LWS → DCD ownership.
func validateSTSChainToDCD(
	sts *appsv1.StatefulSet,
	lws *leaderworkersetv1.LeaderWorkerSet,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
) string {
	if reason := validateLeaderWorkerSetOwner(sts, lws); reason != "" {
		return reason
	}
	return validateDynamoComponentDeploymentOwner(lws, dcd)
}

func getControllerStatefulSet(
	ctx context.Context,
	reader client.Reader,
	obj metav1.Object,
) (*appsv1.StatefulSet, string, error) {
	ref := metav1.GetControllerOf(obj)
	if reason := validateOwnerRef(ref, appsv1.SchemeGroupVersion.String(), statefulSetKind); reason != "" {
		return nil, reason, nil
	}

	sts := &appsv1.StatefulSet{}
	err := reader.Get(ctx, types.NamespacedName{Namespace: obj.GetNamespace(), Name: ref.Name}, sts)
	if apierrors.IsNotFound(err) {
		return nil, ReasonSTSCacheMiss, nil
	}
	if err != nil {
		return nil, "", err
	}
	if sts.UID != ref.UID || sts.Name != ref.Name || sts.Namespace != obj.GetNamespace() {
		return nil, ReasonSTSIdentityMismatch, nil
	}
	return sts, "", nil
}

func getControllerPod(
	ctx context.Context,
	reader client.Reader,
	obj metav1.Object,
) (*corev1.Pod, string, error) {
	ref := metav1.GetControllerOf(obj)
	if reason := validateOwnerRef(ref, corev1.SchemeGroupVersion.String(), podKind); reason != "" {
		return nil, reason, nil
	}

	pod := &corev1.Pod{}
	err := reader.Get(ctx, types.NamespacedName{Namespace: obj.GetNamespace(), Name: ref.Name}, pod)
	if apierrors.IsNotFound(err) {
		return nil, ReasonLeaderPodCacheMiss, nil
	}
	if err != nil {
		return nil, "", err
	}
	if pod.UID != ref.UID || pod.Name != ref.Name || pod.Namespace != obj.GetNamespace() {
		return nil, ReasonLeaderPodIdentityMismatch, nil
	}
	return pod, "", nil
}

func validateLeaderWorkerSetOwner(sts *appsv1.StatefulSet, lws *leaderworkersetv1.LeaderWorkerSet) string {
	ref := metav1.GetControllerOf(sts)
	if reason := validateOwnerRef(ref, leaderWorkerSetAPIVersion, leaderWorkerSetKind); reason != "" {
		return reason
	}
	if ref.Name != lws.Name || ref.UID != lws.UID {
		return ReasonLWSIdentityMismatch
	}
	if lws.Namespace != sts.Namespace {
		return ReasonLWSIdentityMismatch
	}
	return ""
}

func validateDynamoComponentDeploymentOwner(
	lws *leaderworkersetv1.LeaderWorkerSet,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
) string {
	ref := metav1.GetControllerOf(lws)
	if ref == nil {
		return ReasonMissingOwner
	}
	if ref.Controller == nil || !*ref.Controller {
		return ReasonOwnerNotController
	}
	if ref.Name == "" || ref.UID == "" {
		return ReasonMissingOwner
	}
	if ref.Kind != nvidiacomv1beta1.DynamoComponentDeploymentGVK.Kind {
		return ReasonOwnerAPIKindMismatch
	}
	if ref.APIVersion != nvidiacomv1beta1.GroupVersion.String() &&
		ref.APIVersion != nvidiacomv1alpha1.GroupVersion.String() {
		return ReasonOwnerAPIKindMismatch
	}
	if ref.Name != dcd.Name || ref.UID != dcd.UID {
		return ReasonDCDIdentityMismatch
	}
	if lws.Namespace != dcd.Namespace {
		return ReasonDCDIdentityMismatch
	}
	return ""
}

func validateOwnerRef(ref *metav1.OwnerReference, apiVersion, kind string) string {
	if ref == nil {
		return ReasonMissingOwner
	}
	if ref.Controller == nil || !*ref.Controller {
		return ReasonOwnerNotController
	}
	if ref.Name == "" || ref.UID == "" {
		return ReasonMissingOwner
	}
	if ref.APIVersion != apiVersion || ref.Kind != kind {
		return ReasonOwnerAPIKindMismatch
	}
	return ""
}
