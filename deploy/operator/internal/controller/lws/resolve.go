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
	if dcd.UID == "" || lws.UID == "" || dcd.Generation == 0 || lws.Generation == 0 {
		return resolvedPod{}, ReasonMissingRootIdentity, nil
	}
	if dcd.DeletionTimestamp != nil || lws.DeletionTimestamp != nil {
		return resolvedPod{}, ReasonRootDeleting, nil
	}
	if lws.Status.ObservedGeneration != lws.Generation {
		return resolvedPod{}, ReasonLWSObservedGenerationLag, nil
	}
	if pod.Namespace != dcd.Namespace || pod.Namespace != lws.Namespace {
		return resolvedPod{}, ReasonDCDIdentityMismatch, nil
	}

	setName := pod.Labels[leaderworkersetv1.SetNameLabelKey]
	if setName == "" {
		return resolvedPod{}, ReasonMissingSetNameLabel, nil
	}
	if setName != lws.Name {
		return resolvedPod{}, ReasonLWSIdentityMismatch, nil
	}

	groupLabel := pod.Labels[leaderworkersetv1.GroupIndexLabelKey]
	if groupLabel == "" {
		return resolvedPod{}, ReasonMissingGroupIndexLabel, nil
	}
	groupIndex, err := strconv.Atoi(groupLabel)
	if err != nil || groupIndex < 0 {
		return resolvedPod{}, ReasonGroupIndexInvalid, nil
	}

	workerLabel := pod.Labels[leaderworkersetv1.WorkerIndexLabelKey]
	if workerLabel == "" {
		return resolvedPod{}, ReasonMissingWorkerIndexLabel, nil
	}
	workerIndex, err := strconv.Atoi(workerLabel)
	if err != nil || workerIndex < 0 {
		return resolvedPod{}, ReasonWorkerIndexInvalid, nil
	}

	// Compare in int space so large label values cannot wrap through int32.
	replicas := 1
	if lws.Spec.Replicas != nil {
		replicas = int(*lws.Spec.Replicas)
	}
	if groupIndex >= replicas {
		return resolvedPod{}, ReasonGroupIndexOutOfRange, nil
	}

	size := 1
	if lws.Spec.LeaderWorkerTemplate.Size != nil {
		size = int(*lws.Spec.LeaderWorkerTemplate.Size)
	}
	if size < 1 || workerIndex >= size {
		return resolvedPod{}, ReasonWorkerIndexOutOfRange, nil
	}

	podRevision := pod.Labels[leaderworkersetv1.RevisionKey]
	if podRevision == "" {
		return resolvedPod{}, ReasonMissingRevision, nil
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
		reason, err = validateLeaderWorkerSetOwner(owningSTS, lws)
		if err != nil || reason != "" {
			return resolvedPod{}, reason, err
		}
		reason = validateDynamoComponentDeploymentOwner(lws, dcd)
		if reason != "" {
			return resolvedPod{}, reason, nil
		}
	} else {
		// Follower: Pod → worker STS → leader Pod → leader STS → LWS → DCD.
		leaderPod, reason, err := getControllerPod(ctx, reader, owningSTS)
		if err != nil || reason != "" {
			return resolvedPod{}, reason, err
		}
		if leaderPod.DeletionTimestamp != nil {
			return resolvedPod{}, ReasonDependencyDeleting, nil
		}
		if leaderPod.Labels[leaderworkersetv1.WorkerIndexLabelKey] != "0" {
			return resolvedPod{}, ReasonLeaderWorkerIndexMismatch, nil
		}
		if leaderPod.Labels[leaderworkersetv1.SetNameLabelKey] != lws.Name {
			return resolvedPod{}, ReasonLWSIdentityMismatch, nil
		}
		if leaderPod.Labels[leaderworkersetv1.GroupIndexLabelKey] != groupLabel {
			return resolvedPod{}, ReasonGroupIndexInvalid, nil
		}
		leaderRevision := leaderPod.Labels[leaderworkersetv1.RevisionKey]
		if leaderRevision == "" {
			return resolvedPod{}, ReasonMissingRevision, nil
		}
		if leaderRevision != podRevision {
			return resolvedPod{}, ReasonRevisionMismatch, nil
		}

		leaderSTS, reason, err := getControllerStatefulSet(ctx, reader, leaderPod)
		if err != nil || reason != "" {
			return resolvedPod{}, reason, err
		}
		if leaderSTS.DeletionTimestamp != nil {
			return resolvedPod{}, ReasonDependencyDeleting, nil
		}
		leaderSTSRevision := leaderSTS.Spec.Template.Labels[leaderworkersetv1.RevisionKey]
		if leaderSTSRevision == "" || leaderSTSRevision != podRevision {
			return resolvedPod{}, ReasonLeaderSTSTemplateRevisionMismatch, nil
		}
		reason, err = validateLeaderWorkerSetOwner(leaderSTS, lws)
		if err != nil || reason != "" {
			return resolvedPod{}, reason, err
		}
		reason = validateDynamoComponentDeploymentOwner(lws, dcd)
		if reason != "" {
			return resolvedPod{}, reason, nil
		}
		leaderUID = leaderPod.UID
	}

	return resolvedPod{
		groupIndex: groupIndex,
		revision:   podRevision,
		leaderUID:  leaderUID,
	}, "", nil
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

func validateLeaderWorkerSetOwner(sts *appsv1.StatefulSet, lws *leaderworkersetv1.LeaderWorkerSet) (string, error) {
	ref := metav1.GetControllerOf(sts)
	if reason := validateOwnerRef(ref, leaderWorkerSetAPIVersion, leaderWorkerSetKind); reason != "" {
		return reason, nil
	}
	if ref.Name != lws.Name || ref.UID != lws.UID {
		return ReasonLWSIdentityMismatch, nil
	}
	if lws.Namespace != sts.Namespace {
		return ReasonLWSIdentityMismatch, nil
	}
	return "", nil
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
