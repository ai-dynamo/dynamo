/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lws

import (
	"context"
	"sort"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
)

// Observe reads current LeaderWorkerSet Pod evidence from the manager cache.
//
// dcd and lws must be non-nil. Observe does not mutate them. Cache NotFound and
// incomplete identity/revision/time evidence become Unknown. Real Get/List and
// index configuration errors are returned to the caller.
func Observe(
	ctx context.Context,
	reader client.Reader,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	lws *leaderworkersetv1.LeaderWorkerSet,
	now time.Time,
) ([]Observation, error) {
	root := rootObservation(dcd, lws)

	if dcd.UID == "" || lws.UID == "" || dcd.Generation == 0 || lws.Generation == 0 {
		return []Observation{unknown(root, ReasonMissingRootIdentity)}, nil
	}
	if dcd.DeletionTimestamp != nil || lws.DeletionTimestamp != nil {
		return []Observation{unknown(root, ReasonRootDeleting)}, nil
	}
	if lws.Status.ObservedGeneration != lws.Generation {
		return []Observation{unknown(root, ReasonLWSObservedGenerationLag)}, nil
	}

	pods, err := ListPodsBySetName(ctx, reader, lws.Namespace, lws.Name)
	if err != nil {
		return nil, err
	}
	if len(pods) == 0 {
		return []Observation{unknown(root, ReasonNoPodsVisible)}, nil
	}

	observations := make([]Observation, 0, len(pods))
	for i := range pods {
		obs, obsErr := observePod(ctx, reader, dcd, lws, &pods[i], now)
		if obsErr != nil {
			return nil, obsErr
		}
		observations = append(observations, obs)
	}

	// One post-dependency root Get validates the frozen input epoch is still current.
	if reason, recheckErr := recheckRoots(ctx, reader, dcd, lws); recheckErr != nil {
		return nil, recheckErr
	} else if reason != "" {
		for i := range observations {
			observations[i] = unknown(observations[i], reason)
		}
	}

	return dedupeCurrentFailures(observations), nil
}

func observePod(
	ctx context.Context,
	reader client.Reader,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	lws *leaderworkersetv1.LeaderWorkerSet,
	pod *corev1.Pod,
	now time.Time,
) (Observation, error) {
	base := rootObservation(dcd, lws)
	base.PodName = pod.Name
	base.PodUID = pod.UID
	base.PodResourceVersion = pod.ResourceVersion

	if pod.UID == "" {
		return unknown(base, ReasonMissingPodUID), nil
	}

	resolved, reason, err := resolvePod(ctx, reader, dcd, lws, pod)
	if err != nil {
		return Observation{}, err
	}
	if reason != "" {
		return unknown(base, reason), nil
	}

	groupIndex := resolved.groupIndex
	base.GroupIndex = &groupIndex
	base.Revision = resolved.revision
	base.LeaderUID = resolved.leaderUID

	if pod.DeletionTimestamp != nil {
		return noCurrentFailure(base, ReasonPodDeleting), nil
	}
	if pod.CreationTimestamp.IsZero() {
		return unknown(base, ReasonMissingCreationTimestamp), nil
	}

	ready, readyLTT, readyReason := podReadyEvidence(pod)
	base.Ready = ready
	if readyLTT != nil {
		base.ReadyLastTransitionTime = readyLTT.DeepCopy()
	}

	main := mainContainerStatus(pod)
	if main == nil || main.State.Terminated == nil {
		return noCurrentFailure(base, ReasonNoCurrentMainTerminated), nil
	}

	base.Container = main.Name
	exitCode := main.State.Terminated.ExitCode
	base.ExitCode = &exitCode
	if !main.State.Terminated.FinishedAt.IsZero() {
		finishedAt := main.State.Terminated.FinishedAt
		base.FinishedAt = finishedAt.DeepCopy()
	}

	if exitCode == 0 {
		return noCurrentFailure(base, ReasonMainExitZero), nil
	}
	if ready == corev1.ConditionTrue {
		return noCurrentFailure(base, ReasonReadyTrue), nil
	}
	if ready != corev1.ConditionFalse {
		if readyReason != "" {
			return unknown(base, readyReason), nil
		}
		return unknown(base, ReasonReadyNotFalse), nil
	}
	if readyLTT == nil || readyLTT.IsZero() {
		return unknown(base, ReasonMissingReadyLastTransitionTime), nil
	}
	if timeReason := validateEventTime(pod.CreationTimestamp, *readyLTT, now, ReasonReadyLTTBeforeCreation, ReasonReadyLTTInFuture); timeReason != "" {
		return unknown(base, timeReason), nil
	}
	if base.FinishedAt == nil || base.FinishedAt.IsZero() {
		return unknown(base, ReasonMissingFinishedAt), nil
	}
	if timeReason := validateEventTime(pod.CreationTimestamp, *base.FinishedAt, now, ReasonFinishedAtBeforeCreation, ReasonFinishedAtInFuture); timeReason != "" {
		return unknown(base, timeReason), nil
	}

	base.Verdict = VerdictCurrentFailure
	base.Reason = ReasonCurrentMainNonzeroExit
	return base, nil
}

func rootObservation(
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	lws *leaderworkersetv1.LeaderWorkerSet,
) Observation {
	return Observation{
		Namespace:     dcd.Namespace,
		DCDName:       dcd.Name,
		DCDUID:        dcd.UID,
		DCDGeneration: dcd.Generation,
		LWSName:       lws.Name,
		LWSUID:        lws.UID,
		LWSGeneration: lws.Generation,
		ReadOrigin:    ReadOriginManagerCache,
	}
}

func unknown(base Observation, reason string) Observation {
	base.Verdict = VerdictUnknown
	base.Reason = reason
	return base
}

func noCurrentFailure(base Observation, reason string) Observation {
	base.Verdict = VerdictNoCurrentFailureEvidence
	base.Reason = reason
	return base
}

func mainContainerStatus(pod *corev1.Pod) *corev1.ContainerStatus {
	for i := range pod.Status.ContainerStatuses {
		if pod.Status.ContainerStatuses[i].Name == commonconsts.MainContainerName {
			return &pod.Status.ContainerStatuses[i]
		}
	}
	return nil
}

func podReadyEvidence(pod *corev1.Pod) (corev1.ConditionStatus, *metav1.Time, string) {
	for i := range pod.Status.Conditions {
		if pod.Status.Conditions[i].Type != corev1.PodReady {
			continue
		}
		condition := pod.Status.Conditions[i]
		if condition.LastTransitionTime.IsZero() {
			return condition.Status, nil, ReasonMissingReadyLastTransitionTime
		}
		ltt := condition.LastTransitionTime
		return condition.Status, &ltt, ""
	}
	return "", nil, ReasonReadyNotFalse
}

func validateEventTime(
	creation metav1.Time,
	eventTime metav1.Time,
	now time.Time,
	beforeReason, futureReason string,
) string {
	if eventTime.IsZero() {
		return beforeReason
	}
	if creation.IsZero() {
		return ReasonMissingCreationTimestamp
	}
	if eventTime.Time.Before(creation.Time) {
		return beforeReason
	}
	if eventTime.Time.After(now.Add(clockSkewAllowance)) {
		return futureReason
	}
	return ""
}

// recheckRoots reads the current cached DCD/LWS once and compares them to the
// frozen caller epoch. It does not claim multi-object cache atomicity.
func recheckRoots(
	ctx context.Context,
	reader client.Reader,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	lws *leaderworkersetv1.LeaderWorkerSet,
) (string, error) {
	currentDCD := &nvidiacomv1beta1.DynamoComponentDeployment{}
	err := reader.Get(ctx, types.NamespacedName{Namespace: dcd.Namespace, Name: dcd.Name}, currentDCD)
	if apierrors.IsNotFound(err) {
		return ReasonRootCacheMiss, nil
	}
	if err != nil {
		return "", err
	}

	currentLWS := &leaderworkersetv1.LeaderWorkerSet{}
	err = reader.Get(ctx, types.NamespacedName{Namespace: lws.Namespace, Name: lws.Name}, currentLWS)
	if apierrors.IsNotFound(err) {
		return ReasonLWSCacheMiss, nil
	}
	if err != nil {
		return "", err
	}

	if currentDCD.Namespace != dcd.Namespace || currentDCD.Name != dcd.Name ||
		currentDCD.UID != dcd.UID || currentDCD.Generation != dcd.Generation {
		return ReasonRootEpochChanged, nil
	}
	if currentLWS.Namespace != lws.Namespace || currentLWS.Name != lws.Name ||
		currentLWS.UID != lws.UID || currentLWS.Generation != lws.Generation {
		return ReasonRootEpochChanged, nil
	}
	if currentDCD.DeletionTimestamp != nil || currentLWS.DeletionTimestamp != nil {
		return ReasonRootDeleting, nil
	}
	if currentLWS.Status.ObservedGeneration != lws.Status.ObservedGeneration ||
		currentLWS.Status.ObservedGeneration != currentLWS.Generation {
		return ReasonLWSObservedGenerationLag, nil
	}
	// OwnerReferences can change without a generation bump; re-validate against the cached DCD.
	if reason := validateDynamoComponentDeploymentOwner(currentLWS, currentDCD); reason != "" {
		return reason, nil
	}
	return "", nil
}

// dedupeCurrentFailures keeps one deterministic witness per known incarnation.
func dedupeCurrentFailures(in []Observation) []Observation {
	bestFailure := map[incarnationKey]Observation{}
	order := make([]incarnationKey, 0)
	out := make([]Observation, 0, len(in))

	for _, obs := range in {
		if obs.Verdict != VerdictCurrentFailure {
			out = append(out, obs)
			continue
		}
		if obs.GroupIndex == nil || obs.LeaderUID == "" {
			out = append(out, unknown(obs, ReasonMissingGroupIndexLabel))
			continue
		}
		key := incarnationKey{
			dcdUID:        obs.DCDUID,
			dcdGeneration: obs.DCDGeneration,
			lwsUID:        obs.LWSUID,
			lwsGeneration: obs.LWSGeneration,
			revision:      obs.Revision,
			groupIndex:    *obs.GroupIndex,
			leaderUID:     obs.LeaderUID,
		}
		existing, found := bestFailure[key]
		if !found {
			bestFailure[key] = obs
			order = append(order, key)
			continue
		}
		if witnessLess(obs, existing) {
			bestFailure[key] = obs
		}
	}

	for _, key := range order {
		out = append(out, bestFailure[key])
	}

	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Verdict != out[j].Verdict {
			return out[i].Verdict < out[j].Verdict
		}
		if out[i].PodName != out[j].PodName {
			return out[i].PodName < out[j].PodName
		}
		return string(out[i].PodUID) < string(out[j].PodUID)
	})
	return out
}

func witnessLess(a, b Observation) bool {
	if a.PodName != b.PodName {
		return a.PodName < b.PodName
	}
	return string(a.PodUID) < string(b.PodUID)
}
