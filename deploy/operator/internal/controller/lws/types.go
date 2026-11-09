/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lws

import (
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

const (
	// ReadOriginManagerCache records that evidence came from the shared informer cache.
	ReadOriginManagerCache = "manager_cache"

	// clockSkewAllowance tolerates small kubelet/API clock skew for FinishedAt and Ready LTT.
	clockSkewAllowance = 5 * time.Second

	leaderWorkerSetAPIVersion = "leaderworkerset.x-k8s.io/v1"
	leaderWorkerSetKind       = "LeaderWorkerSet"
	statefulSetKind           = "StatefulSet"
	podKind                   = "Pod"
)

// Verdict is the current-evidence classification for one Observe result.
type Verdict string

const (
	// VerdictCurrentFailure is Ready=False with a current main nonzero termination.
	VerdictCurrentFailure Verdict = "CurrentFailure"
	// VerdictNoCurrentFailureEvidence means this Pod supplies no current failure proof.
	// It is never proof that the group is healthy or recovered.
	VerdictNoCurrentFailureEvidence Verdict = "NoCurrentFailureEvidence"
	// VerdictUnknown means identity, revision, or time evidence is incomplete.
	VerdictUnknown Verdict = "Unknown"
)

// Fixed observation reasons used by Observe and its tests.
const (
	ReasonCurrentMainNonzeroExit          = "current_main_nonzero_exit"
	ReasonReadyTrue                       = "ready_true"
	ReasonReadyNotFalse                   = "ready_not_false"
	ReasonNoCurrentMainTerminated         = "no_current_main_terminated"
	ReasonMainExitZero                    = "main_exit_zero"
	ReasonPodDeleting                     = "pod_deleting"
	ReasonMissingFinishedAt               = "missing_finished_at"
	ReasonMissingReadyLastTransitionTime  = "missing_ready_last_transition_time"
	ReasonFinishedAtBeforeCreation        = "finished_at_before_creation"
	ReasonFinishedAtInFuture              = "finished_at_in_future"
	ReasonReadyLTTBeforeCreation          = "ready_ltt_before_creation"
	ReasonReadyLTTInFuture                = "ready_ltt_in_future"
	ReasonMissingOwner                    = "missing_owner"
	ReasonOwnerNotController              = "owner_not_controller"
	ReasonOwnerAPIKindMismatch            = "owner_api_kind_mismatch"
	ReasonSTSCacheMiss                    = "sts_cache_miss"
	ReasonSTSIdentityMismatch             = "sts_identity_mismatch"
	ReasonLeaderPodCacheMiss              = "leader_pod_cache_miss"
	ReasonLeaderPodIdentityMismatch       = "leader_pod_identity_mismatch"
	ReasonLWSIdentityMismatch             = "lws_identity_mismatch"
	ReasonDCDIdentityMismatch             = "dcd_identity_mismatch"
	ReasonLWSObservedGenerationLag        = "lws_observed_generation_lag"
	ReasonMissingRevision                 = "missing_revision"
	ReasonOwningSTSTemplateRevisionMismatch = "owning_sts_template_revision_mismatch"
	ReasonLeaderSTSTemplateRevisionMismatch = "leader_sts_template_revision_mismatch"
	ReasonRevisionMismatch                = "revision_mismatch"
	ReasonMissingSetNameLabel             = "missing_set_name_label"
	ReasonMissingGroupIndexLabel          = "missing_group_index_label"
	ReasonMissingWorkerIndexLabel         = "missing_worker_index_label"
	ReasonGroupIndexInvalid               = "group_index_invalid"
	ReasonWorkerIndexInvalid              = "worker_index_invalid"
	ReasonGroupIndexOutOfRange            = "group_index_out_of_range"
	ReasonWorkerIndexOutOfRange           = "worker_index_out_of_range"
	ReasonLeaderWorkerIndexMismatch       = "leader_worker_index_mismatch"
	ReasonRootDeleting                    = "root_deleting"
	ReasonDependencyDeleting              = "dependency_deleting"
	ReasonMissingRootIdentity             = "missing_root_identity"
	ReasonRootCacheMiss                   = "root_cache_miss"
	ReasonLWSCacheMiss                    = "lws_cache_miss"
	ReasonRootEpochChanged                = "root_epoch_changed"
	ReasonMissingCreationTimestamp        = "missing_creation_timestamp"
	ReasonMissingPodUID                   = "missing_pod_uid"
	ReasonNoPodsVisible                   = "no_pods_visible"
)

// Observation is one typed current-evidence result from Observe.
type Observation struct {
	Verdict                 Verdict
	Reason                  string
	Namespace               string
	DCDName                 string
	DCDUID                  types.UID
	DCDGeneration           int64
	LWSName                 string
	LWSUID                  types.UID
	LWSGeneration           int64
	GroupIndex              *int
	Revision                string
	LeaderUID               types.UID
	PodName                 string
	PodUID                  types.UID
	PodResourceVersion      string
	Container               string
	ExitCode                *int32
	FinishedAt              *metav1.Time
	Ready                   corev1.ConditionStatus
	ReadyLastTransitionTime *metav1.Time
	ReadOrigin              string
}

type incarnationKey struct {
	dcdUID        types.UID
	dcdGeneration int64
	lwsUID        types.UID
	lwsGeneration int64
	revision      string
	groupIndex    int
	leaderUID     types.UID
}
