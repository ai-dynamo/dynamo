/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lws

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	// FailureRecordConflictRequeue is the bounded retry after a UID/RV CAS conflict.
	FailureRecordConflictRequeue = 2 * time.Second
)

// PersistResult describes one post-status persistence attempt.
type PersistResult struct {
	Written           bool
	Noop              bool
	CoverageLimited   bool
	Unavailable       bool
	Conflict          bool
	RequeueAfter      time.Duration
	UnavailableReason string
}

// PersistFailureRecords merges CurrentFailure observations into the DCD annotation after status write.
//
// dcd must be non-nil and the object returned from the successful status write (current UID/RV).
// batch is the immutable Observe result from the earlier cache read; it is never relabeled.
// A nil batch means observation failed and persistence is skipped. Empty batch may still prune.
func PersistFailureRecords(
	ctx context.Context,
	c client.Client,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	batch []Observation,
	now time.Time,
) (PersistResult, error) {
	if batch == nil {
		return PersistResult{Noop: true}, nil
	}
	if dcd.UID == "" || dcd.ResourceVersion == "" {
		return PersistResult{Unavailable: true, UnavailableReason: "missing_root_identity"}, nil
	}
	if dcd.DeletionTimestamp != nil {
		return PersistResult{Noop: true}, nil
	}

	// Reject silently relabeling a stale observation batch onto a new root epoch.
	if reason := validateBatchRoot(batch, dcd); reason != "" {
		logs := log.FromContext(ctx)
		logs.Error(nil, "LeaderWorkerSet failure record batch root mismatch",
			"reason", reason,
			"namespace", dcd.Namespace,
			"dcdName", dcd.Name,
			"dcdUID", dcd.UID,
			"dcdGeneration", dcd.Generation,
		)
		return PersistResult{Unavailable: true, UnavailableReason: reason}, nil
	}

	existing := ""
	if dcd.Annotations != nil {
		existing = dcd.Annotations[FailureRecordsAnnotationKey]
	}
	outcome := MergeFailureRecords(dcd.UID, existing, dcd.Annotations, batch, now)
	result := PersistResult{
		CoverageLimited:   outcome.CoverageLimited,
		Unavailable:       outcome.Unavailable,
		UnavailableReason: outcome.UnavailableReason,
	}
	if outcome.NextExpiry != nil {
		result.RequeueAfter = outcome.NextExpiry.Sub(now)
		if result.RequeueAfter < time.Second {
			result.RequeueAfter = time.Second
		}
	}

	logs := log.FromContext(ctx)
	if outcome.Unavailable {
		logs.Error(nil, "LeaderWorkerSet failure records unavailable",
			"reason", outcome.UnavailableReason,
			"namespace", dcd.Namespace,
			"dcdName", dcd.Name,
			"dcdUID", dcd.UID,
			"readOrigin", ReadOriginManagerCache,
		)
		return result, nil
	}
	if !outcome.Changed {
		result.Noop = true
		if outcome.CoverageLimited {
			logs.Info("LeaderWorkerSet failure record coverage limited",
				"namespace", dcd.Namespace,
				"dcdName", dcd.Name,
				"dcdUID", dcd.UID,
				"recordCount", outcome.RecordCount,
			)
		}
		return result, nil
	}

	patch, err := failureRecordsMergePatch(dcd.UID, dcd.ResourceVersion, outcome.Encoded)
	if err != nil {
		return PersistResult{}, fmt.Errorf("build failure records patch: %w", err)
	}
	if err := c.Patch(ctx, dcd, client.RawPatch(types.MergePatchType, patch)); err != nil {
		if apierrors.IsConflict(err) {
			logs.Error(err, "LeaderWorkerSet failure record patch conflict",
				"namespace", dcd.Namespace,
				"dcdName", dcd.Name,
				"dcdUID", dcd.UID,
				"resourceVersion", dcd.ResourceVersion,
			)
			result.Conflict = true
			result.RequeueAfter = FailureRecordConflictRequeue
			return result, nil
		}
		if apierrors.IsInvalid(err) ||
			apierrors.IsForbidden(err) ||
			apierrors.IsUnauthorized(err) ||
			apierrors.IsRequestEntityTooLargeError(err) {
			logs.Error(err, "LeaderWorkerSet failure record patch rejected",
				"namespace", dcd.Namespace,
				"dcdName", dcd.Name,
				"dcdUID", dcd.UID,
			)
			result.Unavailable = true
			result.UnavailableReason = "patch_rejected"
			result.RequeueAfter = 0
			return result, nil
		}
		logs.Error(err, "LeaderWorkerSet failure record patch failed",
			"namespace", dcd.Namespace,
			"dcdName", dcd.Name,
			"dcdUID", dcd.UID,
		)
		return PersistResult{}, err
	}

	result.Written = true
	if outcome.CoverageLimited {
		logs.Info("LeaderWorkerSet failure records updated with coverage limit",
			"namespace", dcd.Namespace,
			"dcdName", dcd.Name,
			"dcdUID", dcd.UID,
			"recordCount", outcome.RecordCount,
		)
	} else {
		logs.V(1).Info("LeaderWorkerSet failure records updated",
			"namespace", dcd.Namespace,
			"dcdName", dcd.Name,
			"dcdUID", dcd.UID,
			"recordCount", outcome.RecordCount,
		)
	}
	return result, nil
}

func validateBatchRoot(batch []Observation, dcd *nvidiacomv1beta1.DynamoComponentDeployment) string {
	for i := range batch {
		obs := batch[i]
		if obs.Verdict == VerdictCurrentFailure {
			if obs.Namespace == "" || obs.DCDName == "" || obs.DCDUID == "" || obs.DCDGeneration == 0 {
				return "incomplete_current_failure_root"
			}
		}
		if obs.DCDUID != "" && obs.DCDUID != dcd.UID {
			return "batch_dcd_uid_mismatch"
		}
		if obs.DCDGeneration != 0 && obs.DCDGeneration != dcd.Generation {
			return "batch_dcd_generation_mismatch"
		}
		if obs.Namespace != "" && obs.Namespace != dcd.Namespace {
			return "batch_namespace_mismatch"
		}
		if obs.DCDName != "" && obs.DCDName != dcd.Name {
			return "batch_dcd_name_mismatch"
		}
	}
	return ""
}

// failureRecordsMergePatch builds a merge patch that explicitly carries UID and resourceVersion
// and touches only the owned annotation key.
func failureRecordsMergePatch(uid types.UID, resourceVersion, encoded string) ([]byte, error) {
	body := map[string]any{
		"metadata": map[string]any{
			"uid":             string(uid),
			"resourceVersion": resourceVersion,
			"annotations": map[string]string{
				FailureRecordsAnnotationKey: encoded,
			},
		},
	}
	return json.Marshal(body)
}
