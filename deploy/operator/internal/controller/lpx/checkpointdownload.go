/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const checkpointDownloadPendingMessage = "Waiting for checkpoint downloads to complete"

// reconcileCheckpointDownloads gates rendering on the checkpoints of resolved
// workloads. Resolution accepts a checkpoint only for a hybrid Cyborg runtime,
// so this never downloads a checkpoint that no conductor consumes. A
// successful observation remains usable during periodic refreshes of a Ready
// deployment. deployment is non-nil.
func (r *graphReconciler) reconcileCheckpointDownloads(
	ctx context.Context,
	deployment *v1alpha1.LPXGraphDeployment,
	workloads map[string]*lpx.Workload,
) (ctrl.Result, error) {
	var (
		lastCheckedAt *metav1.Time
		existing      []string
		recheck       bool
	)

	if checkpointDownload := deployment.Status.CheckpointDownload; checkpointDownload != nil {
		ready := meta.IsStatusConditionTrue(deployment.Status.Conditions, v1alpha1.LPXReadyCondition)
		lastCheckedAt = checkpointDownload.LastCheckedAt
		recheck = ready && deployment.Status.ObservedGeneration == deployment.Generation && lastCheckedAt != nil

		if lastCheckedAt != nil && time.Since(lastCheckedAt.Time) < modelDownloadRefreshInterval {
			existing = checkpointDownload.Checkpoints
		}
	}

	downloaded, pending, err := ensureCheckpointsDownloaded(ctx, workloadCheckpoints(workloads), r.modelRegistry, existing)
	if err != nil || len(pending) > 0 {
		if recheck {
			if err != nil {
				log.FromContext(ctx).Error(err, "Unable to refresh checkpoint downloads")
			}
			return ctrl.Result{}, nil
		}

		deployment.Status.CheckpointDownload = &v1alpha1.CheckpointDownloadStatus{Checkpoints: downloaded}
		if err != nil {
			return ctrl.Result{}, err
		}

		setReadyCondition(deployment, v1beta1.DGDStatePending, checkpointDownloadPendingMessage+": "+strings.Join(pending, ", "))
		return ctrl.Result{RequeueAfter: modelDownloadRequeueAfter}, nil
	}

	if len(downloaded) == 0 {
		deployment.Status.CheckpointDownload = nil
		return ctrl.Result{}, nil
	}

	if len(existing) == 0 {
		lastCheckedAt = new(metav1.Now())
	}

	deployment.Status.CheckpointDownload = &v1alpha1.CheckpointDownloadStatus{Checkpoints: downloaded, LastCheckedAt: lastCheckedAt}
	return ctrl.Result{}, nil
}

// ensureCheckpointsDownloaded checks the checkpoints whose keys are not in
// existing within one shared budget. It returns the downloaded keys and the
// keys still in progress, both sorted.
func ensureCheckpointsDownloaded(
	ctx context.Context,
	checkpoints []*v1beta1.LPXCheckpoint,
	registry lpx.ModelRegistry,
	existing []string,
) ([]string, []string, error) {
	if len(checkpoints) == 0 {
		return nil, nil, nil
	}

	var (
		downloaded     = make([]string, 0, len(checkpoints))
		pending        []string
		downloadErrors []error
	)

	ctx, cancel := context.WithTimeout(ctx, modelDownloadCheckTimeout)
	defer cancel()

	perCheckpointTimeout := modelDownloadCheckTimeout / time.Duration(len(checkpoints))

	for _, checkpoint := range checkpoints {
		key := lpx.CheckpointKey(checkpoint)

		if slices.Contains(existing, key) {
			downloaded = append(downloaded, key)
			continue
		}

		ctx, cancel := context.WithTimeout(ctx, perCheckpointTimeout)
		checkpointDownloaded, err := registry.EnsureCheckpointDownloaded(ctx, checkpoint)
		cancel()

		if err != nil {
			downloadErrors = append(downloadErrors, fmt.Errorf("ensure checkpoint %q is downloaded: %w", key, err))
			continue
		}
		if !checkpointDownloaded {
			pending = append(pending, key)
			continue
		}

		downloaded = append(downloaded, key)
	}

	return downloaded, pending, errors.Join(downloadErrors...)
}

// workloadCheckpoints returns the distinct checkpoints of the resolved
// workloads' model projections, sorted by key.
func workloadCheckpoints(workloads map[string]*lpx.Workload) []*v1beta1.LPXCheckpoint {
	checkpointsByKey := make(map[string]*v1beta1.LPXCheckpoint)
	for _, workload := range workloads {
		for _, projection := range workload.ModelProjections() {
			if checkpoint := projection.Checkpoint(); checkpoint != nil {
				checkpointsByKey[lpx.CheckpointKey(checkpoint)] = checkpoint
			}
		}
	}

	keys := slices.Sorted(maps.Keys(checkpointsByKey))
	checkpoints := make([]*v1beta1.LPXCheckpoint, len(keys))
	for index, key := range keys {
		checkpoints[index] = checkpointsByKey[key]
	}
	return checkpoints
}
