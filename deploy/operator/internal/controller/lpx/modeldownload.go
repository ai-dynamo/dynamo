/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"net/url"
	"slices"
	"sort"
	"strings"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	modelpb "github.com/ai-dynamo/modelexpress/modelexpress_client/go/gen/modelexpress/model"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	modelDownloadRequeueAfter          = 5 * time.Second
	modelDownloadPendingMessage string = "Waiting for model downloads to complete"

	modelDownloadRefreshInterval = 24 * time.Hour
	modelDownloadCheckTimeout    = 30 * time.Second
)

// newLPXModelRegistry constructs a registry from the non-nil operator configuration.
func newLPXModelRegistry(config *configv1alpha1.OperatorConfiguration) (lpx.ModelRegistry, error) {
	// Configure the optional Model Express client.
	var mxClient modelpb.ModelServiceClient
	var err error
	if config.Infrastructure.ModelExpressURL != "" {
		mxClient, err = lpx.NewModelExpressClient(config.Infrastructure.ModelExpressURL)
		if err != nil {
			return nil, fmt.Errorf("unable to create Model Express client: %w", err)
		}
		log.Log.WithName("setup").Info("LPX Model Express client configured", "modelExpressURL", config.Infrastructure.ModelExpressURL)
	}

	// Construct the registry with the configured download client.
	registry, err := lpx.NewModelRegistry(config.LPX.ModelRegistryURL, mxClient)
	if err != nil {
		return nil, fmt.Errorf("unable to create LPX model registry client: %w", err)
	}
	return registry, nil
}

// reconcileModelDownloads gates startup on downloaded builds and checkpoints. A
// successful observation remains usable during periodic refreshes of a Ready
// deployment. deployment and dgd are non-nil.
func (r *graphReconciler) reconcileModelDownloads(
	ctx context.Context,
	deployment *v1alpha1.LPXGraphDeployment,
	dgd *v1beta1.DynamoGraphDeployment,
) (ctrl.Result, error) {
	var (
		lastCheckedAt *metav1.Time
		existing      v1alpha1.ModelDownloadStatus
		recheck       bool
	)

	if modelDownload := deployment.Status.ModelDownload; modelDownload != nil {
		ready := meta.IsStatusConditionTrue(deployment.Status.Conditions, v1alpha1.LPXReadyCondition)
		lastCheckedAt = modelDownload.LastCheckedAt
		recheck = ready && deployment.Status.ObservedGeneration == deployment.Generation && lastCheckedAt != nil

		if lastCheckedAt != nil && time.Since(lastCheckedAt.Time) < modelDownloadRefreshInterval {
			existing = v1alpha1.ModelDownloadStatus{Builds: modelDownload.Builds, Checkpoints: modelDownload.Checkpoints}
		}
	}

	downloaded, pending, err := ensureModelsDownloaded(ctx, dgd, r.modelRegistry, existing)
	if err != nil || len(pending) > 0 {
		if recheck {
			if err != nil {
				log.FromContext(ctx).Error(err, "Unable to refresh model downloads")
			}
			return ctrl.Result{}, nil
		}

		deployment.Status.ModelDownload = &downloaded
		if err != nil {
			return ctrl.Result{}, err
		}

		setReadyCondition(deployment, v1beta1.DGDStatePending, modelDownloadPendingMessage+": "+strings.Join(pending, ", "))
		return ctrl.Result{RequeueAfter: modelDownloadRequeueAfter}, nil
	}

	if len(downloaded.Builds) == 0 && len(downloaded.Checkpoints) == 0 {
		deployment.Status.ModelDownload = nil
		return ctrl.Result{}, nil
	}

	if len(existing.Builds) == 0 && len(existing.Checkpoints) == 0 {
		lastCheckedAt = new(metav1.Now())
	}

	downloaded.LastCheckedAt = lastCheckedAt
	deployment.Status.ModelDownload = &downloaded
	return ctrl.Result{}, nil
}

// ensureModelsDownloaded checks the DGD's remote builds and checkpoints that are
// not in existing. It returns the downloaded subset, without LastCheckedAt, and
// the identities of downloads still in progress, builds before checkpoints.
// dgd is non-nil.
func ensureModelsDownloaded(
	ctx context.Context,
	dgd *v1beta1.DynamoGraphDeployment,
	registry lpx.ModelRegistry,
	existing v1alpha1.ModelDownloadStatus,
) (v1alpha1.ModelDownloadStatus, []string, error) {
	builds, err := collectBuilds(dgd, registry)
	if err != nil {
		return v1alpha1.ModelDownloadStatus{}, nil, err
	}
	checkpoints := collectCheckpoints(dgd)

	if len(builds) == 0 && len(checkpoints) == 0 {
		return v1alpha1.ModelDownloadStatus{}, nil, nil
	}

	var (
		downloaded     v1alpha1.ModelDownloadStatus
		pending        []string
		downloadErrors []error
	)

	// Share one check budget across every build and checkpoint download.
	ctx, cancel := context.WithTimeout(ctx, modelDownloadCheckTimeout)
	defer cancel()
	perDownloadTimeout := modelDownloadCheckTimeout / time.Duration(len(builds)+len(checkpoints))

	for _, buildURL := range builds {
		build := buildURL.String()

		if slices.Contains(existing.Builds, build) {
			downloaded.Builds = append(downloaded.Builds, build)
			continue
		}

		ctx, cancel := context.WithTimeout(ctx, perDownloadTimeout)
		buildDownloaded, err := registry.EnsureDownloaded(ctx, buildURL)
		cancel()

		if err != nil {
			downloadErrors = append(downloadErrors, fmt.Errorf("ensure model %q is downloaded: %w", build, err))
			continue
		}
		if !buildDownloaded {
			pending = append(pending, build)
			continue
		}

		downloaded.Builds = append(downloaded.Builds, build)
	}

	for _, checkpoint := range checkpoints {
		key := lpx.CheckpointKey(checkpoint)

		if slices.Contains(existing.Checkpoints, key) {
			downloaded.Checkpoints = append(downloaded.Checkpoints, key)
			continue
		}

		ctx, cancel := context.WithTimeout(ctx, perDownloadTimeout)
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

		downloaded.Checkpoints = append(downloaded.Checkpoints, key)
	}

	return downloaded, pending, errors.Join(downloadErrors...)
}

func collectBuilds(dgd *v1beta1.DynamoGraphDeployment, registry lpx.ModelRegistry) ([]url.URL, error) {
	buildsByKey := make(map[string]url.URL)

	for _, component := range dgd.Spec.Components {
		config := component.LPX
		if config == nil {
			continue
		}

		// Each LPX component owns one build, including an agent-only draft.
		buildURL, err := registry.BuildURL(config.BuildID)
		if err != nil {
			return nil, fmt.Errorf("resolve LPX build URL %q: %w", config.BuildID, err)
		}

		if buildURL.Scheme == lpx.BuildSchemeGCS {
			buildsByKey[buildURL.String()] = *buildURL
		}
	}

	builds := make([]url.URL, 0, len(buildsByKey))
	for _, build := range buildsByKey {
		builds = append(builds, build)
	}
	sort.Slice(builds, func(i, j int) bool {
		return builds[i].String() < builds[j].String()
	})

	return builds, nil
}

// collectCheckpoints returns the DGD's distinct checkpoints sorted by key. dgd is non-nil.
func collectCheckpoints(dgd *v1beta1.DynamoGraphDeployment) []*v1beta1.LPXCheckpoint {
	checkpointsByKey := make(map[string]*v1beta1.LPXCheckpoint)
	for _, component := range dgd.Spec.Components {
		if component.LPX != nil && component.LPX.Checkpoint != nil {
			checkpointsByKey[lpx.CheckpointKey(component.LPX.Checkpoint)] = component.LPX.Checkpoint
		}
	}

	keys := slices.Sorted(maps.Keys(checkpointsByKey))
	checkpoints := make([]*v1beta1.LPXCheckpoint, len(keys))
	for index, key := range keys {
		checkpoints[index] = checkpointsByKey[key]
	}
	return checkpoints
}
