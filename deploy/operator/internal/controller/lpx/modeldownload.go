/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"sort"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	modelpb "github.com/ai-dynamo/modelexpress/modelexpress_client/go/gen/modelexpress/model"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	modelDownloadRequeueAfter          = 5 * time.Second
	modelDownloadPendingReason  string = "model_download_in_progress"
	modelDownloadPendingMessage string = "Waiting for model downloads to complete"

	modelDownloadRefreshInterval = 24 * time.Hour
	modelDownloadCheckTimeout    = 30 * time.Second
)

type lpxModelRegistry interface {
	BuildURL(id string) (*url.URL, error)
	EnsureDownloaded(ctx context.Context, buildURL url.URL) (bool, error)
	AcquireBuildSnapshot(ctx context.Context, id string) (*lpx.BuildSnapshot, error)
}

// newLPXModelRegistry constructs a registry from the non-nil operator configuration.
func newLPXModelRegistry(config *configv1alpha1.OperatorConfiguration) (*lpx.ModelRegistry, error) {
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

// reconcileModelDownloads checks the nonnil DGD's remote builds and updates the
// owned, nonnil child's download status without mutating the DGD. forceCheck
// bypasses the refresh interval until an immutable LPX selection exists.
func (r *graphReconciler) reconcileModelDownloads(
	ctx context.Context,
	deployment *v1alpha1.LPXGraphDeployment,
	dgd *v1beta1.DynamoGraphDeployment,
	forceCheck bool,
) (bool, error) {
	current := deployment.Status.ObservedGeneration == deployment.Generation
	previous := deployment.Status.ModelDownload
	if !forceCheck && current && previous != nil && previous.LastCheckedAt != nil &&
		time.Since(previous.LastCheckedAt.Time) < modelDownloadRefreshInterval {
		return true, nil
	}
	running := !forceCheck && current && meta.IsStatusConditionTrue(deployment.Status.Conditions, "Ready")
	checkCtx, cancel := context.WithTimeout(ctx, modelDownloadCheckTimeout)
	defer cancel()

	downloaded, ready, err := ensureModelsDownloaded(checkCtx, dgd, r.modelRegistry)
	checkSucceeded := ready && err == nil
	// Retain the running child's last observation when refresh is incomplete.
	if !running || checkSucceeded {
		deployment.Status.ModelDownload = &v1beta1.ModelDownloadStatus{Builds: downloaded}
	} else if previous == nil {
		deployment.Status.ModelDownload = &v1beta1.ModelDownloadStatus{}
	}
	if checkSucceeded {
		checkedAt := metav1.Now()
		deployment.Status.ModelDownload.LastCheckedAt = &checkedAt
	}
	if !running {
		return ready, err
	}
	if err != nil {
		log.FromContext(ctx).Error(err, "failed to refresh model downloads", "dgd", dgd.Name)
	}
	return true, nil
}

func ensureModelsDownloaded(
	ctx context.Context,
	dgd *v1beta1.DynamoGraphDeployment,
	registry lpxModelRegistry,
) ([]string, bool, error) {
	builds, err := collectBuilds(dgd, registry)
	if err != nil {
		return nil, false, err
	}

	downloaded := make([]string, 0, len(builds))
	var downloadErrors []error
	for i, buildURL := range builds {
		build := buildURL.String()
		buildCtx := ctx
		cancel := func() {}
		if deadline, ok := ctx.Deadline(); ok {
			perBuildTimeout := time.Until(deadline) / time.Duration(len(builds)-i)
			buildCtx, cancel = context.WithTimeout(ctx, perBuildTimeout)
		}
		buildDownloaded, err := registry.EnsureDownloaded(buildCtx, buildURL)
		cancel()
		if err != nil {
			downloadErrors = append(downloadErrors, fmt.Errorf("ensure model %q is downloaded: %w", build, err))
			continue
		}
		if !buildDownloaded {
			continue
		}

		downloaded = append(downloaded, build)
	}

	return downloaded, len(downloaded) == len(builds), errors.Join(downloadErrors...)
}

func collectBuilds(dgd *v1beta1.DynamoGraphDeployment, registry lpxModelRegistry) ([]url.URL, error) {
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
