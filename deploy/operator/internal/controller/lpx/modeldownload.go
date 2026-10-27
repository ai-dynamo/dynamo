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
	"slices"
	"sort"
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

// reconcileModelDownloads gates startup on downloaded builds. A successful
// observation remains usable during periodic refreshes of a Ready deployment.
// deployment and dgd are non-nil.
func (r *graphReconciler) reconcileModelDownloads(
	ctx context.Context,
	deployment *v1alpha1.LPXGraphDeployment,
	dgd *v1beta1.DynamoGraphDeployment,
) (ctrl.Result, error) {
	var (
		lastCheckedAt  *metav1.Time
		existingBuilds []string
		recheck        bool
	)

	if modelDownload := deployment.Status.ModelDownload; modelDownload != nil {
		ready := meta.IsStatusConditionTrue(deployment.Status.Conditions, v1alpha1.LPXReadyCondition)
		lastCheckedAt = modelDownload.LastCheckedAt
		recheck = ready && deployment.Status.ObservedGeneration == deployment.Generation && lastCheckedAt != nil

		if lastCheckedAt != nil && time.Since(lastCheckedAt.Time) < modelDownloadRefreshInterval {
			existingBuilds = modelDownload.Builds
		}
	}

	downloaded, ready, err := ensureModelsDownloaded(ctx, dgd, r.modelRegistry, existingBuilds)
	if err != nil || !ready {
		if recheck {
			if err != nil {
				log.FromContext(ctx).Error(err, "Unable to refresh model downloads")
			}
			return ctrl.Result{}, nil
		}

		deployment.Status.ModelDownload = &v1alpha1.ModelDownloadStatus{Builds: downloaded}
		if err != nil {
			return ctrl.Result{}, err
		}

		setReadyCondition(deployment, v1beta1.DGDStatePending, modelDownloadPendingMessage)
		return ctrl.Result{RequeueAfter: modelDownloadRequeueAfter}, nil
	}

	if len(downloaded) == 0 {
		deployment.Status.ModelDownload = nil
		return ctrl.Result{}, nil
	}

	if len(existingBuilds) == 0 {
		lastCheckedAt = new(metav1.Now())
	}

	deployment.Status.ModelDownload = &v1alpha1.ModelDownloadStatus{Builds: downloaded, LastCheckedAt: lastCheckedAt}
	return ctrl.Result{}, nil
}

// ensureModelsDownloaded requires a non-nil dgd. registry may be nil unless
// the graph contains LPX components.
func ensureModelsDownloaded(
	ctx context.Context,
	dgd *v1beta1.DynamoGraphDeployment,
	registry lpx.ModelRegistry,
	existingBuilds []string,
) ([]string, bool, error) {
	builds, err := collectBuilds(dgd, registry)
	if err != nil {
		return nil, false, err
	}

	if len(builds) == 0 {
		return nil, true, nil
	}

	downloaded, _, err := ensureDownloaded(ctx, builds, existingBuilds, "model",
		func(build url.URL) string { return build.String() }, registry.EnsureDownloaded)
	return downloaded, len(downloaded) == len(builds), err
}

// ensureDownloaded checks the items whose keys are not in existing, in order,
// within one shared budget. It returns the downloaded and pending keys.
func ensureDownloaded[T any](
	ctx context.Context,
	items []T,
	existing []string,
	kind string,
	key func(T) string,
	download func(context.Context, T) (bool, error),
) ([]string, []string, error) {
	if len(items) == 0 {
		return nil, nil, nil
	}

	var (
		downloaded     = make([]string, 0, len(items))
		pending        []string
		downloadErrors []error
	)

	ctx, cancel := context.WithTimeout(ctx, modelDownloadCheckTimeout)
	defer cancel()

	// Cached items count toward the divisor, so each check gets the same share.
	perItemTimeout := modelDownloadCheckTimeout / time.Duration(len(items))

	for _, item := range items {
		itemKey := key(item)

		if slices.Contains(existing, itemKey) {
			downloaded = append(downloaded, itemKey)
			continue
		}

		ctx, cancel := context.WithTimeout(ctx, perItemTimeout)
		itemDownloaded, err := download(ctx, item)
		cancel()

		if err != nil {
			downloadErrors = append(downloadErrors, fmt.Errorf("ensure %s %q is downloaded: %w", kind, itemKey, err))
			continue
		}
		if !itemDownloaded {
			pending = append(pending, itemKey)
			continue
		}

		downloaded = append(downloaded, itemKey)
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
