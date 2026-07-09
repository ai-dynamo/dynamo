// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/scale"
	ctrl "sigs.k8s.io/controller-runtime"

	configapi "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gpu"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/modelendpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/rbac"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/secret"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/secrets"
)

// SetupControllersOpts holds optional dependencies for SetupControllers.
type SetupControllersOpts struct {
	RuntimeConfig         *commoncontroller.RuntimeConfig
	DockerSecretRetriever *secrets.DockerSecretIndexer
	SSHKeyManager         *secret.SSHKeyManager
	RBACManager           *rbac.Manager
	ScaleClient           scale.ScalesGetter
	OperatorImage         string
	OperatorPullPolicy    v1.PullPolicy
}

// SetupControllers sets up the core controllers. It returns the name of the
// controller that failed to create and an error, if any.
func SetupControllers(mgr ctrl.Manager, cfg *configapi.OperatorConfiguration, opts SetupControllersOpts) (string, error) {
	if err := (&DynamoComponentDeploymentReconciler{
		Client:                mgr.GetClient(),
		Recorder:              mgr.GetEventRecorderFor("dynamocomponentdeployment"),
		Config:                cfg,
		RuntimeConfig:         opts.RuntimeConfig,
		DockerSecretRetriever: opts.DockerSecretRetriever,
	}).SetupWithManager(mgr); err != nil {
		return "dynamocomponentdeployment", err
	}

	if err := (&DynamoGraphDeploymentReconciler{
		Client:                mgr.GetClient(),
		Recorder:              mgr.GetEventRecorderFor("dynamographdeployment"),
		Config:                cfg,
		RuntimeConfig:         opts.RuntimeConfig,
		RestConfig:            mgr.GetConfig(),
		DockerSecretRetriever: opts.DockerSecretRetriever,
		ScaleClient:           opts.ScaleClient,
		SSHKeyManager:         opts.SSHKeyManager,
		RBACManager:           opts.RBACManager,
	}).SetupWithManager(mgr); err != nil {
		return "dynamographdeployment", err
	}

	if err := (&DynamoGraphDeploymentScalingAdapterReconciler{
		Client:        mgr.GetClient(),
		Scheme:        mgr.GetScheme(),
		Recorder:      mgr.GetEventRecorderFor("dgdscalingadapter"),
		Config:        cfg,
		RuntimeConfig: opts.RuntimeConfig,
	}).SetupWithManager(mgr); err != nil {
		return "dgdscalingadapter", err
	}

	if err := (&DynamoGraphDeploymentRequestReconciler{
		Client:                  mgr.GetClient(),
		APIReader:               mgr.GetAPIReader(),
		Recorder:                mgr.GetEventRecorderFor("dynamographdeploymentrequest"),
		Config:                  cfg,
		RuntimeConfig:           opts.RuntimeConfig,
		GPUDiscoveryCache:       gpu.NewGPUDiscoveryCache(),
		GPUDiscovery:            gpu.NewGPUDiscovery(gpu.ScrapeMetricsEndpoint),
		OperatorImage:           opts.OperatorImage,
		OperatorImagePullPolicy: opts.OperatorPullPolicy,
		RBACManager:             opts.RBACManager,
	}).SetupWithManager(mgr); err != nil {
		return "dynamographdeploymentrequest", err
	}

	if err := (&DynamoModelReconciler{
		Client:         mgr.GetClient(),
		Recorder:       mgr.GetEventRecorderFor("dynamomodel"),
		EndpointClient: modelendpoint.NewClient(),
		Config:         cfg,
		RuntimeConfig:  opts.RuntimeConfig,
	}).SetupWithManager(mgr); err != nil {
		return "dynamomodel", err
	}

	if err := (&CheckpointReconciler{
		Client:        mgr.GetClient(),
		Config:        cfg,
		RuntimeConfig: opts.RuntimeConfig,
		Recorder:      mgr.GetEventRecorderFor("checkpoint"),
	}).SetupWithManager(mgr); err != nil {
		return "checkpoint", err
	}

	if err := (&PodSnapshotReconciler{
		Client:        mgr.GetClient(),
		Config:        cfg,
		RuntimeConfig: opts.RuntimeConfig,
		Recorder:      mgr.GetEventRecorderFor("snapshot"),
	}).SetupWithManager(mgr); err != nil {
		return "podsnapshot", err
	}

	if opts.RuntimeConfig.GroveEnabled {
		if err := NewFailoverCascadeReconciler(
			mgr.GetClient(),
			mgr.GetEventRecorderFor("gms-failover-cascade"),
		).SetupWithManager(mgr); err != nil {
			return "gms-failover-cascade", err
		}
	}

	if err := (&TopologyLabelReconciler{
		Client:        mgr.GetClient(),
		NodeReader:    mgr.GetAPIReader(),
		Config:        cfg,
		RuntimeConfig: opts.RuntimeConfig,
		Recorder:      mgr.GetEventRecorderFor("topology-label"),
	}).SetupWithManager(mgr); err != nil {
		return "topology-label", err
	}

	return "", nil
}
