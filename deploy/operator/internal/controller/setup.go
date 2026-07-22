/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"fmt"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gpu"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/modelendpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/secret"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/scale"
	ctrl "sigs.k8s.io/controller-runtime"
)

type SetupOptions struct {
	Config        *configv1alpha1.OperatorConfiguration
	RuntimeConfig *commoncontroller.RuntimeConfig
}

type DynamoComponentDeploymentSetupOptions struct {
	SetupOptions
	DockerSecretRetriever DockerSecretRetriever
}

type DynamoGraphDeploymentSetupOptions struct {
	SetupOptions
	DockerSecretRetriever DockerSecretRetriever
	ScaleClient           scale.ScalesGetter
	RBACManager           RBACManager
	SSHKeyManager         *secret.SSHKeyManager
}

type DynamoGraphDeploymentRequestSetupOptions struct {
	SetupOptions
	RBACManager             RBACManager
	GPUDiscoveryCache       *gpu.GPUDiscoveryCache
	GPUDiscovery            *gpu.GPUDiscovery
	OperatorImage           string
	OperatorImagePullPolicy corev1.PullPolicy
}

type DynamoModelSetupOptions struct {
	SetupOptions
	ModelEndpointClient *modelendpoint.Client
}

func (o DynamoGraphDeploymentRequestSetupOptions) gpuDiscoveryCache() *gpu.GPUDiscoveryCache {
	if o.GPUDiscoveryCache != nil {
		return o.GPUDiscoveryCache
	}
	return gpu.NewGPUDiscoveryCache()
}

func (o DynamoGraphDeploymentRequestSetupOptions) gpuDiscovery() *gpu.GPUDiscovery {
	if o.GPUDiscovery != nil {
		return o.GPUDiscovery
	}
	return gpu.NewGPUDiscovery(gpu.ScrapeMetricsEndpoint)
}

func (o DynamoModelSetupOptions) modelEndpointClient() *modelendpoint.Client {
	if o.ModelEndpointClient != nil {
		return o.ModelEndpointClient
	}
	return modelendpoint.NewClient()
}

func SetupDynamoComponentDeployment(mgr ctrl.Manager, opts DynamoComponentDeploymentSetupOptions) error {
	if err := (&DynamoComponentDeploymentReconciler{
		Client:                mgr.GetClient(),
		Recorder:              mgr.GetEventRecorderFor("dynamocomponentdeployment"),
		Config:                opts.Config,
		RuntimeConfig:         opts.RuntimeConfig,
		DockerSecretRetriever: opts.DockerSecretRetriever,
	}).SetupWithManager(mgr); err != nil {
		return fmt.Errorf("unable to create DynamoComponentDeployment controller: %w", err)
	}
	return nil
}

func SetupDynamoGraphDeployment(mgr ctrl.Manager, opts DynamoGraphDeploymentSetupOptions) error {
	if err := (&DynamoGraphDeploymentReconciler{
		Client:                mgr.GetClient(),
		Recorder:              mgr.GetEventRecorderFor("dynamographdeployment"),
		Config:                opts.Config,
		RuntimeConfig:         opts.RuntimeConfig,
		RestConfig:            mgr.GetConfig(),
		DockerSecretRetriever: opts.DockerSecretRetriever,
		ScaleClient:           opts.ScaleClient,
		SSHKeyManager:         opts.SSHKeyManager,
		RBACManager:           opts.RBACManager,
	}).SetupWithManager(mgr); err != nil {
		return fmt.Errorf("unable to create DynamoGraphDeployment controller: %w", err)
	}
	return nil
}

func SetupDynamoGraphDeploymentScalingAdapter(mgr ctrl.Manager, opts SetupOptions) error {
	if err := (&DynamoGraphDeploymentScalingAdapterReconciler{
		Client:        mgr.GetClient(),
		Scheme:        mgr.GetScheme(),
		Recorder:      mgr.GetEventRecorderFor("dgdscalingadapter"),
		Config:        opts.Config,
		RuntimeConfig: opts.RuntimeConfig,
	}).SetupWithManager(mgr); err != nil {
		return fmt.Errorf("unable to create DGDScalingAdapter controller: %w", err)
	}
	return nil
}

func SetupDynamoGraphDeploymentRequest(mgr ctrl.Manager, opts DynamoGraphDeploymentRequestSetupOptions) error {
	if err := (&DynamoGraphDeploymentRequestReconciler{
		Client:                  mgr.GetClient(),
		APIReader:               mgr.GetAPIReader(),
		Recorder:                mgr.GetEventRecorderFor("dynamographdeploymentrequest"),
		Config:                  opts.Config,
		RuntimeConfig:           opts.RuntimeConfig,
		GPUDiscoveryCache:       opts.gpuDiscoveryCache(),
		GPUDiscovery:            opts.gpuDiscovery(),
		OperatorImage:           opts.OperatorImage,
		OperatorImagePullPolicy: opts.OperatorImagePullPolicy,
		RBACManager:             opts.RBACManager,
	}).SetupWithManager(mgr); err != nil {
		return fmt.Errorf("unable to create DynamoGraphDeploymentRequest controller: %w", err)
	}
	return nil
}

func SetupDynamoModel(mgr ctrl.Manager, opts DynamoModelSetupOptions) error {
	if err := (&DynamoModelReconciler{
		Client:         mgr.GetClient(),
		Recorder:       mgr.GetEventRecorderFor("dynamomodel"),
		EndpointClient: opts.modelEndpointClient(),
		Config:         opts.Config,
		RuntimeConfig:  opts.RuntimeConfig,
	}).SetupWithManager(mgr); err != nil {
		return fmt.Errorf("unable to create DynamoModel controller: %w", err)
	}
	return nil
}

func SetupDynamoCheckpoint(mgr ctrl.Manager, opts SetupOptions) error {
	if err := (&CheckpointReconciler{
		Client:        mgr.GetClient(),
		Config:        opts.Config,
		RuntimeConfig: opts.RuntimeConfig,
		Recorder:      mgr.GetEventRecorderFor("checkpoint"),
	}).SetupWithManager(mgr); err != nil {
		return fmt.Errorf("unable to create DynamoCheckpoint controller: %w", err)
	}
	return nil
}

func SetupPodSnapshot(mgr ctrl.Manager, opts SetupOptions) error {
	if err := (&PodSnapshotReconciler{
		Client:        mgr.GetClient(),
		Config:        opts.Config,
		RuntimeConfig: opts.RuntimeConfig,
		Recorder:      mgr.GetEventRecorderFor("snapshot"),
	}).SetupWithManager(mgr); err != nil {
		return fmt.Errorf("unable to create PodSnapshot controller: %w", err)
	}
	return nil
}

func SetupFailoverCascade(mgr ctrl.Manager) error {
	if err := NewFailoverCascadeReconciler(
		mgr.GetClient(),
		mgr.GetEventRecorderFor("gms-failover-cascade"),
	).SetupWithManager(mgr); err != nil {
		return fmt.Errorf("unable to create GMS FailoverCascade controller: %w", err)
	}
	return nil
}

func SetupTopologyLabel(mgr ctrl.Manager, opts SetupOptions) error {
	if err := (&TopologyLabelReconciler{
		Client:        mgr.GetClient(),
		NodeReader:    mgr.GetAPIReader(),
		Config:        opts.Config,
		RuntimeConfig: opts.RuntimeConfig,
		Recorder:      mgr.GetEventRecorderFor("topology-label"),
	}).SetupWithManager(mgr); err != nil {
		return fmt.Errorf("unable to create TopologyLabel controller: %w", err)
	}
	return nil
}
