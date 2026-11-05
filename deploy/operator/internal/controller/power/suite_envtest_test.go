//go:build !clustertest

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package power

import (
	"context"
	"os"
	"testing"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gpu"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorenv"
	webhookdefaulting "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/defaulting"
	webhooksetup "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/setup"
	ctrl "sigs.k8s.io/controller-runtime"
)

type fixedHardwareDiscoverer struct{}

func (fixedHardwareDiscoverer) Discover(context.Context, v1beta1.GPUSKUType) (*gpu.GPUInfo, error) {
	return &gpu.GPUInfo{System: "h200_sxm", GPUsPerNode: 8, NodesWithGPUs: 1, TotalGPUs: 8, SelectedShapeGPUs: 8}, nil
}

// sharedEnv serves CRD conversion without admission, so fixtures describe observed state directly.
var sharedEnv = operatorenv.New(operatorenv.Options{
	SetupWebhooks: func(mgr ctrl.Manager, opts operatorenv.WebhookSetupOptions) error {
		return webhooksetup.Setup(mgr, webhooksetup.Options{
			Config:            opts.OperatorConfig,
			RuntimeConfig:     opts.RuntimeConfig,
			OperatorVersion:   opts.OperatorVersion,
			OperatorPrincipal: opts.OperatorPrincipal,
			DGDRV1beta2:       webhookdefaulting.DGDRV1beta2Options{Hardware: fixedHardwareDiscoverer{}},
		})
	},
})

func TestMain(m *testing.M) {
	os.Exit(sharedEnv.RunM(m))
}
