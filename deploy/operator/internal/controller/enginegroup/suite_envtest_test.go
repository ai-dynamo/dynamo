//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"os"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorenv"
	webhooksetup "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/setup"
	ctrl "sigs.k8s.io/controller-runtime"
)

var sharedEnv = operatorenv.New(operatorenv.Options{
	Admission:     operatorenv.AdmissionWebhooks{Mutating: true, Validating: true},
	SetupWebhooks: setupProductionWebhooks,
})

func setupProductionWebhooks(mgr ctrl.Manager, opts operatorenv.WebhookSetupOptions) error {
	return webhooksetup.Setup(mgr, webhooksetup.Options{
		Config:            opts.OperatorConfig,
		RuntimeConfig:     opts.RuntimeConfig,
		OperatorVersion:   opts.OperatorVersion,
		OperatorPrincipal: opts.OperatorPrincipal,
	})
}

func TestMain(m *testing.M) {
	os.Exit(sharedEnv.RunM(m))
}
