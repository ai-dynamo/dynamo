/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Command dgdr-run-publisher is the sidecar of a DGDR(v2) run Job. It is a
// controller-runtime controller for one DynamoGraphDeploymentRun: it observes the
// Sweeper's pod-local snapshot file and reconciles it into DynamoGraphDeploymentCandidates
// and DynamoGraphDeploymentRun.status. See internal/dgdrrunpublisher.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdrrunpublisher"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
)

func main() {
	os.Exit(run())
}

func run() int {
	var cfg dgdrrunpublisher.Config
	logOptions := zap.Options{}
	logOptions.BindFlags(flag.CommandLine)
	flag.StringVar(&cfg.SnapshotDir, "snapshot-dir", "", "directory shared with the Sweeper container")
	flag.StringVar(&cfg.RunName, "run-name", "", "name of the DynamoGraphDeploymentRun")
	flag.StringVar(&cfg.RunUID, "run-uid", "", "UID of the DynamoGraphDeploymentRun this pod serves")
	flag.StringVar(&cfg.SweeperContainer, "sweeper-container", "sweeper", "name of the Sweeper container in this pod")
	flag.DurationVar(&cfg.MinPollInterval, "poll-interval", time.Second, "snapshot check interval after a change")
	flag.DurationVar(&cfg.MaxPollInterval, "max-poll-interval", 5*time.Second, "longest snapshot check interval while the snapshot is unchanged")
	flag.Parse()
	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&logOptions)))

	cfg.Namespace, cfg.PodName = os.Getenv("POD_NAMESPACE"), os.Getenv("POD_NAME")
	if cfg.SnapshotDir == "" || cfg.RunName == "" || cfg.RunUID == "" || cfg.Namespace == "" || cfg.PodName == "" {
		fmt.Fprintln(os.Stderr, "--snapshot-dir, --run-name, --run-uid and the POD_NAMESPACE/POD_NAME environment variables are required")
		return dgdrrunpublisher.ExitReconcileFailed
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()

	err := dgdrrunpublisher.Run(ctx, ctrl.GetConfigOrDie(), cfg)
	if err != nil {
		ctrl.Log.WithName("dgdr-run-publisher").Error(err, "publisher stopped")
	}
	return dgdrrunpublisher.ExitCode(err)
}
