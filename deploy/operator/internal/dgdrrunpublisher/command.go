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

package dgdrrunpublisher

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
)

// Subcommand is the operator-manager subcommand that runs one Run publisher.
const Subcommand = "dgdr-run-publisher"

// RunSubcommand parses args and runs the publisher until it reaches a terminal outcome.
func RunSubcommand(args []string) int {
	var cfg Config
	var terminationLog string
	flags := flag.NewFlagSet(Subcommand, flag.ContinueOnError)
	logOptions := zap.Options{}
	logOptions.BindFlags(flags)
	flags.StringVar(&cfg.SnapshotDir, "snapshot-dir", "", "directory shared with the Sweeper container")
	flags.StringVar(&cfg.RunName, "run-name", "", "name of the DynamoGraphDeploymentRun")
	flags.StringVar(&cfg.RunUID, "run-uid", "", "UID of the DynamoGraphDeploymentRun this pod serves")
	flags.StringVar(&cfg.SweeperContainer, "sweeper-container", "sweeper", "name of the Sweeper container in this pod")
	flags.DurationVar(&cfg.MinPollInterval, "poll-interval", time.Second, "snapshot check interval after a change")
	flags.DurationVar(
		&cfg.MaxPollInterval, "max-poll-interval", 5*time.Second,
		"longest snapshot check interval while the snapshot is unchanged",
	)
	flags.StringVar(
		&terminationLog, "termination-log", DefaultTerminationLog,
		"file the failure outcome is written to as the container termination message",
	)

	// Keep the helper's flags isolated from the manager's global flag set.
	if err := flags.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return ExitAcknowledged
		}
		return ExitReconcileFailed
	}
	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&logOptions)))

	// Resolve the downward-API identity supplied by the Run Job.
	cfg.Namespace, cfg.PodName = os.Getenv("POD_NAMESPACE"), os.Getenv("POD_NAME")
	if cfg.SnapshotDir == "" || cfg.RunName == "" || cfg.RunUID == "" || cfg.Namespace == "" || cfg.PodName == "" {
		fmt.Fprintln(os.Stderr, "--snapshot-dir, --run-name, --run-uid and the POD_NAMESPACE/POD_NAME "+
			"environment variables are required")
		return ExitReconcileFailed
	}

	// Stop the pod-scoped controller cleanly when Kubernetes terminates the sidecar.
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()

	err := Run(ctx, ctrl.GetConfigOrDie(), cfg)
	if err != nil {
		ctrl.Log.WithName(Subcommand).Error(err, "publisher stopped")
		if writeErr := WriteTerminationMessage(terminationLog, err); writeErr != nil {
			ctrl.Log.WithName(Subcommand).Error(writeErr, "writing termination message")
		}
	}
	return ExitCode(err)
}
