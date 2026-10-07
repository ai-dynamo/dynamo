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

// Command dgdr-run-publisher is the sidecar of a DGDR(v2) run Job. It watches the
// Sweeper's pod-local snapshot file and reconciles it into DynamoGraphDeploymentCandidates
// and DynamoGraphDeploymentRun.status. See internal/dgdrrunpublisher.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdrrunpublisher"
	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdrrunpublisher/placeholderapi"
	"k8s.io/apimachinery/pkg/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func main() {
	os.Exit(run())
}

func run() int {
	var (
		snapshotDir      string
		runName          string
		sweeperContainer string
		pollInterval     time.Duration
	)
	flag.StringVar(&snapshotDir, "snapshot-dir", "", "directory shared with the Sweeper container")
	flag.StringVar(&runName, "run-name", "", "name of the DynamoGraphDeploymentRun")
	flag.StringVar(&sweeperContainer, "sweeper-container", "sweeper", "name of the Sweeper container in this pod")
	flag.DurationVar(&pollInterval, "poll-interval", time.Second, "snapshot poll interval")
	flag.Parse()

	namespace, podName := os.Getenv("POD_NAMESPACE"), os.Getenv("POD_NAME")
	if snapshotDir == "" || runName == "" || namespace == "" || podName == "" {
		fmt.Fprintln(os.Stderr, "--snapshot-dir, --run-name and the POD_NAMESPACE/POD_NAME environment variables are required")
		return dgdrrunpublisher.ExitReconcileFailed
	}

	scheme := runtime.NewScheme()
	for _, add := range []func(*runtime.Scheme) error{clientgoscheme.AddToScheme, v1beta1.AddToScheme, v1beta2.AddToScheme} {
		if err := add(scheme); err != nil {
			log.Printf("building scheme: %v", err)
			return dgdrrunpublisher.ExitReconcileFailed
		}
	}
	kube, err := client.New(ctrl.GetConfigOrDie(), client.Options{Scheme: scheme})
	if err != nil {
		log.Printf("creating client: %v", err)
		return dgdrrunpublisher.ExitReconcileFailed
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()

	publisher := &dgdrrunpublisher.Publisher{
		Cluster: &dgdrrunpublisher.KubeCluster{
			Client:           kube,
			Namespace:        namespace,
			RunName:          runName,
			PodName:          podName,
			SweeperContainer: sweeperContainer,
		},
		SnapshotDir:  snapshotDir,
		RunName:      runName,
		PollInterval: pollInterval,
	}
	err = publisher.Run(ctx)
	if err != nil {
		log.Printf("publisher: %v", err)
	}
	return dgdrrunpublisher.ExitCode(err)
}
