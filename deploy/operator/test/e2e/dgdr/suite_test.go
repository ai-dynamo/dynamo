/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package dgdr

import (
	"context"
	"flag"
	"fmt"
	"testing"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/clientcmd"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// CLI flags — mirrors the Python --dgdr-* options.
var (
	flagNamespace        string
	flagImage            string
	flagModel            string
	flagBackend          string
	flagNoMocker         bool
	flagProfilingTimeout int
	flagDeployTimeout    int
	flagKubeconfig       string
)

// Package-level clients initialised in BeforeSuite.
var (
	k8sClient   client.Client
	typedClient kubernetes.Interface
	ctx         context.Context
	cancel      context.CancelFunc
)

func init() {
	flag.StringVar(&flagNamespace, "dgdr-namespace", "", "Kubernetes namespace for DGDR resources (required)")
	flag.StringVar(&flagImage, "dgdr-image", "", "Container image for profiling/deployment workers (required)")
	flag.StringVar(&flagModel, "dgdr-model", "Qwen/Qwen3-0.6B", "HuggingFace model ID")
	flag.StringVar(&flagBackend, "dgdr-backend", "vllm", "Default backend (auto|vllm|sglang|trtllm)")
	flag.BoolVar(&flagNoMocker, "dgdr-no-mocker", false, "Disable mocker mode (requires real GPUs)")
	flag.IntVar(&flagProfilingTimeout, "dgdr-profiling-timeout", 3600, "Max seconds to wait for profiling")
	flag.IntVar(&flagDeployTimeout, "dgdr-deploy-timeout", 600, "Max seconds to wait for deployment")
	flag.StringVar(&flagKubeconfig, "kubeconfig", "", "Path to kubeconfig (uses default if empty)")
}

func useMocker() bool { return !flagNoMocker }

func TestDGDR(t *testing.T) {
	RegisterFailHandler(Fail)
	_, _ = fmt.Fprintf(GinkgoWriter, "Starting DGDR e2e suite\n")
	RunSpecs(t, "DGDR e2e Suite")
}

var _ = BeforeSuite(func() {
	if flagNamespace == "" {
		Skip("--dgdr-namespace is required to run DGDR tests")
	}
	if flagImage == "" {
		Skip("--dgdr-image is required to run DGDR tests")
	}

	ctx, cancel = context.WithCancel(context.Background())

	// Build kubeconfig
	loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
	if flagKubeconfig != "" {
		loadingRules.ExplicitPath = flagKubeconfig
	}
	kubeConfig := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(loadingRules, nil)
	restCfg, err := kubeConfig.ClientConfig()
	Expect(err).NotTo(HaveOccurred(), "failed to load kubeconfig")

	// Register CRD scheme
	scheme := runtime.NewScheme()
	Expect(clientgoscheme.AddToScheme(scheme)).To(Succeed())
	Expect(v1alpha1.AddToScheme(scheme)).To(Succeed())
	Expect(v1beta1.AddToScheme(scheme)).To(Succeed())

	// controller-runtime typed client (for DGDR CRs)
	k8sClient, err = client.New(restCfg, client.Options{Scheme: scheme})
	Expect(err).NotTo(HaveOccurred())

	// standard clientset (for ConfigMaps, CRDs, etc.)
	typedClient, err = kubernetes.NewForConfig(restCfg)
	Expect(err).NotTo(HaveOccurred())

	_, _ = fmt.Fprintf(GinkgoWriter, "DGDR e2e: namespace=%s image=%s model=%s mocker=%v\n",
		flagNamespace, flagImage, flagModel, useMocker())
})

var _ = AfterSuite(func() {
	if cancel != nil {
		cancel()
	}
})
