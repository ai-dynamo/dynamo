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

package disaggregatedset

import (
	"context"
	"flag"
	"fmt"
	"strings"
	"testing"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/discovery"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/clientcmd"
	"sigs.k8s.io/controller-runtime/pkg/client"
	disaggregatedsetv1 "sigs.k8s.io/lws/api/disaggregatedset/v1"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
)

var (
	flagNamespace      string
	flagLWSVersion     string
	flagVolcanoVersion string
	flagWorkloadImage  string
	flagKubeconfig     string
	flagReadyTimeout   time.Duration

	k8sClient       client.Client
	discoveryClient discovery.DiscoveryInterface
	ctx             context.Context
	cancel          context.CancelFunc
)

func init() {
	flag.StringVar(&flagNamespace, "disaggregatedset-namespace", "", "namespace for test resources (required)")
	flag.StringVar(&flagLWSVersion, "disaggregatedset-lws-version", "v0.9.0", "required LWS controller image version")
	flag.StringVar(
		&flagVolcanoVersion,
		"disaggregatedset-volcano-version",
		"v1.14.0",
		"required Volcano scheduler image version",
	)
	flag.StringVar(&flagWorkloadImage, "disaggregatedset-workload-image", "busybox:1.36", "CPU-only test workload image")
	flag.StringVar(
		&flagKubeconfig,
		"disaggregatedset-kubeconfig",
		"",
		"path to kubeconfig (uses the default loading rules when empty)",
	)
	flag.DurationVar(
		&flagReadyTimeout,
		"disaggregatedset-ready-timeout",
		5*time.Minute,
		"maximum wait for each rollout step",
	)
}

func TestDisaggregatedSet(t *testing.T) {
	RegisterFailHandler(Fail)
	_, _ = fmt.Fprintln(GinkgoWriter, "Starting DisaggregatedSet e2e suite")
	RunSpecs(t, "DisaggregatedSet e2e Suite")
}

var _ = BeforeSuite(func() {
	if flagNamespace == "" {
		Skip("--disaggregatedset-namespace is required")
	}
	if flagLWSVersion == "" {
		Skip("--disaggregatedset-lws-version is required")
	}
	if flagVolcanoVersion == "" {
		Skip("--disaggregatedset-volcano-version is required")
	}

	ctx, cancel = context.WithCancel(context.Background())

	loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
	if flagKubeconfig != "" {
		loadingRules.ExplicitPath = flagKubeconfig
	}
	kubeConfig := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(loadingRules, nil)
	restConfig, err := kubeConfig.ClientConfig()
	Expect(err).NotTo(HaveOccurred(), "failed to load kubeconfig")

	scheme := runtime.NewScheme()
	Expect(clientgoscheme.AddToScheme(scheme)).To(Succeed())
	Expect(apiextensionsv1.AddToScheme(scheme)).To(Succeed())
	Expect(nvidiacomv1beta1.AddToScheme(scheme)).To(Succeed())
	Expect(disaggregatedsetv1.AddToScheme(scheme)).To(Succeed())
	Expect(leaderworkersetv1.AddToScheme(scheme)).To(Succeed())

	k8sClient, err = client.New(restConfig, client.Options{Scheme: scheme})
	Expect(err).NotTo(HaveOccurred(), "failed to create Kubernetes client")
	discoveryClient, err = discovery.NewDiscoveryClientForConfig(restConfig)
	Expect(err).NotTo(HaveOccurred(), "failed to create discovery client")

	By("verifying the live cluster serves Dynamo, DisaggregatedSet, and LeaderWorkerSet APIs")
	Expect(apiResourceExists("nvidia.com/v1beta1", "dynamographdeployments")).To(BeTrue())
	Expect(apiResourceExists("disaggregatedset.x-k8s.io/v1", "disaggregatedsets")).To(BeTrue())
	Expect(apiResourceExists("leaderworkerset.x-k8s.io/v1", "leaderworkersets")).To(BeTrue())
	Expect(apiResourceExists("scheduling.volcano.sh/v1beta1", "podgroups")).To(BeTrue())

	By("verifying the requested LWS controller version is running")
	Eventually(func() string { return findReadyControllerImage("/lws") }, time.Minute, 2*time.Second).
		Should(ContainSubstring(":"+flagLWSVersion), "no running LWS controller image uses version %s", flagLWSVersion)

	By("verifying the requested Volcano scheduler version is running")
	Eventually(func() string { return findReadyControllerImage("/vc-scheduler") }, time.Minute, 2*time.Second).
		Should(
			ContainSubstring(":"+flagVolcanoVersion),
			"no running Volcano scheduler image uses version %s",
			flagVolcanoVersion,
		)

	By("ensuring the test namespace exists")
	namespace := &corev1.Namespace{}
	err = k8sClient.Get(ctx, client.ObjectKey{Name: flagNamespace}, namespace)
	if err == nil && !namespace.DeletionTimestamp.IsZero() {
		Eventually(func() bool {
			err := k8sClient.Get(ctx, client.ObjectKey{Name: flagNamespace}, &corev1.Namespace{})
			return apierrors.IsNotFound(err)
		}, flagReadyTimeout, time.Second).Should(BeTrue(), "timed out waiting for the terminating test namespace")
		err = k8sClient.Get(ctx, client.ObjectKey{Name: flagNamespace}, namespace)
	}
	if apierrors.IsNotFound(err) {
		namespace = &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{
			Name:   flagNamespace,
			Labels: map[string]string{"test.dynamo/managed": "true"},
		}}
		Expect(k8sClient.Create(ctx, namespace)).To(Succeed())
	} else {
		Expect(err).NotTo(HaveOccurred())
	}

	_, _ = fmt.Fprintf(
		GinkgoWriter,
		"DisaggregatedSet e2e: namespace=%s lwsVersion=%s volcanoVersion=%s workloadImage=%s\n",
		flagNamespace,
		flagLWSVersion,
		flagVolcanoVersion,
		flagWorkloadImage,
	)
})

var _ = AfterSuite(func() {
	if cancel != nil {
		cancel()
	}
})

func apiResourceExists(groupVersion, resource string) bool {
	resources, err := discoveryClient.ServerResourcesForGroupVersion(groupVersion)
	if err != nil {
		return false
	}
	for _, candidate := range resources.APIResources {
		if candidate.Name == resource {
			return true
		}
	}
	return false
}

func findReadyControllerImage(imageFragment string) string {
	pods := &corev1.PodList{}
	if err := k8sClient.List(ctx, pods); err != nil {
		return ""
	}
	for i := range pods.Items {
		if pods.Items[i].Status.Phase != corev1.PodRunning || !podReady(&pods.Items[i]) {
			continue
		}
		for _, container := range pods.Items[i].Spec.Containers {
			if strings.Contains(container.Image, imageFragment) {
				return container.Image
			}
		}
	}
	return ""
}

func podReady(pod *corev1.Pod) bool {
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady {
			return condition.Status == corev1.ConditionTrue
		}
	}
	return false
}
