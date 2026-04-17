package main

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	"sigs.k8s.io/yaml"

	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

func loadRunContext(ctx context.Context, manifestPath string, namespaceOverride string, kubeContext string) (*corev1.Pod, kubernetes.Interface, string, snapshotprotocol.Storage, error) {
	pod, err := loadPod(manifestPath)
	if err != nil {
		return nil, nil, "", snapshotprotocol.Storage{}, err
	}

	clientset, currentNamespace, err := loadClientset(kubeContext)
	if err != nil {
		return nil, nil, "", snapshotprotocol.Storage{}, err
	}

	namespace := currentNamespace
	if namespace == "" {
		namespace = corev1.NamespaceDefault
	}
	if pod.Namespace != "" {
		namespace = pod.Namespace
	}
	if namespaceOverride != "" {
		namespace = namespaceOverride
	}

	storage, err := discoverSnapshotStorage(ctx, clientset, namespace)
	if err != nil {
		return nil, nil, "", snapshotprotocol.Storage{}, err
	}
	return pod, clientset, namespace, storage, nil
}

func loadClientset(kubeContext string) (kubernetes.Interface, string, error) {
	loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
	clientConfig := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(loadingRules, &clientcmd.ConfigOverrides{
		CurrentContext: strings.TrimSpace(kubeContext),
	})
	restConfig, err := clientConfig.ClientConfig()
	if err != nil {
		return nil, "", fmt.Errorf("load kubeconfig: %w", err)
	}
	restConfig.Timeout = 30 * time.Second

	namespace, _, err := clientConfig.Namespace()
	if err != nil {
		return nil, "", fmt.Errorf("resolve current namespace: %w", err)
	}
	if strings.TrimSpace(namespace) == "" {
		namespace = corev1.NamespaceDefault
	}

	clientset, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		return nil, "", fmt.Errorf("create kubernetes client: %w", err)
	}
	return clientset, namespace, nil
}

func discoverSnapshotStorage(ctx context.Context, clientset kubernetes.Interface, namespace string) (snapshotprotocol.Storage, error) {
	daemonSets, err := clientset.AppsV1().DaemonSets(namespace).List(ctx, metav1.ListOptions{
		LabelSelector: snapshotprotocol.SnapshotAgentLabelSelector,
	})
	if err != nil {
		return snapshotprotocol.Storage{}, fmt.Errorf("list snapshot-agent daemonsets in %s: %w", namespace, err)
	}

	return snapshotprotocol.DiscoverStorageFromDaemonSets(namespace, daemonSets.Items)
}

func loadPod(manifestPath string) (*corev1.Pod, error) {
	content, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, fmt.Errorf("read manifest %s: %w", manifestPath, err)
	}

	var pod corev1.Pod
	if err := yaml.Unmarshal(content, &pod); err != nil {
		return nil, fmt.Errorf("parse manifest %s: %w", manifestPath, err)
	}
	if kind := strings.TrimSpace(pod.Kind); kind != "" && kind != "Pod" {
		return nil, fmt.Errorf("manifest %s is kind %q, expected Pod", manifestPath, kind)
	}
	workerContainer := snapshotprotocol.ResolveMainContainer(&pod.Spec)
	if workerContainer == nil {
		return nil, fmt.Errorf(
			"manifest %s has no worker containers; snapshotctl requires at least one worker container",
			manifestPath,
		)
	}
	// In multi-container manifests the author must pick the worker explicitly
	// by naming it "main"; otherwise the ResolveMainContainer fallback would
	// silently target whichever container happens to be first.
	if len(pod.Spec.Containers) > 1 && workerContainer.Name != snapshotprotocol.MainContainerName {
		return nil, fmt.Errorf(
			"manifest %s has %d containers; snapshotctl requires a worker container named %q",
			manifestPath,
			len(pod.Spec.Containers),
			snapshotprotocol.MainContainerName,
		)
	}
	if strings.TrimSpace(workerContainer.Image) == "" {
		return nil, fmt.Errorf("manifest %s: worker container image is required", manifestPath)
	}
	if strings.TrimSpace(pod.Name) == "" {
		return nil, fmt.Errorf("manifest %s: metadata.name is required", manifestPath)
	}

	pod.Namespace = strings.TrimSpace(pod.Namespace)
	return &pod, nil
}
