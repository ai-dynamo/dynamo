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
)

type snapshotStorage struct {
	PVCName  string
	BasePath string
}

func loadRunContext(ctx context.Context, manifestPath string, namespaceOverride string, kubeContext string) (*corev1.Pod, kubernetes.Interface, string, snapshotStorage, error) {
	pod, err := loadPod(manifestPath)
	if err != nil {
		return nil, nil, "", snapshotStorage{}, err
	}

	clientset, currentNamespace, err := loadClientset(kubeContext)
	if err != nil {
		return nil, nil, "", snapshotStorage{}, err
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
		return nil, nil, "", snapshotStorage{}, err
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

func discoverSnapshotStorage(ctx context.Context, clientset kubernetes.Interface, namespace string) (snapshotStorage, error) {
	daemonSets, err := clientset.AppsV1().DaemonSets(namespace).List(ctx, metav1.ListOptions{
		LabelSelector: "app.kubernetes.io/name=snapshot",
	})
	if err != nil {
		return snapshotStorage{}, fmt.Errorf("list snapshot-agent daemonsets in %s: %w", namespace, err)
	}
	if len(daemonSets.Items) == 0 {
		return snapshotStorage{}, fmt.Errorf("no snapshot-agent daemonset found in namespace %s", namespace)
	}

	for _, daemonSet := range daemonSets.Items {
		mountPaths := map[string]string{}
		for _, container := range daemonSet.Spec.Template.Spec.Containers {
			if container.Name != "agent" {
				continue
			}
			for _, mount := range container.VolumeMounts {
				if strings.TrimSpace(mount.MountPath) == "" {
					continue
				}
				mountPaths[mount.Name] = mount.MountPath
			}
		}

		for _, volume := range daemonSet.Spec.Template.Spec.Volumes {
			if volume.PersistentVolumeClaim == nil {
				continue
			}
			basePath, ok := mountPaths[volume.Name]
			if !ok || strings.TrimSpace(basePath) == "" {
				continue
			}
			claimName := strings.TrimSpace(volume.PersistentVolumeClaim.ClaimName)
			if claimName == "" {
				continue
			}
			return snapshotStorage{
				PVCName:  claimName,
				BasePath: strings.TrimRight(basePath, "/"),
			}, nil
		}
	}

	names := make([]string, 0, len(daemonSets.Items))
	for _, daemonSet := range daemonSets.Items {
		names = append(names, daemonSet.Name)
	}
	return snapshotStorage{}, fmt.Errorf(
		"snapshot-agent daemonset in %s does not mount a PVC-backed checkpoint volume (%s)",
		namespace,
		strings.Join(names, ", "),
	)
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
	if len(pod.Spec.Containers) != 1 {
		return nil, fmt.Errorf(
			"manifest %s has %d containers; snapshotctl requires exactly one worker container",
			manifestPath,
			len(pod.Spec.Containers),
		)
	}
	if strings.TrimSpace(pod.Spec.Containers[0].Image) == "" {
		return nil, fmt.Errorf("manifest %s: worker container image is required", manifestPath)
	}
	if strings.TrimSpace(pod.Name) == "" {
		return nil, fmt.Errorf("manifest %s: metadata.name is required", manifestPath)
	}

	pod.Namespace = strings.TrimSpace(pod.Namespace)
	return &pod, nil
}
