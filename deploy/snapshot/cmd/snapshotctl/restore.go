package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"

	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

type restoreOptions struct {
	ManifestPath string
	PodName      string
	Namespace    string
	KubeContext  string
	CheckpointID string
	Timeout      time.Duration
}

func runRestoreFlow(ctx context.Context, opts restoreOptions) (*result, error) {
	createPodFromManifest := strings.TrimSpace(opts.ManifestPath) != ""
	targetExistingPod := strings.TrimSpace(opts.PodName) != ""
	if createPodFromManifest == targetExistingPod {
		return nil, fmt.Errorf("restore requires exactly one of --manifest or --pod")
	}
	if strings.TrimSpace(opts.CheckpointID) == "" {
		return nil, fmt.Errorf("missing required flags: --checkpoint-id")
	}
	if opts.Timeout <= 0 {
		return nil, fmt.Errorf("--timeout must be greater than zero")
	}

	checkpointID := strings.TrimSpace(opts.CheckpointID)
	clientset, currentNamespace, err := loadClientset(opts.KubeContext)
	if err != nil {
		return nil, err
	}
	namespace := currentNamespace
	if namespace == "" {
		namespace = corev1.NamespaceDefault
	}
	if strings.TrimSpace(opts.Namespace) != "" {
		namespace = strings.TrimSpace(opts.Namespace)
	}

	podName := strings.TrimSpace(opts.PodName)
	pod := &corev1.Pod{}
	if createPodFromManifest {
		pod, err = loadPod(opts.ManifestPath)
		if err != nil {
			return nil, err
		}
		if strings.TrimSpace(pod.Namespace) != "" && strings.TrimSpace(opts.Namespace) == "" {
			namespace = strings.TrimSpace(pod.Namespace)
		}
		podName = pod.Name
	}

	storage, err := discoverSnapshotStorage(ctx, clientset, namespace)
	if err != nil {
		return nil, err
	}
	resolvedStorage, err := snapshotprotocol.ResolveRestoreStorage(checkpointID, snapshotprotocol.DefaultCheckpointArtifactVersion, "", snapshotprotocol.Storage{
		Type:     snapshotprotocol.StorageTypePVC,
		PVCName:  storage.PVCName,
		BasePath: storage.BasePath,
	})
	if err != nil {
		return nil, err
	}

	if createPodFromManifest {
		restorePod := snapshotprotocol.NewRestorePod(&corev1.Pod{
			TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Pod"},
			ObjectMeta: metav1.ObjectMeta{
				Name:        pod.Name,
				Labels:      pod.Labels,
				Annotations: pod.Annotations,
			},
			Spec: *pod.Spec.DeepCopy(),
		}, snapshotprotocol.PodOptions{
			Namespace:       namespace,
			CheckpointID:    checkpointID,
			ArtifactVersion: snapshotprotocol.DefaultCheckpointArtifactVersion,
			Storage:         resolvedStorage,
			SeccompProfile:  snapshotprotocol.DefaultSeccompLocalhostProfile,
		})
		_, err = clientset.CoreV1().Pods(namespace).Create(ctx, restorePod, metav1.CreateOptions{})
		if apierrors.IsAlreadyExists(err) {
			return nil, fmt.Errorf("restore pod %s/%s already exists", namespace, pod.Name)
		}
		if err != nil {
			return nil, err
		}
	} else {
		pod, err = clientset.CoreV1().Pods(namespace).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			return nil, fmt.Errorf("get restore target pod %s/%s: %w", namespace, podName, err)
		}
		if len(pod.Spec.Containers) == 0 {
			return nil, fmt.Errorf("restore target pod %s/%s has no containers", namespace, podName)
		}
		if err := snapshotprotocol.ValidateRestorePodSpec(&pod.Spec, resolvedStorage, snapshotprotocol.DefaultSeccompLocalhostProfile); err != nil {
			return nil, fmt.Errorf("restore target pod %s/%s is not snapshot-compatible: %w", namespace, podName, err)
		}

		labels := map[string]string{}
		for key, value := range pod.Labels {
			labels[key] = value
		}
		annotations := map[string]string{}
		for key, value := range pod.Annotations {
			annotations[key] = value
		}
		snapshotprotocol.ApplyRestoreTargetMetadata(labels, annotations, true, checkpointID, snapshotprotocol.DefaultCheckpointArtifactVersion)
		patch, err := json.Marshal(map[string]any{
			"metadata": map[string]any{
				"labels":      labels,
				"annotations": annotations,
			},
		})
		if err != nil {
			return nil, fmt.Errorf("encode restore target metadata patch: %w", err)
		}
		if _, err := clientset.CoreV1().Pods(namespace).Patch(ctx, podName, types.MergePatchType, patch, metav1.PatchOptions{}); err != nil {
			return nil, fmt.Errorf("patch restore target pod %s/%s: %w", namespace, podName, err)
		}
	}

	waitCtx, cancel := context.WithTimeout(ctx, opts.Timeout)
	defer cancel()
	status, err := waitForRestore(waitCtx, clientset, namespace, podName)
	if err != nil {
		return nil, err
	}

	return &result{
		Name:               podName,
		Namespace:          namespace,
		CheckpointID:       checkpointID,
		CheckpointLocation: resolvedStorage.Location,
		RestorePod:         podName,
		Status:             status,
	}, nil
}

func waitForRestore(ctx context.Context, clientset kubernetes.Interface, namespace string, podName string) (string, error) {
	var status string
	err := watchNamedObject(
		ctx,
		podName,
		&corev1.Pod{},
		func(ctx context.Context, options metav1.ListOptions) (runtime.Object, error) {
			return clientset.CoreV1().Pods(namespace).List(ctx, options)
		},
		func(ctx context.Context, options metav1.ListOptions) (watch.Interface, error) {
			return clientset.CoreV1().Pods(namespace).Watch(ctx, options)
		},
		func(event watch.Event) (bool, error) {
			if event.Type == watch.Error {
				return false, apierrors.FromObject(event.Object)
			}

			pod, ok := event.Object.(*corev1.Pod)
			if !ok {
				return false, fmt.Errorf("unexpected restore watch object %T", event.Object)
			}

			status = strings.TrimSpace(pod.Annotations[snapshotprotocol.RestoreStatusAnnotation])
			if status == snapshotprotocol.RestoreStatusCompleted {
				return true, nil
			}
			if status == snapshotprotocol.RestoreStatusFailed {
				return false, fmt.Errorf("restore pod %s/%s failed", namespace, podName)
			}
			if pod.Status.Phase == corev1.PodFailed {
				return false, fmt.Errorf("restore pod %s/%s entered phase Failed (%s)", namespace, podName, pod.Status.Reason)
			}
			return false, nil
		},
	)
	if err != nil {
		if !errors.Is(err, context.DeadlineExceeded) {
			return "", err
		}
		return "", fmt.Errorf("restore pod %s/%s timed out: %s", namespace, podName, restoreTimeoutSummary(clientset, namespace, podName, status))
	}
	return status, nil
}

func restoreTimeoutSummary(clientset kubernetes.Interface, namespace string, podName string, status string) string {
	summaryCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	pod, err := clientset.CoreV1().Pods(namespace).Get(summaryCtx, podName, metav1.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			return fmt.Sprintf("restore_status=%q pod not found", status)
		}
		return "unable to get restore pod: " + err.Error()
	}

	parts := []string{
		fmt.Sprintf("restore_status=%q", status),
		fmt.Sprintf("pod=%s phase=%s", pod.Name, pod.Status.Phase),
	}
	if reason := strings.TrimSpace(pod.Status.Reason); reason != "" {
		parts = append(parts, fmt.Sprintf("reason=%s", reason))
	}
	for _, condition := range pod.Status.Conditions {
		if condition.Status == corev1.ConditionTrue || condition.Status == corev1.ConditionFalse {
			parts = append(parts, fmt.Sprintf("%s=%s", condition.Type, condition.Status))
		}
	}
	for _, containerStatus := range pod.Status.ContainerStatuses {
		if containerStatus.State.Waiting != nil {
			parts = append(parts, fmt.Sprintf("container=%s waiting=%s", containerStatus.Name, containerStatus.State.Waiting.Reason))
		}
		if containerStatus.State.Terminated != nil {
			parts = append(parts, fmt.Sprintf("container=%s terminated=%s", containerStatus.Name, containerStatus.State.Terminated.Reason))
		}
	}
	return strings.Join(parts, " ")
}
