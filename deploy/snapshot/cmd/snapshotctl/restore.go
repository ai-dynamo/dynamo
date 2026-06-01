package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

type restoreOptions struct {
	ManifestPath string
	PodName      string
	Namespace    string
	KubeContext  string
	CheckpointID string
	Containers   string
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
		// Stamp (or validate) the required snapshot-target-containers
		// annotation on the manifest before handing it to the protocol.
		targetValue, err := reconcileTargetContainers(pod.Annotations, opts.Containers, 1, 0)
		if err != nil {
			return nil, err
		}
		annotations := map[string]string{}
		for k, v := range pod.Annotations {
			annotations[k] = v
		}
		annotations[snapshotprotocol.TargetContainersAnnotation] = targetValue

		restorePod, err := snapshotprotocol.NewRestorePod(&corev1.Pod{
			TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Pod"},
			ObjectMeta: metav1.ObjectMeta{
				Name:         pod.Name,
				GenerateName: pod.GenerateName,
				Labels:       pod.Labels,
				Annotations:  annotations,
			},
			Spec: *pod.Spec.DeepCopy(),
		}, snapshotprotocol.PodOptions{
			Namespace:       namespace,
			CheckpointID:    checkpointID,
			ArtifactVersion: snapshotprotocol.DefaultCheckpointArtifactVersion,
			Storage:         resolvedStorage,
			SeccompProfile:  snapshotprotocol.DefaultSeccompLocalhostProfile,
		})
		if err != nil {
			return nil, err
		}
		restorePod, err = clientset.CoreV1().Pods(namespace).Create(ctx, restorePod, metav1.CreateOptions{})
		if apierrors.IsAlreadyExists(err) {
			return nil, fmt.Errorf("restore pod %s/%s already exists", namespace, pod.Name)
		}
		if err != nil {
			return nil, err
		}
		podName = restorePod.Name
	} else {
		pod, err = clientset.CoreV1().Pods(namespace).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			return nil, fmt.Errorf("get restore target pod %s/%s: %w", namespace, podName, err)
		}
		if len(pod.Spec.Containers) == 0 {
			return nil, fmt.Errorf("restore target pod %s/%s has no containers", namespace, podName)
		}
		targetValue, err := reconcileTargetContainers(pod.Annotations, opts.Containers, 1, 0)
		if err != nil {
			return nil, err
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
		annotations[snapshotprotocol.TargetContainersAnnotation] = targetValue
		if err := snapshotprotocol.ValidateRestorePodSpec(&pod.Spec, annotations, resolvedStorage, snapshotprotocol.DefaultSeccompLocalhostProfile); err != nil {
			return nil, fmt.Errorf("restore target pod %s/%s is not snapshot-compatible: %w", namespace, podName, err)
		}
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

	return &result{
		Name:               podName,
		Namespace:          namespace,
		CheckpointID:       checkpointID,
		CheckpointLocation: resolvedStorage.Location,
		RestorePod:         podName,
		Status:             "requested",
	}, nil
}
