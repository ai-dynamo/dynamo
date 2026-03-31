package workload

import (
	"context"
	"fmt"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	SnapshotAgentLabelKey      = "app.kubernetes.io/name"
	SnapshotAgentLabelValue    = "snapshot"
	SnapshotAgentContainerName = "agent"
	SnapshotAgentLabelSelector = SnapshotAgentLabelKey + "=" + SnapshotAgentLabelValue
)

func DiscoverStorageFromDaemonSets(namespace string, daemonSets []appsv1.DaemonSet) (protocol.Storage, error) {
	if len(daemonSets) == 0 {
		return protocol.Storage{}, fmt.Errorf("no snapshot-agent daemonset found in namespace %s", namespace)
	}

	names := make([]string, 0, len(daemonSets))
	for _, daemonSet := range daemonSets {
		names = append(names, daemonSet.Name)

		mountPaths := map[string]string{}
		for _, container := range daemonSet.Spec.Template.Spec.Containers {
			if container.Name != SnapshotAgentContainerName {
				continue
			}
			for _, mount := range container.VolumeMounts {
				if strings.TrimSpace(mount.MountPath) == "" {
					continue
				}
				mountPaths[mount.Name] = strings.TrimRight(mount.MountPath, "/")
			}
		}

		for _, volume := range daemonSet.Spec.Template.Spec.Volumes {
			if volume.PersistentVolumeClaim == nil {
				continue
			}

			basePath, ok := mountPaths[volume.Name]
			if !ok || basePath == "" {
				continue
			}

			pvcName := strings.TrimSpace(volume.PersistentVolumeClaim.ClaimName)
			if pvcName == "" {
				continue
			}

			return protocol.Storage{
				Type:     protocol.StorageTypePVC,
				PVCName:  pvcName,
				BasePath: basePath,
			}, nil
		}
	}

	return protocol.Storage{}, fmt.Errorf(
		"snapshot-agent daemonset in %s does not mount a PVC-backed checkpoint volume (%s)",
		namespace,
		strings.Join(names, ", "),
	)
}

func PrepareRestorePodSpecForCheckpoint(
	ctx context.Context,
	reader ctrlclient.Reader,
	namespace string,
	podSpec *corev1.PodSpec,
	container *corev1.Container,
	checkpointID string,
	artifactVersion string,
	seccompProfile string,
	placeholder bool,
) error {
	if reader == nil {
		return fmt.Errorf("snapshot client is required")
	}

	daemonSets := &appsv1.DaemonSetList{}
	if err := reader.List(
		ctx,
		daemonSets,
		ctrlclient.InNamespace(namespace),
		ctrlclient.MatchingLabels{SnapshotAgentLabelKey: SnapshotAgentLabelValue},
	); err != nil {
		return fmt.Errorf("list snapshot-agent daemonsets in %s: %w", namespace, err)
	}

	storage, err := DiscoverStorageFromDaemonSets(namespace, daemonSets.Items)
	if err != nil {
		return err
	}

	resolvedStorage, err := protocol.ResolveCheckpointStorage(checkpointID, artifactVersion, storage)
	if err != nil {
		return err
	}

	PrepareRestorePodSpec(podSpec, container, resolvedStorage, seccompProfile, placeholder)
	return nil
}
