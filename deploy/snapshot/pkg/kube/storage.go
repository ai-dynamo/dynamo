package kube

import (
	"context"
	"fmt"
	"path"
	"slices"
	"strings"

	appsv1 "k8s.io/api/apps/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const DefaultArtifactVersion = "1"
const defaultCheckpointBasePath = "/checkpoints"

type SnapshotStorage struct {
	PVCName  string
	BasePath string
}

func (s SnapshotStorage) ArtifactVersion(version string) string {
	version = strings.TrimSpace(version)
	if version == "" {
		return DefaultArtifactVersion
	}
	return version
}

func (s SnapshotStorage) CheckpointLocation(snapshotID string, version string) string {
	return path.Join(s.BasePath, snapshotID, "versions", s.ArtifactVersion(version))
}

func DiscoverSnapshotStorage(ctx context.Context, kubeClient client.Client, namespace string) (SnapshotStorage, error) {
	daemonSets := &appsv1.DaemonSetList{}
	if err := kubeClient.List(ctx, daemonSets, client.InNamespace(namespace), client.MatchingLabels{
		"app.kubernetes.io/name": "snapshot",
	}); err != nil {
		return SnapshotStorage{}, fmt.Errorf("list snapshot-agent daemonsets in %s: %w", namespace, err)
	}
	if len(daemonSets.Items) == 0 {
		return SnapshotStorage{}, fmt.Errorf("no snapshot-agent daemonset found in namespace %s", namespace)
	}

	slices.SortFunc(daemonSets.Items, func(a, b appsv1.DaemonSet) int {
		return strings.Compare(a.Name, b.Name)
	})

	for i := range daemonSets.Items {
		storage, ok := SnapshotStorageFromDaemonSet(&daemonSets.Items[i])
		if ok {
			return storage, nil
		}
	}

	names := make([]string, 0, len(daemonSets.Items))
	for _, daemonSet := range daemonSets.Items {
		names = append(names, daemonSet.Name)
	}
	return SnapshotStorage{}, fmt.Errorf(
		"snapshot-agent daemonset in %s does not mount a PVC-backed checkpoint volume (%s)",
		namespace,
		strings.Join(names, ", "),
	)
}

func SnapshotStorageFromDaemonSet(daemonSet *appsv1.DaemonSet) (SnapshotStorage, bool) {
	if daemonSet == nil {
		return SnapshotStorage{}, false
	}

	mountPaths := map[string]string{}
	for _, container := range daemonSet.Spec.Template.Spec.Containers {
		if container.Name != "agent" {
			continue
		}
		for _, mount := range container.VolumeMounts {
			mountPaths[mount.Name] = mount.MountPath
		}
	}

	firstPVCMount := SnapshotStorage{}
	foundPVCMount := false

	for _, volume := range daemonSet.Spec.Template.Spec.Volumes {
		if volume.PersistentVolumeClaim == nil {
			continue
		}
		mountPath := mountPaths[volume.Name]
		if strings.TrimSpace(mountPath) == "" {
			continue
		}

		storage := SnapshotStorage{
			PVCName:  volume.PersistentVolumeClaim.ClaimName,
			BasePath: mountPath,
		}
		if !foundPVCMount {
			firstPVCMount = storage
			foundPVCMount = true
		}
		if mountPath == defaultCheckpointBasePath || volume.Name == "checkpoints" {
			return storage, true
		}
	}

	return firstPVCMount, foundPVCMount
}
