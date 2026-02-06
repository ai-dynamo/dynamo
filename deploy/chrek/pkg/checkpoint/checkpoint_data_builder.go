// checkpoint_data_builder provides checkpoint data construction.
package checkpoint

import (
	"context"
	"strings"

	"github.com/sirupsen/logrus"

	checkpointk8s "github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint/k8s"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
)

// CheckpointDataBuilderConfig holds per-checkpoint identifiers for building checkpoint data.
type CheckpointDataBuilderConfig struct {
	CheckpointID  string
	NodeName      string
	ContainerID   string
	ContainerName string
	PodName       string
	PodNamespace  string
	PID           int
}

// BuildCheckpointData constructs checkpoint data from container state and config.
// Static config fields (CRIU options, rootfs exclusions) are copied directly from checkpointCfg.
// Dynamic fields are populated from container introspection.
func BuildCheckpointData(
	ctx context.Context,
	cfg CheckpointDataBuilderConfig,
	checkpointCfg *config.CheckpointConfig,
	containerInfo *checkpointk8s.ContainerInfo,
	mounts []MountMapping,
	namespaces map[NamespaceType]*NamespaceInfo,
	k8sClient *checkpointk8s.K8sClient,
	log *logrus.Entry,
) *config.CheckpointData {
	data := config.NewCheckpointData(cfg.CheckpointID)

	// ========== STATIC: Copy from CheckpointConfig ==========
	// These are the actual config values, not hardcoded defaults
	if checkpointCfg != nil {
		data.CRIU = checkpointCfg.CRIU
		data.RootfsExclusions = checkpointCfg.RootfsExclusions
	}

	// ========== DYNAMIC: Fill from container introspection ==========
	data.SourceNode = cfg.NodeName
	data.ContainerID = cfg.ContainerID
	data.PodName = cfg.PodName
	data.PodNamespace = cfg.PodNamespace
	data.PID = cfg.PID

	if containerInfo == nil {
		return data
	}
	data.Image = containerInfo.Image

	// Populate OCI spec derived paths
	data.MaskedPaths = containerInfo.GetMaskedPaths()
	data.ReadonlyPaths = containerInfo.GetReadonlyPaths()

	// Build mount metadata
	ociMountByDest := buildOCIMountLookup(containerInfo, data)

	// Get K8s volume types if available
	k8sVolumes := getK8sVolumes(ctx, k8sClient, cfg, log)

	// Add mount metadata
	for _, mount := range mounts {
		mountMeta := buildMountMetadata(mount, k8sVolumes, ociMountByDest)
		data.Mounts = append(data.Mounts, mountMeta)
	}

	// Add namespace metadata
	for nsType, nsInfo := range namespaces {
		data.Namespaces = append(data.Namespaces, config.NamespaceMetadata{
			Type:       string(nsType),
			Inode:      nsInfo.Inode,
			IsExternal: nsInfo.IsExternal,
		})
	}

	return data
}

// buildOCIMountLookup builds a lookup map from OCI mounts and populates bind mount destinations.
func buildOCIMountLookup(containerInfo *checkpointk8s.ContainerInfo, data *config.CheckpointData) map[string]checkpointk8s.MountInfo {
	ociMounts := containerInfo.GetMounts()
	ociMountByDest := make(map[string]checkpointk8s.MountInfo)
	for _, m := range ociMounts {
		ociMountByDest[m.Destination] = m
		if m.Type == "bind" {
			data.BindMountDests = append(data.BindMountDests, m.Destination)
		}
	}
	return ociMountByDest
}

// getK8sVolumes fetches volume types from K8s API if available.
func getK8sVolumes(ctx context.Context, k8sClient *checkpointk8s.K8sClient, cfg CheckpointDataBuilderConfig, log *logrus.Entry) map[string]*checkpointk8s.VolumeInfo {
	if k8sClient == nil || cfg.PodNamespace == "" || cfg.PodName == "" || cfg.ContainerName == "" {
		return nil
	}

	k8sVolumes, err := k8sClient.GetPodVolumes(ctx, cfg.PodNamespace, cfg.PodName, cfg.ContainerName)
	if err != nil {
		log.WithError(err).Warn("Failed to get volume types from K8s API, falling back to path-based detection")
		return nil
	}
	log.WithField("volume_count", len(k8sVolumes)).Debug("Got volume types from K8s API")
	return k8sVolumes
}

// buildMountMetadata constructs metadata for a single mount.
func buildMountMetadata(mount MountMapping, k8sVolumes map[string]*checkpointk8s.VolumeInfo, ociMountByDest map[string]checkpointk8s.MountInfo) config.MountMetadata {
	var volumeType, volumeName string

	// Try K8s API first for accurate volume types
	if k8sVolumes != nil {
		if volInfo, ok := k8sVolumes[mount.InsidePath]; ok {
			volumeType = volInfo.VolumeType
			volumeName = volInfo.VolumeName
		}
	}

	// Fall back to path-based detection if K8s API didn't provide info
	if volumeType == "" {
		volumeType, volumeName = checkpointk8s.DetectVolumeTypeFromPath(mount.OutsidePath)
	}

	mountMeta := config.MountMetadata{
		ContainerPath: mount.InsidePath,
		HostPath:      mount.OutsidePath,
		VolumeType:    volumeType,
		VolumeName:    volumeName,
		FSType:        mount.FSType,
		ReadOnly:      strings.Contains(mount.Options, "ro"),
	}

	// Cross-reference with OCI spec mount if available
	if ociMount, ok := ociMountByDest[mount.InsidePath]; ok {
		mountMeta.OCISource = ociMount.Source
		mountMeta.OCIType = ociMount.Type
		mountMeta.OCIOptions = ociMount.Options
	}

	return mountMeta
}
