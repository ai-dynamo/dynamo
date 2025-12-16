// metadata_builder provides checkpoint metadata construction.
package checkpoint

import (
	"context"
	"strings"

	"github.com/sirupsen/logrus"

	checkpointk8s "github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint/k8s"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/common"
)

// MetadataBuilderConfig holds configuration for building checkpoint metadata.
type MetadataBuilderConfig struct {
	CheckpointID  string
	NodeName      string
	ContainerID   string
	ContainerName string
	PodName       string
	PodNamespace  string
	PID           int
	CUDAPluginDir string
}

// BuildCheckpointMetadata constructs checkpoint metadata from container state.
func BuildCheckpointMetadata(
	ctx context.Context,
	cfg MetadataBuilderConfig,
	containerInfo *checkpointk8s.ContainerInfo,
	mounts []MountMapping,
	namespaces map[NamespaceType]*NamespaceInfo,
	k8sClient *checkpointk8s.K8sClient,
	log *logrus.Entry,
) *common.CheckpointMetadata {
	meta := common.NewCheckpointMetadata(cfg.CheckpointID)
	meta.SourceNode = cfg.NodeName
	meta.ContainerID = cfg.ContainerID
	meta.PodName = cfg.PodName
	meta.PodNamespace = cfg.PodNamespace
	meta.PID = cfg.PID
	meta.Image = containerInfo.Image

	// Populate OCI spec derived paths
	meta.MaskedPaths = containerInfo.GetMaskedPaths()
	meta.ReadonlyPaths = containerInfo.GetReadonlyPaths()

	// Build mount metadata
	ociMountByDest := buildOCIMountLookup(containerInfo, meta)

	// Get K8s volume types if available
	k8sVolumes := getK8sVolumes(ctx, k8sClient, cfg, log)

	// Add mount metadata
	for _, mount := range mounts {
		mountMeta := buildMountMetadata(mount, k8sVolumes, ociMountByDest)
		meta.Mounts = append(meta.Mounts, mountMeta)
	}

	// Add namespace metadata
	for nsType, nsInfo := range namespaces {
		meta.Namespaces = append(meta.Namespaces, common.NamespaceMetadata{
			Type:       string(nsType),
			Inode:      nsInfo.Inode,
			IsExternal: nsInfo.IsExternal,
		})
	}

	// Set CRIU options (hardcoded as always-on for K8s, stored for compatibility)
	meta.CRIUOptions = common.CRIUOptionsMetadata{
		TcpEstablished: false, // Always false - we close TCP connections
		TcpClose:       true,  // Always true - pod IPs change on restore
		ShellJob:       true,  // Always true - containers are session leaders
		FileLocks:      true,  // Always true - apps use file locks
		LeaveRunning:   true,  // Always true - keep process running after checkpoint
		LinkRemap:      true,  // Always true - handle deleted-but-open files
		ExtMasters:     true,  // Always true - external bind mount masters
	}

	return meta
}

// buildOCIMountLookup builds a lookup map from OCI mounts and populates bind mount destinations.
func buildOCIMountLookup(containerInfo *checkpointk8s.ContainerInfo, meta *common.CheckpointMetadata) map[string]checkpointk8s.MountInfo {
	ociMounts := containerInfo.GetMounts()
	ociMountByDest := make(map[string]checkpointk8s.MountInfo)
	for _, m := range ociMounts {
		ociMountByDest[m.Destination] = m
		if m.Type == "bind" {
			meta.BindMountDests = append(meta.BindMountDests, m.Destination)
		}
	}
	return ociMountByDest
}

// getK8sVolumes fetches volume types from K8s API if available.
func getK8sVolumes(ctx context.Context, k8sClient *checkpointk8s.K8sClient, cfg MetadataBuilderConfig, log *logrus.Entry) map[string]*checkpointk8s.VolumeInfo {
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
func buildMountMetadata(mount MountMapping, k8sVolumes map[string]*checkpointk8s.VolumeInfo, ociMountByDest map[string]checkpointk8s.MountInfo) common.MountMetadata {
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

	mountMeta := common.MountMetadata{
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

