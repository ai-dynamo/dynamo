package restore

import (
	"fmt"

	criurpc "github.com/checkpoint-restore/go-criu/v7/rpc"
	"google.golang.org/protobuf/proto"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/common"
)

// GenerateExtMountMaps generates external mount mappings for CRIU restore.
// It parses /proc/1/mountinfo (the restore container's mounts) and adds
// mappings for all mount points plus masked/readonly/bind mount paths from checkpoint metadata.
func GenerateExtMountMaps(data *checkpoint.CheckpointMetadata) ([]*criurpc.ExtMountMap, error) {
	var maps []*criurpc.ExtMountMap
	addedMounts := make(map[string]bool)

	// Add root filesystem mapping first
	maps = append(maps, &criurpc.ExtMountMap{
		Key: proto.String("/"),
		Val: proto.String("."),
	})
	addedMounts["/"] = true

	// Parse /proc/1/mountinfo for all current mount points
	restoreMounts, err := common.ParseMountInfoFile("/proc/1/mountinfo")
	if err != nil {
		return nil, fmt.Errorf("failed to parse mountinfo: %w", err)
	}

	// Add all current mount points
	for _, m := range restoreMounts {
		if addedMounts[m.Path] || m.Path == "/" {
			continue
		}
		maps = append(maps, &criurpc.ExtMountMap{
			Key: proto.String(m.Path),
			Val: proto.String(m.Path),
		})
		addedMounts[m.Path] = true
	}

	// Use masked paths from checkpoint metadata (OCI spec derived)
	// Fall back to defaults for backwards compatibility
	maskedPaths := common.DefaultMaskedPaths()
	if data != nil && len(data.Filesystem.MaskedPaths) > 0 {
		maskedPaths = data.Filesystem.MaskedPaths
	}

	for _, path := range maskedPaths {
		if addedMounts[path] {
			continue
		}
		maps = append(maps, &criurpc.ExtMountMap{
			Key: proto.String(path),
			Val: proto.String(path),
		})
		addedMounts[path] = true
	}

	// Also add readonly paths from checkpoint metadata if available
	if data != nil {
		for _, path := range data.Filesystem.ReadonlyPaths {
			if addedMounts[path] {
				continue
			}
			maps = append(maps, &criurpc.ExtMountMap{
				Key: proto.String(path),
				Val: proto.String(path),
			})
			addedMounts[path] = true
		}

		// Add bind mount destinations from checkpoint metadata
		for _, path := range data.Filesystem.BindMountDests {
			if addedMounts[path] {
				continue
			}
			maps = append(maps, &criurpc.ExtMountMap{
				Key: proto.String(path),
				Val: proto.String(path),
			})
			addedMounts[path] = true
		}

		// Also add container paths from mount metadata
		for _, mount := range data.Mounts {
			if addedMounts[mount.ContainerPath] {
				continue
			}
			maps = append(maps, &criurpc.ExtMountMap{
				Key: proto.String(mount.ContainerPath),
				Val: proto.String(mount.ContainerPath),
			})
			addedMounts[mount.ContainerPath] = true
		}
	}

	return maps, nil
}

// AddExtMountMap is a helper to create a single ExtMountMap entry.
func AddExtMountMap(key, val string) *criurpc.ExtMountMap {
	return &criurpc.ExtMountMap{
		Key: proto.String(key),
		Val: proto.String(val),
	}
}
