package restore

import (
	"fmt"

	criurpc "github.com/checkpoint-restore/go-criu/v7/rpc"
	"google.golang.org/protobuf/proto"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/common"
)

// GenerateExtMountMaps generates external mount mappings for CRIU restore.
// It parses /proc/1/mountinfo (the restore container's mounts) and adds
// mappings for all mount points plus masked/readonly paths from common.
//
// If meta is nil or doesn't have OCI-derived paths, falls back to defaults.
func GenerateExtMountMaps(meta *common.CheckpointMetadata) ([]*criurpc.ExtMountMap, error) {
	var maps []*criurpc.ExtMountMap
	addedMounts := make(map[string]bool)

	// Add root filesystem mapping first
	maps = append(maps, &criurpc.ExtMountMap{
		Key: proto.String("/"),
		Val: proto.String("."),
	})
	addedMounts["/"] = true

	// Parse /proc/1/mountinfo for all current mount points
	mountPoints, err := common.GetMountPointPaths("/proc/1/mountinfo")
	if err != nil {
		return nil, fmt.Errorf("failed to parse mountinfo: %w", err)
	}

	for _, mountPoint := range mountPoints {
		if addedMounts[mountPoint] || mountPoint == "/" {
			continue
		}
		maps = append(maps, &criurpc.ExtMountMap{
			Key: proto.String(mountPoint),
			Val: proto.String(mountPoint),
		})
		addedMounts[mountPoint] = true
	}

	// Use masked paths from checkpoint metadata (OCI spec derived)
	// Fall back to defaults for backwards compatibility
	maskedPaths := common.DefaultMaskedPaths()
	if meta != nil && len(meta.MaskedPaths) > 0 {
		maskedPaths = meta.MaskedPaths
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

	// Also add readonly paths from metadata if available
	if meta != nil {
		for _, path := range meta.ReadonlyPaths {
			if addedMounts[path] {
				continue
			}
			maps = append(maps, &criurpc.ExtMountMap{
				Key: proto.String(path),
				Val: proto.String(path),
			})
			addedMounts[path] = true
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
