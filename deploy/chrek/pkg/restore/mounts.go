package restore

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	criurpc "github.com/checkpoint-restore/go-criu/v7/rpc"
	"google.golang.org/protobuf/proto"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/common"
)

// findNvidiaCtkhookFromDumpLog parses the CRIU dump.log to find nvidia-ctk-hook mount paths
func findNvidiaCtkhookFromDumpLog(checkpointPath string) string {
	dumpLogPath := filepath.Join(checkpointPath, "dump.log")
	file, err := os.Open(dumpLogPath)
	if err != nil {
		return ""
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		// Look for lines like: "@ ./run/nvidia-ctk-hook..."
		if strings.Contains(line, "nvidia-ctk-hook") && strings.Contains(line, "@ ./") {
			// Extract the path after "@ ./"
			idx := strings.Index(line, "@ ./")
			if idx >= 0 {
				pathPart := line[idx+4:] // Skip "@ ./"
				// Find the end of the path (space or end of line)
				endIdx := strings.Index(pathPart, " ")
				if endIdx > 0 {
					return "/" + pathPart[:endIdx]
				}
				return "/" + pathPart
			}
		}
	}
	return ""
}

// GenerateExtMountMaps generates external mount mappings for CRIU restore.
// It parses /proc/1/mountinfo (the restore container's mounts) and adds
// mappings for all mount points plus masked/readonly/bind mount paths from metadata.
//
// If meta is nil or doesn't have OCI-derived paths, falls back to defaults.
// checkpointPath is used to read the dump.log for nvidia-ctk-hook path detection.
func GenerateExtMountMaps(meta *common.CheckpointMetadata, checkpointPath string) ([]*criurpc.ExtMountMap, error) {
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

	// Find nvidia-ctk-hook mount in restore container (for mapping checkpoint's version)
	var restoreNvidiaCtkhookPath string
	for _, m := range restoreMounts {
		if strings.Contains(m.Path, "nvidia-ctk-hook") {
			restoreNvidiaCtkhookPath = m.Path
			break
		}
	}

	// Find nvidia-ctk-hook mount from checkpoint's dump.log (has different UUID)
	checkpointNvidiaCtkhookPath := ""
	if checkpointPath != "" {
		checkpointNvidiaCtkhookPath = findNvidiaCtkhookFromDumpLog(checkpointPath)
	}

	// If both paths exist and are different, add explicit mapping
	if checkpointNvidiaCtkhookPath != "" && restoreNvidiaCtkhookPath != "" &&
		checkpointNvidiaCtkhookPath != restoreNvidiaCtkhookPath {
		maps = append(maps, &criurpc.ExtMountMap{
			Key: proto.String(checkpointNvidiaCtkhookPath),
			Val: proto.String(restoreNvidiaCtkhookPath),
		})
		addedMounts[checkpointNvidiaCtkhookPath] = true
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

		// Add bind mount destinations from checkpoint metadata
		// These are critical for CRIU to properly map checkpoint mounts to restore mounts
		for _, path := range meta.BindMountDests {
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
		// This ensures all mounts from the checkpoint have mappings
		// Special handling for nvidia-ctk-hook paths which have random UUIDs
		for _, mount := range meta.Mounts {
			if addedMounts[mount.ContainerPath] {
				continue
			}

			// Check if this is an nvidia-ctk-hook mount with different UUID
			if strings.Contains(mount.ContainerPath, "nvidia-ctk-hook") && restoreNvidiaCtkhookPath != "" {
				// Map checkpoint's nvidia-ctk-hook path to restore container's path
				maps = append(maps, &criurpc.ExtMountMap{
					Key: proto.String(mount.ContainerPath),
					Val: proto.String(restoreNvidiaCtkhookPath),
				})
				addedMounts[mount.ContainerPath] = true
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

// PrepareNvidiaCtkhookSymlink creates a symlink from the checkpoint's nvidia-ctk-hook path
// to the restore container's actual nvidia-ctk-hook path. This is needed because CRIU's
// bind-mount restoration expects the source path to exist, even with ext-mount-map mappings.
// Returns the checkpoint path and restore path (for logging), or empty strings if no symlink needed.
func PrepareNvidiaCtkhookSymlink(checkpointPath string) (string, string, error) {
	// Find checkpoint's nvidia-ctk-hook path from dump.log
	checkpointNvidiaCtkhookPath := findNvidiaCtkhookFromDumpLog(checkpointPath)
	if checkpointNvidiaCtkhookPath == "" {
		return "", "", nil // No nvidia-ctk-hook in checkpoint
	}

	// Parse restore container's mounts to find its nvidia-ctk-hook path
	restoreMounts, err := common.ParseMountInfoFile("/proc/1/mountinfo")
	if err != nil {
		return "", "", fmt.Errorf("failed to parse mountinfo: %w", err)
	}

	var restoreNvidiaCtkhookPath string
	for _, m := range restoreMounts {
		if strings.Contains(m.Path, "nvidia-ctk-hook") && strings.Contains(m.Path, "/run/") {
			restoreNvidiaCtkhookPath = m.Path
			break
		}
	}

	if restoreNvidiaCtkhookPath == "" {
		return "", "", nil // No nvidia-ctk-hook in restore container
	}

	// If paths are the same, no symlink needed
	if checkpointNvidiaCtkhookPath == restoreNvidiaCtkhookPath {
		return "", "", nil
	}

	// Check if checkpoint path already exists (might be a mount or previous symlink)
	if _, err := os.Lstat(checkpointNvidiaCtkhookPath); err == nil {
		// Path exists, check if it's a symlink pointing to the right place
		if target, err := os.Readlink(checkpointNvidiaCtkhookPath); err == nil {
			if target == restoreNvidiaCtkhookPath {
				return checkpointNvidiaCtkhookPath, restoreNvidiaCtkhookPath, nil // Already correct symlink
			}
		}
		// Path exists but not the right symlink - can't create symlink
		return "", "", nil
	}

	// Create parent directory if needed
	parentDir := filepath.Dir(checkpointNvidiaCtkhookPath)
	if err := os.MkdirAll(parentDir, 0755); err != nil {
		return "", "", fmt.Errorf("failed to create parent directory %s: %w", parentDir, err)
	}

	// Create symlink from checkpoint path to restore path
	if err := os.Symlink(restoreNvidiaCtkhookPath, checkpointNvidiaCtkhookPath); err != nil {
		return "", "", fmt.Errorf("failed to create symlink %s -> %s: %w",
			checkpointNvidiaCtkhookPath, restoreNvidiaCtkhookPath, err)
	}

	return checkpointNvidiaCtkhookPath, restoreNvidiaCtkhookPath, nil
}
