// namespaces provides Linux namespace introspection for CRIU checkpoint.
package checkpoint

import (
	"fmt"

	"golang.org/x/sys/unix"
)

// NamespaceMetadata stores namespace information saved in checkpoint metadata.
type NamespaceMetadata struct {
	Type       string `yaml:"type"`       // net, pid, mnt, etc.
	Inode      uint64 `yaml:"inode"`      // Namespace inode
	IsExternal bool   `yaml:"isExternal"` // Whether namespace is external (shared)
}

// NamespaceType represents a Linux namespace type
type NamespaceType string

const (
	NamespaceNet    NamespaceType = "net"
	NamespacePID    NamespaceType = "pid"
	NamespaceMnt    NamespaceType = "mnt"
	NamespaceUTS    NamespaceType = "uts"
	NamespaceIPC    NamespaceType = "ipc"
	NamespaceUser   NamespaceType = "user"
	NamespaceCgroup NamespaceType = "cgroup"
)

// NamespaceInfo holds namespace identification information
type NamespaceInfo struct {
	Type       NamespaceType
	Inode      uint64
	IsExternal bool // Whether NS is external (shared with pause container)
}

// NewNamespaceMetadata constructs namespace metadata from introspected namespaces.
func NewNamespaceMetadata(namespaces map[NamespaceType]*NamespaceInfo) []NamespaceMetadata {
	if len(namespaces) == 0 {
		return nil
	}

	result := make([]NamespaceMetadata, 0, len(namespaces))
	for nsType, nsInfo := range namespaces {
		result = append(result, NamespaceMetadata{
			Type:       string(nsType),
			Inode:      nsInfo.Inode,
			IsExternal: nsInfo.IsExternal,
		})
	}
	return result
}

// GetNamespaceInode returns the inode number for a namespace
func GetNamespaceInode(pid int, nsType NamespaceType) (uint64, error) {
	nsPath := fmt.Sprintf("%s/%d/ns/%s", HostProcPath, pid, nsType)
	var stat unix.Stat_t
	if err := unix.Stat(nsPath, &stat); err != nil {
		return 0, fmt.Errorf("failed to stat namespace %s: %w", nsPath, err)
	}

	return stat.Ino, nil
}

// GetNamespaceInfo returns detailed namespace information
func GetNamespaceInfo(pid int, nsType NamespaceType) (*NamespaceInfo, error) {
	nsPath := fmt.Sprintf("%s/%d/ns/%s", HostProcPath, pid, nsType)

	// Get inode
	var stat unix.Stat_t
	if err := unix.Stat(nsPath, &stat); err != nil {
		return nil, fmt.Errorf("failed to stat namespace %s: %w", nsPath, err)
	}

	// Check if this is different from init's namespace (PID 1)
	initNsPath := fmt.Sprintf("%s/1/ns/%s", HostProcPath, nsType)
	var initStat unix.Stat_t
	isExternal := false
	if err := unix.Stat(initNsPath, &initStat); err == nil {
		// If the inode is different from init's, it's an external namespace
		isExternal = stat.Ino != initStat.Ino
	}

	return &NamespaceInfo{
		Type:       nsType,
		Inode:      stat.Ino,
		IsExternal: isExternal,
	}, nil
}

// GetAllNamespaces returns information about all namespaces for a process
func GetAllNamespaces(pid int) (map[NamespaceType]*NamespaceInfo, error) {
	nsTypes := []NamespaceType{
		NamespaceNet,
		NamespacePID,
		NamespaceMnt,
		NamespaceUTS,
		NamespaceIPC,
		NamespaceUser,
		NamespaceCgroup,
	}

	namespaces := make(map[NamespaceType]*NamespaceInfo)
	for _, nsType := range nsTypes {
		info, err := GetNamespaceInfo(pid, nsType)
		if err != nil {
			// Some namespaces might not exist, skip them
			continue
		}
		namespaces[nsType] = info
	}

	return namespaces, nil
}

