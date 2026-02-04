// namespaces provides Linux namespace introspection for CRIU checkpoint.
package checkpoint

import (
	"fmt"
	"os"
	"strings"

	"golang.org/x/sys/unix"
)

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
	Path       string
	IsExternal bool // Whether NS is external (shared with pause container)
}

// GetNamespaceInode returns the inode number for a namespace
func GetNamespaceInode(pid int, nsType NamespaceType, hostProc string) (uint64, error) {
	if hostProc == "" {
		hostProc = "/proc"
	}

	nsPath := fmt.Sprintf("%s/%d/ns/%s", hostProc, pid, nsType)
	var stat unix.Stat_t
	if err := unix.Stat(nsPath, &stat); err != nil {
		return 0, fmt.Errorf("failed to stat namespace %s: %w", nsPath, err)
	}

	return stat.Ino, nil
}

// GetNamespaceInfo returns detailed namespace information
func GetNamespaceInfo(pid int, nsType NamespaceType, hostProc string) (*NamespaceInfo, error) {
	if hostProc == "" {
		hostProc = "/proc"
	}

	nsPath := fmt.Sprintf("%s/%d/ns/%s", hostProc, pid, nsType)

	// Get inode
	var stat unix.Stat_t
	if err := unix.Stat(nsPath, &stat); err != nil {
		return nil, fmt.Errorf("failed to stat namespace %s: %w", nsPath, err)
	}

	// Read the symlink to get the namespace identifier
	link, err := os.Readlink(nsPath)
	if err != nil {
		return nil, fmt.Errorf("failed to readlink %s: %w", nsPath, err)
	}

	// Check if this is different from init's namespace (PID 1)
	initNsPath := fmt.Sprintf("%s/1/ns/%s", hostProc, nsType)
	var initStat unix.Stat_t
	isExternal := false
	if err := unix.Stat(initNsPath, &initStat); err == nil {
		// If the inode is different from init's, it's an external namespace
		isExternal = stat.Ino != initStat.Ino
	}

	return &NamespaceInfo{
		Type:       nsType,
		Inode:      stat.Ino,
		Path:       link,
		IsExternal: isExternal,
	}, nil
}

// GetAllNamespaces returns information about all namespaces for a process
func GetAllNamespaces(pid int, hostProc string) (map[NamespaceType]*NamespaceInfo, error) {
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
		info, err := GetNamespaceInfo(pid, nsType, hostProc)
		if err != nil {
			// Some namespaces might not exist, skip them
			continue
		}
		namespaces[nsType] = info
	}

	return namespaces, nil
}

// IsNetNamespaceExternal checks if the network namespace is external
// (i.e., shared with the pause container in Kubernetes)
func IsNetNamespaceExternal(pid int, hostProc string) (bool, uint64, error) {
	info, err := GetNamespaceInfo(pid, NamespaceNet, hostProc)
	if err != nil {
		return false, 0, err
	}
	return info.IsExternal, info.Inode, nil
}

// IsPIDNamespaceExternal checks if the PID namespace is external
func IsPIDNamespaceExternal(pid int, hostProc string) (bool, uint64, error) {
	info, err := GetNamespaceInfo(pid, NamespacePID, hostProc)
	if err != nil {
		return false, 0, err
	}
	return info.IsExternal, info.Inode, nil
}

// OpenNamespaceFD opens a file descriptor to a namespace
// The caller is responsible for closing the returned file
func OpenNamespaceFD(pid int, nsType NamespaceType, hostProc string) (*os.File, error) {
	if hostProc == "" {
		hostProc = "/proc"
	}

	nsPath := fmt.Sprintf("%s/%d/ns/%s", hostProc, pid, nsType)
	return os.Open(nsPath)
}

// FormatExternalNamespace formats namespace info for CRIU's External option
// Format: <type>[<inode>]:<key>
func FormatExternalNamespace(nsType NamespaceType, inode uint64) string {
	key := formatNamespaceKey(nsType)
	return fmt.Sprintf("%s[%d]:%s", nsType, inode, key)
}

// formatNamespaceKey creates the CRIU key for external namespaces
// Format: extRoot<Type>NS (e.g., extRootNetNS, extRootPidNS)
func formatNamespaceKey(nsType NamespaceType) string {
	// Capitalize first letter of namespace type
	nsName := string(nsType)
	if len(nsName) > 0 {
		nsName = strings.ToUpper(nsName[:1]) + nsName[1:]
	}
	return "extRoot" + nsName + "NS"
}

// GetNamespaceKey returns the CRIU key for a namespace type
func GetNamespaceKey(nsType NamespaceType) string {
	return formatNamespaceKey(nsType)
}
