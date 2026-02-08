// mounts provides mount parsing from /proc for CRIU checkpoint.
// This is used for runtime mount state that requires /proc inspection.
package checkpoint

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
)

// MountMapping represents an external mount for CRIU
type MountMapping struct {
	InsidePath  string // Path inside container (mount point)
	OutsidePath string // Path on host (source)
	FSType      string // Filesystem type
	Source      string // Mount source
	Options     string // Mount options
}

// System mount types that should be filtered out
var systemMountTypes = map[string]bool{
	"proc":        true,
	"sysfs":       true,
	"devpts":      true,
	"mqueue":      true,
	"tmpfs":       true, // Note: some tmpfs mounts may need special handling
	"cgroup":      true,
	"cgroup2":     true,
	"securityfs":  true,
	"debugfs":     true,
	"tracefs":     true,
	"fusectl":     true,
	"configfs":    true,
	"devtmpfs":    true,
	"hugetlbfs":   true,
	"pstore":      true,
	"bpf":         true,
}

// System mount paths that should always be filtered
var systemMountPaths = map[string]bool{
	"/proc":        true,
	"/sys":         true,
	"/dev":         true,
	"/dev/pts":     true,
	"/dev/shm":     true,
	"/dev/mqueue":  true,
	"/run":         true,
	"/run/secrets": true,
}

// ParseMountInfo parses /proc/<pid>/mountinfo and returns bind mounts
// that need to be handled by CRIU as external mounts
func ParseMountInfo(pid int) ([]MountMapping, error) {
	mountinfoPath := fmt.Sprintf("%s/%d/mountinfo", config.HostProcPath, pid)
	file, err := os.Open(mountinfoPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open mountinfo: %w", err)
	}
	defer file.Close()

	var mounts []MountMapping
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := scanner.Text()
		mount, skip := parseMountInfoLine(line)
		if skip {
			continue
		}
		mounts = append(mounts, mount)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading mountinfo: %w", err)
	}

	return mounts, nil
}

// parseMountInfoLine parses a single line from mountinfo
// Returns the mount mapping and whether to skip this mount
//
// mountinfo format:
// 36 35 98:0 /mnt1 /mnt2 rw,noatime master:1 - ext3 /dev/root rw,errors=continue
// (1)(2)(3)   (4)   (5)      (6)     (7)   (8) (9)   (10)         (11)
//
// (1) mount ID
// (2) parent ID
// (3) major:minor
// (4) root: root of the mount within the filesystem (host-side path for bind mounts)
// (5) mount point: mount point relative to process's root
// (6) mount options
// (7) optional fields (terminated by single hyphen)
// (8) separator (hyphen)
// (9) filesystem type
// (10) mount source (device)
// (11) super options
func parseMountInfoLine(line string) (MountMapping, bool) {
	fields := strings.Fields(line)
	if len(fields) < 10 {
		return MountMapping{}, true
	}

	root := fields[3]       // Host-side path within the filesystem (important for bind mounts)
	mountPoint := fields[4] // Container-side mount point
	mountOptions := fields[5]

	// Find separator (-) to get fstype and source
	sepIdx := -1
	for i, f := range fields {
		if f == "-" {
			sepIdx = i
			break
		}
	}

	if sepIdx == -1 || sepIdx+2 >= len(fields) {
		return MountMapping{}, true
	}

	fsType := fields[sepIdx+1]
	source := fields[sepIdx+2]
	superOptions := ""
	if sepIdx+3 < len(fields) {
		superOptions = fields[sepIdx+3]
	}

	// Skip system mount types
	if systemMountTypes[fsType] {
		return MountMapping{}, true
	}

	// Skip system mount paths
	if systemMountPaths[mountPoint] {
		return MountMapping{}, true
	}

	// Skip /sys and /proc prefixed paths
	if strings.HasPrefix(mountPoint, "/sys/") || strings.HasPrefix(mountPoint, "/proc/") {
		return MountMapping{}, true
	}

	// Skip overlay (the root filesystem itself)
	if fsType == "overlay" && mountPoint == "/" {
		return MountMapping{}, true
	}

	// For bind mounts, the root field contains the actual host path
	// Use root as OutsidePath since it gives us the host-side path for volume mounts
	outsidePath := root
	if root == "/" {
		// If root is /, this isn't a bind mount from a subdirectory
		outsidePath = source
	}

	return MountMapping{
		InsidePath:  mountPoint,
		OutsidePath: outsidePath,
		FSType:      fsType,
		Source:      source,
		Options:     mountOptions + "," + superOptions,
	}, false
}

// GetBindMounts returns only bind mounts (type "bind" or with bind option)
func GetBindMounts(pid int) ([]MountMapping, error) {
	mounts, err := ParseMountInfo(pid)
	if err != nil {
		return nil, err
	}

	var bindMounts []MountMapping
	for _, m := range mounts {
		// Bind mounts typically show the underlying filesystem type
		// and have paths that look like kubelet volume paths
		if strings.Contains(m.OutsidePath, "/var/lib/kubelet/pods/") ||
			strings.Contains(m.OutsidePath, "/volumes/") ||
			strings.Contains(m.Options, "bind") {
			bindMounts = append(bindMounts, m)
		}
	}

	return bindMounts, nil
}

// GetKubernetesVolumeMounts returns mounts that appear to be Kubernetes volumes
func GetKubernetesVolumeMounts(pid int) ([]MountMapping, error) {
	mounts, err := ParseMountInfo(pid)
	if err != nil {
		return nil, err
	}

	var k8sMounts []MountMapping
	for _, m := range mounts {
		// Kubernetes volumes are identified by:
		// 1. Standard kubelet paths: /var/lib/kubelet/pods/
		// 2. Minikube/Docker paths: /var/lib/docker/volumes/minikube/_data/lib/kubelet/pods/
		// 3. Kubernetes volume markers: kubernetes.io~empty-dir, kubernetes.io~configmap, etc.
		if strings.Contains(m.OutsidePath, "/kubelet/pods/") ||
			strings.Contains(m.OutsidePath, "/kubernetes.io~") ||
			strings.Contains(m.OutsidePath, "/containerd/io.containerd") {
			k8sMounts = append(k8sMounts, m)
		}
	}

	return k8sMounts, nil
}

// AllMountInfo represents a mount entry from /proc/<pid>/mountinfo
// This includes ALL mounts without filtering, which CRIU captures during checkpoint.
type AllMountInfo struct {
	MountID      string // Mount ID
	ParentID     string // Parent mount ID
	MountPoint   string // Mount point inside container (container-side path)
	Root         string // Root of mount within filesystem (host-side path for bind mounts)
	FSType       string // Filesystem type
	Source       string // Mount source
	Options      string // Mount options
	SuperOptions string // Super block options
}

// GetAllMountsFromMountinfo parses /proc/<pid>/mountinfo and returns ALL mounts.
// This is used for CRIU checkpoint to mark ALL mounts as external, since CRIU
// captures everything from mountinfo, not just the filtered subset.
// Without marking ALL mounts as external, CRIU restore fails with
// "No mapping for <mount_id>:(null) mountpoint" errors.
func GetAllMountsFromMountinfo(pid int) ([]AllMountInfo, error) {
	mountinfoPath := fmt.Sprintf("%s/%d/mountinfo", config.HostProcPath, pid)
	file, err := os.Open(mountinfoPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open mountinfo: %w", err)
	}
	defer file.Close()

	var mounts []AllMountInfo
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := scanner.Text()
		mount, err := parseAllMountInfoLine(line)
		if err != nil {
			continue // Skip malformed lines
		}
		mounts = append(mounts, mount)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading mountinfo: %w", err)
	}

	return mounts, nil
}

// GetMountsUnderPrefixes returns all mount points that fall under any of the given
// directory prefixes. This is used to enumerate mounts to skip during checkpoint,
// allowing cross-node restore when certain mounts (e.g., nvidia runtime mounts)
// don't exist on the target node.
//
// For example, with prefixes ["/run/nvidia/driver", "/proc/driver/nvidia"]:
//   - /run/nvidia/driver/lib/firmware/nvidia/580.82.07/gsp_tu10x.bin -> included
//   - /run/nvidia/driver/lib/firmware/nvidia/580.82.07/gsp_ga10x.bin -> included
//   - /proc/driver/nvidia/params -> included
//   - /run/nvidia-ctk-hook -> NOT included (doesn't match prefix)
func GetMountsUnderPrefixes(pid int, prefixes []string) ([]string, error) {
	if len(prefixes) == 0 {
		return nil, nil
	}

	mounts, err := GetAllMountsFromMountinfo(pid)
	if err != nil {
		return nil, err
	}

	var matchedMounts []string
	for _, m := range mounts {
		for _, prefix := range prefixes {
			// Check if mount point starts with prefix
			// Use HasPrefix with trailing slash check to avoid partial matches
			// e.g., "/run/nvidia" should match "/run/nvidia/driver/..." but not "/run/nvidia-ctk-hook"
			if strings.HasPrefix(m.MountPoint, prefix+"/") || m.MountPoint == prefix {
				matchedMounts = append(matchedMounts, m.MountPoint)
				break // Don't add same mount twice if it matches multiple prefixes
			}
		}
	}

	return matchedMounts, nil
}

// parseAllMountInfoLine parses a single line from mountinfo without filtering.
// mountinfo format:
// 36 35 98:0 /mnt1 /mnt2 rw,noatime master:1 - ext3 /dev/root rw,errors=continue
// (1)(2)(3)   (4)   (5)      (6)     (7)   (8) (9)   (10)         (11)
func parseAllMountInfoLine(line string) (AllMountInfo, error) {
	fields := strings.Fields(line)
	if len(fields) < 10 {
		return AllMountInfo{}, fmt.Errorf("malformed mountinfo line: %s", line)
	}

	mountID := fields[0]
	parentID := fields[1]
	root := fields[3]       // Host-side path within the filesystem
	mountPoint := fields[4] // Container-side mount point
	mountOptions := fields[5]

	// Find separator (-) to get fstype and source
	sepIdx := -1
	for i, f := range fields {
		if f == "-" {
			sepIdx = i
			break
		}
	}

	if sepIdx == -1 || sepIdx+2 >= len(fields) {
		return AllMountInfo{}, fmt.Errorf("malformed mountinfo line (no separator): %s", line)
	}

	fsType := fields[sepIdx+1]
	source := fields[sepIdx+2]
	superOptions := ""
	if sepIdx+3 < len(fields) {
		superOptions = fields[sepIdx+3]
	}

	return AllMountInfo{
		MountID:      mountID,
		ParentID:     parentID,
		MountPoint:   mountPoint,
		Root:         root,
		FSType:       fsType,
		Source:       source,
		Options:      mountOptions,
		SuperOptions: superOptions,
	}, nil
}
