package runtime

import (
	"fmt"

	"golang.org/x/sys/unix"
)

// GetNetNSInode returns the network namespace inode for a container process via /host/proc.
func GetNetNSInode(pid int) (uint64, error) {
	nsPath := fmt.Sprintf("%s/%d/ns/net", HostProcPath, pid)
	var stat unix.Stat_t
	if err := unix.Stat(nsPath, &stat); err != nil {
		return 0, fmt.Errorf("failed to stat %s: %w", nsPath, err)
	}
	return stat.Ino, nil
}
