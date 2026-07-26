// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package runtime

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"golang.org/x/sys/unix"
)

const checkpointMountAttrs = unix.MOUNT_ATTR_RDONLY |
	unix.MOUNT_ATTR_NOSUID |
	unix.MOUNT_ATTR_NODEV |
	unix.MOUNT_ATTR_NOEXEC

// OpenCheckpointTree creates a detached clone of the mount subtree rooted at
// path (the agent's own checkpoint dir) and returns it as an *os.File so it can
// be handed to a child process via exec.Cmd.ExtraFiles.
//
// The clone is an anonymous mount (attached to no mount namespace), which is the
// only thing that can be grafted into a different mount namespace. A plain bind
// of /proc/self/fd/N fails with EINVAL from inside another namespace because the
// source mount belongs to this (the agent's) namespace; open_tree(OPEN_TREE_CLONE)
// plus move_mount is the kernel API built for exactly this transfer. The full
// hardened sequence requires Linux 5.12+ (mount_setattr).
func OpenCheckpointTree(path string) (*os.File, error) {
	if !filepath.IsAbs(path) || filepath.Clean(path) != path || path == string(os.PathSeparator) {
		return nil, fmt.Errorf("checkpoint tree source must be an absolute, clean, non-root path: %q", path)
	}
	// Resolve the exact version directory once, rejecting symlinks in every
	// component. open_tree then clones from this pinned O_PATH descriptor rather
	// than resolving an attacker-changeable shared-PVC pathname a second time.
	sourceFD, err := unix.Openat2(unix.AT_FDCWD, path, &unix.OpenHow{
		Flags:   unix.O_PATH | unix.O_DIRECTORY | unix.O_CLOEXEC,
		Resolve: unix.RESOLVE_NO_SYMLINKS | unix.RESOLVE_NO_MAGICLINKS,
	})
	if err != nil {
		return nil, fmt.Errorf("open checkpoint tree source %s without symlinks: %w", path, err)
	}
	defer unix.Close(sourceFD)

	fd, err := unix.OpenTree(sourceFD, "",
		uint(unix.OPEN_TREE_CLONE|unix.AT_RECURSIVE|unix.OPEN_TREE_CLOEXEC|unix.AT_EMPTY_PATH))
	if err != nil {
		return nil, fmt.Errorf("open_tree(%s): %w", path, err)
	}
	// open_tree(OPEN_TREE_CLONE) is equivalent to a detached recursive bind
	// mount: it references the same backing filesystem and inherits the source
	// mount's writable attributes. Harden the anonymous tree before it crosses
	// into the untrusted restore container's mount namespace. AT_RECURSIVE
	// applies the attributes to any nested submount too.
	//
	// NOTE: these attributes are set, not MNT_LOCK_*-locked (neither
	// mount_setattr nor move_mount locks; only cross-user-namespace copy_mnt_ns
	// does, and Linux exposes no API to set MNT_LOCK_READONLY explicitly). So
	// read-only holds because a normal restore target lacks CAP_SYS_ADMIN over
	// its own user namespace, which makes `mount -o remount,rw` return EPERM.
	// A target that DID hold CAP_SYS_ADMIN could remount rw, but that is out of
	// scope: such a container is already privileged enough to cause worse harm,
	// and confinement still bounds it to its own versions/<v>. The guarantee this
	// hardening provides is against a normal, unprivileged workload.
	attr := &unix.MountAttr{
		Attr_set:    checkpointMountAttrs,
		Propagation: unix.MS_PRIVATE,
	}
	if err := unix.MountSetattr(
		fd,
		"",
		uint(unix.AT_EMPTY_PATH|unix.AT_RECURSIVE),
		attr,
	); err != nil {
		_ = unix.Close(fd)
		return nil, fmt.Errorf("mount_setattr(%s, read-only): %w", path, err)
	}
	return os.NewFile(uintptr(fd), "checkpoint-tree:"+path), nil
}

// openOrCreateMountTarget securely walks target from the current namespace's
// root, creating missing directories without following symlinks. The returned
// O_PATH fd pins the exact directory used by move_mount, closing the
// MkdirAll/path-replacement race against the untrusted placeholder process.
func openOrCreateMountTarget(target string) (int, error) {
	if !filepath.IsAbs(target) || filepath.Clean(target) != target || target == string(os.PathSeparator) {
		return -1, fmt.Errorf("checkpoint mount target must be an absolute, clean, non-root path: %q", target)
	}

	currentFD, err := unix.Open(
		string(os.PathSeparator),
		unix.O_PATH|unix.O_DIRECTORY|unix.O_CLOEXEC|unix.O_NOFOLLOW,
		0,
	)
	if err != nil {
		return -1, fmt.Errorf("open mount namespace root: %w", err)
	}

	components := strings.Split(strings.TrimPrefix(target, string(os.PathSeparator)), string(os.PathSeparator))
	for _, component := range components {
		if component == "" || component == "." || component == ".." {
			_ = unix.Close(currentFD)
			return -1, fmt.Errorf("checkpoint mount target contains invalid component %q: %q", component, target)
		}
		if err := unix.Mkdirat(currentFD, component, 0o755); err != nil && !errors.Is(err, unix.EEXIST) {
			_ = unix.Close(currentFD)
			return -1, fmt.Errorf("create checkpoint mount target component %q in %s: %w", component, target, err)
		}
		nextFD, err := unix.Openat(
			currentFD,
			component,
			unix.O_PATH|unix.O_DIRECTORY|unix.O_CLOEXEC|unix.O_NOFOLLOW,
			0,
		)
		_ = unix.Close(currentFD)
		if err != nil {
			return -1, fmt.Errorf("open checkpoint mount target component %q in %s without symlinks: %w", component, target, err)
		}
		currentFD = nextFD
	}
	return currentFD, nil
}

// rejectSharedMountTarget fails closed if the mount containing targetFD is a
// shared-propagation mount. Attaching below a shared mount can propagate the
// graft into peer mount namespaces before the new mount's own MS_PRIVATE
// setting can contain subsequent events.
func rejectSharedMountTarget(targetFD int, target string) error {
	fdInfo, err := os.ReadFile(fmt.Sprintf("/proc/self/fdinfo/%d", targetFD))
	if err != nil {
		return fmt.Errorf("read mount information for checkpoint target %s: %w", target, err)
	}
	var mountID string
	for _, line := range strings.Split(string(fdInfo), "\n") {
		fields := strings.Fields(line)
		if len(fields) == 2 && fields[0] == "mnt_id:" {
			if _, err := strconv.ParseUint(fields[1], 10, 64); err != nil {
				return fmt.Errorf("parse mount ID %q for checkpoint target %s: %w", fields[1], target, err)
			}
			mountID = fields[1]
			break
		}
	}
	if mountID == "" {
		return fmt.Errorf("mount ID is unavailable for checkpoint target %s", target)
	}

	mountInfo, err := os.ReadFile("/proc/self/mountinfo")
	if err != nil {
		return fmt.Errorf("read mount propagation for checkpoint target %s: %w", target, err)
	}
	for _, line := range strings.Split(string(mountInfo), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 7 || fields[0] != mountID {
			continue
		}
		for i := 6; i < len(fields) && fields[i] != "-"; i++ {
			if strings.HasPrefix(fields[i], "shared:") {
				return fmt.Errorf("checkpoint mount target %s is on shared-propagation mount %s", target, mountID)
			}
		}
		return nil
	}
	return fmt.Errorf("mount %s for checkpoint target %s is absent from /proc/self/mountinfo", mountID, target)
}

// AttachCheckpointTree grafts the detached mount referred to by treeFD onto
// target inside the CURRENT mount namespace, creating target if absent without
// following symlinks. It consumes treeFD and returns a cleanup func that lazily
// detaches the mount. It is meant to run from inside the target container's
// mount namespace (i.e. from nsrestore), so the checkpoint dir becomes visible
// to CRIU without the workload pod ever mounting the PVC.
func AttachCheckpointTree(treeFD int, target string) (func() error, error) {
	// This is the child-side duplicate inherited through ExtraFiles. Close it
	// as soon as move_mount has attached the tree so nsrestore cannot
	// accidentally retain or pass an O_PATH reference to the checkpoint mount.
	defer unix.Close(treeFD)

	targetFD, err := openOrCreateMountTarget(target)
	if err != nil {
		return nil, err
	}
	defer unix.Close(targetFD)

	if err := rejectSharedMountTarget(targetFD, target); err != nil {
		return nil, err
	}

	if err := unix.MoveMount(
		treeFD,
		"",
		targetFD,
		"",
		unix.MOVE_MOUNT_F_EMPTY_PATH|unix.MOVE_MOUNT_T_EMPTY_PATH,
	); err != nil {
		return nil, fmt.Errorf("move_mount -> %s: %w", target, err)
	}
	return func() error {
		// UMOUNT_NOFOLLOW prevents final-component symlink substitution during
		// cleanup. MNT_DETACH removes the graft from pathname lookup
		// immediately even if CRIU or the workload still holds open files.
		if err := unix.Unmount(target, unix.MNT_DETACH|unix.UMOUNT_NOFOLLOW); err != nil {
			return fmt.Errorf("detach checkpoint mount %s: %w", target, err)
		}
		return nil
	}, nil
}
