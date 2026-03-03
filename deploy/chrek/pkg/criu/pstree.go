package criu

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/checkpoint-restore/go-criu/v8/crit"
	"github.com/checkpoint-restore/go-criu/v8/crit/images/pstree"
)

// PstreeOrderedPIDs parses pstree.img from the checkpoint directory and returns
// all process PIDs in BFS order. The ordering is derived from the pstree.img file
// which records the original process tree structure from checkpoint time.
// CRIU restores processes in the same structural order, so a parallel BFS of the
// restored /proc tree yields a positional old→new PID correspondence.
func PstreeOrderedPIDs(checkpointDir string) ([]uint32, error) {
	imgPath := filepath.Join(checkpointDir, "pstree.img")
	f, err := os.Open(imgPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open pstree.img: %w", err)
	}
	defer f.Close()

	c := crit.New(f, nil, "", false, false)
	img, err := c.Decode(&pstree.PstreeEntry{})
	if err != nil {
		return nil, fmt.Errorf("failed to decode pstree.img: %w", err)
	}

	// Build parent→children map, preserving pstree.img file order for children.
	// File order matches the original /proc children order from checkpoint time.
	type node struct {
		children []uint32
	}
	nodes := make(map[uint32]*node, len(img.Entries))
	var rootPID uint32

	for _, entry := range img.Entries {
		e := entry.Message.(*pstree.PstreeEntry)
		pid := e.GetPid()
		nodes[pid] = &node{}
		// Root process has ppid==0 or ppid==pid
		if e.GetPpid() == 0 || pid == e.GetPpid() {
			rootPID = pid
		}
	}

	// Second pass: populate children lists in file order
	for _, entry := range img.Entries {
		e := entry.Message.(*pstree.PstreeEntry)
		pid := e.GetPid()
		ppid := e.GetPpid()
		if ppid != 0 && pid != ppid {
			if parent, ok := nodes[ppid]; ok {
				parent.children = append(parent.children, pid)
			}
		}
	}

	if rootPID == 0 {
		return nil, fmt.Errorf("no root process found in pstree.img")
	}

	// BFS from root — mirrors ProcessTreePIDs traversal order
	queue := []uint32{rootPID}
	ordered := make([]uint32, 0, len(nodes))
	for len(queue) > 0 {
		pid := queue[0]
		queue = queue[1:]
		ordered = append(ordered, pid)
		if n, ok := nodes[pid]; ok {
			queue = append(queue, n.children...)
		}
	}

	return ordered, nil
}

// BuildPIDMapping builds an old→new PID mapping by aligning two BFS-ordered PID lists.
// originalBFS comes from PstreeOrderedPIDs (checkpoint-time pstree.img), restoredBFS
// comes from ProcessTreePIDs (post-restore /proc walk). Both lists traverse the same
// tree structure, so position i in each corresponds to the same process.
func BuildPIDMapping(originalBFS []uint32, restoredBFS []int) (map[int]int, error) {
	if len(originalBFS) != len(restoredBFS) {
		return nil, fmt.Errorf("pstree BFS has %d entries but restored tree has %d — tree structure mismatch",
			len(originalBFS), len(restoredBFS))
	}
	mapping := make(map[int]int, len(originalBFS))
	for i, oldPID := range originalBFS {
		mapping[int(oldPID)] = restoredBFS[i]
	}
	return mapping, nil
}
