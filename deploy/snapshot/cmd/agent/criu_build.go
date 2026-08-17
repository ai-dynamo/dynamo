package main

import (
	"encoding/json"
	"fmt"
	"os"
)

const criuBuildInfoPath = "/usr/local/share/snapshot/criu-build.json"

type criuBuildInfo struct {
	CRIURef     string `json:"criu_ref"`
	FastRestore struct {
		PrivateVMANativeAIO struct {
			OriginPR string `json:"origin_pr"`
			Depth    int    `json:"depth"`
		} `json:"private_vma_native_aio"`
		ParallelMemfdFill struct {
			OriginPR   string `json:"origin_pr"`
			MaxThreads int    `json:"max_threads"`
		} `json:"parallel_memfd_fill"`
		DirectIOFallback        string `json:"direct_io_fallback"`
		ParallelBufferedReaders int    `json:"parallel_buffered_readers"`
	} `json:"fast_restore"`
}

func readCRIUBuildInfo(path string) (criuBuildInfo, error) {
	var info criuBuildInfo
	data, err := os.ReadFile(path)
	if err != nil {
		return info, err
	}
	if err := json.Unmarshal(data, &info); err != nil {
		return info, fmt.Errorf("decode CRIU build metadata: %w", err)
	}
	bufferedReaders := info.FastRestore.ParallelBufferedReaders
	validBufferedReaders := bufferedReaders == 0 || bufferedReaders == 1 ||
		bufferedReaders == 2 || bufferedReaders == 4 || bufferedReaders == 8
	if info.CRIURef == "" || info.FastRestore.PrivateVMANativeAIO.OriginPR != "3022" ||
		info.FastRestore.PrivateVMANativeAIO.Depth < 64 ||
		info.FastRestore.ParallelMemfdFill.OriginPR != "3021" ||
		info.FastRestore.ParallelMemfdFill.MaxThreads < 8 ||
		info.FastRestore.DirectIOFallback != "buffered" || !validBufferedReaders {
		return info, fmt.Errorf("invalid CRIU fast-restore metadata")
	}
	return info, nil
}
