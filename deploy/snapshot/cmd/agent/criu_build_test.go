package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReadCRIUBuildInfo(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "criu-build.json")
	valid := `{"criu_ref":"b47c692","fast_restore":{"private_vma_native_aio":{"origin_pr":"3022","depth":256},"parallel_memfd_fill":{"origin_pr":"3021","max_threads":32},"direct_io_fallback":"buffered","parallel_buffered_readers":8}}`
	if err := os.WriteFile(path, []byte(valid), 0o600); err != nil {
		t.Fatal(err)
	}
	info, err := readCRIUBuildInfo(path)
	if err != nil {
		t.Fatalf("readCRIUBuildInfo() error = %v", err)
	}
	if info.CRIURef != "b47c692" {
		t.Fatalf("CRIURef = %q, want b47c692", info.CRIURef)
	}
	if info.FastRestore.PrivateVMANativeAIO.Depth != 256 {
		t.Fatalf("AIO depth = %d, want 256", info.FastRestore.PrivateVMANativeAIO.Depth)
	}
	if info.FastRestore.ParallelMemfdFill.MaxThreads != 32 {
		t.Fatalf("memfd max threads = %d, want 32", info.FastRestore.ParallelMemfdFill.MaxThreads)
	}
	if info.FastRestore.ParallelBufferedReaders != 8 {
		t.Fatalf("buffered readers = %d, want 8", info.FastRestore.ParallelBufferedReaders)
	}
}

func TestReadCRIUBuildInfoRejectsInvalidBufferedReaders(t *testing.T) {
	path := filepath.Join(t.TempDir(), "criu-build.json")
	invalid := `{"criu_ref":"b47c692","fast_restore":{"private_vma_native_aio":{"origin_pr":"3022","depth":128},"parallel_memfd_fill":{"origin_pr":"3021","max_threads":16},"direct_io_fallback":"buffered","parallel_buffered_readers":3}}`
	if err := os.WriteFile(path, []byte(invalid), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := readCRIUBuildInfo(path); err == nil {
		t.Fatal("readCRIUBuildInfo() accepted invalid buffered reader count")
	}
}

func TestReadCRIUBuildInfoRejectsIncompleteMetadata(t *testing.T) {
	path := filepath.Join(t.TempDir(), "criu-build.json")
	if err := os.WriteFile(path, []byte(`{"criu_ref":"b47c692"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := readCRIUBuildInfo(path); err == nil {
		t.Fatal("readCRIUBuildInfo() accepted incomplete metadata")
	}
}
