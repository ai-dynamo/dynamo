package criu

import (
	"bytes"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"testing"

	"github.com/checkpoint-restore/go-criu/v8/crit"
	"github.com/checkpoint-restore/go-criu/v8/crit/images/fdinfo"
	"github.com/checkpoint-restore/go-criu/v8/crit/images/fown"
	sk_opts "github.com/checkpoint-restore/go-criu/v8/crit/images/sk-opts"
	sk_unix "github.com/checkpoint-restore/go-criu/v8/crit/images/sk-unix"
	"google.golang.org/protobuf/proto"
)

func TestRewriteCUDASocketName(t *testing.T) {
	tests := []struct {
		name string
		in   []byte
		want []byte
	}{
		{
			name: "abstract with trailing nulls",
			in:   []byte("\x00cuda-uvmfd-4026543509-855\x00\x00"),
			want: []byte("\x00cuda-uvmfd-987654321-855\x00\x00"),
		},
		{name: "pathname", in: []byte("cuda-uvmfd-4026543509-855")},
		{name: "unnamed", in: nil},
		{name: "extra suffix", in: []byte("\x00cuda-uvmfd-4026543509-855-extra")},
		{name: "nonnumeric namespace", in: []byte("\x00cuda-uvmfd-source-855")},
		{name: "nonnumeric pid", in: []byte("\x00cuda-uvmfd-4026543509-main")},
		{name: "missing pid", in: []byte("\x00cuda-uvmfd-4026543509-")},
		{name: "embedded null", in: []byte("\x00cuda-uvmfd-4026543509\x00-855")},
		{name: "similar prefix", in: []byte("\x00cuda-uvmfdx-4026543509-855")},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, rewritten := rewriteCUDASocketName(test.in, 987654321)
			if test.want == nil {
				if rewritten {
					t.Fatalf("rewriteCUDASocketName(%q) unexpectedly rewrote to %q", test.in, got)
				}
				return
			}
			if !rewritten {
				t.Fatalf("rewriteCUDASocketName(%q) did not rewrite", test.in)
			}
			if !bytes.Equal(got, test.want) {
				t.Fatalf("rewriteCUDASocketName(%q) = %q, want %q", test.in, got, test.want)
			}
		})
	}
}

func TestPrepareRestoreImageDir(t *testing.T) {
	checkpointPath := t.TempDir()
	entries := []*fdinfo.FileEntry{
		newUnixSocketEntry(1, []byte("\x00cuda-uvmfd-4026543509-1\x00"), 101, 202),
		newUnixSocketEntry(2, []byte("\x00other-4026543509-1\x00"), 202, 101),
		newUnixSocketEntry(3, []byte("/tmp/cuda-uvmfd-4026543509-3"), 303, 0),
	}
	originalEntries := make([]*fdinfo.FileEntry, len(entries))
	for i, entry := range entries {
		originalEntries[i] = proto.Clone(entry).(*fdinfo.FileEntry)
	}
	canonicalFilesImage := writeFilesImage(t, checkpointPath, entries)
	pageImage := []byte("large page image stand-in")
	if err := os.WriteFile(filepath.Join(checkpointPath, "pages-1.img"), pageImage, 0600); err != nil {
		t.Fatalf("write pages image: %v", err)
	}

	imageDir, cleanup, err := prepareRestoreImageDirForSocketScope(checkpointPath, 987654321)
	if err != nil {
		t.Fatalf("prepareRestoreImageDirForSocketScope: %v", err)
	}
	defer cleanup()
	if imageDir == checkpointPath {
		t.Fatal("matching CUDA socket should use a private image directory")
	}

	privateEntries := readFilesImage(t, imageDir)
	originalEntries[0].Usk.Name = []byte("\x00cuda-uvmfd-987654321-1\x00")
	for i, want := range originalEntries {
		if !proto.Equal(privateEntries[i], want) {
			t.Errorf("private entry %d changed outside the expected name rewrite:\n got: %v\nwant: %v", i, privateEntries[i], want)
		}
	}

	linkInfo, err := os.Lstat(filepath.Join(imageDir, "pages-1.img"))
	if err != nil {
		t.Fatalf("lstat private pages image: %v", err)
	}
	if linkInfo.Mode()&os.ModeSymlink == 0 {
		t.Fatal("pages image was copied instead of exposed by symlink")
	}
	gotPageImage, err := os.ReadFile(filepath.Join(imageDir, "pages-1.img"))
	if err != nil {
		t.Fatalf("read private pages image: %v", err)
	}
	if !bytes.Equal(gotPageImage, pageImage) {
		t.Fatalf("private pages image = %q, want %q", gotPageImage, pageImage)
	}

	gotCanonicalFilesImage, err := os.ReadFile(filepath.Join(checkpointPath, filesImageFilename))
	if err != nil {
		t.Fatalf("read canonical files image: %v", err)
	}
	if !bytes.Equal(gotCanonicalFilesImage, canonicalFilesImage) {
		t.Fatal("canonical files.img was modified")
	}

	cleanup()
	if _, err := os.Stat(imageDir); !os.IsNotExist(err) {
		t.Fatalf("private image directory still exists after cleanup: %v", err)
	}
}

func TestPrepareRestoreImageDirWithoutCUDASocketsUsesCheckpoint(t *testing.T) {
	checkpointPath := t.TempDir()
	writeFilesImage(t, checkpointPath, []*fdinfo.FileEntry{
		newUnixSocketEntry(1, []byte("\x00other-123-1"), 101, 0),
	})

	imageDir, cleanup, err := prepareRestoreImageDirForSocketScope(checkpointPath, 987654321)
	if err != nil {
		t.Fatalf("prepareRestoreImageDirForSocketScope: %v", err)
	}
	defer cleanup()
	if imageDir != checkpointPath {
		t.Fatalf("image directory = %q, want canonical checkpoint %q", imageDir, checkpointPath)
	}
}

func TestPrepareRestoreImageDirConcurrentSocketScopesAreIndependent(t *testing.T) {
	checkpointPath := t.TempDir()
	canonicalFilesImage := writeFilesImage(t, checkpointPath, []*fdinfo.FileEntry{
		newUnixSocketEntry(1, []byte("\x00cuda-uvmfd-4026543509-855"), 101, 0),
	})

	type result struct {
		socketScope uint64
		path        string
		cleanup     func()
		err         error
	}
	results := make(chan result, 2)
	var wg sync.WaitGroup
	for _, socketScope := range []uint64{111111, 222222} {
		wg.Add(1)
		go func() {
			defer wg.Done()
			path, cleanup, err := prepareRestoreImageDirForSocketScope(checkpointPath, socketScope)
			results <- result{socketScope: socketScope, path: path, cleanup: cleanup, err: err}
		}()
	}
	wg.Wait()
	close(results)

	var paths []string
	for result := range results {
		if result.err != nil {
			t.Fatalf("prepare restore view for socket scope %d: %v", result.socketScope, result.err)
		}
		defer result.cleanup()
		paths = append(paths, result.path)
		entries := readFilesImage(t, result.path)
		want := []byte("\x00cuda-uvmfd-" + strconv.FormatUint(result.socketScope, 10) + "-855")
		if got := entries[0].Usk.Name; !bytes.Equal(got, want) {
			t.Fatalf("socket name for scope %d = %q, want %q", result.socketScope, got, want)
		}
	}
	if paths[0] == paths[1] {
		t.Fatalf("concurrent restores shared private image directory %q", paths[0])
	}

	gotCanonicalFilesImage, err := os.ReadFile(filepath.Join(checkpointPath, filesImageFilename))
	if err != nil {
		t.Fatalf("read canonical files image: %v", err)
	}
	if !bytes.Equal(gotCanonicalFilesImage, canonicalFilesImage) {
		t.Fatal("canonical files.img was modified by concurrent restores")
	}
}

func newUnixSocketEntry(id uint32, name []byte, inode, peer uint32) *fdinfo.FileEntry {
	zero32 := uint32(0)
	zero64 := uint64(0)
	socketType := uint32(1)
	socketState := uint32(10)
	return &fdinfo.FileEntry{
		Type: fdinfo.FdTypes_UNIXSK.Enum(),
		Id:   proto.Uint32(id),
		Usk: &sk_unix.UnixSkEntry{
			Id:      proto.Uint32(id),
			Ino:     proto.Uint32(inode),
			Type:    &socketType,
			State:   &socketState,
			Flags:   &zero32,
			Uflags:  &zero32,
			Backlog: &zero32,
			Peer:    proto.Uint32(peer),
			Fown: &fown.FownEntry{
				Uid:     &zero32,
				Euid:    &zero32,
				Signum:  &zero32,
				PidType: &zero32,
				Pid:     &zero32,
			},
			Opts: &sk_opts.SkOptsEntry{
				SoSndbuf:     &zero32,
				SoRcvbuf:     &zero32,
				SoSndTmoSec:  &zero64,
				SoSndTmoUsec: &zero64,
				SoRcvTmoSec:  &zero64,
				SoRcvTmoUsec: &zero64,
				SoMark:       proto.Uint32(77),
			},
			Name: name,
			NsId: proto.Uint32(9),
		},
	}
}

func writeFilesImage(t *testing.T, dir string, entries []*fdinfo.FileEntry) []byte {
	t.Helper()

	file, err := os.OpenFile(filepath.Join(dir, filesImageFilename), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		t.Fatalf("create files image: %v", err)
	}
	imageEntries := make([]*crit.CriuEntry, len(entries))
	for i, entry := range entries {
		imageEntries[i] = &crit.CriuEntry{Message: entry}
	}
	image := &crit.CriuImage{
		Magic:     "FILES",
		Entries:   imageEntries,
		EntryType: &fdinfo.FileEntry{},
	}
	if err := crit.New(nil, file, "", false, false).Encode(image); err != nil {
		_ = file.Close()
		t.Fatalf("encode files image: %v", err)
	}
	if err := file.Close(); err != nil {
		t.Fatalf("close files image: %v", err)
	}
	data, err := os.ReadFile(filepath.Join(dir, filesImageFilename))
	if err != nil {
		t.Fatalf("read files image: %v", err)
	}
	return data
}

func readFilesImage(t *testing.T, dir string) []*fdinfo.FileEntry {
	t.Helper()

	file, err := os.Open(filepath.Join(dir, filesImageFilename))
	if err != nil {
		t.Fatalf("open files image: %v", err)
	}
	defer file.Close()
	image, err := crit.New(file, nil, "", false, false).Decode(&fdinfo.FileEntry{})
	if err != nil {
		t.Fatalf("decode files image: %v", err)
	}
	entries := make([]*fdinfo.FileEntry, len(image.Entries))
	for i, entry := range image.Entries {
		entries[i] = entry.Message.(*fdinfo.FileEntry)
	}
	return entries
}
