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
	sk_inet "github.com/checkpoint-restore/go-criu/v8/crit/images/sk-inet"
	sk_opts "github.com/checkpoint-restore/go-criu/v8/crit/images/sk-opts"
	sk_unix "github.com/checkpoint-restore/go-criu/v8/crit/images/sk-unix"
	"golang.org/x/sys/unix"
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
		newAutobindSocketEntry(4, []byte("\x000a1b2"), 404, uint32(unix.SOCK_DGRAM), linuxUnixSocketStateClose, 0),
		newAutobindSocketEntry(5, []byte("\x00abcde"), 505, uint32(unix.SOCK_DGRAM), linuxUnixSocketStateClose, 0),
		newAutobindSocketEntry(6, []byte("\x0012345"), 606, uint32(unix.SOCK_DGRAM), linuxUnixSocketStateClose, 404),
		newAutobindSocketEntry(7, []byte("\x00f00ba"), 707, uint32(unix.SOCK_STREAM), linuxUnixSocketStateClose, 0),
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
	if got, want := filepath.Dir(imageDir), filepath.Dir(checkpointPath); got != want {
		t.Fatalf("private image directory parent = %q, want %q", got, want)
	}

	privateEntries := readFilesImage(t, imageDir)
	originalEntries[0].Usk.Name = []byte("\x00cuda-uvmfd-987654321-1\x00")
	originalEntries[3].Usk.Name = []byte("\x000a1b2-987654321")
	originalEntries[4].Usk.Name = []byte("\x00abcde-987654321")
	for i, want := range originalEntries {
		if !proto.Equal(privateEntries[i], want) {
			t.Errorf("private entry %d changed outside the expected name rewrite:\n got: %v\nwant: %v", i, privateEntries[i], want)
		}
	}

	canonicalPageInfo, err := os.Lstat(filepath.Join(checkpointPath, "pages-1.img"))
	if err != nil {
		t.Fatalf("lstat canonical pages image: %v", err)
	}
	linkInfo, err := os.Lstat(filepath.Join(imageDir, "pages-1.img"))
	if err != nil {
		t.Fatalf("lstat private pages image: %v", err)
	}
	if linkInfo.Mode()&os.ModeSymlink != 0 || !linkInfo.Mode().IsRegular() {
		t.Fatalf("private pages image mode = %v, want regular non-symlink", linkInfo.Mode())
	}
	if !os.SameFile(canonicalPageInfo, linkInfo) {
		t.Fatal("private pages image is not a hard link to the canonical image")
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
		t.Cleanup(result.cleanup)
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

func TestPrepareRestoreImageDirRewritesEstablishedIPv6ClientPort(t *testing.T) {
	checkpointPath := t.TempDir()
	entries := newTCPRestoreTopology()
	other := newIPv6TCPSocketEntry(4, 104, 40001, 40002)
	other.Isk.State = proto.Uint32(7)
	other.Isk.Family = proto.Uint32(unix.AF_INET)
	other.Isk.SrcAddr, other.Isk.DstAddr = []uint32{0}, []uint32{0}
	entries = append(entries, other)
	canonicalFilesImage := writeFilesImage(t, checkpointPath, entries)
	writeTCPStreamImages(t, checkpointPath, 102, 103)

	imageDir, cleanup, err := prepareRestoreImageDirForSocketScope(checkpointPath, 987654321)
	if err != nil {
		t.Fatalf("prepareRestoreImageDirForSocketScope: %v", err)
	}
	defer cleanup()
	if imageDir == checkpointPath {
		t.Fatal("established TCP pair should use a private image directory")
	}

	privateEntries := readFilesImage(t, imageDir)
	clientPort := privateEntries[1].Isk.GetSrcPort()
	if clientPort == 0 || clientPort > 65535 {
		t.Fatalf("rewritten client port = %d, want 1..65535", clientPort)
	}
	for _, forbidden := range []uint32{46730, 52103, 40001, 40002} {
		if clientPort == forbidden {
			t.Fatalf("rewritten client port = %d, matches canonical port %d", clientPort, forbidden)
		}
	}
	if got := privateEntries[2].Isk.GetDstPort(); got != clientPort {
		t.Fatalf("rewritten server destination port = %d, want client port %d", got, clientPort)
	}

	for i, original := range entries {
		want := proto.Clone(original).(*fdinfo.FileEntry)
		if i == 1 {
			want.Isk.SrcPort = proto.Uint32(clientPort)
		}
		if i == 2 {
			want.Isk.DstPort = proto.Uint32(clientPort)
		}
		if !proto.Equal(privateEntries[i], want) {
			t.Errorf("private entry %d changed outside the expected port rewrite:\n got: %v\nwant: %v", i, privateEntries[i], want)
		}
	}
	gotCanonical, err := os.ReadFile(filepath.Join(checkpointPath, filesImageFilename))
	if err != nil || !bytes.Equal(gotCanonical, canonicalFilesImage) {
		t.Fatalf("canonical files image changed: %v", err)
	}

	if fd, err := bindDualStackTCPPort(clientPort, false); err == nil {
		_ = unix.Close(fd)
		t.Fatalf("strict bind to reserved TCP port %d unexpectedly succeeded", clientPort)
	}
	reuseFD, err := bindDualStackTCPPort(clientPort, true)
	if err != nil {
		t.Fatalf("CRIU-like bind to reserved TCP port %d: %v", clientPort, err)
	}
	_ = unix.Close(reuseFD)

	cleanup()
	fd, err := bindDualStackTCPPort(clientPort, false)
	if err != nil {
		t.Fatalf("bind to released TCP port %d: %v", clientPort, err)
	}
	cleanup()
	if _, err := unix.Getsockname(fd); err != nil {
		t.Fatalf("repeated cleanup closed reused socket FD: %v", err)
	}
	_ = unix.Close(fd)
}

func TestPrepareRestoreImageDirTCPRewriteFailsClosed(t *testing.T) {
	wildcard := []uint32{0, 0, 0, 0}
	for _, name := range []string{"nonreciprocal", "ambiguous", "indeterminate roles"} {
		t.Run(name, func(t *testing.T) {
			entries := newTCPRestoreTopology()
			switch name {
			case "nonreciprocal":
				entries = entries[:2]
			case "ambiguous":
				duplicate := proto.Clone(entries[2]).(*fdinfo.FileEntry)
				duplicate.Id = proto.Uint32(4)
				duplicate.Isk.Id = proto.Uint32(4)
				duplicate.Isk.Ino = proto.Uint32(104)
				entries = append(entries, duplicate)
			case "indeterminate roles":
				listener := newIPv6TCPSocketEntry(4, 104, 46730, 0)
				listener.Isk.State = proto.Uint32(linuxTCPStateListen)
				listener.Isk.SrcAddr, listener.Isk.DstAddr = wildcard, wildcard
				entries = append(entries, listener)
			}
			checkpointPath := t.TempDir()
			writeFilesImage(t, checkpointPath, entries)
			for _, entry := range entries {
				if entry.Isk.GetState() == linuxTCPStateEstablished {
					writeTCPStreamImages(t, checkpointPath, entry.Isk.GetIno())
				}
			}

			imageDir, cleanup, err := prepareRestoreImageDirForSocketScope(checkpointPath, 987654321)
			if err != nil {
				t.Fatalf("prepareRestoreImageDirForSocketScope: %v", err)
			}
			defer cleanup()
			if imageDir != checkpointPath {
				t.Fatalf("unsupported TCP graph used private image directory %q", imageDir)
			}
		})
	}
}

func TestPrepareRestoreImageDirConcurrentTCPPortsAreIndependent(t *testing.T) {
	checkpointPath := t.TempDir()
	writeFilesImage(t, checkpointPath, newTCPRestoreTopology())
	writeTCPStreamImages(t, checkpointPath, 102, 103)

	type result struct {
		path    string
		cleanup func()
		err     error
	}
	results := make(chan result, 3)
	for i := 0; i < 3; i++ {
		go func() {
			path, cleanup, err := prepareRestoreImageDirForSocketScope(checkpointPath, 987654321)
			results <- result{path, cleanup, err}
		}()
	}

	ports := make(map[uint32]struct{})
	for range 3 {
		result := <-results
		if result.err != nil {
			t.Fatalf("prepare restore view: %v", result.err)
		}
		t.Cleanup(result.cleanup)
		port := readFilesImage(t, result.path)[1].Isk.GetSrcPort()
		if _, exists := ports[port]; exists {
			t.Fatalf("concurrent restore views shared rewritten TCP port %d", port)
		}
		ports[port] = struct{}{}
	}
	if len(ports) != 3 {
		t.Fatalf("concurrent restores used %d TCP ports, want 3", len(ports))
	}
}

func newAutobindSocketEntry(id uint32, name []byte, inode, socketType, state, peer uint32) *fdinfo.FileEntry {
	entry := newUnixSocketEntry(id, name, inode, peer)
	entry.Usk.Type = proto.Uint32(socketType)
	entry.Usk.State = proto.Uint32(state)
	return entry
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

func newTCPRestoreTopology() []*fdinfo.FileEntry {
	wildcard := []uint32{0, 0, 0, 0}
	podAddress := []uint32{0, 0, 0xffff0000, 0x0100007f}
	entries := []*fdinfo.FileEntry{
		newIPv6TCPSocketEntry(1, 101, 52103, 0),
		newIPv6TCPSocketEntry(2, 102, 46730, 52103),
		newIPv6TCPSocketEntry(3, 103, 52103, 46730),
	}
	entries[0].Isk.State = proto.Uint32(linuxTCPStateListen)
	entries[0].Isk.SrcAddr, entries[0].Isk.DstAddr = wildcard, wildcard
	entries[1].Isk.SrcAddr, entries[1].Isk.DstAddr = podAddress, podAddress
	entries[2].Isk.SrcAddr, entries[2].Isk.DstAddr = podAddress, podAddress
	return entries
}

func newIPv6TCPSocketEntry(id, inode, srcPort, dstPort uint32) *fdinfo.FileEntry {
	base := newUnixSocketEntry(id, nil, inode, 0).Usk
	return &fdinfo.FileEntry{
		Type: fdinfo.FdTypes_INETSK.Enum(),
		Id:   proto.Uint32(id),
		Isk: &sk_inet.InetSkEntry{
			Id:      proto.Uint32(id),
			Ino:     proto.Uint32(inode),
			Family:  proto.Uint32(uint32(unix.AF_INET6)),
			Type:    proto.Uint32(uint32(unix.SOCK_STREAM)),
			Proto:   proto.Uint32(uint32(unix.IPPROTO_TCP)),
			State:   proto.Uint32(linuxTCPStateEstablished),
			SrcPort: proto.Uint32(srcPort),
			DstPort: proto.Uint32(dstPort),
			Flags:   base.Flags,
			Backlog: base.Backlog,
			Fown:    base.Fown,
			Opts:    base.Opts,
			V6Only:  proto.Bool(false),
			NsId:    proto.Uint32(9),
		},
	}
}

func writeTCPStreamImages(t *testing.T, dir string, inodes ...uint32) {
	t.Helper()
	for _, inode := range inodes {
		name := filepath.Join(dir, "tcp-stream-"+strconv.FormatUint(uint64(inode), 16)+".img")
		if err := os.WriteFile(name, []byte("tcp stream"), 0600); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}
}

func bindDualStackTCPPort(port uint32, reuse bool) (int, error) {
	fd, err := unix.Socket(unix.AF_INET6, unix.SOCK_STREAM|unix.SOCK_CLOEXEC, unix.IPPROTO_TCP)
	if err != nil {
		return -1, err
	}
	if err := unix.SetsockoptInt(fd, unix.IPPROTO_IPV6, unix.IPV6_V6ONLY, 0); err != nil {
		_ = unix.Close(fd)
		return -1, err
	}
	address := &unix.SockaddrInet6{Port: int(port)}
	if reuse {
		for _, option := range []int{unix.SO_REUSEADDR, unix.SO_REUSEPORT} {
			if err := unix.SetsockoptInt(fd, unix.SOL_SOCKET, option, 1); err != nil {
				_ = unix.Close(fd)
				return -1, err
			}
		}
		address.Addr = [16]byte{10: 0xff, 11: 0xff, 12: 127, 15: 1}
	}
	if err := unix.Bind(fd, address); err != nil {
		_ = unix.Close(fd)
		return -1, err
	}
	return fd, nil
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
	defer func() {
		if err := file.Close(); err != nil {
			t.Errorf("close files image: %v", err)
		}
	}()
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
