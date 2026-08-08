package criu

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"

	"github.com/checkpoint-restore/go-criu/v8/crit"
	"github.com/checkpoint-restore/go-criu/v8/crit/images/fdinfo"
	sk_inet "github.com/checkpoint-restore/go-criu/v8/crit/images/sk-inet"
	sk_unix "github.com/checkpoint-restore/go-criu/v8/crit/images/sk-unix"
	"golang.org/x/sys/unix"
)

const (
	filesImageFilename            = "files.img"
	placeholderMountNamespacePath = "/proc/self/ns/mnt"
	cudaSocketNameStart           = "\x00cuda-uvmfd-"
	// CRIU records an unconnected AF_UNIX socket with Linux's CLOSE state value.
	linuxUnixSocketStateClose = 7
	linuxTCPStateEstablished  = 1
	linuxTCPStateListen       = 10
)

type tcpPortRewrite struct {
	client *sk_inet.InetSkEntry
	server *sk_inet.InetSkEntry
}

func prepareRestoreImageDir(checkpointPath string) (string, func(), error) {
	// The placeholder mount namespace remains container-specific with shareProcessNamespace,
	// so its inode uniquely scopes names that would collide in the shared network namespace.
	var stat unix.Stat_t
	if err := unix.Stat(placeholderMountNamespacePath, &stat); err != nil {
		return "", nil, fmt.Errorf("failed to stat placeholder mount namespace at %s: %w", placeholderMountNamespacePath, err)
	}
	return prepareRestoreImageDirForSocketScope(checkpointPath, stat.Ino)
}

func prepareRestoreImageDirForSocketScope(checkpointPath string, restoreSocketScope uint64) (string, func(), error) {
	checkpointPath, err := filepath.Abs(checkpointPath)
	if err != nil {
		return "", nil, fmt.Errorf("failed to resolve checkpoint path: %w", err)
	}

	filesImage, err := os.Open(filepath.Join(checkpointPath, filesImageFilename))
	if err != nil {
		return "", nil, fmt.Errorf("failed to open %s: %w", filesImageFilename, err)
	}

	image, err := crit.New(filesImage, nil, "", false, false).Decode(&fdinfo.FileEntry{})
	closeErr := filesImage.Close()
	if err != nil {
		return "", nil, fmt.Errorf("failed to decode %s: %w", filesImageFilename, err)
	}
	if closeErr != nil {
		return "", nil, fmt.Errorf("failed to close %s: %w", filesImageFilename, closeErr)
	}

	tcpRewrite, forbiddenPorts := planTCPPortRewrite(image, checkpointPath)
	reservationFD := -1
	var restorePort uint32
	if tcpRewrite != nil {
		restorePort, reservationFD, err = reserveDualStackTCPPort(forbiddenPorts)
		if err != nil {
			return "", nil, fmt.Errorf("failed to reserve restore TCP port: %w", err)
		}
	}

	rewritten := false
	for _, entry := range image.Entries {
		fileEntry := entry.Message.(*fdinfo.FileEntry)
		if fileEntry.GetType() != fdinfo.FdTypes_UNIXSK || fileEntry.Usk == nil {
			continue
		}
		if name, ok := rewriteCUDASocketName(fileEntry.Usk.Name, restoreSocketScope); ok {
			fileEntry.Usk.Name = name
			rewritten = true
		}
		if rewriteLinuxAutobindDatagramSocketName(fileEntry.Usk, restoreSocketScope) {
			rewritten = true
		}
	}
	if tcpRewrite != nil {
		*tcpRewrite.client.SrcPort = restorePort
		*tcpRewrite.server.DstPort = restorePort
		rewritten = true
	}
	if !rewritten {
		return checkpointPath, func() {}, nil
	}

	privateDir, err := os.MkdirTemp(filepath.Dir(checkpointPath), ".dynamo-criu-restore-*")
	if err != nil {
		if reservationFD >= 0 {
			_ = unix.Close(reservationFD)
		}
		return "", nil, fmt.Errorf("failed to create private CRIU image directory: %w", err)
	}
	var cleanupOnce sync.Once
	cleanup := func() {
		cleanupOnce.Do(func() {
			if reservationFD >= 0 {
				_ = unix.Close(reservationFD)
			}
			_ = os.RemoveAll(privateDir)
		})
	}
	fail := func(err error) (string, func(), error) {
		cleanup()
		return "", nil, err
	}

	entries, err := os.ReadDir(checkpointPath)
	if err != nil {
		return fail(fmt.Errorf("failed to read checkpoint directory: %w", err))
	}
	for _, entry := range entries {
		name := entry.Name()
		if name == filesImageFilename || !strings.HasSuffix(name, ".img") {
			continue
		}
		// CRIU does not modify restore images; hard links keep them visible after it
		// enters the restored mount namespace without copying large page images.
		if err := os.Link(filepath.Join(checkpointPath, name), filepath.Join(privateDir, name)); err != nil {
			return fail(fmt.Errorf("failed to hard-link CRIU image %s: %w", name, err))
		}
	}

	privateFilesImage, err := os.OpenFile(filepath.Join(privateDir, filesImageFilename), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return fail(fmt.Errorf("failed to create private %s: %w", filesImageFilename, err))
	}
	if err := crit.New(nil, privateFilesImage, "", false, false).Encode(image); err != nil {
		_ = privateFilesImage.Close()
		return fail(fmt.Errorf("failed to encode private %s: %w", filesImageFilename, err))
	}
	if err := privateFilesImage.Close(); err != nil {
		return fail(fmt.Errorf("failed to close private %s: %w", filesImageFilename, err))
	}

	return privateDir, cleanup, nil
}

func planTCPPortRewrite(image *crit.CriuImage, checkpointPath string) (*tcpPortRewrite, map[uint32]struct{}) {
	var files []*fdinfo.FileEntry
	idOwners := make(map[uint32]int)
	inodeOwners := make(map[uint32]int)
	forbiddenPorts := make(map[uint32]struct{})

	for _, entry := range image.Entries {
		file, ok := entry.Message.(*fdinfo.FileEntry)
		if !ok {
			continue
		}
		if file.GetType() != fdinfo.FdTypes_INETSK {
			continue
		}
		files = append(files, file)
		if id := file.GetId(); id != 0 {
			idOwners[id]++
		}
		if file.Isk == nil {
			continue
		}
		if id := file.Isk.GetId(); id != 0 && id != file.GetId() {
			idOwners[id]++
		}
		if inode := file.Isk.GetIno(); inode != 0 {
			inodeOwners[inode]++
		}
		if port := file.Isk.GetSrcPort(); port != 0 {
			forbiddenPorts[port] = struct{}{}
		}
		if port := file.Isk.GetDstPort(); port != 0 {
			forbiddenPorts[port] = struct{}{}
		}
	}

	var rewrites []tcpPortRewrite
	for _, endpoint := range files {
		if !isEligibleEstablished(endpoint, idOwners, inodeOwners, checkpointPath) {
			continue
		}
		peer, ok := soleReciprocal(endpoint.Isk, files)
		if !ok || endpoint.GetId() > peer.GetId() {
			continue
		}
		if !isEligibleEstablished(peer, idOwners, inodeOwners, checkpointPath) {
			continue
		}
		reverse, ok := soleReciprocal(peer.Isk, files)
		if !ok || reverse != endpoint {
			continue
		}
		endpointRole := listenerRole(endpoint.Isk, files, idOwners, inodeOwners)
		peerRole := listenerRole(peer.Isk, files, idOwners, inodeOwners)
		switch {
		case endpointRole == 0 && peerRole == 1:
			rewrites = append(rewrites, tcpPortRewrite{client: endpoint.Isk, server: peer.Isk})
		case endpointRole == 1 && peerRole == 0:
			rewrites = append(rewrites, tcpPortRewrite{client: peer.Isk, server: endpoint.Isk})
		}
	}
	if len(rewrites) != 1 {
		return nil, forbiddenPorts
	}
	return &rewrites[0], forbiddenPorts
}

func isEligibleINET(
	file *fdinfo.FileEntry,
	idOwners, inodeOwners map[uint32]int,
) bool {
	if file == nil {
		return false
	}
	socket := file.Isk
	return file.GetType() == fdinfo.FdTypes_INETSK &&
		file.GetId() != 0 &&
		socket != nil &&
		socket.GetId() != 0 &&
		file.GetId() == socket.GetId() &&
		socket.GetIno() != 0 &&
		socket.SrcPort != nil &&
		socket.DstPort != nil &&
		socket.V6Only != nil &&
		socket.NsId != nil &&
		socket.GetFamily() == uint32(unix.AF_INET6) &&
		socket.GetType() == uint32(unix.SOCK_STREAM) &&
		socket.GetProto() == uint32(unix.IPPROTO_TCP) &&
		!socket.GetV6Only() &&
		idOwners[socket.GetId()] == 1 &&
		inodeOwners[socket.GetIno()] == 1
}

func isEligibleEstablished(
	file *fdinfo.FileEntry,
	idOwners, inodeOwners map[uint32]int,
	checkpointPath string,
) bool {
	socket := file.Isk
	return isEligibleINET(file, idOwners, inodeOwners) &&
		isIPv4MappedEstablished(socket) &&
		hasRegularTCPStreamImage(checkpointPath, socket.GetIno())
}

func isIPv4MappedEstablished(socket *sk_inet.InetSkEntry) bool {
	return socket != nil &&
		socket.GetFamily() == uint32(unix.AF_INET6) &&
		socket.GetType() == uint32(unix.SOCK_STREAM) &&
		socket.GetProto() == uint32(unix.IPPROTO_TCP) &&
		socket.GetState() == linuxTCPStateEstablished &&
		socket.GetSrcPort() > 0 &&
		socket.GetSrcPort() <= 65535 &&
		socket.GetDstPort() > 0 &&
		socket.GetDstPort() <= 65535 &&
		ipv4MappedAddress(socket.SrcAddr) &&
		ipv4MappedAddress(socket.DstAddr)
}

func hasRegularTCPStreamImage(checkpointPath string, inode uint32) bool {
	info, err := os.Lstat(filepath.Join(checkpointPath, fmt.Sprintf("tcp-stream-%x.img", inode)))
	return err == nil && info.Mode().IsRegular()
}

func soleReciprocal(
	endpoint *sk_inet.InetSkEntry,
	files []*fdinfo.FileEntry,
) (*fdinfo.FileEntry, bool) {
	var peer *fdinfo.FileEntry
	for _, candidate := range files {
		socket := candidate.Isk
		if !isIPv4MappedEstablished(socket) ||
			endpoint == socket ||
			endpoint.GetNsId() != socket.GetNsId() ||
			endpoint.GetSrcPort() != socket.GetDstPort() ||
			endpoint.GetDstPort() != socket.GetSrcPort() ||
			!slices.Equal(endpoint.SrcAddr, socket.DstAddr) ||
			!slices.Equal(endpoint.DstAddr, socket.SrcAddr) {
			continue
		}
		if peer != nil {
			return nil, false
		}
		peer = candidate
	}
	return peer, peer != nil
}

func listenerRole(
	endpoint *sk_inet.InetSkEntry,
	files []*fdinfo.FileEntry,
	idOwners, inodeOwners map[uint32]int,
) int {
	role := 0
	for _, file := range files {
		socket := file.Isk
		if socket == nil ||
			socket.GetType() != uint32(unix.SOCK_STREAM) ||
			socket.GetProto() != uint32(unix.IPPROTO_TCP) ||
			socket.GetState() != linuxTCPStateListen ||
			socket.GetNsId() != endpoint.GetNsId() ||
			socket.GetSrcPort() != endpoint.GetSrcPort() {
			continue
		}
		if role != 0 || !isEligibleWildcardListener(file, idOwners, inodeOwners) {
			return -1
		}
		role = 1
	}
	return role
}

func isEligibleWildcardListener(
	file *fdinfo.FileEntry,
	idOwners, inodeOwners map[uint32]int,
) bool {
	socket := file.Isk
	return isEligibleINET(file, idOwners, inodeOwners) &&
		socket.GetState() == linuxTCPStateListen &&
		socket.GetSrcPort() > 0 &&
		socket.GetSrcPort() <= 65535 &&
		socket.GetDstPort() == 0 &&
		!socket.GetV6Only() &&
		slices.Equal(socket.SrcAddr, []uint32{0, 0, 0, 0}) &&
		slices.Equal(socket.DstAddr, []uint32{0, 0, 0, 0})
}

func ipv4MappedAddress(address []uint32) bool {
	return len(address) == 4 &&
		address[0] == 0 &&
		address[1] == 0 &&
		address[2] == 0xffff0000
}

func reserveDualStackTCPPort(forbidden map[uint32]struct{}) (uint32, int, error) {
	var rejected []int
	defer func() {
		for _, fd := range rejected {
			_ = unix.Close(fd)
		}
	}()

	for {
		fd, err := unix.Socket(unix.AF_INET6, unix.SOCK_STREAM|unix.SOCK_CLOEXEC, unix.IPPROTO_TCP)
		if err != nil {
			return 0, -1, err
		}
		if err := unix.SetsockoptInt(fd, unix.IPPROTO_IPV6, unix.IPV6_V6ONLY, 0); err != nil {
			_ = unix.Close(fd)
			return 0, -1, err
		}
		if err := unix.Bind(fd, &unix.SockaddrInet6{}); err != nil {
			_ = unix.Close(fd)
			return 0, -1, err
		}

		boundAddress, err := unix.Getsockname(fd)
		if err != nil {
			_ = unix.Close(fd)
			return 0, -1, err
		}
		address, ok := boundAddress.(*unix.SockaddrInet6)
		if !ok {
			_ = unix.Close(fd)
			return 0, -1, fmt.Errorf("unexpected bound socket address %T", boundAddress)
		}
		port := uint32(address.Port)
		if port == 0 || port > 65535 {
			_ = unix.Close(fd)
			return 0, -1, fmt.Errorf("kernel selected invalid TCP port %d", port)
		}
		if _, exists := forbidden[port]; exists {
			// Keep the bind until selection finishes so port 0 cannot immediately
			// return the same forbidden port.
			rejected = append(rejected, fd)
			continue
		}
		if err := unix.SetsockoptInt(fd, unix.SOL_SOCKET, unix.SO_REUSEADDR, 1); err != nil {
			_ = unix.Close(fd)
			return 0, -1, err
		}
		return port, fd, nil
	}
}

func rewriteCUDASocketName(name []byte, restoreSocketScope uint64) ([]byte, bool) {
	if !bytes.HasPrefix(name, []byte(cudaSocketNameStart)) {
		return nil, false
	}

	end := len(name)
	for end > len(cudaSocketNameStart) && name[end-1] == 0 {
		end--
	}
	numericFields := name[len(cudaSocketNameStart):end]
	separator := bytes.IndexByte(numericFields, '-')
	if separator <= 0 ||
		separator == len(numericFields)-1 ||
		!decimalDigits(numericFields[:separator]) ||
		!decimalDigits(numericFields[separator+1:]) {
		return nil, false
	}

	rewritten := make([]byte, 0, len(name)+20)
	rewritten = append(rewritten, cudaSocketNameStart...)
	rewritten = strconv.AppendUint(rewritten, restoreSocketScope, 10)
	rewritten = append(rewritten, numericFields[separator:]...)
	rewritten = append(rewritten, name[end:]...)
	return rewritten, true
}

func rewriteLinuxAutobindDatagramSocketName(entry *sk_unix.UnixSkEntry, restoreSocketScope uint64) bool {
	if entry == nil ||
		entry.Type == nil ||
		entry.State == nil ||
		entry.Peer == nil ||
		entry.Backlog == nil ||
		entry.Uflags == nil ||
		*entry.Type != uint32(unix.SOCK_DGRAM) ||
		*entry.State != linuxUnixSocketStateClose ||
		*entry.Peer != 0 ||
		*entry.Backlog != 0 ||
		*entry.Uflags != 0 ||
		entry.FilePerms != nil ||
		entry.NameDir != nil ||
		(entry.Deleted != nil && *entry.Deleted) ||
		len(entry.Name) != 6 ||
		entry.Name[0] != 0 {
		return false
	}
	for _, digit := range entry.Name[1:] {
		if (digit < '0' || digit > '9') && (digit < 'a' || digit > 'f') {
			return false
		}
	}

	rewritten := make([]byte, 0, len(entry.Name)+1+20)
	rewritten = append(rewritten, entry.Name...)
	rewritten = append(rewritten, '-')
	rewritten = strconv.AppendUint(rewritten, restoreSocketScope, 10)
	entry.Name = rewritten
	return true
}

func decimalDigits(value []byte) bool {
	for _, digit := range value {
		if digit < '0' || digit > '9' {
			return false
		}
	}
	return len(value) > 0
}
