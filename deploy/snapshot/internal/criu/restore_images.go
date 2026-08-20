package criu

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"net/netip"
	"os"
	"path/filepath"
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
	cudaUVMFDSocketNamePrefix     = "\x00cuda-uvmfd-"
	// NCCL's bootstrap sockets are abstract and named
	// "\0tmp/nccl-socket-<rank>-<hash>", zero-padded to the full sun_path.
	ncclAbstractSocketNamePrefix = "\x00tmp/nccl-"
	linuxUnixSocketStateListen    = 10
	// A kernel autobind address is "\0" followed by exactly five hex digits.
	autobindNameHexDigits = 5
	linuxTCPStateEstablished      = 1
	linuxTCPStateClose            = 7
	linuxTCPStateListen           = 10
)

type tcpPortRewrite struct {
	socket *sk_inet.InetSkEntry
	peers  []*sk_inet.InetSkEntry
	port   uint32
}

func prepareRestoreImageDir(checkpointPath string) (string, func(), error) {
	// The placeholder mount namespace remains container-specific with shareProcessNamespace.
	var stat unix.Stat_t
	if err := unix.Stat(placeholderMountNamespacePath, &stat); err != nil {
		return "", nil, fmt.Errorf("failed to stat placeholder mount namespace at %s: %w", placeholderMountNamespacePath, err)
	}
	return prepareRestoreImageDirForRestoreID(checkpointPath, stat.Ino)
}

func prepareRestoreImageDirForRestoreID(checkpointPath string, restoreID uint64) (string, func(), error) {
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

	// CRIU recreates bound sockets from files.img. Give each clone a private
	// view so containers sharing a Pod network namespace get unique identities.
	tcpRewrites, tcpDisconnects, forbiddenPorts := planTCPPortRewrites(image)
	var reservationFDs []int
	for i := range tcpRewrites {
		var reservationFD int
		tcpRewrites[i].port, reservationFD, err = reserveDualStackTCPPort(forbiddenPorts)
		if err != nil {
			closeFDs(reservationFDs)
			return "", nil, fmt.Errorf("failed to reserve replacement TCP port: %w", err)
		}
		reservationFDs = append(reservationFDs, reservationFD)
		forbiddenPorts[tcpRewrites[i].port] = struct{}{}
	}

	rewritten := false
	for _, entry := range image.Entries {
		fileEntry := entry.Message.(*fdinfo.FileEntry)
		if fileEntry.GetType() != fdinfo.FdTypes_UNIXSK || fileEntry.Usk == nil {
			continue
		}
		did := rewriteCloneConflictingUnixSocketAddress(fileEntry.Usk, restoreID)
		if did {
			rewritten = true
		}
		// DIAGNOSTIC, temporary. Clone-restore keeps failing on unix sockets
		// whose address CRIU prints as empty, and the log alone cannot say
		// which entries the predicate matched. Dump the raw shape of every
		// unix socket so the ones that collide can be identified precisely
		// rather than inferred from the symptom.
		u := fileEntry.Usk
		fmt.Fprintf(os.Stderr,
			"DYNDBG unixsk id=%d ino=%d type=%d state=%d peer=%d namelen=%d name=%x rewritten=%v\n",
			u.GetId(), u.GetIno(), u.GetType(), u.GetState(), u.GetPeer(),
			len(u.GetName()), u.GetName(), did)
	}
	for _, rewrite := range tcpRewrites {
		*rewrite.socket.SrcPort = rewrite.port
		for _, peer := range rewrite.peers {
			*peer.DstPort = rewrite.port
		}
		rewritten = true
	}
	for _, socket := range tcpDisconnects {
		// A remote peer cannot follow a cloned connection to its new tuple.
		// Preserve the FD, but restore it as an unconnected TCP socket.
		*socket.State = linuxTCPStateClose
		*socket.SrcPort = 0
		*socket.DstPort = 0
		clear(socket.SrcAddr)
		clear(socket.DstAddr)
		rewritten = true
	}
	if !rewritten {
		return checkpointPath, func() {}, nil
	}

	privateDir, err := os.MkdirTemp(filepath.Dir(checkpointPath), ".dynamo-criu-restore-*")
	if err != nil {
		closeFDs(reservationFDs)
		return "", nil, fmt.Errorf("failed to create private CRIU image directory: %w", err)
	}
	var cleanupOnce sync.Once
	cleanup := func() {
		cleanupOnce.Do(func() {
			closeFDs(reservationFDs)
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

func planTCPPortRewrites(image *crit.CriuImage) (
	[]tcpPortRewrite,
	[]*sk_inet.InetSkEntry,
	map[uint32]struct{},
) {
	var sockets []*sk_inet.InetSkEntry
	forbiddenPorts := make(map[uint32]struct{})

	for _, entry := range image.Entries {
		file, ok := entry.Message.(*fdinfo.FileEntry)
		if !ok || file.GetType() != fdinfo.FdTypes_INETSK || file.Isk == nil {
			continue
		}
		sockets = append(sockets, file.Isk)
		if port := file.Isk.GetSrcPort(); port != 0 {
			forbiddenPorts[port] = struct{}{}
		}
		if port := file.Isk.GetDstPort(); port != 0 {
			forbiddenPorts[port] = struct{}{}
		}
	}

	var rewrites []tcpPortRewrite
	var disconnects []*sk_inet.InetSkEntry
	// Keep listener ports stable. Only rewrite connections whose reciprocal
	// endpoint is also in the image, so both halves retain a valid TCP tuple.
	for _, socket := range sockets {
		if !isEstablishedTCP(socket) || !hasSupportedTCPAddresses(socket) {
			continue
		}
		rewrite := tcpPortRewrite{socket: socket}
		for _, peer := range sockets {
			if reciprocalTCPPair(socket, peer) {
				rewrite.peers = append(rewrite.peers, peer)
			}
		}
		if len(rewrite.peers) == 0 {
			disconnects = append(disconnects, socket)
		} else if len(rewrite.peers) == 1 && !hasTCPListener(socket, sockets) {
			rewrites = append(rewrites, rewrite)
		}
	}
	return rewrites, disconnects, forbiddenPorts
}

func isTCPSocket(socket *sk_inet.InetSkEntry) bool {
	return socket != nil &&
		socket.SrcPort != nil &&
		socket.DstPort != nil &&
		socket.NsId != nil &&
		(socket.GetFamily() == uint32(unix.AF_INET) ||
			socket.GetFamily() == uint32(unix.AF_INET6)) &&
		socket.GetType() == uint32(unix.SOCK_STREAM) &&
		socket.GetProto() == uint32(unix.IPPROTO_TCP)
}

func isEstablishedTCP(socket *sk_inet.InetSkEntry) bool {
	return isTCPSocket(socket) &&
		socket.GetState() == linuxTCPStateEstablished &&
		socket.GetSrcPort() > 0 &&
		socket.GetDstPort() > 0
}

func reciprocalTCPPair(a, b *sk_inet.InetSkEntry) bool {
	aSrc, aSrcOK := normalizedIPAddress(a.GetFamily(), a.SrcAddr)
	aDst, aDstOK := normalizedIPAddress(a.GetFamily(), a.DstAddr)
	bSrc, bSrcOK := normalizedIPAddress(b.GetFamily(), b.SrcAddr)
	bDst, bDstOK := normalizedIPAddress(b.GetFamily(), b.DstAddr)
	return a != b &&
		isEstablishedTCP(a) &&
		isEstablishedTCP(b) &&
		aSrcOK && aDstOK && bSrcOK && bDstOK &&
		a.GetNsId() == b.GetNsId() &&
		a.GetSrcPort() == b.GetDstPort() &&
		a.GetDstPort() == b.GetSrcPort() &&
		aSrc == bDst &&
		aDst == bSrc
}

func hasSupportedTCPAddresses(socket *sk_inet.InetSkEntry) bool {
	_, srcOK := normalizedIPAddress(socket.GetFamily(), socket.SrcAddr)
	_, dstOK := normalizedIPAddress(socket.GetFamily(), socket.DstAddr)
	return srcOK && dstOK
}

func normalizedIPAddress(family uint32, words []uint32) (netip.Addr, bool) {
	switch family {
	case unix.AF_INET:
		if len(words) != 1 {
			return netip.Addr{}, false
		}
		var address [4]byte
		binary.LittleEndian.PutUint32(address[:], words[0])
		return netip.AddrFrom4(address), true
	case unix.AF_INET6:
		if len(words) != 4 {
			return netip.Addr{}, false
		}
		var address [16]byte
		for i, word := range words {
			binary.LittleEndian.PutUint32(address[i*4:], word)
		}
		return netip.AddrFrom16(address).Unmap(), true
	default:
		return netip.Addr{}, false
	}
}

func hasTCPListener(
	endpoint *sk_inet.InetSkEntry,
	sockets []*sk_inet.InetSkEntry,
) bool {
	for _, socket := range sockets {
		if !isTCPSocket(socket) ||
			socket.GetState() != linuxTCPStateListen ||
			socket.GetFamily() != endpoint.GetFamily() ||
			socket.GetNsId() != endpoint.GetNsId() ||
			socket.GetSrcPort() != endpoint.GetSrcPort() ||
			socket.GetDstPort() != 0 {
			continue
		}
		return true
	}
	return false
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

func closeFDs(fds []int) {
	for _, fd := range fds {
		_ = unix.Close(fd)
	}
}

func rewriteCloneConflictingUnixSocketAddress(entry *sk_unix.UnixSkEntry, restoreID uint64) bool {
	if !isCUDAUVMFDListener(entry) && !isAutoboundSocket(entry) && !isNCCLAbstractSocket(entry) {
		return false
	}

	// Only the clone-private address changes; the file descriptor is preserved.
	input := make([]byte, 8+len(entry.Name))
	binary.BigEndian.PutUint64(input, restoreID)
	copy(input[8:], entry.Name)
	digest := sha256.Sum256(input)
	entry.Name = hex.AppendEncode([]byte("\x00dynamo-"), digest[:])
	return true
}

func isCUDAUVMFDListener(entry *sk_unix.UnixSkEntry) bool {
	return entry != nil &&
		entry.Type != nil &&
		entry.State != nil &&
		entry.Peer != nil &&
		*entry.Type == uint32(unix.SOCK_SEQPACKET) &&
		*entry.State == linuxUnixSocketStateListen &&
		*entry.Peer == 0 &&
		bytes.HasPrefix(entry.Name, []byte(cudaUVMFDSocketNamePrefix))
}

// isAutoboundSocket matches a socket the kernel named itself. bind() with a
// zero-length address yields an abstract address of exactly "\0" followed by
// five hex digits, so the name carries no meaning and no peer can resolve it:
// it is anonymous by construction.
//
// Restoring one checkpoint into sibling containers otherwise rebuilds the same
// autobind name in a shared namespace and the second bind fails with
// EADDRINUSE -- which is exactly how intra-pod shadow failover fails:
//
//	unix: bind id 0x1fc ino 4156147439 addr :
//	Error (criu/sk-unix.c:1691): unix: Can't bind ... Address already in use
//
// Giving each clone a distinct address is safe here in a way that renaming a
// deliberately-named listener is not -- there is nothing to look the address up
// with.
//
// Caveat: a datagram socket can be addressed by a sender that learned the
// address at runtime without appearing as a peer. Such a sender would have to
// live outside the checkpoint and have cached the address, which does not
// happen for the per-process sockets observed here.
func isAutoboundSocket(entry *sk_unix.UnixSkEntry) bool {
	if entry == nil || entry.Peer == nil || *entry.Peer != 0 {
		return false
	}
	name := bytes.TrimSuffix(entry.Name, []byte{0})
	if len(name) != 1+autobindNameHexDigits || name[0] != 0 {
		return false
	}
	for _, c := range name[1:] {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}

// isNCCLAbstractSocket matches NCCL's abstract bootstrap sockets, named
// "\0tmp/nccl-socket-<rank>-<hash>" and zero-padded to the full sun_path.
//
// Both engines of an intra-pod failover worker restore the same checkpoint, so
// they rebuild identical NCCL socket names in a shared namespace and the second
// bind fails with EADDRINUSE:
//
//	unix: Can't bind id 0x620 ino 1228807150 addr : Address already in use
//
// CRIU prints the address as empty because it is abstract -- the name begins
// with a NUL and the printer stops there -- which makes these look exactly like
// the autobind case in the log while having a completely different shape.
//
// Renaming is safe here for the same reason it is for the CUDA UVM listener:
// the entry has no peer in the image, so nothing being restored resolves it by
// name, and the engine rebuilds its collectives after restore anyway --
// checkpoint_prepare tears the communicators down before the dump and
// checkpoint_restore re-establishes them afterwards.
func isNCCLAbstractSocket(entry *sk_unix.UnixSkEntry) bool {
	return entry != nil &&
		entry.Peer != nil &&
		*entry.Peer == 0 &&
		bytes.HasPrefix(entry.Name, []byte(ncclAbstractSocketNamePrefix))
}
