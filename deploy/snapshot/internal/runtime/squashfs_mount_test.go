package runtime

import (
	"errors"
	"os"
	"testing"

	"golang.org/x/sys/unix"
)

func stubLoopAllocation(t *testing.T) {
	t.Helper()
	oldOpenControl := openLoopControl
	oldRetInt := loopIoctlRetInt
	oldOpenDevice := openLoopDevice
	oldCreateNode := createLoopNode
	oldStatDevice := statLoopDevice
	oldConfigure := configureLoop
	oldClear := clearLoop
	oldMount := mountSquashfs
	oldUnmount := unmountPath
	oldOpenTree := openMountTree
	t.Cleanup(func() {
		openLoopControl = oldOpenControl
		loopIoctlRetInt = oldRetInt
		openLoopDevice = oldOpenDevice
		createLoopNode = oldCreateNode
		statLoopDevice = oldStatDevice
		configureLoop = oldConfigure
		clearLoop = oldClear
		mountSquashfs = oldMount
		unmountPath = oldUnmount
		openMountTree = oldOpenTree
	})

	control, err := os.Open("/dev/null")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { control.Close() })
	device, err := os.Open("/dev/null")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { device.Close() })
	openLoopControl = func(string, int, uint32) (int, error) {
		return unix.Dup(int(control.Fd()))
	}
	loopIoctlRetInt = func(int, uint) (int, error) { return 7, nil }
	openLoopDevice = func(string, int, uint32) (int, error) {
		return unix.Dup(int(device.Fd()))
	}
	statLoopDevice = func(_ int, stat *unix.Stat_t) error {
		stat.Mode = unix.S_IFBLK
		stat.Rdev = unix.Mkdev(7, 7)
		return nil
	}
}

func testRootfsImage(t *testing.T) *os.File {
	t.Helper()
	image, err := os.CreateTemp(t.TempDir(), "image")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { image.Close() })
	return image
}

func assertDirectoryEmpty(t *testing.T, path string) {
	t.Helper()
	entries, err := os.ReadDir(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("staging directory contains %d entries, want empty", len(entries))
	}
}

func TestConfigureAvailableLoopRetriesPastEightBusyDevices(t *testing.T) {
	stubLoopAllocation(t)
	attempts := 0
	configureLoop = func(_ int, _ *unix.LoopConfig) error {
		attempts++
		if attempts <= 9 {
			return unix.EBUSY
		}
		return nil
	}

	fd, _, err := configureAvailableLoop(testRootfsImage(t))
	if err != nil {
		t.Fatal(err)
	}
	if err := unix.Close(fd); err != nil {
		t.Fatal(err)
	}
	if attempts != 10 {
		t.Fatalf("LOOP_CONFIGURE attempts = %d, want 10", attempts)
	}
}

func TestPrepareDetachedRootfsMountClearsConfiguredLoopOnMountFailure(t *testing.T) {
	stubLoopAllocation(t)
	stagingDir := t.TempDir()
	mountErr := errors.New("mount failed")
	configuredFD := -1
	configureLoop = func(fd int, _ *unix.LoopConfig) error {
		configuredFD = fd
		return nil
	}
	clearedFD := -1
	clearLoop = func(fd int, request uint, value int) error {
		if request != unix.LOOP_CLR_FD || value != 0 {
			t.Fatalf("clear loop request = (%d, %d)", request, value)
		}
		clearedFD = fd
		return nil
	}
	mountSquashfs = func(string, string, string, uintptr, string) error {
		return mountErr
	}

	_, err := prepareDetachedRootfsMount(testRootfsImage(t), stagingDir)
	if !errors.Is(err, mountErr) {
		t.Fatalf("prepare error = %v, want %v", err, mountErr)
	}
	if clearedFD != configuredFD {
		t.Fatalf("cleared loop FD = %d, want configured FD %d", clearedFD, configuredFD)
	}
	assertDirectoryEmpty(t, stagingDir)
}

func TestPrepareDetachedRootfsMountUnmountsStagingOnOpenTreeFailure(t *testing.T) {
	stubLoopAllocation(t)
	stagingDir := t.TempDir()
	openTreeErr := errors.New("open_tree failed")
	configureLoop = func(int, *unix.LoopConfig) error { return nil }
	cleared := false
	clearLoop = func(int, uint, int) error {
		cleared = true
		return nil
	}
	mountSquashfs = func(string, string, string, uintptr, string) error { return nil }
	openMountTree = func(int, string, uint) (int, error) { return -1, openTreeErr }
	unmounted := false
	unmountPath = func(path string, flags int) error {
		if path == "" || flags != unix.MNT_DETACH {
			t.Fatalf("unmount staging = (%q, %d)", path, flags)
		}
		unmounted = true
		return nil
	}

	_, err := prepareDetachedRootfsMount(testRootfsImage(t), stagingDir)
	if !errors.Is(err, openTreeErr) {
		t.Fatalf("prepare error = %v, want %v", err, openTreeErr)
	}
	if !unmounted {
		t.Fatal("staging mount was not unmounted")
	}
	if !cleared {
		t.Fatal("configured loop was not cleared")
	}
	assertDirectoryEmpty(t, stagingDir)
}

func TestPrepareDetachedRootfsMountJoinsCleanupErrors(t *testing.T) {
	stubLoopAllocation(t)
	stagingDir := t.TempDir()
	openTreeErr := errors.New("open_tree failed")
	unmountErr := errors.New("unmount failed")
	clearErr := errors.New("clear failed")
	configureLoop = func(int, *unix.LoopConfig) error { return nil }
	clearLoop = func(int, uint, int) error { return clearErr }
	mountSquashfs = func(string, string, string, uintptr, string) error { return nil }
	openMountTree = func(int, string, uint) (int, error) { return -1, openTreeErr }
	unmountPath = func(string, int) error { return unmountErr }

	_, err := prepareDetachedRootfsMount(testRootfsImage(t), stagingDir)
	for _, want := range []error{openTreeErr, unmountErr, clearErr} {
		if !errors.Is(err, want) {
			t.Fatalf("prepare error = %v, want joined %v", err, want)
		}
	}
	assertDirectoryEmpty(t, stagingDir)
}
