package runtime

import (
	"errors"
	"os"
	"testing"

	"golang.org/x/sys/unix"
)

func TestPrepareDetachedRootfsMountCleansLoopOnMountFailure(t *testing.T) {
	oldOpenControl := openLoopControl
	oldRetInt := loopIoctlRetInt
	oldOpenDevice := openLoopDevice
	oldCreateNode := createLoopNode
	oldStatDevice := statLoopDevice
	oldConfigure := configureLoop
	oldClear := clearLoop
	oldMount := mountSquashfs
	t.Cleanup(func() {
		openLoopControl = oldOpenControl
		loopIoctlRetInt = oldRetInt
		openLoopDevice = oldOpenDevice
		createLoopNode = oldCreateNode
		statLoopDevice = oldStatDevice
		configureLoop = oldConfigure
		clearLoop = oldClear
		mountSquashfs = oldMount
	})

	control, err := os.Open("/dev/null")
	if err != nil {
		t.Fatal(err)
	}
	device, err := os.Open("/dev/null")
	if err != nil {
		t.Fatal(err)
	}
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
	configureLoop = func(_ int, config *unix.LoopConfig) error {
		want := uint32(unix.LO_FLAGS_READ_ONLY | unix.LO_FLAGS_AUTOCLEAR)
		if config.Info.Flags != want {
			t.Fatalf("loop flags = %#x, want %#x", config.Info.Flags, want)
		}
		return nil
	}
	cleared := false
	clearLoop = func(int, uint, int) error {
		cleared = true
		return nil
	}
	mountSquashfs = func(string, string, string, uintptr, string) error {
		return errors.New("mount failed")
	}

	image, err := os.CreateTemp(t.TempDir(), "image")
	if err != nil {
		t.Fatal(err)
	}
	defer image.Close()
	if _, err := PrepareDetachedRootfsMount(image); err == nil {
		t.Fatal("expected mount failure")
	}
	if !cleared {
		t.Fatal("loop device was not cleared after mount failure")
	}
}
