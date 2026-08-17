"""Privileged CRIU regression for parallel buffered restore task identity."""

import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time
import unittest


_SOURCE = r'''
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

struct worker { unsigned char *map; size_t bytes; pthread_barrier_t *barrier; int fd; const char *stop; uint64_t checksum; };

static long tid(void) { return syscall(SYS_gettid); }
static int stopped(const char *path) { return access(path, F_OK) == 0; }
static void line(int fd, const char *kind, uint64_t sum) {
  char value[160];
  int n = snprintf(value, sizeof(value), "%s %ld %ld %llu\n", kind, (long)getpid(), tid(), (unsigned long long)sum);
  if (n > 0 && (size_t)n < sizeof(value)) (void)write(fd, value, (size_t)n);
}
static void *heartbeat(void *opaque) {
  struct worker *w = opaque;
  pthread_barrier_wait(w->barrier);
  line(w->fd, "B", w->checksum);
  while (!stopped(w->stop)) {
    line(w->fd, "H", w->checksum);
    struct timespec wait = {.tv_sec = 0, .tv_nsec = 20000000};
    nanosleep(&wait, NULL);
  }
  return NULL;
}
static uint64_t fill_and_checksum(unsigned char *map, size_t bytes, unsigned int seed) {
  uint64_t state = 0x9e3779b97f4a7c15ULL ^ seed, sum = 1469598103934665603ULL;
  for (size_t i = 0; i < bytes; ++i) {
    state ^= state << 13; state ^= state >> 7; state ^= state << 17;
    map[i] = (unsigned char)state;
    sum = (sum ^ map[i]) * 1099511628211ULL;
  }
  return sum;
}
static void child(const char *status, const char *stop, size_t bytes, unsigned threads, unsigned seed) {
  unsigned char *map = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (map == MAP_FAILED || madvise(map, bytes, MADV_NOHUGEPAGE)) _exit(2);
  int fd = open(status, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0600);
  if (fd < 0) _exit(3);
  uint64_t checksum = fill_and_checksum(map, bytes, seed);
  pthread_t *ids = calloc(threads, sizeof(*ids));
  struct worker worker = {.map = map, .bytes = bytes, .fd = fd, .stop = stop, .checksum = checksum};
  pthread_barrier_t barrier;
  if (!ids || pthread_barrier_init(&barrier, NULL, threads)) _exit(4);
  worker.barrier = &barrier;
  for (unsigned i = 0; i < threads; ++i) if (pthread_create(&ids[i], NULL, heartbeat, &worker)) _exit(5);
  for (unsigned i = 0; i < threads; ++i) pthread_join(ids[i], NULL);
  munmap(map, bytes); close(fd); _exit(0);
}
int main(int argc, char **argv) {
  if (argc != 6) return 64;
  const char *status = argv[1], *stop = argv[2];
  unsigned children = (unsigned)strtoul(argv[3], NULL, 10), threads = (unsigned)strtoul(argv[4], NULL, 10);
  const size_t bytes = (size_t)strtoull(argv[5], NULL, 10);
  if (!children || !threads || bytes < (320UL * 1024UL * 1024UL)) return 64;
  for (unsigned i = 0; i < children; ++i) { pid_t pid = fork(); if (pid < 0) return 1; if (!pid) child(status, stop, bytes, threads, i + 1); }
  while (!stopped(stop)) { struct timespec wait = {.tv_sec = 0, .tv_nsec = 20000000}; nanosleep(&wait, NULL); }
  while (wait(NULL) > 0 || errno == EINTR) {}
  return 0;
}
'''


def _parallel_restore_patch():
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "deploy/snapshot/criu-parallel-buffered-restore.patch"
        if candidate.is_file():
            return candidate.read_text()
    raise RuntimeError("CRIU parallel restore patch not found")


class ParallelRestorePatchTest(unittest.TestCase):
    """Keep the cross-task identity barrier fail-closed in the candidate patch."""

    def test_patch_counts_alive_tasks_and_aborts_the_shared_barrier(self):
        patch = _parallel_restore_patch()
        self.assertIn("futex_t buffered_restore_barrier;", patch)
        self.assertIn("int buffered_restore_participants;", patch)
        self.assertIn("if (task_alive(item))", patch)
        self.assertIn("futex_abort_and_wake(&task_entries_local->buffered_restore_barrier);", patch)
        self.assertIn("wait_buffered_restore_barrier", patch)

    def test_patch_treats_reader_echild_as_a_restore_failure(self):
        patch = _parallel_restore_patch()
        echild = patch.index("if (waited == -ECHILD)")
        following = patch[echild : echild + 400]
        self.assertIn("shared->error = -1;", following)
        self.assertNotIn("already reaped", following)
        self.assertNotIn("continue;", following)


class MultiprocessRestoreBarrierTest(unittest.TestCase):
    """Exercise a CRIU binary built with the buffered parallel restorer."""

    children = 3
    threads_per_child = 16

    def setUp(self):
        if os.environ.get("CRIU_MULTIPROCESS_REGRESSION") != "1":
            self.skipTest("set CRIU_MULTIPROCESS_REGRESSION=1 to run privileged CRIU regression")
        if os.geteuid() != 0:
            self.skipTest("CRIU dump/restore requires root")
        self.criu = os.environ.get("CRIU_BIN", "criu")
        if shutil.which(self.criu) is None:
            self.skipTest("CRIU_BIN is not executable")
        self.mapping_mib = int(os.environ.get("CRIU_MULTIPROCESS_MAPPING_MIB", "320"))
        if self.mapping_mib < 320:
            self.fail("CRIU_MULTIPROCESS_MAPPING_MIB must be at least 320")
        configured_threads = int(os.environ.get("CRIU_MULTIPROCESS_THREADS", str(self.threads_per_child)))
        if configured_threads < 16:
            self.fail("CRIU_MULTIPROCESS_THREADS must be at least 16")
        self.threads_per_child = configured_threads

    @staticmethod
    def _records(path, offset=0):
        with path.open("rb") as stream:
            stream.seek(offset)
            return [tuple(line.decode().split()) for line in stream if line.strip()]

    def _wait_for(self, predicate, message, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.05)
        self.fail(message)

    @staticmethod
    def _reader_counts(logs):
        counts = {}
        pattern = re.compile(r"^.*?\s(?P<owner>[0-9]+):.*Buffered VMA reader started with pid [0-9]+", re.MULTILINE)
        for match in pattern.finditer(logs):
            owner = match.group("owner")
            counts[owner] = counts.get(owner, 0) + 1
        return counts

    def test_parallel_buffered_dump_restore_preserves_each_process_thread_barrier(self):
        with tempfile.TemporaryDirectory(prefix="criu-multiprocess-") as temporary:
            root = Path(temporary)
            source, binary = root / "fixture.c", root / "fixture"
            status, stop, images = root / "status", root / "stop", root / "images"
            source.write_text(_SOURCE)
            subprocess.run(["gcc", "-O2", "-pthread", str(source), "-o", str(binary)], check=True, capture_output=True)
            images.mkdir()
            driver = subprocess.Popen([
                str(binary), str(status), str(stop), str(self.children), str(self.threads_per_child),
                str(self.mapping_mib * 1024 * 1024),
            ])
            try:
                expected = self.children * self.threads_per_child
                before = self._wait_for(
                    lambda: self._records(status) if status.exists() and len([r for r in self._records(status) if r[0] == "B"]) == expected else None,
                    "all process-local pthread barriers did not release",
                )
                barrier = {(record[1], record[2], record[3]) for record in before if record[0] == "B"}
                self.assertEqual(len(barrier), expected, "each child must report every distinct TID")
                offset = status.stat().st_size
                dumped = subprocess.run(
                    [self.criu, "dump", "-t", str(driver.pid), "-D", str(images), "--shell-job", "--leave-running", "-v4", "--log-file", "dump.log"],
                    text=True, capture_output=True, timeout=180,
                )
                self.assertEqual(dumped.returncode, 0, dumped.stderr)
                self.assertTrue(
                    any(path.stat().st_size for path in images.glob("pages-*.img")),
                    "dump did not create a non-empty page image for the mapped fixture data",
                )
                # Keep the tree running through dump, then let its leader stop
                # and reap every child before restore. A normal dump leaves
                # child zombies owned by container init; their retained PIDs
                # would collide with clone3(set_tid), masking reader/barrier
                # bugs.
                stop.touch()
                driver.wait(timeout=10)
                restored = subprocess.run(
                    [self.criu, "restore", "-D", str(images), "--restore-detached", "--shell-job", "-v4", "--log-file", "restore.log"],
                    text=True, capture_output=True, timeout=180,
                )
                restore_log = (images / "restore.log").read_text(errors="replace")
                self.assertEqual(restored.returncode, 0, restored.stderr + "\n" + restore_log)
                after = self._wait_for(
                    lambda: self._records(status, offset) if barrier <= {(r[1], r[2], r[3]) for r in self._records(status, offset) if r[0] == "H"} else None,
                    "restored tree lost a child/TID heartbeat or changed an incompressible mapping checksum",
                    timeout=180,
                )
                self.assertTrue(any(record[0] == "H" for record in after))
                restore_logs = "\n".join(path.read_text(errors="replace") for path in images.glob("*restore*.log"))
                reader_counts = self._reader_counts(restore_logs)
                self.assertGreaterEqual(
                    sum(count >= 8 for count in reader_counts.values()), 2,
                    "parallel buffered restore did not log eight readers for at least two task processes",
                )
            finally:
                stop.touch()
                try:
                    driver.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.kill(driver.pid, signal.SIGKILL)
                    driver.wait()


if __name__ == "__main__":
    unittest.main()
