import os
import subprocess
import sys

_WATCHDOG = """
import os
import select
import signal
import sys

parent_pid = int(sys.argv[1])
timeout = float(sys.argv[2])
readable, _, _ = select.select([sys.stdin], [], [], timeout)
if not readable and os.getppid() == parent_pid:
    os.kill(parent_pid, signal.SIGKILL)
"""


class ShutdownWatchdog:
    """Enforce a fatal-shutdown deadline outside the worker's Python GIL."""

    def __init__(self, timeout):
        self.process = subprocess.Popen(
            [sys.executable, "-c", _WATCHDOG, str(os.getpid()), str(timeout)],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=5)
