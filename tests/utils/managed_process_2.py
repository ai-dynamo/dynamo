import logging
import os
import subprocess
import sys
from logging.handlers import WatchedFileHandler
from threading import Thread

# Configure logging to file and stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Logs to stdout
        WatchedFileHandler("managed_process.log"),  # Logs to file
    ],
)


class ManagedProcess:
    def __init__(
        self,
        command: List[str],
        env: Optional[dict] = None,
        check_ports: Optional[List[int]] = None,
        timeout: int = 300,
        cwd: Optional[str] = None,
        data_dir: Optional[str] = None,
        ensure_not_running: bool = False,
    ):
        self.command = command
        self.env = env or os.environ.copy()
        self.check_ports = check_ports or []
        self.timeout = timeout
        self.cwd = cwd
        self.data_dir = data_dir
        self.ensure_not_running = ensure_not_running
        self.proc = None
        self.stdout_pipe = None
        self.stderr_pipe = None
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        if self.data_dir:
            cleanup_directory(self.data_dir)

        if self.ensure_not_running:
            if is_process_running(self.command[0]):
                raise RuntimeError("Process is already running")

        # Create pipes for subprocess output
        self.stdout_pipe = os.pipe()
        self.stderr_pipe = os.pipe()

        # Start logging threads
        self.stdout_thread = Thread(
            target=self._log_stream, args=("STDOUT", self.stdout_pipe[0])
        )
        self.stderr_thread = Thread(
            target=self._log_stream, args=("STDERR", self.stderr_pipe[0])
        )
        self.stdout_thread.daemon = True
        self.stderr_thread.daemon = True
        self.stdout_thread.start()
        self.stderr_thread.start()

        # Start process with redirected output
        self.proc = subprocess.Popen(
            self.command,
            env=self.env,
            cwd=self.cwd,
            stdout=self.stdout_pipe[1],
            stderr=self.stderr_pipe[1],
            text=True,  # For Python 3.7+
        )

        # Close write ends in parent
        os.close(self.stdout_pipe[1])
        os.close(self.stderr_pipe[1])

        # Wait for ports (if needed)
        if self.check_ports:
            self._wait_for_ports()

        return self.proc

    def _log_stream(self, stream_name: str, pipe_fd: int) -> None:
        while True:
            try:
                data = os.read(pipe_fd, 1024)
                if not data:
                    break
                clean_data = data.decode().strip()
                if clean_data:
                    for line in clean_data.split("\n"):
                        logging.info(f"[{stream_name}] {line}")
            except Exception as e:
                logging.error(f"Logging error: {e}")
                break
        os.close(pipe_fd)

    def _wait_for_ports(self):
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            if all(is_port_open(p) for p in self.check_ports):
                logging.info(f"All ports {self.check_ports} are ready")
                break
            time.sleep(0.1)
        else:
            self.proc.terminate()
            raise TimeoutError(f"Ports {self.check_ports} not ready in time")

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.info(f"Terminating process: {self.command[0]}")
        self.proc.terminate()

        # Wait for logging threads to finish
        self.stdout_thread.join(timeout=1)
        self.stderr_thread.join(timeout=1)

        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logging.warning("Process did not terminate gracefully, killing it")
            self.proc.kill()
            self.proc.wait()

        if self.data_dir:
            cleanup_directory(self.data_dir)


def is_process_running(name: str) -> bool:
    """Check if any process with the given name is running."""
    return any(proc.name() == name for proc in psutil.process_iter(["name"]))


# Existing helper functions (is_port_open, cleanup_directory, etc.)
