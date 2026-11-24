import time


class HealthTestWorkerExtension:
    """Block the execution queue without modifying model weights."""

    def health_test_block(self, delay: float):
        time.sleep(delay)
