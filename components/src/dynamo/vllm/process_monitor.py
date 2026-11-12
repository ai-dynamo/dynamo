import logging

import psutil

logger = logging.getLogger(__name__)


class LocalEngineProcesses:
    """Track owned local EngineCore and fixed-topology execution processes."""

    def __init__(self, engine_client):
        resources = getattr(
            getattr(engine_client, "engine_core", None), "resources", None
        )
        manager = getattr(resources, "engine_manager", None)
        self.processes = []
        parallel_config = engine_client.vllm_config.parallel_config
        track_workers = not getattr(parallel_config, "enable_elastic_ep", False)
        for handle in getattr(manager, "processes", ()):
            if handle.pid is None:
                continue
            core = psutil.Process(handle.pid)
            core.create_time()
            self.processes.append(core)
            if track_workers:
                for child in core.children(recursive=True):
                    if child.name().startswith(("VLLM::Worker", "VllmWorker")):
                        child.create_time()
                        self.processes.append(child)
        if not self.processes:
            logger.warning(
                "No owned local vLLM processes available; independent process "
                "monitoring is unavailable for this engine topology"
            )

    def check(self):
        for process in self.processes:
            try:
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    continue
            except psutil.NoSuchProcess:
                pass
            raise RuntimeError(f"vLLM process {process.pid} exited")
