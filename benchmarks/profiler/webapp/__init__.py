import importlib.metadata

from benchmarks.profiler.webapp.main import main

__version__ = importlib.metadata.version("aiconfigurator")

main()
