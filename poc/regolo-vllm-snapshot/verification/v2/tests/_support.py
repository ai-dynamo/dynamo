import importlib.util
import pathlib
import sys


V2_ROOT = pathlib.Path(__file__).resolve().parents[1]
HARNESS = V2_ROOT / "harness" / "v2_harness.py"


def load_harness():
    if not HARNESS.is_file():
        raise FileNotFoundError(f"missing required V2-A harness: {HARNESS}")
    spec = importlib.util.spec_from_file_location("v2_harness_under_test", HARNESS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {HARNESS}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
