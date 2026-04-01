"""Build metrics JSON parser — extracted from BuildMetricsProcessor._extract_build_metrics."""

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional


def parse_build_metrics_json(path: Path) -> Optional[Dict[str, Any]]:
    """
    Parse build metrics from a JSON file **or** a ZIP archive containing one.

    In the cron path *path* is a ZIP downloaded from GitHub artifacts.
    In the CI push path *path* is the JSON file directly on disk.

    Returns:
        Dict with ``container``, ``stages``, ``layers`` keys, or ``None``.
    """
    if not path.exists():
        print(f"Build metrics file not found: {path}")
        return None

    # Direct JSON file
    if path.suffix == ".json":
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading build metrics JSON {path}: {e}")
            return None

    # ZIP archive (artifact download path)
    if path.suffix == ".zip":
        return _extract_from_zip(path)

    # Try as JSON regardless of extension
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return _extract_from_zip(path)


def _extract_from_zip(zip_path: Path) -> Optional[Dict[str, Any]]:
    """Extract build metrics JSON from a ZIP archive."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for file_name in zip_ref.namelist():
                if (
                    file_name.endswith("build_metrics.json")
                    or file_name == "build_metrics.json"
                    or (file_name.startswith("metrics-") and file_name.endswith(".json"))
                    or file_name.endswith("-metrics.json")
                    or (file_name.startswith("build-") and file_name.endswith(".json"))
                ):
                    with zip_ref.open(file_name) as f:
                        return json.load(f)
        return None
    except Exception as e:
        print(f"Error extracting build metrics from ZIP {zip_path}: {e}")
        return None
