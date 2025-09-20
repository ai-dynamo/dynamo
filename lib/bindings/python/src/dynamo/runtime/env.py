import os
import logging


def get_env(name_new: str, name_old: str | None = None, default: str | None = None) -> str | None:
    if (val := os.getenv(name_new)) is not None:
        return val
    if name_old is not None and (val := os.getenv(name_old)) is not None:
        logging.warning(
            f"Environment variable '{name_old}' is deprecated, use '{name_new}' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return val
    return default
