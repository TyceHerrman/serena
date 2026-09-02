from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path


def atomic_write_text(
    path: str | Path,
    content: str,
    *,
    encoding: str,
    newline: str | None = None,
) -> None:
    """Durably stage text beside its destination, then replace it atomically."""
    destination = Path(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    descriptor_open = True
    try:
        if destination.exists():
            os.fchmod(descriptor, stat.S_IMODE(destination.stat().st_mode))
        with os.fdopen(
            descriptor,
            "w",
            encoding=encoding,
            newline=newline,
        ) as handle:
            descriptor_open = False
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if descriptor_open:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
