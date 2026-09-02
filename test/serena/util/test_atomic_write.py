from pathlib import Path
from unittest.mock import patch

import pytest

from serena.util.atomic_write import atomic_write_text


def test_atomic_write_replaces_complete_file_and_preserves_mode(tmp_path: Path) -> None:
    target = tmp_path / "state.txt"
    target.write_text("before\n", encoding="utf-8")
    target.chmod(0o640)

    atomic_write_text(target, "after\n", encoding="utf-8")

    assert target.read_text(encoding="utf-8") == "after\n"
    assert target.stat().st_mode & 0o777 == 0o640
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_write_failure_preserves_original_and_cleans_temporary(
    tmp_path: Path,
) -> None:
    target = tmp_path / "state.txt"
    target.write_text("before\n", encoding="utf-8")

    with (
        patch("serena.util.atomic_write.os.replace", side_effect=OSError("injected")),
        pytest.raises(OSError, match="injected"),
    ):
        atomic_write_text(target, "after\n", encoding="utf-8")

    assert target.read_text(encoding="utf-8") == "before\n"
    assert list(tmp_path.iterdir()) == [target]
