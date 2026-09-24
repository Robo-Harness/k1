from pathlib import Path
import shutil

import pytest

from robo_harness.release_audit import APPROVED_ASSETS, audit


@pytest.mark.parametrize("relative", sorted(APPROVED_ASSETS))
def test_reviewed_documentation_assets_pass_audit(tmp_path, relative):
    root = Path(__file__).resolve().parents[1]
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / relative, target)
    assert audit(tmp_path)["passed"]


def test_approved_filename_does_not_allow_unreviewed_content(tmp_path):
    target = tmp_path / "assets" / "logo.png"
    target.parent.mkdir()
    target.write_bytes(b"unreviewed binary content")
    result = audit(tmp_path)
    assert not result["passed"]
    assert result["findings"][0]["rule"] == "unreviewed_asset_content"


def test_other_images_are_not_implicitly_allowed(tmp_path):
    target = tmp_path / "assets" / "episode.png"
    target.parent.mkdir()
    target.write_bytes(b"unreviewed image")
    result = audit(tmp_path)
    assert not result["passed"]
    assert result["findings"][0]["rule"] == "unexpected_file_type"


def test_approved_asset_cannot_be_a_symlink(tmp_path):
    root = Path(__file__).resolve().parents[1]
    target = tmp_path / "assets" / "logo.png"
    target.parent.mkdir()
    target.symlink_to(root / "assets" / "logo.png")
    result = audit(tmp_path)
    assert not result["passed"]
    assert result["findings"][0]["rule"] == "symlink"
