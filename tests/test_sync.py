import json
from pathlib import Path

from scripts.sync_policies import plan_uploads, sha256


def test_plan_detects_new_changed_and_skips_unchanged(tmp_path):
    (tmp_path / "acme_v1.md").write_text("one")
    (tmp_path / "acme_v2.md").write_text("two")
    (tmp_path / "notes.txt").write_text("ignored: not a policy file name")
    manifest = {"acme_v1.md": sha256(tmp_path / "acme_v1.md")}
    assert [f.name for _, _, f in plan_uploads(tmp_path, manifest)] == ["acme_v2.md"]  # v1 unchanged -> skipped
    (tmp_path / "acme_v1.md").write_text("one, edited")
    assert [f.name for _, _, f in plan_uploads(tmp_path, manifest)] == ["acme_v1.md", "acme_v2.md"]  # edit detected, ordered by version
