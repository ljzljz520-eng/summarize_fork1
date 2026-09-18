"""TR-7.x tests for local path source containment and authorization."""

import os
import tempfile
from pathlib import Path

import pytest

from summarizer.exceptions import LocalSourceDenied
from summarizer.security.auth import Identity
from summarizer.security.localfs import (
    authorize_local_source,
    is_local_source_type,
    is_within_tempdir,
    resolve_within_roots,
)
from summarizer.security.settings import (
    MODE_AUTHENTICATED_SERVER,
    MODE_LOCAL_TRUSTED,
    SCOPE_READ,
    SCOPE_SOURCES_LOCAL,
    ServerSettings,
)


@pytest.fixture
def root_layout(tmp_path):
    root = tmp_path / "media"
    root.mkdir()
    inside = root / "video.mp4"
    inside.write_text("data")
    sibling_secret = tmp_path / "secret.mp4"
    sibling_secret.write_text("secret")
    nested = root / "sub"
    nested.mkdir()
    return {
        "root": root,
        "inside": inside,
        "sibling_secret": sibling_secret,
        "nested": nested,
    }


# ── TR-7.1: denial cases ────────────────────────────────────────────────────


def test_tr7_1_empty_roots_denied(root_layout):
    with pytest.raises(LocalSourceDenied):
        resolve_within_roots(root_layout["inside"], [])


def test_tr7_1_absolute_outside_path_denied(root_layout):
    with pytest.raises(LocalSourceDenied):
        resolve_within_roots("/etc/passwd", [root_layout["root"]])


def test_tr7_1_traversal_denied(root_layout):
    evil = root_layout["nested"] / ".." / ".." / "secret.mp4"
    with pytest.raises(LocalSourceDenied):
        resolve_within_roots(evil, [root_layout["root"]])


def test_tr7_1_symlink_escape_denied(root_layout):
    link = root_layout["root"] / "link.mp4"
    link.symlink_to(root_layout["sibling_secret"])
    with pytest.raises(LocalSourceDenied):
        resolve_within_roots(link, [root_layout["root"]])


def test_tr7_1_nonexistent_root_denied(tmp_path):
    with pytest.raises(LocalSourceDenied):
        resolve_within_roots("/etc/passwd", [tmp_path / "missing-root"])


# ── TR-7.2: allowed cases ───────────────────────────────────────────────────


def test_tr7_2_inside_file_returns_resolved(root_layout):
    result = resolve_within_roots(root_layout["inside"], [root_layout["root"]])
    assert result == root_layout["inside"].resolve()
    assert result.is_absolute()


def test_tr7_2_relative_path_within_cwd(root_layout, monkeypatch):
    monkeypatch.chdir(root_layout["root"])
    result = resolve_within_roots("video.mp4", [root_layout["root"]])
    assert result == root_layout["inside"].resolve()


def test_tr7_2_symlink_staying_inside_allowed(root_layout):
    target = root_layout["nested"] / "real.mp4"
    target.write_text("nested")
    link = root_layout["root"] / "alias.mp4"
    link.symlink_to(target)
    result = resolve_within_roots(link, [root_layout["root"]])
    assert result == target.resolve()


def test_tr7_2_multiple_roots(root_layout, tmp_path):
    other_root = tmp_path / "other"
    other_root.mkdir()
    other_file = other_root / "clip.mov"
    other_file.write_text("x")
    assert (
        resolve_within_roots(other_file, [root_layout["root"], other_root])
        == other_file.resolve()
    )
    assert (
        resolve_within_roots(root_layout["inside"], [root_layout["root"], other_root])
        == root_layout["inside"].resolve()
    )


def test_tr7_2_user_expansion(root_layout, monkeypatch):
    home_root = root_layout["root"]
    monkeypatch.setenv("HOME", str(home_root))
    result = resolve_within_roots("~/video.mp4", [home_root])
    assert result == root_layout["inside"].resolve()


# ── temp bypass / type helpers ──────────────────────────────────────────────


def test_tempdir_bypass_detection():
    fd, name = tempfile.mkstemp()
    os.close(fd)
    try:
        assert is_within_tempdir(name)
    finally:
        os.unlink(name)
    with tempfile.NamedTemporaryFile() as handle:
        assert is_within_tempdir(handle.name)
    assert not is_within_tempdir("/etc/passwd")


def test_local_source_type_predicate():
    assert is_local_source_type("Local File")
    assert is_local_source_type("TXT")
    assert not is_local_source_type("YouTube Video")
    assert not is_local_source_type(None)


# ── authorize_local_source policy matrix ────────────────────────────────────


def test_authorize_non_strict_canonicalizes_without_roots(root_layout):
    settings = ServerSettings(mode=MODE_LOCAL_TRUSTED)
    result = authorize_local_source(root_layout["inside"], settings=settings)
    assert result == root_layout["inside"].resolve()


def test_authorize_strict_without_identity_denied(root_layout):
    settings = ServerSettings(
        mode=MODE_AUTHENTICATED_SERVER,
        local_source_roots=(str(root_layout["root"]),),
    )
    with pytest.raises(LocalSourceDenied):
        authorize_local_source(root_layout["inside"], settings=settings)


def test_authorize_strict_without_scope_denied(root_layout):
    settings = ServerSettings(
        mode=MODE_AUTHENTICATED_SERVER,
        local_source_roots=(str(root_layout["root"]),),
    )
    identity = Identity(
        subject="reader", method="token", scopes=frozenset({SCOPE_READ})
    )
    with pytest.raises(LocalSourceDenied):
        authorize_local_source(
            root_layout["inside"], settings=settings, identity=identity
        )


def test_authorize_strict_scope_but_no_roots_denied(root_layout):
    settings = ServerSettings(mode=MODE_AUTHENTICATED_SERVER)
    identity = Identity(
        subject="local-user",
        method="token",
        scopes=frozenset({SCOPE_SOURCES_LOCAL}),
    )
    with pytest.raises(LocalSourceDenied):
        authorize_local_source(
            root_layout["inside"], settings=settings, identity=identity
        )


def test_authorize_strict_scope_and_root_allows(root_layout):
    settings = ServerSettings(
        mode=MODE_AUTHENTICATED_SERVER,
        local_source_roots=(str(root_layout["root"]),),
    )
    identity = Identity(
        subject="local-user",
        method="token",
        scopes=frozenset({SCOPE_SOURCES_LOCAL}),
    )
    result = authorize_local_source(
        root_layout["inside"], settings=settings, identity=identity
    )
    assert result == root_layout["inside"].resolve()


def test_authorize_strict_scope_denies_escape(root_layout):
    settings = ServerSettings(
        mode=MODE_AUTHENTICATED_SERVER,
        local_source_roots=(str(root_layout["root"]),),
    )
    identity = Identity(
        subject="local-user",
        method="token",
        scopes=frozenset({SCOPE_SOURCES_LOCAL}),
    )
    with pytest.raises(LocalSourceDenied):
        authorize_local_source(
            root_layout["sibling_secret"], settings=settings, identity=identity
        )
