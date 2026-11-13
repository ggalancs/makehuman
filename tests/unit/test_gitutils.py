"""lib/gitutils.py: finding git, reading branch and commit, clone and pull."""

import os
import shutil
import subprocess
import sys

import pytest

import gitutils

COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    monkeypatch.setattr(gitutils, "_gitcmd", None)
    monkeypatch.setattr(gitutils, "_gitdir", None)
    monkeypatch.delenv("GIT_COMMAND", raising=False)


def make_exe(folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    exe = folder / name
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    return str(exe)


# --- finding git -------------------------------------------------------------------

def test_git_command_from_environment(monkeypatch):
    monkeypatch.setenv("GIT_COMMAND", "/opt/custom/git")
    assert gitutils.findPathToGit() == "/opt/custom/git"
    monkeypatch.setenv("GIT_COMMAND", "/elsewhere/git")
    assert gitutils.findPathToGit() == "/opt/custom/git"  # cached


def test_git_found_on_path(tmp_path, monkeypatch):
    git = make_exe(tmp_path / "bin2", "git")
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("PATH", os.pathsep.join([str(tmp_path / "bin1"), '"%s"' % (tmp_path / "bin2")]))
    assert gitutils.findPathToGit() == git


def test_git_exe_found_on_windows(tmp_path, monkeypatch):
    exe = make_exe(tmp_path / "Git" / "cmd", "git.exe")
    make_exe(tmp_path / "unix", "git")  # not what Windows looks for
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("PATH", os.pathsep.join([str(tmp_path / "unix"), str(tmp_path / "Git" / "cmd")]))
    assert gitutils.findPathToGit() == exe


def test_no_git_anywhere(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("PATH", str(tmp_path))
    assert gitutils.findPathToGit() is None


def test_git_dir_is_next_to_the_source_folder(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    (tmp_path / "makehuman").mkdir()
    monkeypatch.chdir(tmp_path / "makehuman")
    assert gitutils.findGitDir() == os.path.abspath(str(tmp_path / ".git"))
    monkeypatch.chdir(tmp_path)
    assert gitutils.findGitDir() == os.path.abspath(str(tmp_path / ".git"))  # cached


def test_no_git_dir(tmp_path, monkeypatch):
    (tmp_path / "makehuman").mkdir()
    monkeypatch.chdir(tmp_path / "makehuman")
    assert gitutils.findGitDir() is None


# --- reading .git without the git command ------------------------------------------------

@pytest.fixture
def gitdir(tmp_path, monkeypatch):
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "refs" / "heads" / "main").write_text(COMMIT + "\n")
    monkeypatch.setattr(gitutils, "_gitdir", str(git))
    return git


def test_branch_and_commit_from_files(gitdir):
    assert gitutils.getBranchFromFile() == "main"
    assert gitutils.getCommitFromFile() == COMMIT[:8]
    assert gitutils.getCommitFromFile(short=False) == COMMIT


def test_detached_head_from_files(gitdir):
    (gitdir / "HEAD").write_text(COMMIT + "\n")
    assert gitutils.getBranchFromFile() == "HEAD"
    assert gitutils.getCommitFromFile(short=False) == COMMIT


def test_missing_or_empty_files(gitdir):
    (gitdir / "refs" / "heads" / "main").unlink()
    assert gitutils.getCommitFromFile() is None
    (gitdir / "HEAD").write_text("")
    assert gitutils.getBranchFromFile() is None
    (gitdir / "HEAD").unlink()
    assert gitutils.getBranchFromFile() is None


def test_nothing_to_read_without_git_dir():
    assert gitutils.getBranchFromFile() is None
    assert gitutils.getCommitFromFile() is None


def test_branch_with_slash_from_files(gitdir):
    (gitdir / "HEAD").write_text("ref: refs/heads/feature/fast\n")
    (gitdir / "refs" / "heads" / "feature").mkdir()
    (gitdir / "refs" / "heads" / "feature" / "fast").write_text(COMMIT + "\n")
    assert gitutils.getBranchFromFile() == "feature/fast"
    assert gitutils.getCommitFromFile() == COMMIT[:8]


def test_current_branch_and_commit_fall_back_to_files(gitdir, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(sys, "platform", "linux")
    assert gitutils.getCurrentBranch() == "main"
    assert gitutils.getCurrentCommit() == COMMIT[:8]
    assert gitutils.getCurrentCommit(short=False) == COMMIT


def test_no_git_at_all(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    monkeypatch.chdir(tmp_path / "src")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(sys, "platform", "linux")
    assert gitutils.getCurrentBranch() is None
    assert gitutils.getCurrentCommit() is None


# --- asking the git command ------------------------------------------------------------------

class FakeGit(object):
    def __init__(self, output=b"", error=None):
        self.output = output
        self.error = error
        self.calls = []

    def __call__(self, args, *rest, **kwargs):
        self.calls.append((list(args), os.getcwd()))
        if self.error:
            raise self.error
        return self.output


@pytest.fixture
def with_git(gitdir, monkeypatch):
    monkeypatch.setattr(gitutils, "_gitcmd", "/usr/bin/git-test")


def test_branch_from_git_command(with_git, monkeypatch):
    fake = FakeGit(b"apple-silicon\n")
    monkeypatch.setattr(subprocess, "check_output", fake)
    assert gitutils.getCurrentBranch() == "apple-silicon"
    assert fake.calls[0][0] == ["/usr/bin/git-test", "rev-parse", "--abbrev-ref", "HEAD"]


@pytest.mark.parametrize("short, args", [
    (True, ["/usr/bin/git-test", "rev-parse", "--short", "HEAD"]),
    (False, ["/usr/bin/git-test", "rev-parse", "HEAD"]),
])
def test_commit_from_git_command(with_git, monkeypatch, short, args):
    fake = FakeGit(b"e89e2707\n")
    monkeypatch.setattr(subprocess, "check_output", fake)
    assert gitutils.getCurrentCommit(short) == "e89e2707"
    assert fake.calls[0][0] == args


def test_git_errors_give_none(with_git, monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "check_output", FakeGit(b"fatal: not a git repository\n"))
    assert gitutils.getCurrentBranch() is None
    assert gitutils.getCurrentCommit() is None
    monkeypatch.setattr(subprocess, "check_output", FakeGit(error=OSError("git crashed")))
    assert gitutils.getCurrentBranch() is None
    assert gitutils.getCurrentCommit() is None
    assert capsys.readouterr().out.count("git crashed") == 2


@pytest.mark.skipif(shutil.which("git") is None, reason="needs a git executable")
def test_against_a_real_repository(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "makehuman").mkdir(parents=True)
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.invalid")
    run = lambda *a: subprocess.run(["git"] + list(a), cwd=str(repo), env=env, check=True,
                                    capture_output=True, text=True).stdout.strip()
    run("init", "-q", "-b", "topic")
    (repo / "makehuman" / "f.txt").write_text("x")
    run("add", ".")
    run("commit", "-q", "-m", "one")
    monkeypatch.chdir(repo / "makehuman")
    assert gitutils.findGitDir() == os.path.abspath(str(repo / ".git"))
    assert gitutils.getCurrentBranch() == "topic"
    assert gitutils.getCurrentCommit(False) == run("rev-parse", "HEAD")
    assert gitutils.getCurrentCommit(True) == run("rev-parse", "--short", "HEAD")
    # The file fallback agrees with git itself
    assert gitutils.getBranchFromFile() == "topic"
    assert gitutils.getCommitFromFile(short=False) == run("rev-parse", "HEAD")


# --- LFS, clone and pull ----------------------------------------------------------------------

def test_lfs_support(monkeypatch, capsys):
    monkeypatch.setattr(gitutils, "_gitcmd", "git-x")
    calls = FakeGit()
    monkeypatch.setattr(subprocess, "check_call", calls)
    assert gitutils.hasGitLFSSupport() is True
    assert calls.calls[0][0] == ["git-x", "lfs", "install"]
    monkeypatch.setattr(subprocess, "check_call",
                        FakeGit(error=subprocess.CalledProcessError(1, "git lfs")))
    assert gitutils.hasGitLFSSupport() is False
    assert "git lfs" in capsys.readouterr().out


def test_clone_checks_out_the_branch_and_returns(tmp_path, monkeypatch):
    dest = tmp_path / "dest"
    dest.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gitutils, "_gitcmd", "git-x")
    calls = FakeGit()
    monkeypatch.setattr(subprocess, "check_call", calls)
    gitutils.cloneRepo("https://example.invalid/r.git", str(dest), "dev", ["--depth", "1"])
    assert calls.calls == [
        (["git-x", "clone", "https://example.invalid/r.git", str(dest), "--depth 1"], str(tmp_path)),
        (["git-x", "checkout", "dev"], os.path.realpath(str(dest))),
    ]
    assert os.path.realpath(os.getcwd()) == os.path.realpath(str(tmp_path))


def test_pull_checks_out_and_pulls(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gitutils, "_gitcmd", "git-x")
    calls = FakeGit()
    monkeypatch.setattr(subprocess, "check_call", calls)
    gitutils.pullRepo(str(repo))
    where = os.path.realpath(str(repo))
    assert [(c[0], os.path.realpath(c[1])) for c in calls.calls] == [
        (["git-x", "checkout", "master"], where),
        (["git-x", "pull", "origin", "master"], where),
    ]
    assert os.path.realpath(os.getcwd()) == os.path.realpath(str(tmp_path))
