"""Chain Kaggle notebook sessions: push a part, wait for it, push the next.

    python run_pipeline.py --arm baseline              # parts 1..5, in order
    python run_pipeline.py --arm baseline --parts 3 4  # resume from part 3
    python run_pipeline.py --arm prep                  # the CPU data session
    python run_pipeline.py --arm baseline --dry-run    # print, push nothing

Each part is a separate ~4-hour Kaggle session that resumes from the previous
part's checkpoint dataset. That handoff only works if part N has finished
*pushing* before part N+1 starts, so this waits for a terminal status rather than
firing them off together.

Everything here is idempotent. If the runner dies, rerun it with --parts starting
at whichever part is unfinished; the notebooks themselves resume from the last
checkpoint, so nothing is lost and nothing is repeated.

Progress is appended to run_pipeline.log, because a 20-hour chain outlives the
terminal you started it in.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOKS = HERE / "notebooks"
STAGING = HERE / ".kernel-staging"
LOG = HERE / "run_pipeline.log"

POLL_SECONDS = 60           # parts run ~4h, but a smoke run is ~10 min
MAX_WAIT_HOURS = 13          # Kaggle's own wall is 12h — past this something is wrong

# Which account each arm belongs to. The runner refuses to push to the wrong one
# rather than silently spending the other account's weekly quota.
ARM_ACCOUNT = {
    "repo": None,
    "prep": None,            # either account; costs no GPU quota
    "prep-smoke": None,
    "baseline": "johnandreimartinez",
    "ablation": "zeref824",
}

REPO_SLUG = "liitllm-repo"



# Datasets are PRIVATE and per-account: every account that runs an arm uploads
# its own copy of the repo and corpora.
#
# The alternative — one public copy mounted by both accounts — is less work but
# publishes the code and the corpora to the world as a side effect of a plumbing
# decision. Publishing should be a choice, not a consequence, so the default is
# private and the duplication is the price. It is cheap: `--arm prep` is a CPU
# session, so building the corpora on a second account costs wall-clock and zero
# GPU quota. Pass --public to opt into a single shared copy instead.

# Files that must never go into the repo dataset: kernel outputs contain a full
# copy of the repo, so shipping results/ back up compounds every upload and makes
# the "find the repo" glob ambiguous.
REPO_EXCLUDE = {".kernel-staging", "results", "out", "data", "__pycache__",
                ".git", ".venv", "run_pipeline.log"}


def log(msg: str):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def kaggle(*args, capture=True) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONUTF8": "1"}  # else the CLI dies writing logs
    return subprocess.run(
        [sys.executable, "-m", "kaggle", *args],
        capture_output=capture, text=True, env=env,
    )


def whoami() -> str:
    out = kaggle("config", "view").stdout
    for line in out.splitlines():
        if "username:" in line:
            return line.split("username:")[1].strip()
    raise SystemExit(f"could not read the active Kaggle username from:\n{out}")


def notebook_for(arm: str, part: int | None) -> Path:
    name = {"prep": "00_prep", "prep-smoke": "00_prep_smoke"}.get(arm) or f"{arm}_part{part}"
    path = NOTEBOOKS / f"{name}.ipynb"
    if not path.exists():
        raise SystemExit(f"{path} missing — run: python notebooks/build_notebooks.py")
    return path


def stage(arm: str, part: int | None, user: str) -> tuple[Path, str]:
    """Build a one-notebook directory with kernel-metadata.json beside it."""
    nb = notebook_for(arm, part)
    slug = nb.stem.replace("_", "-")            # slugs must be >= 6 chars
    assert len(slug) >= 6, f"slug {slug!r} is too short for Kaggle"
    d = STAGING / slug
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    shutil.copy(nb, d / "nb.ipynb")

    # The repo travels as a dataset; everything produced by a session travels as
    # that session's kernel output. No API credentials inside the notebook, and
    # nothing that can fail at the end of a 4-hour run.
    sources = [f"{user}/{REPO_SLUG}"]
    kernel_sources = []
    if not arm.startswith("prep"):
        kernel_sources.append(f"{user}/00-prep")          # the corpora + tokenizer
        if part and part > 1:
            kernel_sources.append(f"{user}/{arm}-part{part - 1}")   # the checkpoint

    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": "nb.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": not arm.startswith("prep"),     # prep is CPU: costs no GPU quota
        "enable_internet": True,
        "dataset_sources": sources,
        "kernel_sources": kernel_sources,
        "competition_sources": [],
        "model_sources": [],
    }, indent=1), encoding="utf-8")
    return d, slug


def wait_dataset_ready(user: str, slug: str, timeout_s: int = 900) -> bool:
    """Block until a dataset finishes processing.

    Uploading returns as soon as the bytes are transferred, but Kaggle keeps
    processing for a while afterwards and mounts the dataset EMPTY in the
    meantime. A kernel pushed into that window fails with "found []" — which
    reads like a missing file and is really a race. Cost of waiting: seconds.
    Cost of not waiting: a dead session and a confusing traceback.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        out = (kaggle("datasets", "status", f"{user}/{slug}").stdout or "").lower()
        if "ready" in out:
            return True
        if "error" in out:
            log(f"  dataset {slug} status: ERROR — {out.strip()}")
            return False
        log(f"  waiting for dataset {slug} to be ready ({out.strip() or 'no status'})")
        time.sleep(15)
    log(f"  dataset {slug} not ready after {timeout_s}s")
    return False


def status_of(user: str, slug: str) -> str:
    out = (kaggle("kernels", "status", f"{user}/{slug}").stdout or "").lower()
    for s in ("complete", "error", "cancel", "running", "queued"):
        if s in out:
            return s
    return "unknown"


def wait_for(user: str, slug: str) -> str:
    """Poll until the kernel reaches a terminal state."""
    deadline = time.time() + MAX_WAIT_HOURS * 3600
    last = None
    while time.time() < deadline:
        st = status_of(user, slug)
        if st != last:
            log(f"  {slug}: {st}")
            last = st
        if st in ("complete", "error", "cancel"):
            return st
        time.sleep(POLL_SECONDS)
    log(f"  {slug}: TIMED OUT after {MAX_WAIT_HOURS}h — check it on kaggle.com")
    return "timeout"


def _failure_log(user: str, slug: str, tail: int = 60) -> str:
    """Pull the kernel's own log so the failure can be diagnosed from here.

    Without this the runner reports 'error' and you go hunting on kaggle.com;
    with it the traceback is already in run_pipeline.log next to the timestamps.
    """
    d = STAGING / f"{slug}-log"
    d.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{user}/{slug}", "-p", str(d))
    lines: list[str] = []
    for f in sorted(d.glob("*.log")) + sorted(d.glob("*.txt")):
        try:
            entries = json.loads(f.read_text(encoding="utf-8", errors="replace"))
            lines += [str(e.get("data", e)) for e in entries]
        except Exception:
            lines += f.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return "(no log retrieved — open the kernel on kaggle.com)"
    return "--- last lines of the kernel log ---\n" + "\n".join(lines[-tail:])


def run_one(arm: str, part: int | None, user: str, dry_run: bool) -> bool:
    d, slug = stage(arm, part, user)
    log(f"pushing {slug}")
    if dry_run:
        log(f"  DRY RUN — metadata:\n{(d / 'kernel-metadata.json').read_text()}")
        return True

    # Every mounted source must be finished processing before the kernel starts,
    # or it mounts empty and the strict globs fail with "found []".
    if not wait_dataset_ready(user, REPO_SLUG):
        log(f"  {REPO_SLUG} is not ready — refusing to push {slug} into an empty mount")
        return False

    args = ["kernels", "push", "-p", str(d)]
    if not arm.startswith("prep"):
        # Kaggle's default is a P100 (sm_60) that its own PyTorch cannot use, and
        # everything before the first training step is CPU work — so a doomed
        # session looks healthy for ten minutes. Pin the T4.
        args += ["--accelerator", "NvidiaTeslaT4"]
    r = kaggle(*args)
    if r.returncode != 0:
        log(f"  PUSH FAILED: {r.stdout}{r.stderr}")
        return False

    st = wait_for(user, slug)
    if st != "complete":
        # Stop the chain and surface the actual traceback. Continuing past a
        # failed part would train the next one on a checkpoint that never got
        # written, so the run must not self-heal here — a human decides.
        log(f"\n{'=' * 70}")
        log(f"NEEDS ATTENTION — {slug} ended '{st}'. Chain stopped.")
        log(f"{'=' * 70}")
        log(_failure_log(user, slug))
        log(f"Nothing after this part was pushed. Once it is fixed, rerun with:")
        log(f"    python run_pipeline.py --arm {arm} --parts {part or ''} ...")
        log(f"https://www.kaggle.com/code/{user}/{slug}")
        return False

    out_dir = HERE / "results" / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{user}/{slug}", "-p", str(out_dir))
    log(f"  {slug} complete -> {out_dir}")
    # 'complete' means the kernel finished, not that its output is mountable yet.
    # The next part mounts this one via kernel_sources, so give it a moment —
    # same race as the dataset one above, just harder to observe.
    time.sleep(60)
    return True


def upload_repo(user: str, dry_run: bool, public: bool = False) -> bool:
    """Publish this repo as a Kaggle Dataset so both accounts can mount it.

    Re-run this after any code change: the training kernels copy the repo out of
    the mounted dataset, so an un-uploaded edit simply does not exist as far as
    Kaggle is concerned — the session runs the previous version and looks fine.
    """
    d = STAGING / REPO_SLUG
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for item in HERE.iterdir():
        if item.name in REPO_EXCLUDE or item.name.startswith("."):
            continue
        (shutil.copytree if item.is_dir() else shutil.copy)(item, d / item.name)
    # data/ is excluded wholesale above, but the lexicons are source, not output —
    # taglish.py reads them at import time and the kernels would fail without them.
    shutil.copytree(HERE / "data" / "lexicons", d / "data" / "lexicons")

    # Both `create` and `version` read dataset-metadata.json, so it is always
    # written — the staging dir is rebuilt from scratch on every call.
    (d / "dataset-metadata.json").write_text(json.dumps({
        "title": REPO_SLUG, "id": f"{user}/{REPO_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=1), encoding="utf-8")
    exists = "ready" in (kaggle("datasets", "status", f"{user}/{REPO_SLUG}").stdout or "").lower()

    log(f"uploading repo -> {user}/{REPO_SLUG} ({'version' if exists else 'create'})")
    if dry_run:
        log(f"  DRY RUN — would upload: {sorted(p.name for p in d.iterdir())}")
        return True

    # -r zip on EVERY call. Without it the CLI silently drops every subdirectory
    # and still reports success, so the dataset quietly becomes top-level files.
    if exists:
        r = kaggle("datasets", "version", "-p", str(d), "-r", "zip", "-m", "repo update")
    else:
        args = ["datasets", "create", "-p", str(d), "-r", "zip"]
        r = kaggle(*(args + ["-u"] if public else args))
    if r.returncode != 0:
        log(f"  FAILED: {r.stdout}{r.stderr}")
        return False
    files = kaggle("datasets", "files", f"{user}/{REPO_SLUG}").stdout
    log(f"  uploaded. contents:\n{files}")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True,
                   choices=["repo", "prep-smoke", "prep", "baseline", "ablation"])
    p.add_argument("--parts", type=int, nargs="*", default=None,
                   help="which parts to run (default: all 5, in order)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--public", action="store_true",
                   help="publish the dataset instead of keeping it private")
    p.add_argument("--force-account", action="store_true",
                   help="push even if the live account is not the expected one")
    a = p.parse_args()

    user = whoami()
    if a.arm == "repo":
        raise SystemExit(0 if upload_repo(user, a.dry_run, a.public) else 1)

    want = ARM_ACCOUNT[a.arm]
    if want and user != want and not a.force_account:
        raise SystemExit(
            f"live Kaggle account is '{user}' but the {a.arm} arm belongs on '{want}'.\n"
            f"Running it here would spend the wrong account's weekly GPU quota.\n"
            f"Switch with:  printf '<token>' > ~/.kaggle/access_token\n"
            f"(printf, not echo — a trailing newline breaks the token)\n"
            f"Or pass --force-account if you meant it."
        )

    parts = [None] if a.arm.startswith("prep") else (a.parts or [1, 2, 3, 4, 5])
    log(f"=== {a.arm} on {user}: {len(parts)} session(s) ===")
    for i, part in enumerate(parts, 1):
        label = a.arm if part is None else f"{a.arm} part {part}"
        log(f"--- {label} ({i}/{len(parts)}) ---")
        if not run_one(a.arm, part, user, a.dry_run):
            raise SystemExit(f"chain stopped at {label}")
    log(f"=== {a.arm} finished ===")


if __name__ == "__main__":
    main()
