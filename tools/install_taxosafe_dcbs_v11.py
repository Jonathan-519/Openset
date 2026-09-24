"""Install the audited v11 file set into an existing v10 directory without git.

Resolves a branch to a single commit, validates every download before writing,
and preserves replaced files in a dated backup. Standard library only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from urllib.request import Request, urlopen

REPOSITORY = "Jonathan-519/Openset"
MANIFEST = "DCBS_V11_FILES.sha256"
DEFAULT_REF = "taxosafe-v11-dcbs"


def download(url):
    with urlopen(Request(url, headers={"User-Agent": "TaxoSafe-v11-installer"}), timeout=60) as response:
        return response.read()


def parse_manifest(content):
    entries = []
    seen = set()
    for line in content.decode("utf-8").splitlines():
        if not line.strip():
            continue
        sha, name = line.split(None, 1)
        name = name.strip()
        path = PurePosixPath(name)
        if not re.fullmatch(r"[0-9a-f]{64}", sha) or path.is_absolute() or ".." in path.parts or "\\" in name or name in seen or name != str(path):
            raise ValueError("Invalid delivery manifest entry: " + name)
        if not path.parts or path.parts[0] == "runs" or "raw" in path.parts or path.suffix.lower() in {".pth", ".pt", ".npy", ".jpg", ".jpeg", ".png"}:
            raise ValueError("Delivery cannot replace datasets, binary models or run artifacts: " + name)
        entries.append((sha, name))
        seen.add(name)
    if not entries:
        raise ValueError("Empty delivery manifest")
    return entries


def install(project, ref=DEFAULT_REF):
    project = Path(project).resolve()
    for name in ("train_taxosafe.py", "models/maple.py", "loader/taxosafe_data.py"):
        if not (project / name).is_file():
            raise ValueError("Expected existing v10 project file: " + name)
    if re.fullmatch(r"[0-9a-fA-F]{40}", ref):
        commit = ref.lower()
    else:
        from urllib.parse import quote
        commit = json.loads(download("https://api.github.com/repos/{}/commits/{}".format(REPOSITORY, quote(ref, safe=""))))["sha"]
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("Invalid resolved GitHub commit")
    base = "https://raw.githubusercontent.com/{}/{}/".format(REPOSITORY, commit)
    manifest = download(base + MANIFEST)
    entries = parse_manifest(manifest)
    with tempfile.TemporaryDirectory(prefix="taxosafe-v11-") as temporary:
        staged = Path(temporary)
        for sha, name in entries:
            target = project / name
            target.resolve().relative_to(project)
            content = download(base + name)
            if hashlib.sha256(content).hexdigest() != sha:
                raise ValueError("Download hash mismatch: " + name)
            path = staged / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        backup = project / "dcbs_v11_backups" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup.mkdir(parents=True, exist_ok=False)
        for name in [n for _, n in entries] + [MANIFEST]:
            source = project / name
            source.resolve().relative_to(project)
            if source.exists():
                destination = backup / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        for _, name in entries:
            destination = project / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged / name, destination)
        (project / MANIFEST).write_bytes(manifest)
    print("Installed v11 commit {} into {}".format(commit, project))
    print("Replaced-file backup: " + str(backup))
    return commit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--ref", default=DEFAULT_REF)
    args = parser.parse_args()
    install(args.project, args.ref)
