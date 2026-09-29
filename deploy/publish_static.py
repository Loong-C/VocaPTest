"""Publish a locally built VocaPTest frontend without installing models on the VPS."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile


def publish(release, archive):
    if not re.fullmatch(r"[a-zA-Z0-9._-]{1,70}", release):
        raise ValueError("Invalid release name")
    base = Path("/var/www/linkukai/vocaptest/releases")
    live = Path("/var/www/linkukai/public/VocaPTest")
    previous = live.resolve(strict=True)
    if not live.is_symlink() or not previous.is_relative_to(base):
        raise RuntimeError("Unexpected live deployment")
    target = base / release
    target.mkdir()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not (target / member.name).resolve().is_relative_to(target):
                raise ValueError("Unsafe archive path")
            if not (member.isfile() or member.isdir()):
                raise ValueError("Unsafe archive entry")
        tar.extractall(target, filter="data")
    metadata = json.loads((target / "release.json").read_text())
    if not re.fullmatch(r"[a-f0-9]{40}", metadata["git_commit"]):
        raise ValueError("Missing source provenance")
    assert (target / "index.html").is_file()
    assert (target / "catalog/producers.json").is_file()
    for asset in (previous / "assets").iterdir():
        if asset.is_file() and not (target / "assets" / asset.name).exists():
            shutil.copy2(asset, target / "assets" / asset.name)
    temporary = live.with_name(".VocaPTest-next")
    try:
        temporary.symlink_to(target)
        os.replace(temporary, live)
        subprocess.run([
            "curl", "-fsS", "--max-time", "10", "--cacert",
            "/etc/nginx/ssl/linkukai/fullchain.pem", "--resolve",
            "linkukai.com:443:127.0.0.1", "https://linkukai.com/VocaPTest/release.json",
        ], check=True)
    except BaseException:
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(previous)
        os.replace(temporary, live)
        raise
    print(f"Published {release}; previous release: {previous}")


if __name__ == "__main__":
    publish(*sys.argv[1:])
