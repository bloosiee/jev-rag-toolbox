"""Download and verify a Jev-RAG checkpoint archive from GitHub Releases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

PROJECT = "https://github.com/bloosiee/jev-rag-toolbox"
MANIFEST = Path(__file__).resolve().parents[1] / "models/release-manifest.json"


def download(model, directory):
    manifest = json.loads(MANIFEST.read_text())
    version = "v1" if model == "joint" else model.removeprefix("jev-joint-")
    bundle = manifest["models"][version]
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / bundle["archive"]
    if destination.exists():
        raise FileExistsError(destination)
    archive_hash = hashlib.sha256()
    output = NamedTemporaryFile(mode="wb", dir=directory, prefix=".download-", delete=False)
    partial = Path(output.name)
    try:
        with output:
            for part in bundle["parts"]:
                url = f"{PROJECT}/releases/download/{bundle.get('tag', manifest['tag'])}/{part['name']}"
                with urlopen(url, timeout=90) as response:
                    data = response.read(part["size"] + 1)
                if len(data) != part["size"] or hashlib.sha256(data).hexdigest() != part["sha256"]:
                    raise ValueError(f"SHA-256 check failed for {part['name']}")
                archive_hash.update(data)
                output.write(data)
        if archive_hash.hexdigest() != bundle["sha256"]:
            raise ValueError("Archive SHA-256 check failed")
        os.link(partial, destination)
        partial.unlink()
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    available = tuple(json.loads(MANIFEST.read_text())["models"])
    canonical = tuple(f"jev-joint-{version}" for version in available)
    parser.add_argument("model", choices=(*canonical, *available, "joint"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    print(download(args.model, args.output))
