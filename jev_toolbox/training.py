"""Prepare a reproducible run around the archived single-task trainer."""
from pathlib import Path
import json
import shutil

from .protocol import atomic, file_sha


def verify(directory, manifest):
    for relative, expected in manifest.items():
        if file_sha(directory / relative) != expected:
            raise ValueError(f"SHA-256 mismatch: {directory / relative}")


def prepare(config, config_dir, output, base, initial, mentat_src):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    data = (config_dir / config["data"]).resolve()
    recipe = (config_dir / config["recipe"]).resolve()
    weights = json.loads((recipe.parent / "model_inputs.json").read_text())
    roots = {"base": base.resolve(), "initial": initial.resolve(), "mentat_src": mentat_src.resolve()}
    model_files = {}
    for relative, expected in weights.items():
        role, filename = relative.split("/", 1)
        path = roots[role] / filename
        if file_sha(path) != expected:
            raise ValueError(f"SHA-256 mismatch: {role}/{filename}")
        model_files[str(path)] = expected
    verify(data, json.loads((data / "FILES_SHA256.json").read_text()))
    (output / "code").mkdir(parents=True)
    (output / "data").mkdir()
    shutil.copyfile(recipe, output / "code/train.py")
    shutil.copyfile(Path(__file__).with_name("protocol.py"), output / "code/common.py")
    shutil.copyfile(data / "train.json", output / "data/train.json")
    shutil.copyfile(data / "DATASET_AUDIT.json", output / "DATASET_AUDIT.json")
    if config["task"] == "necessity":
        shutil.copyfile(data / "SPLIT_MANIFEST.json", output / "SPLIT_MANIFEST.json")
    atomic(output / "config.json", {
        **config["training"], "task_kind": config["task"],
        "base_model": str(roots["base"]), "initial_model": str(roots["initial"]),
    })
    atomic(output / "MODEL_INPUT_MANIFEST.json", model_files)
    sources = ["code/common.py", "code/train.py", "config.json", "MODEL_INPUT_MANIFEST.json",
               "DATASET_AUDIT.json", "data/train.json"]
    if config["task"] == "necessity":
        sources.append("SPLIT_MANIFEST.json")
    atomic(output / "SOURCE_MANIFEST.json", {name: file_sha(output / name) for name in sources})
    return {"run": str(output), "trainer": str(output / "code/train.py"), "mentat_src": str(roots["mentat_src"])}
