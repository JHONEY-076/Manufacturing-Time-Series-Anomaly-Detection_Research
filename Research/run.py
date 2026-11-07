"""Portable entry point for the research repository (run from any directory)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools"))


def new_output(stage: str, requested: Path | None) -> Path:
    path = requested.resolve() if requested else ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + stage)
    path.mkdir(parents=True, exist_ok=False)
    return path


def check_inputs():
    manifest = json.loads((ROOT / "data/input_manifest.json").read_text(encoding="utf-8"))
    errors = []
    for name, expected in manifest["files"].items():
        path = ROOT / name
        if not path.is_file():
            errors.append(f"missing: {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"changed: {name}")
    if errors:
        raise ValueError("Input snapshot differs; restore it or explicitly regenerate the manifest for your own experiment:\n" + "\n".join(errors))
    from reproduce_comparison import load_splits
    _, _, counts, _ = load_splits(ROOT / "data/augmentation_split")
    print(json.dumps(counts, indent=2))
    print(f"Verified {len(manifest['files'])} input checksums and source-row-disjoint evaluation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check", "prepare", "baseline", "compare", "full", "generate", "masking"])
    parser.add_argument("--output", type=Path, help="New output directory; existing directories are rejected")
    parser.add_argument("--split-dir", type=Path, help="Alternative augmentation_split directory for baseline/compare")
    parser.add_argument("--trees", type=int, default=300, help="RF trees for baseline/compare")
    parser.add_argument("--normal-count", type=int, default=5000, help="Normal training windows for baseline/compare")
    parser.add_argument("--seed", type=int, default=42, help="RF seed for baseline/compare")
    parser.add_argument("--jobs", type=int, default=4, help="RF workers for baseline/compare")
    args = parser.parse_args()
    is_comparison = args.stage in {"baseline", "compare"}
    if not is_comparison and (args.split_dir or args.trees != 300 or args.normal_count != 5000 or args.seed != 42 or args.jobs != 4):
        parser.error("--split-dir/--trees/--normal-count/--seed/--jobs apply only to baseline/compare")
    if args.trees < 1 or args.normal_count < 1 or args.jobs == 0:
        parser.error("trees and normal-count must be positive; jobs must not be zero")
    if args.stage == "check":
        if args.output:
            parser.error("check does not write output")
        check_inputs()
        return
    if args.stage in {"generate", "masking"}:
        if args.output:
            parser.error("Legacy generator stages write to data/researchXX; --output is unsupported")
        if not importlib.util.find_spec("torch"):
            parser.error("Generator training needs: pip install -r Research/requirements-training.txt")
        script = "research02_generate_data.py" if args.stage == "generate" else "research10_masking_ratio.py"
        print("Generator stages update cached pools/results in data/. Commit or copy inputs before retraining.", flush=True)
        subprocess.run([sys.executable, str(ROOT / "tools" / script)], cwd=ROOT, check=True)
        return
    output = new_output(args.stage, args.output)
    if args.stage == "prepare":
        from prepare_inputs import prepare
        prepare(output)
    elif is_comparison:
        from reproduce_comparison import run
        run(args.stage, output, (args.split_dir or ROOT / "data/augmentation_split").resolve(), args.trees, args.normal_count, args.seed, args.jobs)
    else:
        import research_section43_unified as experiment
        experiment.OUT = output
        experiment.main()
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
