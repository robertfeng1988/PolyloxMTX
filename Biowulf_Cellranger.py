#!/usr/bin/env python3
"""
generate_cellranger_swarm.py
Generate per-sample cellranger count shell scripts and a swarm submission file.

Usage:
    python generate_cellranger_swarm.py [--root DIR] [--ref DIR] [--dry-run]
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FASTQ_PAT = re.compile(
    r"(.+?)_S\d+_L\d{3}_R[12]_\d{3}\.(?:fastq|fq)\.gz$"
)

# Top-level directory names to skip when scanning for FASTQs
EXCLUDE_TOP = {"cellranger_out", "scripts", "swarm_logs", "Ref"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_ref_dir(ref_parent: Path) -> Path:
    """
    Auto-discover a cellranger-compatible reference inside *ref_parent*.
    A valid reference must contain 'genome.fa' or 'fasta/genome.fa'
    and a 'genes/' or 'star/' subdirectory.
    Falls back to returning ref_parent itself if no subdirectory qualifies.
    """
    if not ref_parent.is_dir():
        log.error("Reference parent directory not found: %s", ref_parent)
        sys.exit(1)

    # Common layout: Ref/<species>/<build>/  or  Ref/<build>/
    candidates = []
    for candidate in sorted(ref_parent.rglob("fasta")):
        ref = candidate.parent
        if (ref / "genes").is_dir() or (ref / "star").is_dir():
            candidates.append(ref)

    if len(candidates) == 1:
        return candidates[0].resolve()
    if len(candidates) > 1:
        log.warning(
            "Multiple reference directories found under %s; "
            "using the first one: %s\n  All candidates:\n    %s",
            ref_parent,
            candidates[0],
            "\n    ".join(str(c) for c in candidates),
        )
        return candidates[0].resolve()

    # No structured subdirectory found — treat ref_parent itself as the ref
    log.warning(
        "Could not auto-detect a structured reference under %s; "
        "using it directly as the reference.",
        ref_parent,
    )
    return ref_parent.resolve()


def discover_samples(root: Path) -> dict[str, set[str]]:
    """
    Walk *root* recursively and collect 10x-style FASTQ files.
    Returns {sample_name: {fastq_dir, ...}}.
    """
    samples: dict[str, set[str]] = {}

    for fq in root.rglob("*"):
        if not fq.is_file():
            continue
        if not (fq.name.endswith(".fastq.gz") or fq.name.endswith(".fq.gz")):
            continue

        # Skip excluded top-level directories
        rel = fq.relative_to(root)
        if rel.parts and rel.parts[0] in EXCLUDE_TOP:
            continue

        m = FASTQ_PAT.match(fq.name)
        if not m:
            continue

        sample_name = m.group(1)
        samples.setdefault(sample_name, set()).add(str(fq.parent.resolve()))

    return samples


def write_sample_script(
    scripts_dir: Path,
    root: Path,
    ref_dir: Path,
    sample: str,
    fastq_dirs: str,
) -> Path:
    """Write a per-sample bash script and return its path."""
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", sample)
    script_path = scripts_dir / f"cellranger_count_{safe_id}.sh"

    script_text = f"""\
#!/bin/bash
set -euo pipefail

module purge
module load cellranger

ROOT="{root}"
REF_DIR="{ref_dir}"
FASTQ_DIRS="{fastq_dirs}"
SAMPLE="{sample}"
RUN_ID="{safe_id}"

mkdir -p "$ROOT/cellranger_out"
cd "$ROOT/cellranger_out"

# Use node-local scratch for temp files if available (Biowulf / SLURM)
if [[ -n "${{LSCRATCH:-}}" ]]; then
    export TMPDIR="$LSCRATCH"
fi

cellranger count \\
  --id="$RUN_ID" \\
  --transcriptome="$REF_DIR" \\
  --fastqs="$FASTQ_DIRS" \\
  --sample="$SAMPLE" \\
  --localcores="${{SLURM_CPUS_PER_TASK:-16}}" \\
  --localmem="${{CELLRANGER_MEM_GB:-100}}" \\
  --create-bam=true
"""
    script_path.write_text(script_text)
    os.chmod(script_path, 0o755)
    return script_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cellranger count scripts and a swarm file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root directory to scan for FASTQs.",
    )
    parser.add_argument(
        "--ref",
        type=Path,
        default=None,
        help=(
            "Path to cellranger reference directory. "
            "If omitted, auto-discovered under <root>/Ref/."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing any files.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    root: Path = args.root.resolve()
    if not root.is_dir():
        log.error("Root directory not found: %s", root)
        sys.exit(1)

    # ---- Reference directory --------------------------------------------
    if args.ref:
        ref_dir = args.ref.resolve()
        if not ref_dir.is_dir():
            log.error("Specified reference directory not found: %s", ref_dir)
            sys.exit(1)
    else:
        ref_dir = find_ref_dir(root / "Ref")

    # ---- Discover samples -----------------------------------------------
    samples = discover_samples(root)

    if not samples:
        log.error(
            "No 10x-style FASTQs detected under %s\n"
            "Expected names like: sample_S1_L001_R1_001.fastq.gz",
            root,
        )
        sys.exit(1)

    # ---- Output directories --------------------------------------------
    scripts_dir = root / "scripts"
    out_dir = root / "cellranger_out"
    log_dir = root / "swarm_logs"

    if not args.dry_run:
        for d in (scripts_dir, out_dir, log_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ---- Write per-sample scripts --------------------------------------
    swarm_lines: list[str] = []

    for sample in sorted(samples):
        fastq_dirs = ",".join(sorted(samples[sample]))
        if args.dry_run:
            log.info("[dry-run] Would write script for sample: %s", sample)
            log.info("          FASTQ dirs: %s", fastq_dirs)
        else:
            script_path = write_sample_script(
                scripts_dir, root, ref_dir, sample, fastq_dirs
            )
            swarm_lines.append(f"bash {script_path}")
            log.info("Wrote script: %s", script_path)

    # ---- Write swarm file ---------------------------------------------
    swarm_file = root / "cellranger_count.swarm"

    if not args.dry_run:
        swarm_file.write_text("\n".join(swarm_lines) + "\n")

    # ---- Summary -------------------------------------------------------
    print(f"\nReference directory : {ref_dir}")
    print(f"Detected {len(samples)} sample(s):")
    for s in sorted(samples):
        print(f"  - {s}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
    else:
        print(f"\nWrote:")
        print(f"  {swarm_file}")
        print(f"  {scripts_dir}/cellranger_count_<sample>.sh")

    print("\nSubmit with:")
    print(
        "  swarm -f cellranger_count.swarm -t 16 -g 110 "
        "--time 48:00:00 --gres=lscratch:200 --logdir swarm_logs"
    )


if __name__ == "__main__":
    main()
