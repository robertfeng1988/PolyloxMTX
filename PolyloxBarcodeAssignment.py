#!/usr/bin/env python3
"""
polylox_barcode_assign.py

Assign Polylox barcodes to cells in an AnnData object by matching cell
barcodes to per-sample segmentation/assembly CSV files.

Deduplication strategy (per cell barcode):
    When a cell barcode appears in multiple samples, pick the assignment
    with the LOWEST pGen (rarest barcode), breaking ties by HIGHEST MinCut
    (most supported recombination). This keeps the most informative and
    well-supported barcode call.

Usage:
    python polylox_barcode_assign.py \\
        --adata processed.h5ad \\
        --output annotated.h5ad \\
        --sample-mouse-map sample_mouse.tsv

Sample-mouse map TSV (no header):
    Smp1    Mouse#1
    Smp2    Mouse#1
    SmpA    Mouse#3
    ...
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import scanpy as sc

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SAMPLES = [
    "Smp1", "Smp2", "Smp3", "Smp4",
    "Smp5", "Smp6", "Smp7",
    "SmpA", "SmpB", "SmpC", "SmpD", "SmpE",
]

# Hardcoded fallback mouse map — override via --sample-mouse-map
_DEFAULT_MOUSE_MAP: dict[str, str] = {
    "Smp1": "Mouse#1", "Smp2": "Mouse#1", "Smp3": "Mouse#1", "Smp4": "Mouse#1",
    "Smp5": "Mouse#2", "Smp6": "Mouse#2", "Smp7": "Mouse#2",
    "SmpA": "Mouse#3", "SmpB": "Mouse#3", "SmpC": "Mouse#3", "SmpD": "Mouse#3",
    "SmpE": "Mouse#4",
}

# Columns that may exist in adata.obs from a previous run — cleared on re-run
_STALE_COLS = [
    "barcode", "pGen", "MinCut", "polylox",
    "polylox_new", "Polylox_BC", "MouseBC",
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--adata",    required=True,  help="Input .h5ad file.")
    p.add_argument("--output",   required=True,  help="Output .h5ad file.")
    p.add_argument(
        "--base-path",
        default="/home/jxfeng/DataProcessing/PolyloxData/2025_Processed_Polylox/CSV_files_Snakemake/",
        help="Directory containing <sample>_seg_assemble.csv files.",
    )
    p.add_argument(
        "--rare-barcode-list",
        default="/home/jxfeng/DataProcessing/PolyloxData/2025_Processed_Polylox/merged_polylox_data_pgen.txt",
    )
    p.add_argument(
        "--barcode-lib",
        default="/home/jxfeng/DataProcessing/PolyloxData/polylox_barcodelib.txt",
    )
    p.add_argument(
        "--min-recomb",
        default="/home/jxfeng/DataProcessing/PolyloxData/PolyloxMatlab_2024/min_reconb_list.txt",
    )
    p.add_argument(
        "--sample-mouse-map",
        default=None,
        help=(
            "Optional TSV (no header) mapping sample → mouse ID. "
            "Columns: sample<TAB>mouse. Falls back to hardcoded map if omitted."
        ),
    )
    p.add_argument("--samples",              nargs="*", default=DEFAULT_SAMPLES)
    p.add_argument("--cb-length",            type=int,   default=16,
                   help="Cell barcode length (characters from the start of obs_names).")
    p.add_argument("--missing-barcode-pgen", type=float, default=1e-8,
                   help="pGen assigned to barcodes absent from the library.")
    p.add_argument("--default-mincut",       type=int,   default=2,
                   help="MinCut assigned when value is missing.")
    p.add_argument(
        "--no-reverse-complement",
        action="store_true",
        help="Store polylox sequence as-is instead of reverse-complementing it.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RC_TABLE = str.maketrans("ACGTN", "TGCAN")


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of *seq*. Preserves N; skips NaN."""
    if pd.isna(seq):
        return seq
    seq = str(seq).upper()
    unexpected = set(seq) - set("ACGTN")
    if unexpected:
        log.debug("Unexpected bases in sequence %r: %s", seq, unexpected)
    return seq.translate(_RC_TABLE)[::-1]


def load_sample_mouse_map(path: str | None) -> dict[str, str]:
    """Load sample→mouse mapping from a TSV, or return the hardcoded default."""
    if path is None:
        log.info("No --sample-mouse-map provided; using hardcoded default.")
        return _DEFAULT_MOUSE_MAP.copy()
    df = pd.read_csv(path, sep="\t", header=None, names=["sample", "mouse"])
    mapping = df.set_index("sample")["mouse"].to_dict()
    log.info("Loaded %d sample→mouse mappings from %s", len(mapping), path)
    return mapping


def validate_paths(*paths: Path) -> None:
    """Raise FileNotFoundError for any path that does not exist."""
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")


def strip_sample_suffix(obs_names: pd.Index, cb_length: int) -> pd.Series:
    """
    Extract the plain cell barcode from obs_names.

    AnnData concatenation appends a sample suffix separated by '-' or ':'.
    Examples:
        ACGTACGTACGTACGT-1-Smp1  →  ACGTACGTACGTACGT
        ACGTACGTACGTACGT-Smp1    →  ACGTACGTACGTACGT
        ACGTACGTACGTACGT-1       →  ACGTACGTACGTACGT  (single sample)

    Strategy: take only the first `cb_length` characters of the first
    dash-delimited token, then upper-strip.
    """
    # The raw barcode is always the first `-`-separated token
    raw = obs_names.astype(str).str.split("-").str[0]
    return raw.str[:cb_length].str.strip().str.upper()


# ---------------------------------------------------------------------------
# Barcode library
# ---------------------------------------------------------------------------

def read_barcode_lib(barcode_lib_path: Path, min_recomb_path: Path) -> pd.DataFrame:
    """Load barcode library and MinCut values; return merged DataFrame."""
    barcode_lib = pd.read_csv(barcode_lib_path, sep="\t", header=None)
    min_recomb   = pd.read_csv(min_recomb_path,  sep="\t", header=None)

    n = min(len(barcode_lib), len(min_recomb))
    if len(barcode_lib) != len(min_recomb):
        log.warning(
            "Barcode library (%d rows) and MinCut list (%d rows) differ in length; "
            "using first %d rows of each.",
            len(barcode_lib), len(min_recomb), n,
        )

    df = barcode_lib.iloc[:n, [0]].copy()
    df.columns = ["barcode"]
    df["MinCut"] = pd.to_numeric(min_recomb.iloc[:n, 0], errors="coerce").fillna(2).astype(int)
    df["barcode"] = df["barcode"].astype(str).str.strip().str.upper()

    df = df.drop_duplicates(subset=["barcode"])
    log.info("Barcode library loaded: %d entries", len(df))
    return df


# ---------------------------------------------------------------------------
# Rare barcode list
# ---------------------------------------------------------------------------

_KNOWN_BC_COLS = ["PolyloxBC", "Barcode", "barcode", "polylox"]


def detect_rare_key_column(df: pd.DataFrame) -> str:
    """Return the barcode key column name, checking it is not entirely null."""
    for col in _KNOWN_BC_COLS:
        if col in df.columns:
            if df[col].notna().any():
                return col
            log.warning("Column %r found in rare barcode list but is entirely null.", col)
    raise ValueError(
        f"No usable barcode key column found in rare barcode list. "
        f"Looked for: {_KNOWN_BC_COLS}. "
        f"Available columns: {list(df.columns)}"
    )


# ---------------------------------------------------------------------------
# Per-sample loading
# ---------------------------------------------------------------------------

def load_sample_barcodes(
    samples: list[str],
    base_path: Path,
    rare_barcode_path: Path,
    barcode_lib_df: pd.DataFrame,
    missing_barcode_pgen: float,
    default_mincut: int,
) -> dict[str, pd.DataFrame]:
    """
    For each sample, load <sample>_seg_assemble.csv, merge pGen and MinCut,
    and return a dict of {sample: DataFrame[cb, polylox, pGen, MinCut]}.
    """
    rare_df = pd.read_csv(rare_barcode_path, sep="\t", header=0)
    rare_key = detect_rare_key_column(rare_df)
    rare_df = rare_df.copy()
    rare_df[rare_key] = rare_df[rare_key].astype(str).str.strip().str.upper()

    barcodes: dict[str, pd.DataFrame] = {}

    for sample in samples:
        sample_file = base_path / f"{sample}_seg_assemble.csv"
        if not sample_file.exists():
            log.warning("Sample file not found, skipping: %s", sample_file)
            continue

        df = pd.read_csv(sample_file, header=0)

        missing_cols = {"cb", "polylox"} - set(df.columns)
        if missing_cols:
            log.warning(
                "Sample %s missing required columns %s — skipping.", sample, missing_cols
            )
            continue

        df = df[["cb", "polylox"]].drop_duplicates().copy()
        df["cb"]      = df["cb"].astype(str).str.strip().str.upper()
        df["polylox"] = df["polylox"].astype(str).str.strip().str.upper()

        # Default pGen = 1.0; overridden below for known/rare barcodes
        df["pGen"] = 1.0

        # Merge MinCut from barcode library
        df = df.merge(
            barcode_lib_df[["barcode", "MinCut"]],
            left_on="polylox",
            right_on="barcode",
            how="left",
        )

        # Barcodes absent from the library get a very small pGen
        absent_mask = df["barcode"].isna()
        df.loc[absent_mask, "pGen"] = missing_barcode_pgen
        if absent_mask.any():
            log.debug(
                "Sample %s: %d/%d barcodes not in library (pGen set to %s)",
                sample, absent_mask.sum(), len(df), missing_barcode_pgen,
            )

        # Override pGen from rare barcode list if this sample is a column
        if sample in rare_df.columns:
            rare_map = (
                rare_df[[rare_key, sample]]
                .dropna()
                .drop_duplicates(subset=[rare_key])
                .set_index(rare_key)[sample]
                .to_dict()
            )
            df["pGen"] = df["polylox"].map(rare_map).fillna(df["pGen"])

        df["MinCut"] = pd.to_numeric(df["MinCut"], errors="coerce").fillna(default_mincut).astype(int)
        df["pGen"]   = pd.to_numeric(df["pGen"],   errors="coerce")

        df = df[["cb", "polylox", "pGen", "MinCut"]].drop_duplicates()
        barcodes[sample] = df
        log.info("  %s: %d barcode assignments loaded", sample, len(df))

    return barcodes


# ---------------------------------------------------------------------------
# Deduplication & lookup
# ---------------------------------------------------------------------------

def build_lookup_df(barcodes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all per-sample barcode tables and deduplicate to one row per
    cell barcode.

    Deduplication priority (see module docstring):
        1. Lowest pGen  (rarest / most informative barcode)
        2. Highest MinCut  (best-supported recombination)
        3. First occurrence (stable tie-break)
    """
    combined = pd.concat(barcodes.values(), ignore_index=True)
    combined["cb"]      = combined["cb"].astype(str).str.strip().str.upper()
    combined["polylox"] = combined["polylox"].astype(str).str.strip().str.upper()
    combined["pGen"]    = pd.to_numeric(combined["pGen"],   errors="coerce")
    combined["MinCut"]  = pd.to_numeric(combined["MinCut"], errors="coerce")

    combined["_order"] = range(len(combined))
    combined = combined.sort_values(
        by=["cb", "pGen", "MinCut", "_order"],
        ascending=[True, True, False, True],   # lowest pGen → highest MinCut → first seen
    )

    lookup = (
        combined
        .drop_duplicates(subset="cb", keep="first")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    log.info("Lookup table built: %d unique cell barcodes", len(lookup))
    return lookup


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()
    args = parse_args()

    # Resolve all paths up front and validate before doing any heavy work
    adata_path        = Path(args.adata).resolve()
    output_path       = Path(args.output).resolve()
    base_path         = Path(args.base_path).resolve()
    rare_barcode_path = Path(args.rare_barcode_list).resolve()
    barcode_lib_path  = Path(args.barcode_lib).resolve()
    min_recomb_path   = Path(args.min_recomb).resolve()

    validate_paths(adata_path, rare_barcode_path, barcode_lib_path, min_recomb_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_mouse_map = load_sample_mouse_map(args.sample_mouse_map)

    # ---- Load AnnData -------------------------------------------------------
    log.info("Loading AnnData: %s", adata_path)
    adata = sc.read_h5ad(adata_path)
    log.info("AnnData: %d cells × %d genes", adata.n_obs, adata.n_vars)

    # ---- Clear stale columns from previous runs -----------------------------
    stale = [c for c in _STALE_COLS if c in adata.obs.columns]
    if stale:
        log.info("Dropping stale obs columns from previous run: %s", stale)
        adata.obs.drop(columns=stale, inplace=True)

    # ---- Mouse assignment ---------------------------------------------------
    if "sample" in adata.obs.columns:
        adata.obs["mouse"] = adata.obs["sample"].astype(str).map(sample_mouse_map).fillna("NoAssign")
        n_unassigned = (adata.obs["mouse"] == "NoAssign").sum()
        if n_unassigned:
            log.warning(
                "%d cells could not be assigned to a mouse "
                "(sample values not in map: %s)",
                n_unassigned,
                adata.obs.loc[adata.obs["mouse"] == "NoAssign", "sample"].unique().tolist(),
            )
    else:
        log.warning("'sample' column not found in adata.obs; mouse='NoAssign' for all cells.")
        adata.obs["mouse"] = "NoAssign"

    # ---- Extract plain cell barcodes from obs_names -------------------------
    adata.obs["cb"] = strip_sample_suffix(adata.obs_names, args.cb_length).values

    # ---- Build barcode lookup -----------------------------------------------
    log.info("Loading barcode library and per-sample data...")
    barcode_lib_df = read_barcode_lib(barcode_lib_path, min_recomb_path)
    barcodes       = load_sample_barcodes(
        samples              = args.samples,
        base_path            = base_path,
        rare_barcode_path    = rare_barcode_path,
        barcode_lib_df       = barcode_lib_df,
        missing_barcode_pgen = args.missing_barcode_pgen,
        default_mincut       = args.default_mincut,
    )

    if not barcodes:
        log.error("No sample data loaded — cannot proceed.")
        sys.exit(1)

    lookup = build_lookup_df(barcodes)
    lookup_indexed = lookup.set_index("cb")

    # ---- Map to adata.obs ---------------------------------------------------
    adata.obs["polylox"] = adata.obs["cb"].map(lookup_indexed["polylox"])
    adata.obs["pGen"]    = adata.obs["cb"].map(lookup_indexed["pGen"])
    adata.obs["MinCut"]  = adata.obs["cb"].map(lookup_indexed["MinCut"])

    # Reverse-complement the polylox sequence for the final barcode column
    if args.no_reverse_complement:
        adata.obs["barcode"] = adata.obs["polylox"]
    else:
        adata.obs["barcode"] = adata.obs["polylox"].apply(reverse_complement)

    # ---- Match-rate report --------------------------------------------------
    n_total    = adata.n_obs
    n_matched  = adata.obs["polylox"].notna().sum()
    n_missing  = n_total - n_matched
    pct        = 100 * n_matched / n_total if n_total else 0
    log.info(
        "Barcode assignment: %d / %d cells matched (%.1f%%) — %d unmatched",
        n_matched, n_total, pct, n_missing,
    )
    if pct < 50:
        log.warning(
            "Less than 50%% of cells received a barcode assignment. "
            "Check that --cb-length (%d) and --samples are correct.",
            args.cb_length,
        )

    # ---- Save ---------------------------------------------------------------
    log.info("Writing output: %s", output_path)
    adata.write_h5ad(output_path)
    log.info("Done.")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
# python polylox_barcode_assign.py \
#   --adata ScanpyPolylox.h5ad \
#   --output ScanpyPolylox_annotated.h5ad \
#   --base-path /home/jxfeng/DataProcessing/PolyloxData/2025_Processed_Polylox/CSV_files_Snakemake/ \
#   --samples Smp1 Smp2 Smp3 Smp4 Smp5 Smp6 Smp7 SmpA SmpB SmpC SmpD SmpE
#
# With custom sample→mouse map (sample_mouse.tsv, no header, tab-separated):
# python polylox_barcode_assign.py \
#   --adata ScanpyPolylox.h5ad \
#   --output ScanpyPolylox_annotated.h5ad \
#   --sample-mouse-map sample_mouse.tsv
