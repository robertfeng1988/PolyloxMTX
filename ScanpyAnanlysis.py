#!/usr/bin/env python3
"""
run_scanpy_pipeline.py
Automated Scanpy pipeline for multiple 10x CellRanger outputs.

Usage:
    python run_scanpy_pipeline.py \
        --input-root /path/to/CellRangerCounts \
        --samples Smp1 Smp2 Smp3 \
        --outdir ./scanpy_out

    # With optional doublet filtering:
    python run_scanpy_pipeline.py ... --scrublet
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Must be before any other matplotlib import
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import anndata as ad


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated Scanpy pipeline for multiple 10x CellRanger outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Directory containing per-sample folders (each with filtered_feature_bc_matrix/).",
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="Sample names to process. Auto-detected if omitted.",
    )
    parser.add_argument("--outdir", default="./scanpy_auto_out")

    # QC thresholds
    parser.add_argument("--min-genes",              type=int,   default=200)
    parser.add_argument("--min-counts",             type=int,   default=500,
                        help="Minimum UMI counts per cell.")
    parser.add_argument("--min-cells",              type=int,   default=3)
    parser.add_argument("--max-genes-by-counts",    type=int,   default=8000)
    parser.add_argument("--max-pct-mt",             type=float, default=10.0)

    # Normalization / HVG
    parser.add_argument("--target-sum",             type=float, default=1e4)

    # Dimensionality reduction / clustering
    parser.add_argument("--n-neighbors",            type=int,   default=10)
    parser.add_argument("--n-pcs",                  type=int,   default=40)
    parser.add_argument("--resolution",             type=float, default=0.8)

    # Markers
    parser.add_argument(
        "--markers",
        nargs="+",
        default=["Cdh5", "Ptprc", "Pecam1", "Emcn", "ZsGreen1", "Col1a2", "Runx1"],
        help="Genes to plot on UMAP.",
    )

    # Optional doublet detection
    parser.add_argument(
        "--scrublet",
        action="store_true",
        help="Run Scrublet doublet detection per sample before merging.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)


def setup_logging(outdir: Path) -> None:
    log_file = outdir / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )


def save_fig(outfile: Path) -> None:
    """Save the current matplotlib figure and close all."""
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close("all")
    log.info("Saved figure: %s", outfile)


# ---------------------------------------------------------------------------
# Sample discovery
# ---------------------------------------------------------------------------

def discover_sample_paths(
    input_root: str | Path,
    requested_samples: list[str] | None,
) -> dict[str, Path]:
    """Return {sample_name: path_to_filtered_feature_bc_matrix}."""
    input_root = Path(input_root).resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    candidates = (
        requested_samples
        if requested_samples
        else sorted(p.name for p in input_root.iterdir() if p.is_dir())
    )

    sample_paths: dict[str, Path] = {}
    for sample in candidates:
        matrix_dir = input_root / sample / "filtered_feature_bc_matrix"
        if matrix_dir.is_dir():
            sample_paths[sample] = matrix_dir
        else:
            log.warning("Skipping %s — matrix dir not found: %s", sample, matrix_dir)

    if not sample_paths:
        raise FileNotFoundError(
            f"No valid sample folders found under {input_root}.\n"
            f"Expected: {input_root}/<sample>/filtered_feature_bc_matrix/"
        )

    return sample_paths


# ---------------------------------------------------------------------------
# Per-sample loading + optional doublet detection
# ---------------------------------------------------------------------------

def load_sample(
    sample: str,
    matrix_dir: Path,
    run_scrublet: bool = False,
) -> sc.AnnData:
    """Load one sample, optionally run Scrublet, return AnnData."""
    log.info("Reading sample: %s", sample)
    adata = sc.read_10x_mtx(str(matrix_dir), var_names="gene_symbols", cache=True)
    adata.var_names_make_unique()
    adata.obs["sample"] = sample

    if run_scrublet:
        try:
            import scrublet as scr
            scrub = scr.Scrublet(adata.X)
            doublet_scores, predicted_doublets = scrub.scrub_doublets(verbose=False)
            adata.obs["doublet_score"] = doublet_scores
            adata.obs["predicted_doublet"] = predicted_doublets
            n_doublets = predicted_doublets.sum()
            log.info(
                "  Scrublet: %d predicted doublets (%.1f%%)",
                n_doublets,
                100 * n_doublets / adata.n_obs,
            )
        except ImportError:
            log.warning("scrublet not installed; skipping doublet detection.")

    return adata


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Logging must be set up before anything else writes to the log
    setup_logging(outdir)

    sc.settings.verbosity = 2
    sc.logging.print_header()
    sc.settings.set_figure_params(dpi=100, facecolor="white")

    # Save run parameters
    with open(outdir / "run_parameters.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ------------------------------------------------------------------ #
    # 1. Discover samples                                                  #
    # ------------------------------------------------------------------ #
    sample_paths = discover_sample_paths(args.input_root, args.samples)

    pd.DataFrame(
        {"sample": list(sample_paths.keys()), "path": [str(p) for p in sample_paths.values()]}
    ).to_csv(outdir / "sample_paths.tsv", sep="\t", index=False)

    log.info("Detected %d sample(s):", len(sample_paths))
    for s, p in sample_paths.items():
        log.info("  %s -> %s", s, p)

    # ------------------------------------------------------------------ #
    # 2. Load samples                                                      #
    # ------------------------------------------------------------------ #
    adata_list = [
        load_sample(sample, path, run_scrublet=args.scrublet)
        for sample, path in sample_paths.items()
    ]

    # ------------------------------------------------------------------ #
    # 3. Merge                                                             #
    # ------------------------------------------------------------------ #
    if len(adata_list) == 1:
        adata = adata_list[0].copy()
    else:
        # ad.concat replaces the deprecated AnnData.concatenate()
        adata = ad.concat(
            adata_list,
            label="sample",
            keys=list(sample_paths.keys()),
            join="outer",       # keep all genes; fill missing with 0
            fill_value=0,
            merge="unique",
        )

    adata.var_names_make_unique()
    log.info("Merged object: %d cells × %d genes", adata.n_obs, adata.n_vars)

    # Save raw merged counts before any filtering
    adata.write_h5ad(outdir / "adata_merged_rawcounts.h5ad")

    # ------------------------------------------------------------------ #
    # 4. QC                                                                #
    # ------------------------------------------------------------------ #
    # Remove predicted doublets if Scrublet was run
    if args.scrublet and "predicted_doublet" in adata.obs.columns:
        before = adata.n_obs
        adata = adata[~adata.obs["predicted_doublet"]].copy()
        log.info("Removed %d doublets; %d cells remain", before - adata.n_obs, adata.n_obs)

    sc.pp.filter_cells(adata, min_genes=args.min_genes)
    sc.pp.filter_cells(adata, min_counts=args.min_counts)   # <-- added UMI floor
    sc.pp.filter_genes(adata, min_cells=args.min_cells)

    # Mitochondrial gene detection (handles both human MT- and mouse mt-)
    adata.var["mt"] = adata.var_names.str.startswith(("mt-", "MT-"))
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

    # Save QC metrics before hard filters
    adata.obs.to_csv(outdir / "cell_qc_metrics_before_filter.tsv", sep="\t")

    # Violin plot of QC metrics
    sc.pl.violin(
        adata,
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        jitter=0.4,
        multi_panel=True,
        show=False,
    )
    save_fig(outdir / "01a_qc_violin.png")

    # Scatter: total_counts vs pct_counts_mt
    sc.pl.scatter(adata, x="total_counts", y="pct_counts_mt", show=False)
    save_fig(outdir / "01b_qc_scatter_mt.png")
    sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts", show=False)
    save_fig(outdir / "01c_qc_scatter_genes.png")

    # Apply hard QC filters
    adata = adata[adata.obs.n_genes_by_counts < args.max_genes_by_counts].copy()
    adata = adata[adata.obs.pct_counts_mt < args.max_pct_mt].copy()
    log.info("After QC filters: %d cells × %d genes", adata.n_obs, adata.n_vars)

    # Highest expressed genes
    sc.pl.highest_expr_genes(adata, n_top=20, show=False)
    save_fig(outdir / "02_highest_expr_genes.png")

    # ------------------------------------------------------------------ #
    # 5. Normalize & log-transform                                         #
    # ------------------------------------------------------------------ #
    sc.pp.normalize_total(adata, target_sum=args.target_sum)
    sc.pp.log1p(adata)

    # Freeze the normalized+log1p counts as .raw BEFORE HVG subsetting
    # so marker UMAPs can retrieve expression for all genes
    adata.raw = adata

    # ------------------------------------------------------------------ #
    # 6. Highly variable genes                                             #
    # ------------------------------------------------------------------ #
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.pl.highly_variable_genes(adata, show=False)
    save_fig(outdir / "03_highly_variable_genes.png")

    n_hvg = adata.var.highly_variable.sum()
    log.info("Highly variable genes selected: %d", n_hvg)

    adata = adata[:, adata.var.highly_variable].copy()

    # ------------------------------------------------------------------ #
    # 7. Regress out confounders & scale                                   #
    # ------------------------------------------------------------------ #
    sc.pp.regress_out(adata, ["total_counts", "pct_counts_mt"])
    sc.pp.scale(adata, max_value=10)

    # ------------------------------------------------------------------ #
    # 8. PCA                                                               #
    # ------------------------------------------------------------------ #
    sc.tl.pca(adata, svd_solver="arpack")

    sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True, show=False)
    save_fig(outdir / "04_pca_variance_ratio.png")

    # Colored by sample to spot batch effects early
    sc.pl.pca(adata, color="sample", show=False)
    save_fig(outdir / "04b_pca_by_sample.png")

    # ------------------------------------------------------------------ #
    # 9. Neighbors + UMAP                                                  #
    # ------------------------------------------------------------------ #
    sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, n_pcs=args.n_pcs)
    sc.tl.umap(adata)

    # UMAP colored by sample
    with plt.rc_context({"figure.figsize": (5, 4), "figure.dpi": 120}):
        sc.pl.umap(adata, color="sample", show=False)
        save_fig(outdir / "05_umap_by_sample.png")

    # ------------------------------------------------------------------ #
    # 10. Marker UMAPs (one figure per marker for clarity)                 #
    # ------------------------------------------------------------------ #
    raw_var_names = set(adata.raw.var_names)
    valid_markers = [g for g in args.markers if g in raw_var_names]
    missing_markers = [g for g in args.markers if g not in raw_var_names]

    if missing_markers:
        log.warning("Markers not found, skipped: %s", ", ".join(missing_markers))

    markers_dir = outdir / "marker_umaps"
    markers_dir.mkdir(exist_ok=True)

    for gene in valid_markers:
        with plt.rc_context({"figure.figsize": (4, 4), "figure.dpi": 120}):
            sc.pl.umap(adata, color=[gene], use_raw=True, show=False)
            save_fig(markers_dir / f"umap_{gene}.png")

    # All markers in one panel if there are multiple
    if len(valid_markers) > 1:
        ncols = min(4, len(valid_markers))
        with plt.rc_context({"figure.figsize": (4 * ncols, 4), "figure.dpi": 100}):
            sc.pl.umap(adata, color=valid_markers, use_raw=True, ncols=ncols, show=False)
            save_fig(outdir / "06_umap_markers_panel.png")

    # ------------------------------------------------------------------ #
    # 11. Leiden clustering                                                #
    # ------------------------------------------------------------------ #
    sc.tl.leiden(adata, resolution=args.resolution)

    with plt.rc_context({"figure.figsize": (5, 4), "figure.dpi": 120}):
        sc.pl.umap(
            adata,
            color=["leiden"],
            legend_fontsize="xx-small",
            show=False,
        )
        save_fig(outdir / "07_umap_leiden.png")

    with plt.rc_context({"figure.figsize": (5, 4), "figure.dpi": 120}):
        sc.pl.umap(
            adata,
            color=["leiden"],
            legend_loc="on data",
            legend_fontsize="xx-small",
            show=False,
        )
        save_fig(outdir / "08_umap_leiden_ondata.png")

    # ------------------------------------------------------------------ #
    # 12. Cluster summary & outputs                                        #
    # ------------------------------------------------------------------ #
    cluster_counts = (
        adata.obs.groupby(["leiden", "sample"])
        .size()
        .unstack(fill_value=0)
    )
    cluster_counts["total"] = cluster_counts.sum(axis=1)
    cluster_counts.to_csv(outdir / "cluster_cell_counts.tsv", sep="\t")
    log.info("Cluster sizes:\n%s", cluster_counts.to_string())

    adata.write_h5ad(outdir / "adata_processed.h5ad")
    adata.obs.to_csv(outdir / "adata_obs_processed.tsv", sep="\t")
    adata.var.to_csv(outdir / "adata_var_processed.tsv", sep="\t")

    log.info("Done. All outputs written to: %s", outdir)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
# chmod +x run_scanpy_pipeline.py
#
# python run_scanpy_pipeline.py \
#   --input-root /home/jxfeng/Polylox_2nd/CellRangerCounts \
#   --samples Smp1 Smp2 Smp3 Smp4 Smp5 Smp6 \
#   --outdir /home/jxfeng/Polylox_2nd/Scanpy_Run01
#
# With doublet filtering:
# python run_scanpy_pipeline.py \
#   --input-root /home/jxfeng/Polylox_2nd/CellRangerCounts \
#   --samples Smp1 Smp2 Smp3 Smp4 Smp5 Smp6 \
#   --outdir /home/jxfeng/Polylox_2nd/Scanpy_Run01 \
#   --scrublet
