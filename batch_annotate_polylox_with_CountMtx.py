import pandas as pd
import os

def load_whitelist(whitelist_file):
    with open(whitelist_file) as f:
        return set(line.strip() for line in f if line.strip())

def is_single_barcode(barcode_str):
    barcodes = [b.strip() for b in str(barcode_str).split(',')]
    return len(barcodes) == 1

def all_barcodes_in_whitelist(barcode_str, whitelist):
    barcodes = [b.strip() for b in str(barcode_str).split(',')]
    return all(b in whitelist for b in barcodes)

def annotate_file_pair(seg_csv, aggregated_csv, whitelist):
    # Load files
    seg_df = pd.read_csv(seg_csv)
    agg_df = pd.read_csv(aggregated_csv, header=None)
    agg_df.columns = ['cb', 'count', 'polylox_barcodes']

    # Annotate
    agg_df['one_plx'] = agg_df['polylox_barcodes'].apply(is_single_barcode)
    agg_df['in_white_list'] = agg_df['polylox_barcodes'].apply(lambda x: all_barcodes_in_whitelist(x, whitelist))

    # Merge
    merged = seg_df.merge(agg_df[['cb', 'one_plx', 'in_white_list']], on='cb', how='left')

    # Output
    base, ext = os.path.splitext(seg_csv)
    output_path = f"{base}_annotated{ext}"
    merged.to_csv(output_path, index=False)
    print(f"✅ Annotated {os.path.basename(seg_csv)} → {os.path.basename(output_path)}")

def annotate_polylox_counts(counts_csv, whitelist):
    # Load polylox counts file
    counts_df = pd.read_csv(counts_csv)

    # Identify polylox column (assume first)
    polylox_col = counts_df.columns[0]

    # Annotate
    counts_df['polylox_in_white_list'] = counts_df[polylox_col].apply(lambda x: str(x).strip() in whitelist)

    # Output
    base, ext = os.path.splitext(counts_csv)
    output_path = f"{base}_annotated{ext}"
    counts_df.to_csv(output_path, index=False)
    print(f"✅ Annotated {os.path.basename(counts_csv)} → {os.path.basename(output_path)}")

def batch_annotate(directory, whitelist_file):
    whitelist = load_whitelist(whitelist_file)
    print(f"✅ Loaded {len(whitelist)} whitelist barcodes.")

    files = os.listdir(directory)
    seg_files = sorted(f for f in files if f.endswith("seg_assemble.csv") and not f.endswith("_annotated.csv"))

    for seg_file in seg_files:
        sample_id = seg_file.split('_seg_assemble')[0]
        agg_file = f"{sample_id}._aggregated_data.csv"
        counts_file = f"{sample_id}_polylox_counts.csv"

        seg_path = os.path.join(directory, seg_file)
        agg_path = os.path.join(directory, agg_file)
        counts_path = os.path.join(directory, counts_file)

        if not os.path.exists(agg_path):
            print(f"⚠️ Missing aggregated file for {seg_file}, skipping.")
            continue

        annotate_file_pair(seg_path, agg_path, whitelist)

        if os.path.exists(counts_path):
            annotate_polylox_counts(counts_path, whitelist)
        else:
            print(f"⚠️ Missing polylox counts file: {counts_file}")

# === Run as script ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch annotate Polylox segment and barcode counts.")
    parser.add_argument("directory", help="Directory with Smp* files")
    parser.add_argument("-w", "--whitelist", required=True, help="Path to whitelist file")
    args = parser.parse_args()

    batch_annotate(args.directory, args.whitelist)

