import gzip
import csv
import os
import sys
import logging
import time
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm


# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
    )
    return logging.getLogger("dbnsfp5")


# ─── dbNSFP5 Column Groups ────────────────────────────────────────────────────

# Curated column subsets for common ML tasks.
# "auto" uses all columns; named groups pick a focused subset.
COLUMN_GROUPS = {
    "pathogenicity": [
        "chr", "pos(1-based)", "ref", "alt", "aaref", "aaalt",
        "SIFT4G_score", "SIFT4G_pred",
        "Polyphen2_HDIV_score", "Polyphen2_HDIV_pred",
        "Polyphen2_HVAR_score", "Polyphen2_HVAR_pred",
        "LRT_score", "LRT_pred",
        "MutationTaster_score", "MutationTaster_pred",
        "PROVEAN_score", "PROVEAN_pred",
        "VEST4_score",
        "MetaSVM_score", "MetaSVM_pred",
        "MetaLR_score", "MetaLR_pred",
        "M-CAP_score", "M-CAP_pred",
        "REVEL_score",
        "MutPred_score",
        "MVP_score",
        "MPC_score",
        "PrimateAI_score", "PrimateAI_pred",
        "DEOGEN2_score", "DEOGEN2_pred",
        "ClinPred_score", "ClinPred_pred",
        "LIST-S2_score", "LIST-S2_pred",
        "clinvar_clnsig",
        "Interpro_domain",
        "GTEx_V8_gene", "GTEx_V8_tissue",
    ],
    "conservation": [
        "chr", "pos(1-based)", "ref", "alt",
        "phyloP100way_vertebrate",
        "phyloP30way_mammalian",
        "phyloP17way_primate",
        "phastCons100way_vertebrate",
        "phastCons30way_mammalian",
        "phastCons17way_primate",
        "GERP++_NR", "GERP++_RS",
        "SiPhy_29way_logOdds",
        "bStatistic",
    ],
    "population": [
        "chr", "pos(1-based)", "ref", "alt",
        "AF", "AF_AFR", "AF_AMR", "AF_ASJ", "AF_EAS", "AF_FIN", "AF_NFE",
        "gnomAD_exomes_AF", "gnomAD_exomes_AFR_AF", "gnomAD_exomes_AMR_AF",
        "gnomAD_exomes_EAS_AF", "gnomAD_exomes_NFE_AF",
        "gnomAD_genomes_AF",
    ],
    "splicing": [
        "chr", "pos(1-based)", "ref", "alt",
        "Ensembl_transcriptid",
        "HGVSc_snpEff", "HGVSp_snpEff",
        "MaxEntScan_alt", "MaxEntScan_ref", "MaxEntScan_diff",
        "ada_score", "rf_score",
        "dbscSNV_ADA_SCORE", "dbscSNV_RF_SCORE",
    ],
}

# Columns to always drop — low ML value or redundant
DEFAULT_DROP_COLS = [
    "Uniprot_acc_Polyphen2", "Uniprot_id_Polyphen2", "Uniprot_aapos_Polyphen2",
    "APPRIS", "TSL",
]

# Known sentinel strings that represent missing data in dbNSFP5
MISSING_SENTINELS = [".", "-", "NA", "N/A", "nan", "NaN", "None", ""]


# ─── Core Reader ──────────────────────────────────────────────────────────────

def iter_chunks(
    gz_path: str,
    chunk_size: int = 100_000,
    usecols: Optional[list] = None,
    sep: str = "\t",
) -> Iterator[pd.DataFrame]:
    """
    Lazily yields DataFrame chunks from a dbNSFP5 .gz file.
    Uses pandas chunked reader to avoid loading the entire file into memory.
    """
    log = logging.getLogger("dbnsfp5")
    log.info(f"Opening: {gz_path}")

    reader = pd.read_csv(
        gz_path,
        sep=sep,
        compression="gzip",
        usecols=usecols,
        chunksize=chunk_size,
        low_memory=False,
        na_values=MISSING_SENTINELS,
        dtype=str,          # Read everything as string first; cast later
        comment=None,       # dbNSFP5 has no comment lines
        encoding="utf-8",
    )

    for chunk in reader:
        yield chunk


def get_header(gz_path: str, sep: str = "\t") -> list[str]:
    """Read only the header row without decompressing the full file."""
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=sep)
        return next(reader)


# ─── Cleaning ─────────────────────────────────────────────────────────────────

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: lowercase, replace special chars with underscores."""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\(\)\-\+\/\;]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    return df


def cast_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt numeric conversion on every column.
    Columns with multiple semicolon-delimited values (transcript-level)
    are kept as strings; numeric columns are cast to float32 for ML efficiency.
    """
    for col in df.columns:
        # Skip obvious ID / annotation columns
        if col in ("chr", "ref", "alt", "aaref", "aaalt") or "id" in col or "name" in col:
            continue
        # Try converting; if >50% parse, keep numeric
        converted = pd.to_numeric(df[col], errors="coerce")
        fill_rate = converted.notna().mean()
        if fill_rate >= 0.5:
            df[col] = converted.astype("float32")
    return df


def handle_semicolon_fields(
    df: pd.DataFrame,
    strategy: str = "first",
) -> pd.DataFrame:
    """
    dbNSFP5 stores per-transcript values as semicolon-delimited strings.
    Strategies:
      'first'  — take the first value (usually canonical transcript)
      'max'    — take the maximum numeric value
      'min'    — take the minimum numeric value
      'mean'   — take the mean of all numeric values
      'keep'   — leave as-is (string)
    """
    if strategy == "keep":
        return df

    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        sample = df[col].dropna().head(200)
        has_semi = sample.str.contains(";", na=False).mean() > 0.3
        if not has_semi:
            continue

        if strategy == "first":
            df[col] = df[col].str.split(";").str[0]
        elif strategy in ("max", "min", "mean"):
            def agg_semi(val):
                if pd.isna(val):
                    return np.nan
                parts = [p for p in val.split(";") if p not in MISSING_SENTINELS]
                nums = pd.to_numeric(parts, errors="coerce")
                nums = nums[~np.isnan(nums)]
                if len(nums) == 0:
                    return np.nan
                return getattr(nums, strategy)()
            df[col] = df[col].apply(agg_semi).astype("float32")

    return df


def impute_missing(
    df: pd.DataFrame,
    strategy: str = "none",
) -> pd.DataFrame:
    """
    Missing value imputation for numeric columns.
    strategy: 'none' | 'median' | 'mean' | 'zero'
    """
    if strategy == "none":
        return df

    num_cols = df.select_dtypes(include=[np.floating, np.integer]).columns
    for col in num_cols:
        if df[col].isna().sum() == 0:
            continue
        if strategy == "median":
            fill = df[col].median()
        elif strategy == "mean":
            fill = df[col].mean()
        elif strategy == "zero":
            fill = 0.0
        else:
            continue
        df[col] = df[col].fillna(fill)

    return df


def drop_low_quality_cols(
    df: pd.DataFrame,
    max_missing_frac: float = 0.95,
    extra_drop: list = None,
) -> pd.DataFrame:
    """
    Drop columns where more than max_missing_frac of values are NaN,
    and any explicitly requested extra columns.
    """
    missing_frac = df.isna().mean()
    drop_high_miss = missing_frac[missing_frac > max_missing_frac].index.tolist()
    extra_drop = extra_drop or []
    all_drop = list(set(drop_high_miss + extra_drop))
    existing_drop = [c for c in all_drop if c in df.columns]
    return df.drop(columns=existing_drop)


# ─── Writer ───────────────────────────────────────────────────────────────────

def write_chunk(
    df: pd.DataFrame,
    output_dir: str,
    chunk_idx: int,
    fmt: str = "csv",
    compress: bool = False,
) -> str:
    """Write a single chunk to disk. Returns the output path."""
    os.makedirs(output_dir, exist_ok=True)
    suffix = f".{fmt}" + (".gz" if compress else "")
    filename = f"chunk_{chunk_idx:05d}{suffix}"
    out_path = os.path.join(output_dir, filename)

    if fmt == "csv":
        df.to_csv(out_path, index=False, compression="gzip" if compress else None)
    elif fmt == "parquet":
        df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    elif fmt == "tsv":
        df.to_csv(out_path, sep="\t", index=False, compression="gzip" if compress else None)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    return out_path


# ─── Manifest ─────────────────────────────────────────────────────────────────

def write_manifest(
    output_dir: str,
    chunk_paths: list[str],
    total_rows: int,
    columns: list[str],
    elapsed: float,
) -> None:
    """Write a JSON manifest describing the output chunks."""
    import json
    manifest = {
        "total_rows": total_rows,
        "num_chunks": len(chunk_paths),
        "columns": columns,
        "elapsed_seconds": round(elapsed, 2),
        "chunks": [os.path.basename(p) for p in chunk_paths],
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logging.getLogger("dbnsfp5").info(f"Manifest written → {manifest_path}")


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def convert(
    input_path: str,
    output_dir: str,
    chunk_size: int = 100_000,
    columns: str = "auto",
    semicolon_strategy: str = "first",
    missing_strategy: str = "none",
    max_missing_frac: float = 0.95,
    fmt: str = "csv",
    compress: bool = False,
    drop_cols: list = None,
    log_level: str = "INFO",
) -> None:
    log = setup_logging(log_level)
    start = time.time()

    # ── Resolve column selection ─────────────────────────────────────────────
    if columns == "auto":
        usecols = None
        log.info("Column mode: ALL columns")
    elif columns in COLUMN_GROUPS:
        usecols = COLUMN_GROUPS[columns]
        log.info(f"Column mode: '{columns}' preset ({len(usecols)} columns)")
    else:
        # Treat as comma-separated list
        usecols = [c.strip() for c in columns.split(",")]
        log.info(f"Column mode: custom ({len(usecols)} columns)")

    # ── Peek at header ───────────────────────────────────────────────────────
    log.info("Reading header …")
    try:
        header = get_header(input_path)
        log.info(f"File has {len(header)} columns total")
        if usecols:
            missing_req = [c for c in usecols if c not in header]
            if missing_req:
                log.warning(f"Requested columns not found in file: {missing_req}")
                usecols = [c for c in usecols if c in header]
    except Exception as e:
        log.error(f"Failed to read header: {e}")
        sys.exit(1)

    # ── Process chunks ───────────────────────────────────────────────────────
    chunk_paths = []
    total_rows = 0
    final_columns = None

    for chunk_idx, chunk in enumerate(tqdm(
        iter_chunks(input_path, chunk_size=chunk_size, usecols=usecols),
        desc="Chunks",
        unit="chunk",
    )):
        # 1. Clean column names
        chunk = clean_column_names(chunk)

        # 2. Drop columns (explicit + high-missing)
        drop = (drop_cols or []) + DEFAULT_DROP_COLS
        chunk = drop_low_quality_cols(chunk, max_missing_frac=max_missing_frac, extra_drop=drop)

        # 3. Handle semicolon-delimited transcript values
        chunk = handle_semicolon_fields(chunk, strategy=semicolon_strategy)

        # 4. Cast numerics
        chunk = cast_numeric_columns(chunk)

        # 5. Impute missing
        chunk = impute_missing(chunk, strategy=missing_strategy)

        # 6. Write
        out_path = write_chunk(chunk, output_dir, chunk_idx, fmt=fmt, compress=compress)
        chunk_paths.append(out_path)
        total_rows += len(chunk)

        if final_columns is None:
            final_columns = chunk.columns.tolist()
            log.info(f"Output columns ({len(final_columns)}): {final_columns[:10]} …")

        log.debug(f"Chunk {chunk_idx:05d}: {len(chunk)} rows → {out_path}")

    elapsed = time.time() - start
    log.info(f"Done. {total_rows:,} rows across {len(chunk_paths)} chunks in {elapsed:.1f}s")
    log.info(f"Output directory: {output_dir}")

    # ── Manifest ─────────────────────────────────────────────────────────────
    write_manifest(output_dir, chunk_paths, total_rows, final_columns or [], elapsed)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG — edit these values, then run:  python dbnsfp5_converter.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INPUT_PATH   = "/content/dbNSFP5.3.1a_variant.chr21.gz"   # ← your .gz file here
OUTPUT_DIR   = "./dbnsfp5_chunks"                    # ← where chunks are saved

# Rows per output file. 100_000 is safe for most machines.
# Increase to 500_000 if you have plenty of RAM.
CHUNK_SIZE   = 100_000

# Column preset: "auto" | "pathogenicity" | "conservation" | "population" | "splicing"
# Or a comma-separated list of exact column names, e.g.:
#   "chr,pos(1-based),ref,alt,SIFT4G_score,REVEL_score"
COLUMNS      = "auto"

# How to collapse semicolon-delimited per-transcript values.
# Options: "first" | "max" | "min" | "mean" | "keep"
# "first" picks the canonical transcript value (recommended for ML).
SEMICOLON_STRATEGY = "first"

# Impute missing numeric values.
# Options: "none" | "median" | "mean" | "zero"
MISSING_STRATEGY   = "none"

# Drop columns where this fraction (0–1) of values are missing.
MAX_MISSING_FRAC   = 0.95

# Output file format: "csv" | "parquet" | "tsv"
# Parquet is strongly recommended for ML — faster reads, smaller files,
# preserves dtypes automatically.
FORMAT       = "csv"

# Gzip-compress CSV/TSV output files (ignored for parquet).
COMPRESS     = False

# Any extra column names you want to drop (in addition to defaults).
DROP_COLS    = []

# Logging verbosity: "DEBUG" | "INFO" | "WARNING" | "ERROR"
LOG_LEVEL    = "INFO"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


if __name__ == "__main__":
    if not os.path.isfile(INPUT_PATH):
        print(f"ERROR: Input file not found: {INPUT_PATH}")
        print("Please update INPUT_PATH in the CONFIG section at the top of this file.")
        sys.exit(1)

    convert(
        input_path=INPUT_PATH,
        output_dir=OUTPUT_DIR,
        chunk_size=CHUNK_SIZE,
        columns=COLUMNS,
        semicolon_strategy=SEMICOLON_STRATEGY,
        missing_strategy=MISSING_STRATEGY,
        max_missing_frac=MAX_MISSING_FRAC,
        fmt=FORMAT,
        compress=COMPRESS,
        drop_cols=DROP_COLS,
        log_level=LOG_LEVEL,
    )
