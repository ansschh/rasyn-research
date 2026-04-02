"""
Download public datasets for robust synthesis experiments.

Datasets:
1. ORDerly — cleaned Open Reaction Database (~1M reactions with conditions/yields)
2. Buchwald HTE — dense condition-outcome grids from high-throughput experiments
"""

import os
import subprocess
import sys
import zipfile
import io
from pathlib import Path

import requests
import pandas as pd


DATA_DIR = Path(__file__).parent / "raw"


# ─── ORDerly dataset ──────────────────────────────────────────────────────────

ORDERLY_URLS = {
    # ORDerly cleaned reaction data from the paper:
    # "ORDerly: Data Sets and Benchmarks for Chemical Reaction Data"
    # Available via Zenodo or the ORDerly GitHub
    "orderly_condition": "https://zenodo.org/records/10654455/files/orderly_condition_data.csv.gz",
    "orderly_forward": "https://zenodo.org/records/10654455/files/orderly_forward_data.csv.gz",
}


def download_file(url: str, dest: Path, chunk_size: int = 8192):
    """Download a file with progress indication."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  Already exists: {dest}")
        return

    print(f"  Downloading: {url}")
    print(f"  To: {dest}")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                print(f"\r  {pct:.1f}% ({downloaded // 1024 // 1024}MB)", end="")

    print(f"\n  Done: {dest}")


def download_orderly():
    """Download ORDerly condition prediction dataset."""
    print("\n=== Downloading ORDerly Dataset ===")
    orderly_dir = DATA_DIR / "orderly"
    orderly_dir.mkdir(parents=True, exist_ok=True)

    for name, url in ORDERLY_URLS.items():
        dest = orderly_dir / url.split("/")[-1]
        download_file(url, dest)

    print("ORDerly download complete.")
    return orderly_dir


# ─── Buchwald HTE datasets ───────────────────────────────────────────────────

# These datasets from Doyle, Ahneman, etc. are typically available via
# journal supplementary info or dedicated repositories.
# We provide download stubs — you may need to manually download some.

BUCHWALD_INFO = {
    "doyle_2018": {
        "description": "Pd-catalyzed Buchwald-Hartwig amination (Doyle et al., Science 2018)",
        "url": "https://raw.githubusercontent.com/doylelab/rxnpredict/master/data/FullCV_01.csv",
        "n_reactions": 3955,
        "conditions": ["aryl_halide", "additive", "base", "ligand"],
        "outcome": "yield",
    },
    "ahneman_2018": {
        "description": "Buchwald-Hartwig amination with DFT descriptors (Ahneman et al., Science 2018)",
        "url": None,  # Available in SI of the paper
        "n_reactions": 3955,
        "conditions": ["aryl_halide", "additive", "base", "ligand"],
        "outcome": "yield",
    },
}


def download_buchwald():
    """Download Buchwald HTE datasets (those with public URLs)."""
    print("\n=== Downloading Buchwald HTE Datasets ===")
    buchwald_dir = DATA_DIR / "buchwald"
    buchwald_dir.mkdir(parents=True, exist_ok=True)

    for name, info in BUCHWALD_INFO.items():
        if info["url"] is None:
            print(f"  {name}: No public URL — check paper supplementary info")
            continue
        dest = buchwald_dir / f"{name}.csv"
        download_file(info["url"], dest)

    print("Buchwald download complete.")
    return buchwald_dir


# ─── Suzuki HTE (Perera et al.) ──────────────────────────────────────────────

def download_suzuki_hte():
    """Download Suzuki coupling HTE dataset (Perera et al. 2020) if available."""
    print("\n=== Suzuki HTE Dataset ===")
    dest_dir = DATA_DIR / "suzuki_hte"
    dest_dir.mkdir(parents=True, exist_ok=True)
    print("  Suzuki HTE: Check ACS JACS paper SI for download")
    print("  Perera et al., JACS 2020, DOI: 10.1021/jacs.0c09947")
    return dest_dir


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    """Download all datasets."""
    print("=" * 60)
    print("Robust Synthesis Experiments — Data Download")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    orderly_dir = download_orderly()
    buchwald_dir = download_buchwald()
    suzuki_dir = download_suzuki_hte()

    print("\n" + "=" * 60)
    print("Download summary:")
    print(f"  ORDerly:  {orderly_dir}")
    print(f"  Buchwald: {buchwald_dir}")
    print(f"  Suzuki:   {suzuki_dir}")
    print("=" * 60)

    # Verify what we got
    print("\nFiles downloaded:")
    for p in sorted(DATA_DIR.rglob("*")):
        if p.is_file():
            size_mb = p.stat().st_size / 1024 / 1024
            print(f"  {p.relative_to(DATA_DIR)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
