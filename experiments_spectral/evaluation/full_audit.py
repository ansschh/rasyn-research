"""
Full-corpus atom-index audit of nmrshiftdb2.

Audits ALL molecules (not just 200) to produce definitive statistics
on the atom-index misassignment problem in 13C spectra.
"""

import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem


def run_full_audit(parquet_path: str, output_path: str):
    t0 = time.time()
    df = pd.read_parquet(parquet_path)

    # All 13C spectrum columns
    c13_cols = [c for c in df.columns if c.startswith("Spectrum 13C")]
    print(f"Found {len(c13_cols)} 13C spectrum columns")

    total_assignments = 0
    correct_carbon = 0
    wrong_element = Counter()
    out_of_range = 0
    parse_failures = 0
    mol_failures = 0

    per_mol_stats = []
    wrong_shift_values = []  # collect shifts assigned to wrong elements

    n_with_any_c13 = 0

    for idx, row in df.iterrows():
        # Check if any 13C spectrum exists
        has_c13 = False
        for col in c13_cols:
            if pd.notna(row.get(col)):
                has_c13 = True
                break
        if not has_c13:
            continue

        n_with_any_c13 += 1
        smi = str(row.get("SMILES", ""))
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mol_failures += 1
            continue

        n_atoms = mol.GetNumAtoms()

        # Count carbons in molecule
        n_carbons = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)

        mol_correct = 0
        mol_wrong = 0
        mol_total = 0
        mol_oob = 0

        # Process first valid 13C spectrum
        for col in c13_cols:
            spec = row.get(col)
            if pd.isna(spec):
                continue

            spec = str(spec)
            for p in spec.split("|"):
                p = p.strip()
                if not p:
                    continue
                fields = p.split(";")
                if len(fields) < 3:
                    parse_failures += 1
                    continue
                try:
                    shift = float(fields[0])
                    aidx = int(fields[2])
                except (ValueError, IndexError):
                    parse_failures += 1
                    continue

                total_assignments += 1
                mol_total += 1

                if aidx >= n_atoms:
                    out_of_range += 1
                    mol_oob += 1
                    continue

                atom = mol.GetAtomWithIdx(aidx)
                if atom.GetAtomicNum() == 6:
                    correct_carbon += 1
                    mol_correct += 1
                else:
                    sym = atom.GetSymbol()
                    wrong_element[sym] += 1
                    mol_wrong += 1
                    wrong_shift_values.append({
                        "shift": shift,
                        "element": sym,
                        "mol_idx": idx,
                    })

            break  # only audit first valid spectrum per molecule

        if mol_total > 0:
            per_mol_stats.append({
                "correct_frac": mol_correct / mol_total,
                "n_wrong": mol_wrong,
                "n_total": mol_total,
                "n_oob": mol_oob,
                "n_carbons": n_carbons,
                "n_peaks_vs_carbons": mol_total - n_carbons,
            })

        if idx % 10000 == 0 and idx > 0:
            elapsed = time.time() - t0
            print(f"  Audited {idx}/{len(df)} ({elapsed:.0f}s)")

    elapsed = time.time() - t0

    # Compute statistics
    fracs = [s["correct_frac"] for s in per_mol_stats]
    n_all_correct = sum(1 for f in fracs if f == 1.0)
    n_any_wrong = sum(1 for f in fracs if f < 1.0)
    n_majority_wrong = sum(1 for f in fracs if f < 0.5)

    wrong_shifts = np.array([w["shift"] for w in wrong_shift_values])

    results = {
        "audit_scope": "full_corpus",
        "n_molecules_total": len(df),
        "n_with_c13_spectra": n_with_any_c13,
        "n_valid_molecules": len(per_mol_stats),
        "n_mol_failures": mol_failures,
        "total_shift_assignments": total_assignments,
        "correct_carbon": correct_carbon,
        "correct_pct": round(100 * correct_carbon / max(total_assignments, 1), 2),
        "wrong_element_total": sum(wrong_element.values()),
        "wrong_pct": round(100 * sum(wrong_element.values()) / max(total_assignments, 1), 2),
        "wrong_by_element": dict(wrong_element.most_common()),
        "out_of_range": out_of_range,
        "out_of_range_pct": round(100 * out_of_range / max(total_assignments, 1), 2),
        "parse_failures": parse_failures,
        "per_molecule": {
            "mean_correct_frac": round(float(np.mean(fracs)), 4),
            "median_correct_frac": round(float(np.median(fracs)), 4),
            "all_correct_count": n_all_correct,
            "all_correct_pct": round(100 * n_all_correct / max(len(fracs), 1), 2),
            "any_wrong_count": n_any_wrong,
            "any_wrong_pct": round(100 * n_any_wrong / max(len(fracs), 1), 2),
            "majority_wrong_count": n_majority_wrong,
        },
        "wrong_shift_distribution": {
            "min": round(float(wrong_shifts.min()), 1) if len(wrong_shifts) > 0 else None,
            "max": round(float(wrong_shifts.max()), 1) if len(wrong_shifts) > 0 else None,
            "mean": round(float(wrong_shifts.mean()), 1) if len(wrong_shifts) > 0 else None,
            "percentiles": {
                "10": round(float(np.percentile(wrong_shifts, 10)), 1) if len(wrong_shifts) > 0 else None,
                "25": round(float(np.percentile(wrong_shifts, 25)), 1) if len(wrong_shifts) > 0 else None,
                "50": round(float(np.percentile(wrong_shifts, 50)), 1) if len(wrong_shifts) > 0 else None,
                "75": round(float(np.percentile(wrong_shifts, 75)), 1) if len(wrong_shifts) > 0 else None,
                "90": round(float(np.percentile(wrong_shifts, 90)), 1) if len(wrong_shifts) > 0 else None,
            },
        },
        "elapsed_seconds": round(elapsed, 1),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  FULL CORPUS AUDIT RESULTS")
    print(f"{'='*60}")
    print(f"Molecules audited: {len(per_mol_stats)} (of {n_with_any_c13} with 13C)")
    print(f"Total shift assignments: {total_assignments}")
    print(f"Correct (carbon): {correct_carbon} ({results['correct_pct']}%)")
    print(f"Wrong element: {sum(wrong_element.values())} ({results['wrong_pct']}%)")
    for elem, count in wrong_element.most_common(5):
        print(f"  {elem}: {count}")
    print(f"Out of range: {out_of_range} ({results['out_of_range_pct']}%)")
    print(f"\nPer-molecule:")
    print(f"  All correct: {n_all_correct} ({results['per_molecule']['all_correct_pct']}%)")
    print(f"  Any wrong: {n_any_wrong} ({results['per_molecule']['any_wrong_pct']}%)")
    print(f"  Mean correct fraction: {results['per_molecule']['mean_correct_frac']}")
    print(f"\nWrong-element shifts span: {results['wrong_shift_distribution']['min']} - {results['wrong_shift_distribution']['max']} ppm")
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"Saved to {output_path}")

    return results


if __name__ == "__main__":
    run_full_audit(
        "/workspace/rasyn-research/experiments_spectral/data/raw/nmrshiftdb2/data/nmrshiftdb2.parquet",
        "/workspace/rasyn-research/experiments_spectral/results/full_corpus_audit.json",
    )
