#!/usr/bin/env python3
"""
Add variable_class column to priorityA_master_nss_annotated.csv

Classification rules (strict, conservative):
- EB*, BY*, SB* → "known_binary"
- RR* → "pulsating_star"
- QSO, AGN → "AGN_QSO"
- Otherwise → "candidate_companion"
"""

import csv
import sys

INPUT_FILE = "data/derived/priorityA_master_nss_annotated.csv"
OUTPUT_CLASSES = "data/derived/priorityA_master_annotated_classes.csv"
OUTPUT_FOLLOWUP = "data/derived/priorityA_followup_only.csv"


def classify_variable(simbad_otype: str) -> str:
    """
    Classify a source based on its SIMBAD object type.

    Returns one of:
    - "known_binary"
    - "pulsating_star"
    - "AGN_QSO"
    - "candidate_companion" (default)
    """
    if not simbad_otype or simbad_otype.strip() == "":
        return "candidate_companion"

    otype = simbad_otype.strip().upper()

    # Known binary types
    if "EB*" in otype or "EB" == otype:
        return "known_binary"
    if "BY*" in otype or "BY" == otype:
        return "known_binary"
    if "SB*" in otype or "SB" == otype:
        return "known_binary"

    # Pulsating stars
    if "RR*" in otype or otype.startswith("RR"):
        return "pulsating_star"

    # AGN/QSO
    if "QSO" in otype or "AGN" in otype:
        return "AGN_QSO"

    # Default: candidate_companion
    return "candidate_companion"


def main():
    # Read input CSV
    with open(INPUT_FILE, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"Read {len(rows)} rows from {INPUT_FILE}")

    # Find simbad_otype column
    if "simbad_otype" not in fieldnames:
        print("ERROR: simbad_otype column not found!")
        sys.exit(1)

    # Add variable_class column
    new_fieldnames = list(fieldnames) + ["variable_class"]

    # Classify each row
    class_counts = {}
    for row in rows:
        otype = row.get("simbad_otype", "")
        vclass = classify_variable(otype)
        row["variable_class"] = vclass
        class_counts[vclass] = class_counts.get(vclass, 0) + 1

    print("\nClassification counts:")
    for vclass, count in sorted(class_counts.items()):
        print(f"  {vclass}: {count}")

    # Write full annotated file
    with open(OUTPUT_CLASSES, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUTPUT_CLASSES}")

    # Write follow-up only subset
    followup_rows = [r for r in rows if r["variable_class"] == "candidate_companion"]

    # Sort by S_robust descending
    followup_rows.sort(key=lambda r: float(r.get("S_robust", 0)), reverse=True)

    with open(OUTPUT_FOLLOWUP, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(followup_rows)

    print(f"Wrote {len(followup_rows)} rows to {OUTPUT_FOLLOWUP}")

    # Summary
    total = len(rows)
    excluded = total - len(followup_rows)
    print(f"\n=== SUMMARY ===")
    print(f"Total Priority A objects: {total}")
    print(f"Excluded (known variable classes): {excluded}")
    print(f"Remaining for follow-up: {len(followup_rows)}")

    # Top 5 candidates
    print(f"\nTop 5 candidate_companion targets:")
    print(f"{'Rank':<5} {'TARGETID':<20} {'Gaia SOURCE_ID':<22} {'N':<4} {'S_robust':<10}")
    print("-" * 65)
    for i, row in enumerate(followup_rows[:5], 1):
        tid = row.get("targetid", "?")
        gaia = row.get("gaia_source_id", "")
        n = row.get("n_epochs", "?")
        s_rob = row.get("S_robust", "?")
        print(f"{i:<5} {tid:<20} {gaia:<22} {n:<4} {s_rob:<10}")


if __name__ == "__main__":
    main()
