#!/usr/bin/env python3
"""
check_atom_balance_with_H.py

Reads a CSV file with columns: reactants,products (SMILES).
Counts atoms INCLUDING HYDROGENS and checks atom balance.
"""

import argparse
from collections import Counter
import pandas as pd
from rdkit import Chem
from reaction_space.utils import *

def compare_reaction(reactants: str, products: str):
    r = element_counts(reactants)
    p = element_counts(products)

    balanced = (r == p)

    return {
        "reactant_total_atoms": sum(r.values()),
        "product_total_atoms": sum(p.values()),
        "balanced_by_elements": balanced,
        "reactant_element_counts": dict(sorted(r.items())),
        "product_element_counts": dict(sorted(p.items())),
        "excess_on_products": dict(sorted((p - r).items())),
        "excess_on_reactants": dict(sorted((r - p).items())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="CSV with columns reactants,products")
    parser.add_argument("--out", help="Optional output CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)
    if not {"reactants", "products"}.issubset(df.columns):
        raise ValueError("CSV must contain reactants and products columns")

    rows = []
    for i, row in df.iterrows():
        try:
            info = compare_reaction(row["reactants"], row["products"])
            rows.append({
                "row": i,
                **info
            })
        except Exception as e:
            rows.append({
                "row": i,
                "error": str(e)
            })

    out = pd.DataFrame(rows)

    print(
        f"Total: {len(out)} | "
        f"Balanced: {out['balanced_by_elements'].fillna(False).sum()} | "
        f"Errors: {out['error'].notna().sum() if 'error' in out else 0}"
    )

    if args.out:
        out.to_csv(args.out, index=False)
        print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
