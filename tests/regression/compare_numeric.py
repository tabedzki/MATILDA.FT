#!/usr/bin/env python3
# Copyright (c) 2026 University of Pennsylvania
# Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).
"""Token-wise numeric comparison of two text files.

Usage: compare_numeric.py GOLDEN ACTUAL [--rtol R] [--atol A]

Numeric tokens are compared with |a - g| <= atol + rtol * |g|; everything
else (headers, keywords) must match exactly. Exit 0 on match, 1 on any
mismatch (first few mismatches are printed).
"""
import argparse
import sys


def as_float(tok):
    try:
        return float(tok)
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("golden")
    ap.add_argument("actual")
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--atol", type=float, default=1e-10)
    args = ap.parse_args()

    with open(args.golden) as f:
        glines = f.read().splitlines()
    with open(args.actual) as f:
        alines = f.read().splitlines()

    errors = []
    if len(glines) != len(alines):
        errors.append(f"line count differs: golden {len(glines)} vs actual {len(alines)}")

    for ln, (gl, al) in enumerate(zip(glines, alines), start=1):
        gtok, atok = gl.split(), al.split()
        if len(gtok) != len(atok):
            errors.append(f"line {ln}: token count differs: {len(gtok)} vs {len(atok)}")
            continue
        for col, (g, a) in enumerate(zip(gtok, atok), start=1):
            gf, af = as_float(g), as_float(a)
            if gf is None or af is None:
                if g != a:
                    errors.append(f"line {ln} col {col}: '{g}' != '{a}'")
            elif abs(af - gf) > args.atol + args.rtol * abs(gf):
                errors.append(f"line {ln} col {col}: {g} vs {a}")
        if len(errors) > 10:
            break

    if errors:
        print(f"MISMATCH {args.actual} vs {args.golden}:")
        for e in errors[:10]:
            print("  " + e)
        if len(errors) > 10:
            print(f"  ... ({len(errors)} + mismatches, stopped early)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
