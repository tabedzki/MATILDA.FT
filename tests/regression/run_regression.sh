#!/bin/bash
# Copyright (c) 2026 University of Pennsylvania
# Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).
#
# Golden-file regression harness. For every cases/*/ directory: run
# matilda.ft on case.input in a scratch dir and compare each file listed in
# golden/ against the freshly produced one with compare_numeric.py.
#
#   ./run_regression.sh            run all cases
#   ./run_regression.sh --update   regenerate the golden files in place
#   MATILDA_BIN=/path/to/matilda.ft ./run_regression.sh
#
# FTS/SCFT runs are deterministic (bit-identical on the same GPU), so the
# default tolerances are tight; they exist to absorb FFT rounding differences
# across GPU generations / CUDA versions, not algorithm changes.
set -u
cd "$(dirname "$0")"

BIN=${MATILDA_BIN:-$(cd ../.. && pwd)/matilda.ft}
RTOL=${MATILDA_REG_RTOL:-1e-8}
ATOL=${MATILDA_REG_ATOL:-1e-10}
UPDATE=0
[ "${1:-}" = "--update" ] && UPDATE=1

if [ ! -x "$BIN" ]; then
    echo "matilda.ft binary not found at $BIN (build src/ first, or set MATILDA_BIN)" >&2
    exit 2
fi

fail=0
for case_dir in cases/*/; do
    name=$(basename "$case_dir")
    work=$(mktemp -d "${TMPDIR:-/tmp}/matilda-reg-$name-XXXXXX")
    cp "$case_dir/case.input" "$work/"

    if ! (cd "$work" && "$BIN" -in case.input > stdout.log 2>&1); then
        echo "[FAIL] $name: matilda.ft exited nonzero; log tail:" >&2
        tail -5 "$work/stdout.log" >&2
        fail=1
        continue
    fi

    if [ "$UPDATE" = 1 ]; then
        rm -rf "$case_dir/golden"
        mkdir -p "$case_dir/golden"
        # Golden every data file the run produced (not the input/log).
        for f in "$work"/*.dat; do
            cp "$f" "$case_dir/golden/"
        done
        echo "[GOLDEN] $name: $(ls "$case_dir/golden" | wc -l) files regenerated"
    else
        case_fail=0
        for g in "$case_dir"/golden/*; do
            b=$(basename "$g")
            if [ ! -f "$work/$b" ]; then
                echo "[FAIL] $name: run did not produce $b" >&2
                case_fail=1
            elif ! python3 compare_numeric.py "$g" "$work/$b" --rtol "$RTOL" --atol "$ATOL"; then
                case_fail=1
            fi
        done
        if [ "$case_fail" = 0 ]; then
            echo "[PASS] $name"
        else
            echo "[FAIL] $name (work dir kept: $work)" >&2
            fail=1
            continue
        fi
    fi
    rm -rf "$work"
done

exit $fail
