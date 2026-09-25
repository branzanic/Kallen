#!/usr/bin/env bash
# Regression: the O2 example must reproduce examples/EXPECTED.txt.
#
# The numbers are from OpenMolcas master (biorthogonality-corrected amplitudes).
# Molcas 8.6 gives pole strengths ~0.3% lower and will NOT match -- that is a real
# difference between the builds, not a tolerance problem. See OPENMOLCAS_CHECK.md.
set -u
cd "$(dirname "$0")/.."
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
gunzip -c examples/o2_dyson.openmolcas.out.gz > "$T/om.out"

python3 kallen.py --reference examples/o2_dyson.rasscf.molden \
                  --dyson     examples/o2_dyson.dys.molden.SF.1 \
                  --output    "$T/om.out" \
                  --anion-output examples/o2_anion.out \
                  --pairs 1,2 --terms 2>&1 | sed -n '/poles retained/,$p' > "$T/got.txt"

if diff -q examples/EXPECTED.txt "$T/got.txt" >/dev/null; then
  echo "  PASS  O2 reproduces EXPECTED.txt"
  exit 0
else
  echo "  FAIL  O2 differs from EXPECTED.txt"
  diff examples/EXPECTED.txt "$T/got.txt" | head -20
  exit 1
fi
