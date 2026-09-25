#!/usr/bin/env bash
# Regression on the two worked examples of the OpenMolcas 2027 contribution.
#
# These lock in the numbers the paper quotes, and they exercise --poles best,
# which the O2 test does not: both decks run a single-state neutral against an
# XMS cation manifold, so the neutral emits no MS/XMS row and a pool collected
# by row type would silently lose index 0.  See HERMES CHANGELOG 2026-09-23.
set -u
cd "$(dirname "$0")/.."
fail=0

for m in thiophene pyrazine; do
  case $m in
    thiophene) pair=1,2 ; ea=-1.15  ;;
    pyrazine)  pair=1,4 ; ea=-0.065 ;;
  esac
  got=$(mktemp)
  python3 kallen.py --reference runs/$m/${m}_neutral.rasscf.molden \
                    --dyson     runs/$m/${m}_dyson.dys.molden.SF.1 \
                    --output    runs/$m/${m}_dyson.log \
                    --poles best --ea-ev $ea --pairs $pair --terms 2>&1 \
    | sed -n '/poles retained/,$p' > "$got"
  if diff -q tests/EXPECTED_$m.txt "$got" > /dev/null; then
    echo "  PASS  $m reproduces tests/EXPECTED_$m.txt"
  else
    echo "  FAIL  $m differs"
    diff tests/EXPECTED_$m.txt "$got" | head -20
    fail=1
  fi
  rm -f "$got"
done
exit $fail
