# nao/ — vendored from HERMES

These four modules are a **mirror**, copied unmodified from HERMES. They are not
Kallen's code and must not be edited here: fix things in HERMES and re-copy, so
that `diff` against the upstream tree stays meaningful and drift is detectable.

| field | value |
|---|---|
| source | `HERMES/ergasterion/organon/` |
| upstream repo | https://github.com/branzanic/HERMES |
| commit | `bef0247` |
| copied | 2026-09-25 |

**Uncommitted upstream changes included in this copy:** ergasterion/organon/compute_nao_from_molcas.py 
(the per-`&CASPT2`-block `'best'` pole pool, and the export of `nao_coeffs_ao`
so that SCF orbitals can be resolved in the CASSCF reference's NAO basis). Commit
these in HERMES and update the hash above.

## Why the filenames mention Turbomole and ORCA

They do not mean Kallen reads those packages — it reads Molcas/OpenMolcas only.
The names are historical. The generic pieces live in the files where they were
first written:

| symbol | lives in | what it does |
|---|---|---|
| `AOBasisFunction` | `..._turbomole.py` | AO basis function record |
| `compute_nao_from_density` | `..._turbomole.py` | the NAO construction (OWSO) |
| `transform_mos_to_nao_basis` | `..._turbomole.py` | AO -> NAO transform |
| `derive_nao_tiers` | `..._turbomole.py` | core/valence/Rydberg tiering |
| `BOHR_TO_ANG`, `_parse_gto_lines`, `parse_molden_mos` | `..._orca.py` | the generic Molden parser |

Molcas writes standard Molden with `[5D]/[7F]/[9G]`, the same convention ORCA
writes, so those parsers apply unchanged. Reusing them rather than duplicating is
deliberate: a second copy of the m_l ordering table is how the Gaussian and
Turbomole d-ordering bugs survived for months.

The Turbomole- and ORCA-specific I/O in these files is never executed on a
Molcas run. It is carried so the copy stays byte-identical to upstream.

## Checking for drift

    for f in nao/*.py; do diff -q "$f" /path/to/HERMES/ergasterion/organon/$(basename $f); done

Compare the four files only; a plain `diff -r` also lists the ~14 upstream
modules that were deliberately not vendored. Any difference means the mirror has
been edited or upstream has moved.
