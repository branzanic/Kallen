# Kallen

**Electron-transfer couplings from an OpenMolcas calculation, with the spectral
weights computed instead of assumed.**

If you have run CASSCF/CASPT2 with RASSI on a molecule, Kallen turns that output
into two things the standard output does not give you:

1. **The pole strength of each ionisation**, `γ_n = ‖d^n‖²` — the fraction of
   intensity the main photoelectron band retains. A one-electron picture predicts
   one band per orbital at full intensity; correlation splits that, and the
   remainder appears as shake-up satellites. `γ_n` is what is left in the main
   line, and it is what electron-momentum spectroscopy calls the spectroscopic
   factor.
2. **An atom-to-atom electronic coupling** `G_rs`, the electronic factor of the
   non-adiabatic golden rule, summed over ionisations *weighted by those pole
   strengths* rather than by the unit weights an orbital-based Green's function
   assumes.

The practical problem it solves: computing `G_rs` normally means summing over
molecular orbitals, which silently gives every pole full weight. If correlation
has redistributed intensity — a breakdown of the orbital picture, shake-up
satellites, a contested state ordering — that assumption is wrong, and there is no
way to tell from the orbital calculation itself. Kallen takes the weights from
the ionised states you already computed.

## Install

Clone and run. The only requirement is **numpy**.

```bash
git clone https://github.com/branzanic/Kallen.git
cd Kallen
python3 kallen.py --help
```

Nothing else is needed: the natural-atomic-orbital machinery ships in `nao/`
(see `nao/PROVENANCE.md`).

## Use

Kallen consumes three files, all produced by a single OpenMolcas job:

| file | carries |
|---|---|
| `$Project.rasscf.molden` of the **neutral** | defines the NAO basis |
| `$Project.dys.molden.SF.1` | the Dyson orbitals — the weights |
| the OpenMolcas output | the CASPT2 poles |

```bash
python3 kallen.py \
    --reference     runs/thiophene/thiophene_neutral.rasscf.molden \
    --dyson         runs/thiophene/thiophene_dyson.dys.molden.SF.1 \
    --output        runs/thiophene/thiophene_dyson.log \
    --poles         best \
    --hf-reference  runs/thiophene/thiophene_dyson.scf.molden \
    --ea-ev         -1.15 \
    --pairs         1,2 --terms
```

`--hf-reference` adds the orbital-based sum from the same run's `&SCF` step, so
both forms come out of one command and can be compared directly. Omitting
`--pairs` evaluates every atom pair carrying valence NAOs.

Three points about the deck, each of which is a silent failure if missed, are
documented in `nao/compute_nao_from_molcas.py` and in the paper's SI:
`group=c1` is required; `JOB001` must be the neutral; and the Dyson amplitudes
must **not** be renormalised on read, because their squared norms *are* the pole
strengths.

## Reproducing the published numbers

`runs/` contains the complete OpenMolcas inputs and outputs for both worked
examples, so every number in the paper can be regenerated from a clone:

```bash
bash tests/test_o2.sh              # the O2 regression
bash tests/test_worked_examples.sh # thiophene and pyrazine
python3 runs/all_pairs.py          # G_rs on every atom pair, both forms
```

`runs/*/\*_dyson.input` are the decks as run; re-running them with `pymolcas`
regenerates the outputs from scratch.

## What is in `nao/`

Four modules copied unmodified from [HERMES](https://github.com/branzanic/HERMES),
which builds the orthonormal, atom-resolved natural atomic orbital basis that
`G_rs` between two *atoms* requires — a raw atomic-orbital expansion will not do,
because the per-atom coefficients are then basis-set dependent and not separable.

They are a mirror, not Kallen's code: do not edit them here. `nao/PROVENANCE.md`
records the upstream commit and how to check for drift. Their filenames mention
Turbomole and ORCA for historical reasons only — Kallen reads Molcas/OpenMolcas.

## Licence

MIT. The vendored modules in `nao/` are MIT by the same author.
