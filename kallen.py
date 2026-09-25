#!/usr/bin/env python3
"""Kallen — the Green's function from Dyson orbitals and exact poles.

    G_rs(E) = SUM_n  d_r^n d_s^n / (E - E_n),    E_n = -IP_n

This is the Kallen-Lehmann spectral representation evaluated between two atoms,
with the EXACT ingredients rather than their one-electron stand-ins:

    d^n   Dyson orbital for ionisation to cation state n, from OpenMolcas RASSI
    IP_n  exact ionisation potential, from OpenMolcas CASPT2

Everywhere else the same expression is evaluated with c_rk c_sk in place of the
weights and eps_k in place of the poles, i.e. with orbitals standing in for
states. Kallen is the interface that substitutes the WEIGHTS and not only the
poles, and so is the one that tests the held-fixed-numerator assumption the
others make.

SCOPE, deliberately narrow. This module computes G_rs from Dyson orbitals and
nothing else. There is no axis projection, no sigma/pi/delta decomposition, no
per-plane graph, no shell sum, no pathway routing. Those belong to HERMES, which
consumes the same NAO basis; keeping them out is what makes this file readable
as the description of one interface.

Two conventions that cost real time to discover, both verified against the
OpenMolcas sources (identical to Molcas 8.6, checked 2026-09-22):

  1. The Dyson coefficients in the Molden export are UN-NORMALISED. Their
     squared norm IS the pole strength -- src/rassi/mkdysorb.f:117,
     DYSAMP = SQRT(sum OVLP^2). Normalising them on read silently deletes the
     spectral weights, which are the entire reason for using Dyson orbitals.

  2. The Molden 'Ene=' field must NOT be used as a pole, on two independent
     grounds: it carries the CASSCF-level energy, and
     src/property_util/molden_dysorb.f:814 writes it as F10.4, truncating at
     1e-4 Ha. Poles are taken from the OpenMolcas output at full precision and
     differenced there.

Usage
-----
    python kallen.py --reference PROJ.rasscf.molden \\
                     --dyson PROJ.dys.molden.SF.1 \\
                     --output PROJ.log \\
                     --pairs 1,2

The NAO construction ships in nao/, so this runs standalone; numpy is the only
requirement. Set HERMES_ROOT to override it with a HERMES checkout.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__version__ = "0.1.0"
HARTREE_EV = 27.211386245988


# ── HERMES bridge ────────────────────────────────────────────────────────────
# The NAO construction is HERMES's, not Kallen's. Importing it rather than
# vendoring a copy keeps one implementation of the basis that both use; a second
# copy would drift and the two would stop being comparable, which is the whole
# point of the interface.

def _import_nao():
    """Import the NAO machinery Kallen evaluates its sums in.

    The vendored copy in nao/ is tried FIRST, so a clone of this repository runs
    with nothing else installed. It is a mirror of HERMES -- see nao/PROVENANCE.md
    for the commit it was taken from, and do not edit it in place.

    A HERMES checkout still wins if HERMES_ROOT is set, which is what a developer
    working on both at once wants; everyone else gets the vendored copy.
    """
    candidates = [os.environ.get("HERMES_ROOT"),
                  Path(__file__).resolve().parent / "nao"]
    tried = []
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        tried.append(str(p))
        if (p / "compute_nao_from_molcas.py").exists():
            sys.path.insert(0, str(p))
            from compute_nao_from_molcas import (compute_nao_from_molcas,
                                                 resolve_fermi_energy)
            from compute_nao_from_turbomole import derive_nao_tiers
            from hermes_grs_from_checkpoint import organize_coefficients
            return (compute_nao_from_molcas, organize_coefficients,
                    derive_nao_tiers, resolve_fermi_energy)
    raise ImportError(
        "the NAO modules are missing. They ship with this repository in nao/; "
        "if that directory is absent, re-clone or set HERMES_ROOT to a HERMES "
        f"checkout. Looked in: {tried}")


# ── the spectral sum ─────────────────────────────────────────────────────────

@dataclass
class Term:
    """One pole's contribution to G_rs, kept so the sum can be read."""
    n: int
    pole_ha: float
    weight: float          # d_r^n d_s^n, signed
    denom_ha: float        # E_F - E_n
    contribution: float    # weight / denom

    @property
    def ip_ev(self) -> float:
        return -self.pole_ha * HARTREE_EV


def grs_terms(coeffs: Dict[int, Dict[str, np.ndarray]],
              atom_r: int, atom_s: int,
              ef_ha: float, poles_ha: Sequence[float]) -> List[Term]:
    """Every pole's contribution to G_rs between two atoms, in state order.

    The numerator is contracted PER NAO TYPE and then summed,
        d_r^n d_s^n = SUM_t  d_{r,t}^n d_{s,t}^n ,
    matching a type against the same type on the other atom. This is the
    atom-pair contraction of the spectral representation, and it is meaningful
    only because the basis is orthonormal -- in a basis with overlap the
    products are not residues of the propagator.

    Not the double sum SUM_t SUM_u, and not (SUM_t d_{r,t})(SUM_u d_{s,u}),
    which is the same thing. Both admit cross terms such as s(r) x p_z(s), and
    those are not symmetry-covariant: on pyrazine they made the four equivalent
    N-C pairs of a D2h molecule come out as -0.110, +0.374, +0.569 and +0.775
    instead of a single value. The single sum is s.s + p.p, invariant under
    rotation because the p components contract as a dot product, and it is what
    HERMES forms (hermes_grs_from_checkpoint.compute_grs_per_spin, which loops
    over orb_type and multiplies like against like).
    """
    if atom_r not in coeffs or atom_s not in coeffs:
        raise KeyError(f"atom {atom_r} or {atom_s} carries no valence NAOs")
    shared = [t for t in coeffs[atom_r] if t in coeffs[atom_s]]
    if not shared:
        raise KeyError(f"atoms {atom_r},{atom_s} share no NAO type")
    cr = {t: np.asarray(coeffs[atom_r][t], float) for t in shared}
    cs = {t: np.asarray(coeffs[atom_s][t], float) for t in shared}
    n_pole = min(min(len(v) for v in cr.values()), len(poles_ha))
    out: List[Term] = []
    for n, e in enumerate(poles_ha):
        if n >= n_pole:
            break
        w = float(sum(cr[t][n] * cs[t][n] for t in shared))
        d = ef_ha - float(e)
        out.append(Term(n=n, pole_ha=float(e), weight=w, denom_ha=d,
                        contribution=(w / d if d != 0.0 else float("inf"))))
    return out


def grs(coeffs, atom_r: int, atom_s: int, ef_ha: float,
        poles_ha: Sequence[float]) -> float:
    """The sum. Diverges if E_F sits on a pole; check nearest_pole() first."""
    return float(sum(t.contribution for t in
                     grs_terms(coeffs, atom_r, atom_s, ef_ha, poles_ha)))


def grs_orbital(coeffs: Dict[int, Dict[str, np.ndarray]],
                atom_r: int, atom_s: int, ef_ha: float,
                eps_ha: Sequence[float],
                occupations: Sequence[float],
                occupied_only: bool = True) -> Tuple[float, int]:
    """Yoshizawa's zeroth-order G_rs, for comparison with the Dyson sum.

        G_rs = SUM_k  C_rk C_sk / (E_F - eps_k)

    This is the same spectral sum with every pole assigned UNIT weight: the
    numerator is a product of coefficients of a NORMALISED orbital, so gamma = 1
    is built in rather than computed.  That assumption is what the Dyson form
    replaces, and it is the only difference between this and grs() -- same atoms,
    same basis, same E_F, same contraction.

    occupied_only defaults to True, and it matters.  The Dyson sum built from
    RASSI removal states contains only (N-1)-electron poles, so an orbital sum
    running over virtuals as well would be comparing a different object: the
    virtuals enter with the opposite sign, because E_F sits between them and the
    occupied levels.  Yoshizawa's own analyses do sum over both halves; matching
    the Dyson side means keeping the occupied half here.

    Returns (G_rs, number of poles summed).
    """
    if atom_r not in coeffs or atom_s not in coeffs:
        raise KeyError(f"atoms {atom_r},{atom_s}: one carries no valence NAOs")
    shared = [t for t in coeffs[atom_r] if t in coeffs[atom_s]]
    if not shared:
        raise KeyError(f"atoms {atom_r},{atom_s} share no NAO type")
    # Per-type contraction, as in grs_terms -- see the note there on why the
    # shell sum is not an option.
    cr = np.sum([np.asarray(coeffs[atom_r][t], float) *
                 np.asarray(coeffs[atom_s][t], float) for t in shared], axis=0)
    cs = np.ones_like(cr)
    eps = np.asarray(eps_ha, float)
    keep = np.asarray(occupations, float) > 1e-8 if occupied_only \
        else np.ones(len(eps), bool)
    d = ef_ha - eps[keep]
    if np.any(np.abs(d) < 1e-10):
        raise ValueError("E_F coincides with an orbital energy; G_rs diverges")
    return float(np.sum(cr[keep] * cs[keep] / d)), int(keep.sum())


def nearest_pole(ef_ha: float, poles_ha: Sequence[float]) -> float:
    """min |E_F - E_n| in eV. Report this beside every coupling: a G_rs
    computed a millielectronvolt from a pole is a number about the prescription
    that placed E_F, not about the molecule."""
    return float(np.min(np.abs(np.asarray(poles_ha, float) - ef_ha))) * HARTREE_EV


def completeness(pole_strengths: Sequence[float]) -> float:
    """SUM_n gamma_n over the poles retained.

    The exact spectral function integrates to 1 per spin orbital. What this
    returns is how much of that a truncated pole list recovers, and it is a
    diagnostic of the CASSCF/RASSI state list rather than of the method: a
    value well below the electron count means cation states were omitted, not
    that the theory failed.
    """
    return float(np.sum(np.asarray(pole_strengths, float)))


# ── driver ───────────────────────────────────────────────────────────────────

def parse_pairs(spec: str) -> List[Tuple[int, int]]:
    out = []
    for chunk in spec.split():
        for part in chunk.split(";"):
            if not part.strip():
                continue
            a, b = part.split(",")
            out.append((int(a), int(b)))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="kallen",
        description="G_rs from OpenMolcas Dyson orbitals and CASPT2 poles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[1] if "Usage" in __doc__ else None)
    ap.add_argument("--reference", required=True,
                    help="$Project.rasscf.molden of the N-electron state")
    ap.add_argument("--dyson", required=True,
                    help="Dyson Molden export. OpenMolcas writes "
                         "$Project.dys.molden.SF.<J>; Molcas 8.6 wrote "
                         "Dyson.SF.molden.<J>")
    ap.add_argument("--output", required=True,
                    help="OpenMolcas output, for the poles at full precision")
    ap.add_argument("--poles", default="caspt2",
                    choices=["caspt2", "mscaspt2", "rasscf", "best"],
                    help="which state energies to difference (default: caspt2)")
    ap.add_argument("--pairs", default=None,
                    help="atom pairs 'r,s' separated by ; or space. "
                         "Default: every pair carrying valence NAOs")
    ap.add_argument("--ef", type=float, default=None,
                    help="Fermi energy in Hartree, set directly")
    ap.add_argument("--anion-output", default=None,
                    help="OpenMolcas output for the anion, giving the electron "
                         "affinity so E_F can be placed at the midpoint")
    ap.add_argument("--anion-energy", type=float, default=None,
                    help="anion total energy in Hartree, if not parsing an output")
    ap.add_argument("--ea-ev", type=float, default=None,
                    help="electron affinity in eV, if known independently")
    ap.add_argument("--hf-reference", default=None,
                    help="SCF Molden ($Project.scf.molden) from the SAME run. "
                         "Adds Yoshizawa's orbital G_rs, in which every pole "
                         "carries unit weight, alongside the Dyson result.")
    ap.add_argument("--hf-all-orbitals", action="store_true",
                    help="sum the orbital G_rs over virtuals too. Off by "
                         "default: the Dyson sum has removal poles only, and "
                         "virtuals enter with the opposite sign.")
    ap.add_argument("--terms", action="store_true",
                    help="print every pole's contribution, not just the sum")
    ap.add_argument("--version", action="version", version=f"kallen {__version__}")
    args = ap.parse_args(argv)

    for f in (args.reference, args.dyson, args.output):
        if not Path(f).exists():
            ap.error(f"not found: {f}")

    (read_molcas, organize_coefficients, derive_nao_tiers,
     resolve_ef) = _import_nao()

    print(f"Kallen {__version__} — G_rs from Dyson orbitals")
    print("=" * 62)
    res = read_molcas(args.reference, args.dyson, args.output,
                      pole_source=args.poles)

    D_nao = res["C_nao"]
    poles = np.asarray(res["energies"], float)
    gam = np.asarray(res["pole_strengths"], float)
    # E_F is not a property of the ionisation spectrum. Placing it needs the
    # electron affinity as well, i.e. an anion calculation -- the midgap of the
    # cation poles alone is the least-bound pole itself, where the denominator
    # vanishes and the dominant term is silently lost. So it is resolved
    # explicitly and never defaulted.
    ip_ha = float(-poles[-1])
    e_neutral = res["state_energies"][args.poles][0]
    try:
        ef, ea_ha, ef_source = resolve_ef(
            ip_ha, poles, e_neutral, args.poles, args.output,
            ef=args.ef, anion_output=args.anion_output,
            anion_energy=args.anion_energy, ea_ev=args.ea_ev, verbose=False)
    except ValueError as exc:
        ap.error(str(exc))
    if ef_source.startswith("PLACEHOLDER"):
        ap.error("cannot place E_F: it needs an electron affinity. Supply one of "
                 "--anion-output, --anion-energy, --ea-ev, or set --ef directly. "
                 "The only parameter-free fallback is the least-bound pole, where "
                 "G_rs diverges.")

    try:
        tiers = derive_nao_tiers(res["ao_list"], res["nao_occupancies"])
    except Exception:
        tiers = None
    coeffs = organize_coefficients(D_nao, res["nao_occupancies"], res["ao_list"],
                                   nao_tiers=tiers, selection="tier")

    print(f"\n  poles retained : {len(poles)}  ({args.poles})")
    print(f"  E_F            : {ef:.6f} Ha   ({ef_source})")
    print(f"  nearest pole   : {nearest_pole(ef, poles):.3f} eV away")
    print(f"  SUM gamma_n    : {completeness(gam):.4f}")
    print(f"  atoms with NAOs: {len(coeffs)}")

    pairs = (parse_pairs(args.pairs) if args.pairs
             else [(a, b) for a in sorted(coeffs) for b in sorted(coeffs) if a < b])

    # Optional orbital-based comparison, from the SCF step of the SAME run.
    hf = None
    if args.hf_reference:
        # The SCF orbitals are expressed in the NAO basis of the CASSCF
        # REFERENCE, not in one rebuilt from the SCF density. Letting each side
        # generate its own NAOs would compare two bases as well as two
        # spectral sums: the two differ by up to 0.02 e in NAO occupancy, small
        # but not nothing, and the point of the comparison is that only the
        # weights differ. Both moldens come from the same geometry, basis and
        # AO ordering, so the reference transform applies unchanged.
        from compute_nao_from_molcas import (read_molden_sections,
                                             parse_molden_mos,
                                             transform_mos_to_nao_basis)
        sec = read_molden_sections(args.hf_reference)
        hf_raw = parse_molden_mos({'mo': sec.get('mo', [])[1:]},
                                  len(res["ao_list"]))
        # parse_molden_mos returns the occupied COUNT, not an occupation
        # vector; for a closed-shell SCF reference the first n_occ are doubly
        # occupied and the rest empty.
        hf_occ = np.zeros(hf_raw["C_ao_alpha"].shape[1])
        hf_occ[: int(hf_raw["n_occ_alpha"])] = 2.0
        C_hf_nao = transform_mos_to_nao_basis(hf_raw["C_ao_alpha"],
                                              res["nao_coeffs_ao"], res["S"])
        hf_res = {"energies": hf_raw["energies_alpha"]}
        hf_coeffs = organize_coefficients(C_hf_nao, res["nao_occupancies"],
                                          res["ao_list"], nao_tiers=tiers,
                                          selection="tier")
        hf = (hf_coeffs, np.asarray(hf_res["energies"], float), hf_occ)
        n_occ = int((hf_occ > 1e-8).sum())
        print(f"  orbital ref    : {args.hf_reference}")
        print(f"                   {n_occ} occupied of {len(hf_occ)} orbitals, "
              f"summing {'all' if args.hf_all_orbitals else 'occupied only'}")

    print(f"\n{'pair':>10} {'G_rs (1/Ha)':>14} {'largest term':>14} {'from IP (eV)':>13}")
    print("-" * 56)
    for a, b in pairs:
        ts = grs_terms(coeffs, a, b, ef, poles)
        g = sum(t.contribution for t in ts)
        top = max(ts, key=lambda t: abs(t.contribution)) if ts else None
        print(f"{a:4d}-{b:<5d} {g:>14.6f} "
              f"{(top.contribution if top else float('nan')):>14.6f} "
              f"{(top.ip_ev if top else float('nan')):>13.3f}")
        if hf is not None:
            hc, heps, hocc = hf
            g_orb, npoles = grs_orbital(hc, a, b, ef, heps, hocc,
                                        occupied_only=not args.hf_all_orbitals)
            print(f"{'':>6}orbital (gamma=1, {npoles} poles): {g_orb:>12.6f}"
                  f"   Dyson/orbital = "
                  f"{(g / g_orb if g_orb else float('nan')):>7.3f}")
        if args.terms:
            print(f"{'':>6}{'n':>4} {'IP (eV)':>10} {'gamma':>9} "
                  f"{'d_r d_s':>12} {'1/(E_F-E)':>12} {'term':>12}")
            for t in ts:
                gm = gam[t.n] if t.n < len(gam) else float("nan")
                print(f"{'':>6}{t.n:>4} {t.ip_ev:>10.3f} {gm:>9.4f} "
                      f"{t.weight:>12.6f} {1.0 / t.denom_ha:>12.4f} "
                      f"{t.contribution:>12.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
