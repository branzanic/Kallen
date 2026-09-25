#!/usr/bin/env python3
"""G_rs over every atom pair, in both frames, for the two worked examples.

Frame 1 (Yoshizawa): orbital sum over the complete occupied manifold of the
                     &SCF step, every pole at unit weight.
Frame 2 (Dyson):     the RASSI/CASPT2 sum, weights computed.

Both from the SAME OpenMolcas run, same NAO basis, same E_F, same contraction,
so the only difference between the two numbers for a given pair is the weights.
Hydrogens are carried in the NAO basis but reported separately: the heavy-atom
pairs are what a coupling map is read from.
"""
import sys, io, contextlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    from analyse import load, HA_EV
    from kallen import _import_nao, grs_orbital
    _read, _organize, _tiers, _ = _import_nao()

# E_F from the paper: mu = -(IP_1 + EA)/2, EA the vertical attachment energy.
EF = {"thiophene": -0.136030, "pyrazine": -0.165365}
LABEL = {"thiophene": ["S1", "C2", "C3", "C4", "C5", "H2", "H3", "H4", "H5"],
         "pyrazine":  ["N1", "C2", "C3", "N4", "C5", "C6",
                       "H2", "H3", "H5", "H6"]}
NHEAVY = {"thiophene": 5, "pyrazine": 6}


def frames(mol):
    d = HERE / mol
    with contextlib.redirect_stdout(_buf):
        res, cD = load(d / f"{mol}_neutral.rasscf.molden",
                       d / f"{mol}_dyson.dys.molden.SF.1",
                       d / f"{mol}_dyson.log")
        # SCF orbitals expressed in the CASSCF reference's NAO basis, not in
        # one rebuilt from the SCF density -- otherwise the comparison spans
        # two bases as well as two spectral sums.
        from compute_nao_from_molcas import (read_molden_sections,
                                             parse_molden_mos,
                                             transform_mos_to_nao_basis)
        sec = read_molden_sections(str(d / f"{mol}_dyson.scf.molden"))
        raw = parse_molden_mos({'mo': sec.get('mo', [])[1:]}, len(res["ao_list"]))
        C_hf = transform_mos_to_nao_basis(raw["C_ao_alpha"],
                                          res["nao_coeffs_ao"], res["S"])
        occ = np.zeros(C_hf.shape[1]); occ[: int(raw["n_occ_alpha"])] = 2.0
        cH = _organize(C_hf, res["nao_occupancies"], res["ao_list"],
                       nao_tiers=_tiers(res["ao_list"], res["nao_occupancies"]),
                       selection="tier")
    return (np.asarray(res["energies"], float), cD,
            np.asarray(raw["energies_alpha"], float), occ, cH)


def gsum(c, r, s, poles, ef):
    """Per-type contraction: SUM_t d_{r,t} d_{s,t}. See kallen.grs_terms."""
    shared = [t for t in c[r] if t in c[s]]
    num = np.sum([np.asarray(c[r][t], float) * np.asarray(c[s][t], float)
                  for t in shared], axis=0)
    return float(np.sum(num[:len(poles)] / (ef - np.asarray(poles, float))))


def main():
    for mol in ("thiophene", "pyrazine"):
        pD, cD, eps, occ, cH = frames(mol)
        ef, lab, nh = EF[mol], LABEL[mol], NHEAVY[mol]
        print(f"\n=== {mol}   E_F = {ef*HA_EV:.2f} eV ===")
        print(f"  {'pair':<10}{'Yoshizawa':>12}{'Dyson':>11}{'Dyson/Yosh':>12}"
              f"{'|d| (A)':>10}")
        print("  " + "-" * 55)
        rows = []
        for i in range(1, nh + 1):
            for j in range(i + 1, nh + 1):
                gy, _ = grs_orbital(cH, i, j, ef, eps, occ, occupied_only=True)
                gd = gsum(cD, i, j, pD, ef)
                rows.append((f"{lab[i-1]}-{lab[j-1]}", gy, gd,
                             gd / gy if gy else float("nan")))
        for name, gy, gd, rt in rows:
            print(f"  {name:<10}{gy:>12.4f}{gd:>11.4f}{rt:>12.3f}")
        r = np.array([x[3] for x in rows])
        print(f"  ratio spread: {r.min():.3f} to {r.max():.3f}  "
              f"(median {np.median(r):.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
