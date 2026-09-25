#!/usr/bin/env python3
"""Worked-example analysis for the Kallen contribution.

kallen.py reports G_rs at one E_F.  For the paper we also want the pole-strength
spectrum next to the published ADC(3)/SAC-CI values, and G_rs as a function of
E_F, because E_F is an input to the method and not an output of it.

Kept out of kallen.py on purpose: kallen.py is strictly Dyson -> G_rs.

Note on E_F.  kallen.py builds the midgap from ``poles[-1]`` -- the LAST pole in
JobIph state order, not the first IP.  That is arbitrary (in the O2 example the
state order is interleaved, so 'last' is neither the first nor a chosen pole) and
it drifts as cation roots are added, which is the summation-window problem in
another guise.  Here mu is built from IP_1, the lowest ionisation potential,
which is what the fundamental gap actually means.  Both are printed so the
difference is visible rather than assumed.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kallen import grs_terms, completeness, _import_nao   # noqa: E402

HA_EV = 27.211386245988

# Above this pole strength a line is treated as a quasiparticle (main) line and
# below it as a shake-up satellite.  Thiophene and pyrazine both leave a wide
# empty band around 0.3-0.5, so nothing sits near the cut.
QP_CUT = 0.45


def load(reference, dyson, output, poles="best"):
    read_molcas, organize_coefficients, derive_nao_tiers, _ = _import_nao()
    res = read_molcas(reference, dyson, output, pole_source=poles)
    try:
        tiers = derive_nao_tiers(res["ao_list"], res["nao_occupancies"])
    except Exception:
        tiers = None
    coeffs = organize_coefficients(res["C_nao"], res["nao_occupancies"],
                                   res["ao_list"], nao_tiers=tiers,
                                   selection="tier")
    return res, coeffs


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--dyson", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--poles", default="best")
    ap.add_argument("--pair", required=True, help="two 1-based atom indices, 'r,s'")
    ap.add_argument("--label", default="", help="name for the pair, e.g. 'S1-C2'")
    ap.add_argument("--ea-ev", type=float, required=True,
                    help="vertical electron affinity in eV (negative for a resonance)")
    ap.add_argument("--fig", default=None, help="write a figure here")
    ap.add_argument("--compose", action="store_true",
                    help="print the unsummed per-shell NAO coefficients on both "
                         "atoms -- the only way to tell a node from a cancellation")
    a = ap.parse_args(argv)

    r, s_ = (int(x) for x in a.pair.split(","))
    res, coeffs = load(a.reference, a.dyson, a.output, a.poles)
    poles = np.asarray(res["energies"], float)          # stored as -IP
    gam = np.asarray(res["pole_strengths"], float)
    ips = -poles * HA_EV                                # positive IPs, eV

    ip1 = ips.min()
    mu_first = -(ip1 + a.ea_ev) / 2.0 / HA_EV
    mu_last = -((-poles[-1]) * HA_EV + a.ea_ev) / 2.0 / HA_EV

    name = a.label or f"{r}-{s_}"
    print(f"\n=== {name} ===")
    print(f"  IP_1 = {ip1:.3f} eV   IP(last state) = {(-poles[-1])*HA_EV:.3f} eV"
          f"   EA = {a.ea_ev:+.3f} eV")
    print(f"  E_F from IP_1        : {mu_first*HA_EV:+8.3f} eV  ({mu_first:+.6f} Ha)")
    print(f"  E_F from last pole   : {mu_last*HA_EV:+8.3f} eV  ({mu_last:+.6f} Ha)"
          "   <- kallen.py default")
    print(f"  SUM gamma_n          : {completeness(gam):.4f} over {len(gam)} poles")

    print(f"\n  {'n':>3} {'IP (eV)':>9} {'gamma':>8} {'d_r d_s':>12}"
          f" {'term @mu(IP1)':>14} {'term @mu(last)':>15}")
    print("  " + "-" * 66)
    t_first = grs_terms(coeffs, r, s_, mu_first, poles)
    t_last = grs_terms(coeffs, r, s_, mu_last, poles)
    for tf, tl in zip(t_first, t_last):
        print(f"  {tf.n:>3} {tf.ip_ev:>9.3f} {gam[tf.n]:>8.4f} {tf.weight:>12.6f}"
              f" {tf.contribution:>14.6f} {tl.contribution:>15.6f}")
    g_first = sum(t.contribution for t in t_first)
    g_last = sum(t.contribution for t in t_last)
    print("  " + "-" * 66)
    print(f"  G_{name} = {g_first:+.6f} 1/Ha  at E_F from IP_1")
    print(f"  G_{name} = {g_last:+.6f} 1/Ha  at E_F from the last pole"
          f"   (ratio {g_last/g_first if g_first else float('nan'):+.3f})")

    # Koopmans counterfactual.  An orbital-based G has one pole per occupied
    # orbital, each carrying FULL weight, and no shake-up poles at all.  So the
    # comparison keeps only the quasiparticle lines and sets their gamma to 1;
    # the satellites are not rescaled but dropped, because the orbital picture
    # never produced them.  (Rescaling them instead would divide a correlation
    # artefact by a near-zero gamma and report a meaningless number.)
    qp = [t for t in t_first if gam[t.n] >= QP_CUT]
    sat = [t for t in t_first if gam[t.n] < QP_CUT]
    g_koop = float(sum(t.contribution / gam[t.n] for t in qp))
    print(f"\n  Koopmans counterfactual (gamma -> 1 on the {len(qp)} quasiparticle"
          f" lines, gamma >= {QP_CUT}; {len(sat)} satellites dropped,"
          f" carrying {sum(gam[t.n] for t in sat):.3f} of the weight):")
    print(f"  G_{name} = {g_koop:+.6f} 1/Ha  "
          f"({100*(g_koop-g_first)/abs(g_first):+.1f}% vs the Dyson-weighted value)")
    worst = max(qp, key=lambda t: abs(t.contribution / gam[t.n] - t.contribution))
    print(f"  largest single distortion: pole at {worst.ip_ev:.3f} eV, "
          f"gamma = {gam[worst.n]:.4f}, term {worst.contribution:+.6f} -> "
          f"{worst.contribution/gam[worst.n]:+.6f}")

    if a.compose:
        # Per-shell NAO coefficients on the two atoms, unsummed.
        #
        # Read this before believing any statement about a node.  G_rs uses the
        # SIGNED shell sum on each atom, and for two symmetry-related atoms that
        # sum has different magnitudes at the two sites: under the operation
        # relating them an s NAO keeps its sign while a p NAO flips, so one site
        # gets s+p and the other s-p.  A pair that looks lopsided in the summed
        # coefficient can therefore be exactly symmetric, and the way to tell is
        # sqrt(sum of squares), which is operation-invariant.  A true node shows
        # up as EVERY shell vanishing at both sites, not as a cancellation
        # between shells.
        print(f"\n  per-shell NAO coefficients ({name}), unsummed:")
        for n in range(len(gam)):
            print(f"    pole {n}  IP = {ips[n]:7.3f} eV   gamma = {gam[n]:.4f}")
            for at, tag in ((r, f"atom {r}"), (s_, f"atom {s_}")):
                v = coeffs[at]
                parts = {k: float(x[n]) for k, x in v.items() if abs(float(x[n])) > 1e-6}
                tot = sum(float(x[n]) for x in v.values())
                rms = float(np.sqrt(sum(float(x[n]) ** 2 for x in v.values())))
                print(f"      {tag:>8}: " +
                      ", ".join(f"{k}={x:+.4f}" for k, x in sorted(parts.items()))
                      + f"   | signed sum {tot:+.4f} | norm {rms:.4f}")

    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        # Type 42, not matplotlib's default Type 3. Type 3 embeds text as
        # bitmaps: thin strokes vanish at print scale, extraction breaks, and
        # ACS rejects them. The figure is vector either way; this makes its
        # TEXT vector too.
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        import matplotlib.pyplot as plt
        grid = -np.linspace(ips.min() - 0.6, ips.min() - 6.0, 900)[::-1] / HA_EV
        curve = [sum(t.contribution for t in grs_terms(coeffs, r, s_, e, poles))
                 for e in grid]
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6.2, 6.0),
                                       gridspec_kw={"height_ratios": [1, 1.25]})
        is_qp = gam >= QP_CUT
        ax0.vlines(ips[is_qp], 0, gam[is_qp], lw=2.4, color="#1f3b73")
        ax0.plot(ips[is_qp], gam[is_qp], "o", ms=4.5, color="#1f3b73",
                 label="quasiparticle lines")
        ax0.vlines(ips[~is_qp], 0, gam[~is_qp], lw=2.4, color="#b06a00")
        ax0.plot(ips[~is_qp], gam[~is_qp], "s", ms=4.0, color="#b06a00",
                 label="shake-up satellites")
        ax0.axhline(1.0, ls=":", lw=1.0, color="grey")
        ax0.annotate("Koopmans limit  $\\gamma=1$", xy=(ips.max(), 1.0),
                     xytext=(0, 3), textcoords="offset points",
                     ha="right", va="bottom", fontsize=7.5, color="grey")
        # Name the weakest main line: it is the one the orbital picture gets wrong.
        w = int(np.argmin(np.where(is_qp, gam, 2.0)))
        ax0.annotate(f"$\\gamma$ = {gam[w]:.2f}", xy=(ips[w], gam[w]),
                     xytext=(0, 6), textcoords="offset points", ha="center",
                     fontsize=8, color="#8c1d18")
        ax0.set_ylim(0, 1.14)
        ax0.set_xlabel("ionisation potential  IP$_n$  (eV)")
        ax0.set_ylabel(r"pole strength  $\gamma_n=\|d^n\|^2$")
        ax0.legend(fontsize=7.5, frameon=False, ncol=2, loc="lower center",
                   bbox_to_anchor=(0.5, 1.01))
        ax0.set_title(f"{name}: Dyson spectral weights", fontsize=10, pad=22)

        # The same poles with Koopmans weights: quasiparticle lines only, gamma -> 1.
        qp_n = [t.n for t in t_first if gam[t.n] >= QP_CUT]
        curve_k = [sum(t.contribution / gam[t.n]
                       for t in grs_terms(coeffs, r, s_, e, poles) if t.n in qp_n)
                   for e in grid]
        ax1.plot(grid * HA_EV, curve_k, lw=1.6, ls="--", color="#b06a00",
                 label=r"Koopmans weights ($\gamma_n\equiv1$, main lines only)")
        ax1.plot(grid * HA_EV, curve, lw=2.0, color="#8c1d18",
                 label=r"Dyson weights (exact $\gamma_n$)")
        ax1.axvline(mu_first * HA_EV, ls=":", lw=1.2, color="#1f3b73",
                    label=r"$\mu=-(\mathrm{IP}_1+\mathrm{EA})/2$")
        ax1.axhline(0, lw=0.7, color="grey")
        ax1.set_xlabel("$E_F$ (eV)")
        ax1.set_ylabel(r"$G_{rs}(E_F)$  (Ha$^{-1}$)")
        ax1.legend(fontsize=7.5, frameon=False)
        fig.tight_layout()
        fig.savefig(a.fig, dpi=200)
        print(f"\n  figure -> {a.fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
