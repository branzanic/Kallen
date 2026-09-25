#!/usr/bin/env python3
"""
Compute Natural Atomic Orbitals (NAOs) and Dyson orbitals from Molcas output.

Fourth HERMES framework reader, alongside Gaussian (.fchk), ORCA (.molden) and
Turbomole. Unlike those three, this one does not supply single-determinant
orbitals and orbital energies. It supplies the *exact* Kallen-Lehmann content of
the Green's function HERMES already computes:

    G_rs = SUM_n  d_r^n d_s^n / (E_F - IP_n)

with Dyson orbitals d^n as the spectral weights and exact ionisation potentials
as the poles. Koopmans' theorem is precisely the approximation that collapses
this into the canonical-MO form the other three readers produce.

INPUTS — three, not one:

  1. reference Molden  ($Project.rasscf.molden from the N-electron RASSCF)
     Natural orbitals of the N-electron state. Defines the NAO basis, and is
     the ONLY source of the overlap matrix: S = inv(C @ C.T) needs a square C,
     and the Dyson file is n_ao x n_dyson (7 x 28 for O2/def2-SVP), not square.

  2. Dyson Molden     (Dyson.SF.molden.<J> from RASSI DYSOn + DYSExport)
     The Dyson orbitals themselves, in the SAME AO basis and ordering.

  3. Molcas output    (for the poles — see below)

THREE TRAPS, all of which produce plausible-looking wrong numbers:

  (a) The Dyson coefficients are UN-NORMALISED. Their squared norm IS the pole
      strength (src/rassi/mkdysorb.f:117, DYSAMP = SQRT(sum OVLP^2)). Do not
      normalise them on read: SUM_n d_r d_s /(E_F - IP) picks up the spectral
      weight automatically, and normalising silently deletes every weight.

  (b) The Molden 'Ene=' field carries CASSCF poles, not CASPT2 ones. Verified on
      O2/def2-SVP CAS(12,8): CASSCF gives IP = 10.841 eV (-1.229 eV vs the
      experimental 12.07), CASPT2 gives 12.121 eV (+0.051). Poles are therefore
      parsed from the Molcas output, not from the Molden file.

  (c) 'Ene=' is additionally written as F10.4 (src/property_util/molden_dysorb.f:814),
      truncating at 1e-4 Ha. Two independent reasons to ignore that field.

FRACTIONAL OCCUPATIONS: CASSCF natural orbitals do not have integer occupations.
The single-determinant readers build P = C_occ @ C_occ.T with a hard 'occ >= 1'
cut, which here would count nine O2 orbitals at equal weight (tr(SP) = 9 instead
of 16) and discard the correlated tail entirely. This module builds the
occupation-weighted density P = C @ diag(n) @ C.T instead.

A. M. V. Branzanic - 2026
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_hermes_dir = Path(__file__).resolve().parent
if str(_hermes_dir) not in sys.path:
    sys.path.insert(0, str(_hermes_dir))

from compute_nao_from_turbomole import (
    AOBasisFunction,
    compute_nao_from_density,
    transform_mos_to_nao_basis,
)
# Molcas writes standard Molden with [5D]/[7F]/[9G], the same convention ORCA
# writes, so the ORCA section/GTO/MO parsers apply unchanged. Reuse rather than
# duplicate — a second copy of the m_l ordering table is exactly how the
# Gaussian and Turbomole d-ordering bugs survived for months.
from compute_nao_from_orca import (
    BOHR_TO_ANG,
    _parse_gto_lines,
    parse_molden_mos,
)

HA_TO_EV = 27.211386245988


# ── Molden section reading ───────────────────────────────────────────────────

def read_molden_sections(molden_file: str) -> Dict[str, List[str]]:
    """Split a Molden file into named sections, keeping the tag line first.

    The tag line is retained because units live on it ('[ATOMS] (AU)'), exactly
    as in the ORCA path.
    """
    sections: Dict[str, List[str]] = {}
    current = None
    with open(molden_file) as fh:
        for line in fh:
            stripped = line.rstrip('\n')
            m = re.match(r'^\s*\[([^\]]+)\]', stripped)
            if m:
                current = m.group(1).strip().lower()
                sections.setdefault(current, [])
                sections[current].insert(0, stripped)
            elif current is not None:
                sections[current].append(stripped)
    return sections


def parse_molcas_atoms(sections: Dict[str, List[str]]) -> Tuple[List[str], np.ndarray]:
    """Parse [ATOMS]; Molcas writes '(AU)' on the tag line."""
    lines = sections.get('atoms', [])
    if not lines:
        raise ValueError("[ATOMS] section not found")
    unit_ang = any('ang' in ln.lower() for ln in lines[:2])

    elements: List[str] = []
    coords: List[List[float]] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            # Molcas labels atoms '<element><index>' -- O1, O2, Fe1. The index
            # must be stripped or every atom is looked up as an unknown element,
            # classified Rydberg, and dropped from G_rs.
            elements.append(re.sub(r'\d+$', '', parts[0]).capitalize())
            coords.append([float(parts[3]), float(parts[4]), float(parts[5])])
        except (ValueError, IndexError):
            continue

    if not elements:
        raise ValueError("No atoms parsed from [ATOMS]")

    arr = np.array(coords)
    if not unit_ang:
        arr = arr * BOHR_TO_ANG
    return elements, arr


def parse_molden_occupations(molden_file: str) -> np.ndarray:
    """Occupation numbers in file order, as written (total, both spins)."""
    occs: List[float] = []
    in_mo = False
    with open(molden_file) as fh:
        for line in fh:
            low = line.strip().lower()
            if low.startswith('[mo]'):
                in_mo = True
                continue
            if in_mo and low.startswith('occup='):
                occs.append(float(line.strip()[6:].strip()))
    return np.array(occs)


def parse_molden_energies(molden_file: str) -> np.ndarray:
    """The 'Ene=' field in file order, Hartree.

    NEVER used as a pole value: it is CASSCF-level and truncated to F10.4. It is
    used only as an IDENTIFIER, to work out which final state each Dyson orbital
    belongs to (see assign_poles).
    """
    enes: List[float] = []
    in_mo = False
    with open(molden_file) as fh:
        for line in fh:
            low = line.strip().lower()
            if low.startswith('[mo]'):
                in_mo = True
                continue
            if in_mo and low.startswith('ene='):
                enes.append(float(line.strip()[4:].strip()))
    return np.array(enes)


def assign_poles(dyson_ene_ha: np.ndarray,
                 rasscf: List[float],
                 corrected: List[float],
                 tol_ev: float = 0.02) -> Tuple[np.ndarray, List[int]]:
    """Map each Dyson orbital to its final state, then to a corrected pole.

    Why this is not a slice. The Dyson orbitals appear in Dyson.SF.molden in
    ISTATE order, but the RASSI state list concatenates several JobIphs, so with
    (neutral, 6 doublets, 4 quartets) the exported orbitals run over states
    2,3,4,5,8,9,10 — the quartets interleave. Taking corrected[1:1+n_dyson]
    would therefore attach the wrong pole to the right orbital and still produce
    entirely plausible numbers. Verified on O2: the RASSI table lists states in
    intensity order 2,3,8,9,4,5,10 while the Molden file is in state order.

    The identification is done through the one thing the Molden 'Ene=' field is
    good for: it equals E_CASSCF(ISTATE) - E_CASSCF(1) for the state that orbital
    belongs to. Matching it against the RASSCF energies pins ISTATE exactly, with
    no assumption about ordering; the pole value then comes from the *corrected*
    (CASPT2) list at that index.

    Args:
        dyson_ene_ha: 'Ene=' per Dyson orbital, Hartree.
        rasscf:       RASSCF total energies, RASSI state order (state 1 first).
        corrected:    CASPT2/MS-CASPT2 totals in the SAME order.
        tol_ev:       match tolerance; 'Ene=' is F10.4 Ha ~ 0.003 eV, so 0.02 eV
                      is loose enough for the truncation and tight enough to
                      catch a genuine mis-ordering.

    Returns (poles_ha, state_indices) — poles as E(state) - E(neutral) from the
    corrected list, and the 1-based RASSI state index each orbital matched.
    """
    if not rasscf:
        raise ValueError("no RASSCF energies — cannot identify Dyson final states")
    if len(corrected) != len(rasscf):
        raise ValueError(
            f"{len(corrected)} corrected energies vs {len(rasscf)} RASSCF energies; "
            "they must be in the same state order for the mapping to be valid")

    cas_be = np.array([(e - rasscf[0]) for e in rasscf])          # Ha, index 0 = 0.0
    poles: List[float] = []
    states: List[int] = []
    used: set = set()

    # Matched WITHOUT REPLACEMENT. Degenerate final states (Pi states of a linear
    # molecule, e states of a symmetric top) have identical energies AND identical
    # Dyson intensities, so no scalar can tell them apart — plain argmin would map
    # every member of a multiplet onto its first element. Within a degenerate
    # multiplet the labelling is arbitrary anyway: any unitary mixing of the
    # degenerate subspace is an equally valid basis, and the pole value is common
    # to the whole multiplet. Consuming each state once keeps the labels distinct
    # and the physics unchanged.
    for i, ene in enumerate(dyson_ene_ha):
        order = np.argsort(np.abs(cas_be - ene))
        j = None
        for cand in order:
            if int(cand) not in used:
                j = int(cand)
                break
        if j is None:
            raise ValueError(f"ran out of states to assign Dyson orbital {i + 1}")

        gap_ev = abs(cas_be[j] - ene) * HA_TO_EV
        if gap_ev > tol_ev:
            raise ValueError(
                f"Dyson orbital {i + 1} (Ene= {ene:.4f} Ha = {ene * HA_TO_EV:.3f} eV) "
                f"matches no unused RASSCF state within {tol_ev} eV; nearest free is "
                f"state {j + 1} at {gap_ev:.3f} eV. The state lists are inconsistent — "
                "check that the RASSCF/CASPT2 modules ran in JobIph order.")
        used.add(j)
        poles.append(corrected[j] - corrected[0])
        states.append(j + 1)

    return np.array(poles), states


def parse_basis_info(output_file: str) -> Tuple[Optional[str], Optional[int]]:
    """(basis label, number of basis functions) from a Molcas output.

    Takes the LAST occurrence of each, so a file containing several &GATEWAY
    blocks reports the basis of its final calculation. Used to refuse pairing an
    anion with a neutral from a different basis — a mistake that produces a
    perfectly plausible EA (+4.2 eV instead of -1.6 eV for O2) and cannot be
    caught by any magnitude test, since real valence affinities reach 3.6 eV.
    """
    label = None
    nbas = None
    re_label = re.compile(r'Basis set label:\s*(\S+)')
    re_nbas = re.compile(r'Number of basis functions\s+(\d+)')
    with open(output_file, errors='replace') as fh:
        for line in fh:
            m = re_label.search(line)
            if m:
                label = m.group(1).strip('. ')
            m = re_nbas.search(line)
            if m:
                nbas = int(m.group(1))
    return label, nbas


def chemical_potential(ip_ha: float, ea_ha: float) -> float:
    """mu = -(IP + EA)/2 — the MIDPOINT of the fundamental gap, in Hartree.

    This is the multireference analogue of HERMES's (eps_HOMO + eps_LUMO)/2, and
    the only E_F on this path that comes out of the calculation rather than being
    asserted.

    Do not confuse it with the gap WIDTH. IP - EA is the fundamental gap (11.62 eV
    for O2); EA - IP is its negative. Neither is a chemical potential, and using
    one as E_F lands you on top of the leading pole: for O2, EA - IP = -0.4271 Ha
    sits 0.018 Ha from the X 2Pi_g pole and blows G_rs up by an order of magnitude.

    Sign convention: IP = E(N-1) - E(N) > 0 and EA = E(N) - E(N+1), positive when
    the anion is bound. A negative EA (unbound anion) is still usable here — the
    average halves its error — but must be reported, not hidden.
    """
    return -(ip_ha + ea_ha) / 2.0


def resolve_fermi_energy(ip_ha: float,
                         eps: np.ndarray,
                         e_neutral: float,
                         poles_key: str,
                         reference_output: str,
                         ef: Optional[float] = None,
                         anion_output: Optional[str] = None,
                         anion_energy: Optional[float] = None,
                         ea_ev: Optional[float] = None,
                         verbose: bool = True) -> Tuple[float, Optional[float], str]:
    """Resolve E_F for a Dyson spectrum, preferring computation over assertion.

    Shared by the checkpoint writer and the one-shot driver so that the two cannot
    drift apart. Precedence, most to least principled:

      anion_output   -> EA from that run's energies, with a basis cross-check
      anion_energy   -> EA from a bare total energy, no basis check possible
      ea_ev          -> EA supplied directly
      ef             -> asserted outright
      (none)         -> placeholder on the least-bound pole, with a loud warning

    Note the ordering: an explicitly supplied ``ef`` loses to a computed one. This is
    deliberate. Passing both is a contradiction, and the computed value is the one with
    provenance; asserting E_F is what this path exists to avoid.

    Returns (E_F in Hartree, EA in Hartree or None, a description of the source).
    """
    say = print if verbose else (lambda *a, **k: None)
    ea_ha: Optional[float] = None
    source: str

    if anion_output is not None:
        a_lab, a_nb = parse_basis_info(anion_output)
        n_lab, n_nb = parse_basis_info(reference_output)
        say(f"  basis check: neutral {n_lab} ({n_nb} fn) vs anion {a_lab} ({a_nb} fn)")
        if (a_nb is not None and n_nb is not None and a_nb != n_nb) or \
           (a_lab and n_lab and a_lab != n_lab):
            raise ValueError(
                f"anion ({a_lab}, {a_nb} fn) and neutral ({n_lab}, {n_nb} fn) are in "
                "different basis sets; EA would be meaningless. &GATEWAY is shared per "
                "run, so the anion must come from a run using the same basis.")
        a_e = parse_state_energies(anion_output).get(poles_key, [])
        if not a_e:
            raise ValueError(f"no '{poles_key}' energies in {anion_output}")
        say(f"  taking the LAST {poles_key} energy: {a_e[-1]:.8f} Ha "
            f"({len(a_e)} present -- verify this is the anion)")
        ea_ha = e_neutral - a_e[-1]
        source = 'computed: -(IP+EA)/2, EA from --anion-output'
    elif anion_energy is not None:
        ea_ha = e_neutral - float(anion_energy)
        source = 'computed: -(IP+EA)/2, EA from --anion-energy'
        say("  NOTE: --anion-energy carries no basis information, so no cross-basis "
            "check is possible. Prefer --anion-output.")
    elif ea_ev is not None:
        ea_ha = float(ea_ev) / HA_TO_EV
        source = 'computed: -(IP+EA)/2, EA from --ea-ev'
    elif ef is not None:
        return float(ef), None, 'asserted via --ef'
    else:
        return float(eps[-1]), None, 'PLACEHOLDER: -IP_min, no anion supplied'

    if abs(ea_ha * HA_TO_EV) > 6.0:
        raise ValueError(
            f"implied EA = {ea_ha * HA_TO_EV:.3f} eV is not physical for a valence "
            "anion; the anion and neutral are probably from different calculations.")
    if ef is not None:
        say(f"  NOTE: --ef given but an anion is available; using the computed "
            f"mu = {chemical_potential(ip_ha, ea_ha):.8f} Ha and ignoring "
            f"--ef {ef}.")
    return chemical_potential(ip_ha, ea_ha), ea_ha, source


def degenerate_groups(poles_ha: np.ndarray, tol_ha: float = 1e-5) -> List[List[int]]:
    """Group orbital indices whose poles coincide — the degenerate multiplets.

    Reported rather than warned about: degeneracy is physics here (O2+ X 2Pi_g is
    a doublet pair), and only the group total is basis-independent.
    """
    groups: List[List[int]] = []
    for i, p in enumerate(poles_ha):
        for g in groups:
            if abs(poles_ha[g[0]] - p) < tol_ha:
                g.append(i)
                break
        else:
            groups.append([i])
    return groups


# ── Pole parsing — the part that must NOT come from the Molden file ─────────

def parse_state_energies(output_file: str) -> Dict[str, List[float]]:
    """Parse RASSCF and CASPT2 total energies from a Molcas output, in order.

    Returns {'rasscf': [...], 'caspt2': [...], 'mscaspt2': [...], 'best': [...]}
    in Hartree, at full printed precision. These are the poles; the Molden 'Ene='
    field is both CASSCF-level and truncated to 1e-4 Ha, so it is never used.

    'best' is assembled per &CASPT2 block rather than by row type: each block
    contributes its multistate rows if it printed any, else its single-state rows.
    A deck whose blocks are treated differently -- single-state on the neutral,
    XMS on the cation manifold -- therefore still yields one pool in state order.
    That deck is not exotic; it is forced. IFMSCOUP is switched off for a lone
    root (src/caspt2/caspt2.f:426, `IF(NLYROOT.NE.0) IFMSCOUP=.FALSE.`), so a
    one-state neutral CANNOT emit an MS/XMS row however it is asked to, and a
    flat 'mscaspt2' pool silently loses its index 0 -- the neutral, i.e. the
    reference every pole is measured from.
    """
    rasscf: List[float] = []
    caspt2: List[float] = []
    mscaspt2: List[float] = []
    best: List[float] = []
    blk_pt2: List[float] = []   # single-state rows of the current &CASPT2 block
    blk_ms: List[float] = []    # multistate rows of the current &CASPT2 block

    def close_block() -> None:
        """Flush the current block into 'best': multistate if it has any."""
        if blk_ms or blk_pt2:
            best.extend(blk_ms if blk_ms else blk_pt2)
        blk_ms.clear()
        blk_pt2.clear()

    re_rasscf = re.compile(r'RASSCF root number\s+\d+\s+Total energy:\s*(-?\d+\.\d+)')
    re_pt2 = re.compile(r'::\s*CASPT2 Root\s+\d+\s+Total energy:\s*(-?\d+\.\d+)')
    # X?MS- so that XMS-CASPT2 (extended multi-state) rows are collected too; a run
    # mixing single-state and XMS blocks then assembles as MS neutral + XMS cation
    # + MS anion, which is the ladder the Dyson binding energies refer to.
    re_ms = re.compile(r'::\s*X?MS-CASPT2 Root\s+\d+\s+Total energy:\s*(-?\d+\.\d+)')

    re_banner = re.compile(r'^\s*&CASPT2\b')

    with open(output_file, errors='replace') as fh:
        for line in fh:
            if re_banner.search(line):
                close_block()          # a new block begins; flush the previous one
                continue
            m = re_rasscf.search(line)
            if m:
                rasscf.append(float(m.group(1)))
                continue
            m = re_ms.search(line)
            if m:
                v = float(m.group(1))
                mscaspt2.append(v)
                blk_ms.append(v)
                continue
            m = re_pt2.search(line)
            if m:
                v = float(m.group(1))
                caspt2.append(v)
                blk_pt2.append(v)
    close_block()                      # the last block has no banner after it

    return {'rasscf': rasscf, 'caspt2': caspt2, 'mscaspt2': mscaspt2,
            'best': best}


def parse_dyson_amplitudes(output_file: str) -> List[Tuple[int, int, float, float]]:
    """Parse the RASSI 'Dyson amplitudes (spin-free states)' table.

    Returns [(from_state, to_state, BE_eV, intensity), ...] in printed order,
    which is the same order the Dyson orbitals appear in Dyson.SF.molden.<J>
    ONLY if the file is trusted to match — it is not. The orbital order is taken
    from the Molden file itself; this table is parsed for cross-checking the
    pole assignment and for the intensities.
    """
    rows: List[Tuple[int, int, float, float]] = []
    in_table = False
    with open(output_file, errors='replace') as fh:
        for line in fh:
            if 'Dyson amplitudes' in line:
                in_table = True
                continue
            if in_table:
                if 'Special properties' in line or line.strip().startswith('****'):
                    in_table = False
                    continue
                parts = line.split()
                if len(parts) == 4:
                    try:
                        rows.append((int(parts[0]), int(parts[1]),
                                     float(parts[2]), float(parts[3])))
                    except ValueError:
                        pass
    return rows


# ── Main entry point ────────────────────────────────────────────────────────

def compute_nao_from_molcas(reference_molden: str,
                            dyson_molden: Optional[str] = None,
                            molcas_output: Optional[str] = None,
                            pole_source: str = 'caspt2',
                            n_electrons_neutral: Optional[int] = None) -> Dict:
    """Build the NAO basis from a CASSCF reference and express Dyson orbitals in it.

    Args:
        reference_molden: $Project.rasscf.molden of the N-electron state.
        dyson_molden:     Dyson.SF.molden.<J>. If None, only the reference is
                          processed (useful for checking the NAO basis alone).
        molcas_output:    Molcas output, for the poles. Required with dyson_molden.
        pole_source:      'caspt2' (default, recommended), 'mscaspt2', 'rasscf',
                          or 'best' -- per-&CASPT2-block, multistate where the
                          block has it. Use 'best' when the neutral is a single
                          state and the cations are MS/XMS, which OpenMolcas
                          forces (see parse_state_energies).
        n_electrons_neutral: if given, checked against sum(occupations).

    Returns dict with the usual reader keys plus Dyson-specific ones:
        C_nao          (n_nao x n_dyson) Dyson orbitals in the NAO basis,
                       or the reference natural orbitals if no Dyson file
        energies       (n_dyson,) poles in Hartree, as -IP (see note below)
        pole_strengths (n_dyson,) ||d^n||^2
        ao_list, elements, coords_ang, nao_occupancies, n_occ
    """
    print(f"\n{'=' * 62}")
    print("HERMES NAO FROM MOLCAS (CASSCF/CASPT2 + Dyson orbitals)")
    print(f"{'=' * 62}")
    print(f"  reference : {reference_molden}")
    print(f"  dyson     : {dyson_molden or '(none — reference only)'}")
    print(f"  output    : {molcas_output or '(none)'}")

    # ── 1. Reference file: atoms, basis, orbitals ────────────────────────────
    print("\n1. Reading reference Molden (defines the NAO basis)...")
    ref_sections = read_molden_sections(reference_molden)
    if '5d' not in ref_sections:
        print("   WARNING: no [5D] marker — d ordering may be wrong. "
              "Verify PER COMPONENT; shell sums cannot detect this.")

    elements, coords_ang = parse_molcas_atoms(ref_sections)
    n_atoms = len(elements)
    print(f"   n_atoms = {n_atoms}  ({', '.join(sorted(set(elements)))})")

    ao_list = _parse_gto_lines(ref_sections.get('gto', [])[1:], elements)
    n_ao = len(ao_list)
    print(f"   n_ao = {n_ao}")

    ref_mo = parse_molden_mos({'mo': ref_sections.get('mo', [])[1:]}, n_ao)
    C_ref = ref_mo['C_ao_alpha']
    n_mo_ref = C_ref.shape[1]
    print(f"   n_mo = {n_mo_ref}")

    if n_ao != n_mo_ref:
        raise ValueError(
            f"Reference Molden is not square (n_ao={n_ao}, n_mo={n_mo_ref}). "
            "The overlap is recovered as S = inv(C @ C.T) and needs all AOs' "
            "worth of orbitals. Use the full $Project.rasscf.molden."
        )

    # ── 2. Overlap from MO orthonormality ────────────────────────────────────
    print("\n2. Recovering overlap S = inv(C @ C.T)...")
    S = np.linalg.inv(C_ref @ C_ref.T)
    orth_err = np.max(np.abs(C_ref.T @ S @ C_ref - np.eye(n_mo_ref)))
    print(f"   |C.T S C - I|_max = {orth_err:.2e}")
    if orth_err > 1e-8:
        print("   WARNING: reference orbitals are not orthonormal to 1e-8.")

    # ── 3. Occupation-WEIGHTED density ───────────────────────────────────────
    # This is the one genuine departure from the single-determinant readers.
    occs = parse_molden_occupations(reference_molden)
    if len(occs) != n_mo_ref:
        raise ValueError(f"{len(occs)} occupations for {n_mo_ref} orbitals")

    n_elec = float(occs.sum())
    print(f"\n3. Occupation-weighted density P = C diag(n) C.T")
    print(f"   sum(occupations) = {n_elec:.4f} electrons")
    frac = occs[(occs > 1e-6) & (occs < 1.999)]
    print(f"   {len(frac)} orbitals with fractional occupation "
          f"(range {frac.min():.5f}–{frac.max():.5f})" if len(frac) else
          "   no fractional occupations — is this really a CASSCF reference?")

    if n_electrons_neutral is not None and abs(n_elec - n_electrons_neutral) > 1e-3:
        print(f"   WARNING: expected {n_electrons_neutral} electrons, got {n_elec:.4f}")

    P = (C_ref * occs) @ C_ref.T
    tr = float(np.trace(S @ P))
    print(f"   tr(S P) = {tr:.4f}   (must equal sum(occupations))")
    if abs(tr - n_elec) > 1e-3:
        print(f"   WARNING: tr(SP) != sum(occ) — density is inconsistent.")

    # What the integer-cut readers would have produced, for the record.
    n_occ_naive = int((occs >= 1.0).sum())
    print(f"   [an integer 'occ>=1' cut would give tr(SP) = {n_occ_naive} "
          f"and discard {float(occs[occs < 1.0].sum()):.4f} electrons]")

    # ── 4. NAOs ──────────────────────────────────────────────────────────────
    print("\n4. Computing NAOs from the correlated density...")
    nao_coeffs_ao, nao_occupancies = compute_nao_from_density(P, S, ao_list)
    print(f"   NAO total occupation: {nao_occupancies.sum():.4f} "
          f"(intra-atomic projection; below n_electrons is expected)")

    result: Dict = {
        "ao_list": ao_list,
        "elements": elements,
        "coords_ang": coords_ang,
        "nao_occupancies": nao_occupancies,
        "n_occ": n_occ_naive,
        "S": S,
        "reference_occupations": occs,
        # The NAO basis itself, in AO coefficients. Returned so that orbitals
        # from a DIFFERENT calculation on the same geometry and basis -- an SCF
        # set, say -- can be expressed in THIS basis rather than in one built
        # from their own density. Two NAO bases built from two densities are
        # close but not identical, and comparing quantities across them
        # compares the bases as well as the physics.
        "nao_coeffs_ao": nao_coeffs_ao,
    }

    # ── 5. Reference-only mode ───────────────────────────────────────────────
    if dyson_molden is None:
        print("\n5. No Dyson file — returning reference natural orbitals in NAO basis.")
        result["C_nao"] = transform_mos_to_nao_basis(C_ref, nao_coeffs_ao, S)
        result["energies"] = ref_mo['energies_alpha']
        result["pole_strengths"] = occs
        return result

    if molcas_output is None:
        raise ValueError("molcas_output is required when dyson_molden is given: "
                         "the poles must come from it, not from the Molden file.")

    # ── 6. Dyson orbitals ────────────────────────────────────────────────────
    print("\n5. Reading Dyson orbitals...")
    dys_sections = read_molden_sections(dyson_molden)
    dys_elements, _ = parse_molcas_atoms(dys_sections)
    if dys_elements != elements:
        raise ValueError(f"Atom lists differ between reference and Dyson file: "
                         f"{elements} vs {dys_elements}")

    dys_ao = _parse_gto_lines(dys_sections.get('gto', [])[1:], dys_elements)
    if len(dys_ao) != n_ao:
        raise ValueError(f"AO basis differs: reference {n_ao}, Dyson {len(dys_ao)}")
    for a, b in zip(ao_list, dys_ao):
        if (a.atom_idx, a.angular_momentum, a.magnetic_quantum) != \
           (b.atom_idx, b.angular_momentum, b.magnetic_quantum):
            raise ValueError("AO ordering differs between reference and Dyson file")
    print(f"   AO basis matches the reference ({n_ao} functions, same ordering)")

    dys_mo = parse_molden_mos({'mo': dys_sections.get('mo', [])[1:]}, n_ao)
    D_ao = dys_mo['C_ao_alpha']
    n_dyson = D_ao.shape[1]
    print(f"   {n_dyson} Dyson orbitals")

    # Pole strengths ARE the squared norms. Verify against the written Occup=
    # field rather than trusting either alone.
    norms2 = np.einsum('ai,ab,bi->i', D_ao, S, D_ao)
    occ_written = parse_molden_occupations(dyson_molden)
    print(f"\n6. Pole strengths (squared norms — NOT normalised away):")
    for i in range(n_dyson):
        flag = "" if abs(norms2[i] - occ_written[i]) < 1e-3 else "   <-- MISMATCH"
        print(f"   orbital {i + 1}:  ||d||^2 = {norms2[i]:.5f}   "
              f"Occup= {occ_written[i]:.5f}{flag}")
    print(f"   sum of pole strengths = {norms2.sum():.4f}  (must be <= n_electrons)")
    if norms2.sum() > n_elec + 1e-6:
        print("   WARNING: pole strengths exceed the electron count.")

    # ── 7. Poles from the output, at full precision ──────────────────────────
    print(f"\n7. Poles from '{pole_source}' in the Molcas output "
          f"(NOT from the Molden Ene= field)...")
    energies = parse_state_energies(molcas_output)
    pool = energies.get(pole_source, [])
    if not pool:
        raise ValueError(f"No '{pole_source}' energies found in {molcas_output}. "
                         f"Available: "
                         f"{ {k: len(v) for k, v in energies.items()} }")

    # Each Dyson orbital is matched to its final state through the CASSCF binding
    # energy, then given the corrected pole for THAT state. Not a slice — see
    # assign_poles for why a slice silently mis-assigns.
    dyson_ene = parse_molden_energies(dyson_molden)
    # With RASSI's EJOB keyword the effective-Hamiltonian diagonal is taken from the
    # CASPT2 JobMix, so the binding energies written into Dyson.SF.molden are already
    # the corrected ones and must be matched against `pool`, not against RASSCF. Detect
    # it from the reference ladder rather than from a user flag, which could disagree
    # with the file actually supplied.
    def _span(ref):
        return np.array([e - ref[0] for e in ref])
    err_ras = np.abs(_span(energies['rasscf'])[None, :] - dyson_ene[:, None]).min(axis=1).max()
    err_cor = np.abs(_span(pool)[None, :] - dyson_ene[:, None]).min(axis=1).max()
    if err_cor < err_ras:
        print(f"   Dyson binding energies match {pole_source.upper()} "
              f"({err_cor * HA_TO_EV:.4f} eV) better than RASSCF "
              f"({err_ras * HA_TO_EV:.4f} eV): reading them as EJOB-corrected.")
        ips, states = assign_poles(dyson_ene, pool, pool)
    else:
        ips, states = assign_poles(dyson_ene, energies['rasscf'], pool)

    print(f"   neutral (state 1) = {pool[0]:.8f} Ha")
    print(f"   orbital -> state   CASSCF BE      {pole_source.upper()} pole")
    for i in range(n_dyson):
        cas_be = dyson_ene[i] * HA_TO_EV
        print(f"     {i + 1:2d}    ->  {states[i]:2d}      "
              f"{cas_be:7.3f} eV     {ips[i] * HA_TO_EV:7.3f} eV")

    if len(set(states)) != len(states):
        print("   WARNING: two Dyson orbitals matched the same state.")

    groups = degenerate_groups(ips)
    multiplets = [g for g in groups if len(g) > 1]
    if multiplets:
        print(f"\n   degenerate multiplets (labels arbitrary within each, "
              f"only the group total is basis-independent):")
        for g in multiplets:
            print(f"     orbitals {[i + 1 for i in g]} -> states "
                  f"{[states[i] for i in g]}   pole {ips[g[0]] * HA_TO_EV:.3f} eV   "
                  f"summed strength {float(norms2[g].sum()):.5f}")

    amps = parse_dyson_amplitudes(molcas_output)
    if amps:
        print(f"\n   RASSI Dyson table ({len(amps)} rows, CASSCF-level BE) — "
              f"cross-check on the assignment:")
        for f, t, be, inten in amps[:12]:
            print(f"     {f} -> {t:2d}   BE = {be:7.3f} eV   intensity = {inten:.5f}")
        # The set of states RASSI reports must equal the set we assigned.
        if set(t for _, t, _, _ in amps) != set(states):
            print(f"   WARNING: assigned states {sorted(set(states))} differ from "
                  f"RASSI's {sorted(set(t for _, t, _, _ in amps))}")
        else:
            print(f"   assignment agrees with the RASSI table: "
                  f"states {sorted(set(states))}")

    # ── 8. Transform Dyson orbitals into the NAO basis ───────────────────────
    print("\n8. Transforming Dyson orbitals to the NAO basis...")
    D_nao = transform_mos_to_nao_basis(D_ao, nao_coeffs_ao, S)
    print(f"   shape {D_nao.shape}  (n_nao x n_dyson)")

    # Norms must survive the transformation; NAOs are orthonormal so the NAO-basis
    # squared norms are a plain column sum of squares.
    norms2_nao = np.einsum('ai,ai->i', D_nao, D_nao)
    drift = float(np.max(np.abs(norms2_nao - norms2)))
    print(f"   max |‖d‖²_NAO - ‖d‖²_AO| = {drift:.2e}  "
          f"({'OK' if drift < 1e-8 else 'WARNING — weights not preserved'})")

    result["C_nao"] = D_nao
    result["pole_strengths"] = norms2
    result["dyson_amplitudes"] = amps
    result["state_energies"] = energies
    result["pole_source"] = pole_source
    result["n_dyson"] = n_dyson
    # Stored as orbital-energy analogues: eps = -IP, so that E_F - eps has the
    # same sign convention as the single-determinant readers.
    result["energies"] = -ips[:n_dyson] if len(ips) >= n_dyson else -ips
    result["ips_ha"] = ips

    print(f"\n{'=' * 62}")
    print("NAO / DYSON COMPUTATION COMPLETE")
    print(f"{'=' * 62}\n")
    return result


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="NAOs and Dyson orbitals from Molcas CASSCF/CASPT2 + RASSI")
    ap.add_argument('reference_molden', help='$Project.rasscf.molden (N-electron)')
    ap.add_argument('--dyson', help='Dyson.SF.molden.<J> from RASSI DYSExport')
    ap.add_argument('--output', help='Molcas output file (required with --dyson)')
    ap.add_argument('--poles', default='caspt2',
                    choices=['caspt2', 'mscaspt2', 'rasscf', 'best'],
                    help='which energies supply the poles (default: caspt2)')
    ap.add_argument('--nelec', type=int, default=None,
                    help='expected electron count of the neutral, for checking')
    args = ap.parse_args()

    compute_nao_from_molcas(args.reference_molden, args.dyson, args.output,
                            args.poles, args.nelec)


if __name__ == '__main__':
    main()
