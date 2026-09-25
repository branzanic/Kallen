#!/usr/bin/env python3
"""
Compute Natural Atomic Orbitals (NAOs) from an ORCA Molden file.

Implements the same pre-NAO algorithm as compute_nao_from_turbomole.py
and compute_nao_from_fchk.py:
  - Per-atom Löwdin orthogonalisation
  - Block-diagonalise density matrix by (atom, l, m_l)
  - Two-step inter-atomic Löwdin on NMB (NBO7-consistent)

The overlap matrix is derived from MO orthonormality: S = inv(C @ C.T),
valid because ORCA writes all n_ao MOs in the Molden file (square C).

Input: an ORCA canonical Molden file produced by orca_2mkl
  (the non-_loc file, which holds the SCF canonical orbitals).
  The file must contain [5D] / [7F] markers (spherical harmonics).

Usage (standalone check):
    python compute_nao_from_orca.py benzene_boys.molden.input
"""

import re
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import core NAO computation routines from the canonical Turbomole module
hermes_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(hermes_dir))
from compute_nao_from_turbomole import (
    AOBasisFunction,
    compute_nao_from_density,
    transform_mos_to_nao_basis,
)


# ── Molden m_l ordering within each shell type ───────────────────────────────
# With [5D][7F][9G] markers (spherical harmonics, as written by ORCA):
#   p:  px py pz              → m = +1, -1,  0
#   d:  d0 d+1 d-1 d+2 d-2   → m =  0, +1, -1, +2, -2
#   f:  f0 f+1 f-1 f+2 f-2 f+3 f-3  → m = 0, +1, -1, +2, -2, +3, -3
MOLDEN_M_ORDER = {
    0: [0],
    1: [+1, -1,  0],
    2: [ 0, +1, -1, +2, -2],
    3: [ 0, +1, -1, +2, -2, +3, -3],
}

# Bohr → Angstrom
BOHR_TO_ANG = 0.529177249

# Element symbol → atomic number (subset covering benchmark molecules)
SYMBOL_TO_ANUM = {
    'H': 1,  'He': 2,
    'Li': 3, 'Be': 4, 'B': 5,  'C': 6,  'N': 7,  'O': 8,  'F': 9,  'Ne': 10,
    'Na': 11,'Mg': 12,'Al': 13,'Si': 14,'P': 15, 'S': 16, 'Cl': 17,'Ar': 18,
    'K': 19, 'Ca': 20,
    'Sc': 21,'Ti': 22,'V': 23, 'Cr': 24,'Mn': 25,
    'Fe': 26,'Co': 27,'Ni': 28,'Cu': 29,'Zn': 30,
    'Ga': 31,'Ge': 32,'As': 33,'Se': 34,'Br': 35,'Kr': 36,
    'Rb': 37,'Sr': 38,
    'Y': 39, 'Zr': 40,'Nb': 41,'Mo': 42,'Tc': 43,
    'Ru': 44,'Rh': 45,'Pd': 46,'Ag': 47,'Cd': 48,
    'In': 49,'Sn': 50,'Sb': 51,'Te': 52,'I': 53, 'Xe': 54,
    'Cs': 55,'Ba': 56,
    'Hf': 72,'Ta': 73,'W': 74, 'Re': 75,'Os': 76,
    'Ir': 77,'Pt': 78,'Au': 79,'Hg': 80,
}


def parse_ecp_from_orca_out(out_path: str) -> Dict[str, int]:
    """
    Parse ECP core electrons from an ORCA .out file.

    Looks for 'NewECP' blocks or the ECP summary table:
        ECP Coverage  Coverage  N_core  ...
        Ru  def2-ecp     28
    Returns {element: ncore}, empty dict if no .out or no ECPs found.
    """
    result: Dict[str, int] = {}
    try:
        with open(out_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return result

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('NewECP') or stripped.startswith('newECP'):
            parts = stripped.split()
            if len(parts) >= 2:
                elem = parts[1].capitalize()
                for j in range(i + 1, min(i + 10, len(lines))):
                    m = re.search(r'N_core\s+(\d+)', lines[j], re.IGNORECASE)
                    if m:
                        result[elem] = int(m.group(1))
                        break
        m = re.match(r'\s*(\w+)\s+\S+ecp\S*\s+(\d+)', stripped, re.IGNORECASE)
        if m and 'ecp' in stripped.lower():
            elem = m.group(1).capitalize()
            ncore = int(m.group(2))
            if ncore > 0 and elem not in result:
                result[elem] = ncore

    return result


# ── Molden parser ─────────────────────────────────────────────────────────────

def _read_sections(molden_file: str) -> Dict[str, List[str]]:
    """Split a Molden file into named sections (lowercased key)."""
    sections: Dict[str, List[str]] = {}
    current = None
    with open(molden_file) as fh:
        for line in fh:
            stripped = line.rstrip('\n')
            if stripped.strip().startswith('[') and stripped.strip().endswith(']'):
                current = stripped.strip()[1:-1].lower()
                sections.setdefault(current, [])
            elif current is not None:
                sections[current].append(stripped)
    return sections


def parse_molden_atoms(sections: Dict) -> Tuple[List[str], np.ndarray]:
    """
    Parse [Atoms] section.

    Returns:
        elements   : list of element symbols (length n_atoms)
        coords_ang : (n_atoms, 3) float array, Angstrom
    """
    lines = sections.get('atoms', [])
    if not lines:
        raise ValueError("[Atoms] section not found in Molden file")

    # First non-empty line may be the unit specifier (already consumed by
    # section key, but orca_2mkl writes "AU" or "Angs" on the [Atoms] line
    # itself, so we need to check the header line that orca_2mkl already
    # includes *inside* the section in some versions).
    # Safest: look for 'AU' or 'Angs' in the section key line in the original
    # file; we can also infer from the coordinate magnitude.
    # orca_2mkl always writes AU for ORCA outputs.
    # The section tag is "[Atoms] AU" — orca_2mkl includes the unit on the
    # same line as the tag, which our splitter captures as part of the key.
    # So the key will be "atoms] au" after stripping brackets — let's handle
    # both possibilities.

    unit_ang = False   # default: Bohr (AU)
    for key in sections:
        if key.startswith('atoms'):
            if 'angs' in key.lower():
                unit_ang = True
            break

    elements: List[str] = []
    coords: List[List[float]] = []

    for line in lines:
        parts = line.split()
        if len(parts) < 6:
            continue
        # format: element  atom_num  atomic_num  x  y  z
        try:
            elem = parts[0].capitalize()
            x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
            elements.append(elem)
            coords.append([x, y, z])
        except (ValueError, IndexError):
            continue

    if not elements:
        raise ValueError("No atoms parsed from [Atoms] section")

    coords_arr = np.array(coords)
    if not unit_ang:
        coords_arr *= BOHR_TO_ANG

    return elements, coords_arr


def parse_molden_gto(sections: Dict, elements: List[str]) -> List[AOBasisFunction]:
    """
    Parse [GTO] section and build the ordered AO basis list.

    Handles s, p, d, f (spherical) and sp shells.
    Contraction index tracks how many contractions of the same (atom, l) have
    been seen — needed for NMB identification in the inter-atomic Löwdin step.

    Returns:
        ao_list : list of AOBasisFunction in Molden AO ordering
    """
    lines = sections.get('gto', [])
    if not lines:
        raise ValueError("[GTO] section not found in Molden file")

    ao_list: List[AOBasisFunction] = []
    atom_idx_0 = -1          # current 0-based atom index
    ctr_count: Dict[int, Dict[int, int]] = {}  # ctr_count[atom][l] = next contraction_idx
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line:
            # blank line between atoms or at end
            continue

        # Atom header: "atom_num  0"
        parts = line.split()
        if len(parts) >= 1 and parts[0].isdigit():
            atom_idx_0 = int(parts[0]) - 1   # convert 1-based → 0-based
            ctr_count.setdefault(atom_idx_0, {})
            continue

        if atom_idx_0 < 0:
            continue

        element = elements[atom_idx_0]

        # Shell header: "type  n_prim  scale"
        # type is one of: s, p, d, f, g, sp, ...
        if len(parts) >= 2:
            sh_type_str = parts[0].lower()
            try:
                n_prim = int(parts[1])
            except ValueError:
                continue

            # Skip primitive exponent/coefficient lines
            for _ in range(n_prim):
                if i < len(lines):
                    i += 1

            # Map shell type to angular momenta
            if sh_type_str == 's':
                l_list = [0]
            elif sh_type_str == 'p':
                l_list = [1]
            elif sh_type_str == 'd':
                l_list = [2]
            elif sh_type_str == 'f':
                l_list = [3]
            elif sh_type_str == 'g':
                l_list = [4]
            elif sh_type_str == 'sp':
                l_list = [0, 1]   # combined sp shell
            else:
                continue   # ignore unknown shell types

            for l in l_list:
                d = ctr_count[atom_idx_0]
                ctr_idx = d.get(l, 0)
                d[l] = ctr_idx + 1

                if l not in MOLDEN_M_ORDER:
                    # g or higher — add a single placeholder with m=0
                    ao_list.append(AOBasisFunction(atom_idx_0, element, l, 0, ctr_idx))
                    continue

                for m in MOLDEN_M_ORDER[l]:
                    ao_list.append(AOBasisFunction(atom_idx_0, element, l, m, ctr_idx))

    return ao_list


def parse_molden_mos(sections: Dict, n_ao: int) -> Dict:
    """
    Parse [MO] section, splitting alpha and beta MO sets.

    Returns:
        dict with keys:
          'C_ao_alpha'      : (n_ao, n_mo_alpha) alpha MO coefficient matrix
          'energies_alpha'  : (n_mo_alpha,) alpha orbital energies [Hartree]
          'n_occ_alpha'     : number of occupied alpha MOs
          'C_ao_beta'       : (n_ao, n_mo_beta) beta MO coefficients, or None
          'energies_beta'   : (n_mo_beta,) beta orbital energies, or None
          'n_occ_beta'      : number of occupied beta MOs, or None
          'is_unrestricted' : True if beta MOs were found
    """
    lines = sections.get('mo', [])
    if not lines:
        raise ValueError("[MO] section not found in Molden file")

    mos_energy: List[float] = []
    mos_coeffs: List[np.ndarray] = []
    mos_occ: List[float] = []
    mos_spin: List[str] = []

    cur_energy: Optional[float] = None
    cur_occ: Optional[float] = None
    cur_spin: str = 'alpha'
    cur_coeffs: List[Tuple[int, float]] = []   # (1-based AO index, coeff)

    def _flush():
        if cur_energy is None:
            return
        c = np.zeros(n_ao)
        for idx1, val in cur_coeffs:
            if 1 <= idx1 <= n_ao:
                c[idx1 - 1] = val
        mos_energy.append(cur_energy)
        mos_coeffs.append(c)
        mos_occ.append(cur_occ if cur_occ is not None else 0.0)
        mos_spin.append(cur_spin)

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        low = stripped.lower()
        if low.startswith('sym='):
            _flush()
            cur_energy = None
            cur_occ    = None
            cur_spin   = 'alpha'
            cur_coeffs = []
        elif low.startswith('ene='):
            val = stripped[4:].strip()
            try:
                cur_energy = float(val)
            except ValueError:
                cur_energy = 0.0
        elif low.startswith('spin='):
            cur_spin = stripped[5:].strip().lower()
        elif low.startswith('occup='):
            try:
                cur_occ = float(stripped[6:].strip())
            except ValueError:
                cur_occ = 0.0
        else:
            # Coefficient line: "  idx   coeff"
            parts = stripped.split()
            if len(parts) == 2:
                try:
                    idx1 = int(parts[0])
                    val  = float(parts[1])
                    cur_coeffs.append((idx1, val))
                except ValueError:
                    pass

    _flush()   # last MO

    # Split by spin channel
    alpha_indices = [i for i, s in enumerate(mos_spin) if s in ('alpha', '')]
    beta_indices = [i for i, s in enumerate(mos_spin) if s == 'beta']

    energies_a = np.array([mos_energy[i] for i in alpha_indices])
    C_ao_a = np.column_stack([mos_coeffs[i] for i in alpha_indices])
    occs_a = [mos_occ[i] for i in alpha_indices]
    n_occ_a = sum(1 for o in occs_a if o >= 1.0)

    has_beta = len(beta_indices) > 0

    if has_beta:
        energies_b = np.array([mos_energy[i] for i in beta_indices])
        C_ao_b = np.column_stack([mos_coeffs[i] for i in beta_indices])
        occs_b = [mos_occ[i] for i in beta_indices]
        n_occ_b = sum(1 for o in occs_b if o >= 1.0)
    else:
        energies_b = None
        C_ao_b = None
        n_occ_b = None

    return {
        'C_ao_alpha': C_ao_a,
        'energies_alpha': energies_a,
        'n_occ_alpha': n_occ_a,
        'C_ao_beta': C_ao_b,
        'energies_beta': energies_b,
        'n_occ_beta': n_occ_b,
        'is_unrestricted': has_beta,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def is_molden_unrestricted(molden_file: str) -> bool:
    """Quickly scan a Molden file for beta MOs (Spin=Beta or Spin= Beta)."""
    with open(molden_file) as fh:
        for line in fh:
            low = line.strip().lower()
            if low.startswith('spin=') and 'beta' in low:
                return True
    return False


def compute_nao_from_orca(molden_file: str) -> Dict:
    """
    Compute Natural Atomic Orbitals from an ORCA canonical Molden file.

    Steps:
      1. Parse [Atoms], [GTO], [MO] sections
      2. Recover overlap S = inv(C_ao @ C_ao.T)  (MO orthonormality)
      3. Build density matrix P = C_occ @ C_occ.T
      4. Compute NAOs via block-diagonalisation + inter-atomic Löwdin
      5. Transform MO coefficients to NAO basis

    Args:
        molden_file: Path to ORCA canonical Molden file (*.molden.input).

    Returns:
        dict with keys:
          "C_nao"           : (n_nao × n_mo)  MO coefficients in NAO basis
          "energies"        : (n_mo,) orbital energies  [Hartree]
          "n_occ"           : number of occupied MOs
          "ao_list"         : list of AOBasisFunction objects
          "elements"        : list of element symbols (n_atoms)
          "coords_ang"      : (n_atoms, 3) Angstrom
          "nao_occupancies" : (n_ao,) pre-NAO occupancies
    """
    print(f"\n{'='*60}")
    print("HERMES NAO FROM ORCA MOLDEN FILE")
    print(f"{'='*60}")
    print(f"  File: {molden_file}")

    # ── 1. Read and split into sections ───────────────────────────────────────
    print("\n1. Reading Molden sections...")
    # Need to handle "[Atoms] AU" — the unit is on the same line as the tag.
    # Re-read to build sections with the full tag line for unit detection.
    sections_raw: Dict[str, List[str]] = {}
    current_key = None
    with open(molden_file) as fh:
        for line in fh:
            stripped = line.rstrip('\n')
            m = re.match(r'^\s*\[([^\]]+)\]', stripped)
            if m:
                current_key = m.group(1).strip().lower()
                sections_raw.setdefault(current_key, [])
                # Store the full tag line as first entry for unit detection
                sections_raw[current_key].insert(0, stripped)
            elif current_key is not None:
                sections_raw[current_key].append(stripped)

    # Check for spherical harmonic markers
    has_5d = '5d' in sections_raw
    has_7f = '7f' in sections_raw
    if not has_5d:
        print("  WARNING: [5D] marker not found — d-function ordering may be wrong")

    # ── 2. Parse atoms ─────────────────────────────────────────────────────────
    print("2. Parsing atoms...")
    # Build unit-aware atoms section
    atoms_lines = sections_raw.get('atoms', [])
    unit_ang = any('angs' in ln.lower() for ln in atoms_lines[:2])

    elements: List[str] = []
    coords_list: List[List[float]] = []
    for line in atoms_lines[1:]:    # skip tag line
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            elem = parts[0].capitalize()
            x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
            elements.append(elem)
            coords_list.append([x, y, z])
        except (ValueError, IndexError):
            continue

    if not elements:
        raise ValueError("No atoms found in [Atoms] section")

    coords_ang = np.array(coords_list)
    if not unit_ang:
        coords_ang *= BOHR_TO_ANG

    n_atoms = len(elements)
    print(f"   n_atoms = {n_atoms}  ({', '.join(sorted(set(elements)))})")

    # ── 3. Parse GTO → AO basis list ──────────────────────────────────────────
    print("3. Parsing GTO section → AO basis list...")
    gto_lines = sections_raw.get('gto', [])
    ao_list = _parse_gto_lines(gto_lines[1:], elements)   # skip tag line
    n_ao = len(ao_list)
    print(f"   n_ao = {n_ao}")

    # ── 4. Parse MO coefficients ───────────────────────────────────────────────
    print("4. Parsing MO coefficients...")
    mo_lines = sections_raw.get('mo', [])
    mo_data = parse_molden_mos({'mo': mo_lines[1:]}, n_ao)

    C_ao = mo_data['C_ao_alpha']
    energies = mo_data['energies_alpha']
    n_occ = mo_data['n_occ_alpha']
    n_mo = len(energies)
    print(f"   n_mo = {n_mo},  n_occ = {n_occ}")

    if mo_data['is_unrestricted']:
        print(f"   WARNING: Unrestricted MOs detected but compute_nao_from_orca() "
              f"handles restricted only. Use compute_nao_from_orca_uhf() instead.")
        print(f"   Proceeding with alpha channel only.")

    if n_ao != n_mo:
        raise ValueError(
            f"n_ao ({n_ao}) ≠ n_mo ({n_mo}). "
            "Need a square MO coefficient matrix to recover S = inv(C @ C.T)."
        )

    # ── 5. Recover overlap ────────────────────────────────────────────────────
    print("\n5. Recovering overlap S = inv(C @ C.T)...")
    S = np.linalg.inv(C_ao @ C_ao.T)
    s_eig = np.linalg.eigvalsh(S)
    print(f"   S diagonal range: {S.diagonal().min():.4f} – {S.diagonal().max():.4f}")
    if s_eig.min() < -1e-6:
        print(f"   WARNING: S has negative eigenvalue {s_eig.min():.2e}")

    # Sanity: C.T @ S @ C should be identity
    orth_err = np.max(np.abs(C_ao.T @ S @ C_ao - np.eye(n_mo)))
    print(f"   Orthonormality check |C.T S C - I|_max = {orth_err:.2e}")

    # ── 6. Density matrix ─────────────────────────────────────────────────────
    C_occ = C_ao[:, :n_occ]
    P = C_occ @ C_occ.T
    print(f"\n6. Density matrix from {n_occ} occupied MOs, tr(S P) = {np.trace(S @ P):.4f}")

    # ── 7. Compute NAOs ───────────────────────────────────────────────────────
    print("\n7. Computing NAOs...")
    nao_coeffs_ao, nao_occupancies = compute_nao_from_density(P, S, ao_list)
    print(f"   NAO total occupation: {nao_occupancies.sum():.4f}  (intra-atomic projection; ~17 for benzene is normal)")

    # ── 8. Transform MOs to NAO basis ─────────────────────────────────────────
    print("\n8. Transforming MO coefficients to NAO basis...")
    C_nao = transform_mos_to_nao_basis(C_ao, nao_coeffs_ao, S)
    print(f"   C_nao shape: {C_nao.shape}  (n_nao × n_mo)")

    print(f"\n{'='*60}")
    print("NAO COMPUTATION COMPLETE")
    print(f"{'='*60}\n")

    return {
        "C_nao":           C_nao,
        "energies":        energies,
        "n_occ":           n_occ,
        "ao_list":         ao_list,
        "elements":        elements,
        "coords_ang":      coords_ang,
        "nao_occupancies": nao_occupancies,
    }


def compute_nao_from_orca_uhf(molden_file: str,
                              nao_basis: str = "shared") -> Dict:
    """
    Compute NAOs from an unrestricted (UHF/UKS) ORCA canonical Molden file.

    NAOs are computed from the TOTAL density (alpha + beta), then both
    alpha and beta MO coefficients are transformed to this common NAO basis.

    Args:
        molden_file: Path to ORCA canonical Molden file (*.molden.input).

    Returns:
        dict with keys:
          "C_nao_alpha"       : Alpha MO coefficients in NAO basis (n_nao × n_mo)
          "C_nao_beta"        : Beta MO coefficients in NAO basis  (n_nao × n_mo)
          "energies_alpha"    : Alpha orbital energies (n_mo,)
          "energies_beta"     : Beta orbital energies  (n_mo,)
          "n_occ_alpha"       : Number of alpha occupied orbitals
          "n_occ_beta"        : Number of beta occupied orbitals
          "ao_list"           : List of AOBasisFunction objects
          "elements"          : List of element symbols (n_atoms)
          "coords_ang"        : Atom coordinates in Angstrom (n_atoms × 3)
          "nao_occupancies"   : NAO occupancies from total density (n_ao,)
    """
    print(f"\n{'='*60}")
    print("HERMES NAO FROM ORCA MOLDEN FILE (UNRESTRICTED)")
    print(f"{'='*60}")
    print(f"  File: {molden_file}")

    # ── 1. Read and split into sections ───────────────────────────────────────
    print("\n1. Reading Molden sections...")
    sections_raw: Dict[str, List[str]] = {}
    current_key = None
    with open(molden_file) as fh:
        for line in fh:
            stripped = line.rstrip('\n')
            m = re.match(r'^\s*\[([^\]]+)\]', stripped)
            if m:
                current_key = m.group(1).strip().lower()
                sections_raw.setdefault(current_key, [])
                sections_raw[current_key].insert(0, stripped)
            elif current_key is not None:
                sections_raw[current_key].append(stripped)

    has_5d = '5d' in sections_raw
    if not has_5d:
        print("  WARNING: [5D] marker not found — d-function ordering may be wrong")

    # ── 2. Parse atoms ─────────────────────────────────────────────────────────
    print("2. Parsing atoms...")
    atoms_lines = sections_raw.get('atoms', [])
    unit_ang = any('angs' in ln.lower() for ln in atoms_lines[:2])

    elements: List[str] = []
    coords_list: List[List[float]] = []
    for line in atoms_lines[1:]:
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            elem = parts[0].capitalize()
            x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
            elements.append(elem)
            coords_list.append([x, y, z])
        except (ValueError, IndexError):
            continue

    if not elements:
        raise ValueError("No atoms found in [Atoms] section")

    coords_ang = np.array(coords_list)
    if not unit_ang:
        coords_ang *= BOHR_TO_ANG

    n_atoms = len(elements)
    print(f"   n_atoms = {n_atoms}  ({', '.join(sorted(set(elements)))})")

    # ── 3. Parse GTO → AO basis list ──────────────────────────────────────────
    print("3. Parsing GTO section → AO basis list...")
    gto_lines = sections_raw.get('gto', [])
    ao_list = _parse_gto_lines(gto_lines[1:], elements)
    n_ao = len(ao_list)
    print(f"   n_ao = {n_ao}")

    # ── 4. Parse MO coefficients (both spin channels) ─────────────────────────
    print("4. Parsing MO coefficients...")
    mo_lines = sections_raw.get('mo', [])
    mo_data = parse_molden_mos({'mo': mo_lines[1:]}, n_ao)

    C_ao_a = mo_data['C_ao_alpha']
    energies_a = mo_data['energies_alpha']
    n_occ_a = mo_data['n_occ_alpha']
    n_mo_a = len(energies_a)

    if not mo_data['is_unrestricted']:
        raise ValueError(
            "No beta MOs found in Molden file. "
            "Use compute_nao_from_orca() for restricted calculations."
        )

    C_ao_b = mo_data['C_ao_beta']
    energies_b = mo_data['energies_beta']
    n_occ_b = mo_data['n_occ_beta']
    n_mo_b = len(energies_b)

    print(f"   Alpha: n_mo = {n_mo_a},  n_occ = {n_occ_a}")
    print(f"   Beta:  n_mo = {n_mo_b},  n_occ = {n_occ_b}")
    print(f"   Unpaired electrons: {n_occ_a - n_occ_b}")

    if n_ao != n_mo_a:
        raise ValueError(
            f"n_ao ({n_ao}) ≠ n_mo_alpha ({n_mo_a}). "
            "Need a square MO coefficient matrix to recover S = inv(C @ C.T)."
        )

    # ── 5. Recover overlap from alpha MOs ─────────────────────────────────────
    print("\n5. Recovering overlap S = inv(C_alpha @ C_alpha.T)...")
    S = np.linalg.inv(C_ao_a @ C_ao_a.T)
    s_eig = np.linalg.eigvalsh(S)
    print(f"   S diagonal range: {S.diagonal().min():.4f} – {S.diagonal().max():.4f}")
    if s_eig.min() < -1e-6:
        print(f"   WARNING: S has negative eigenvalue {s_eig.min():.2e}")

    # ── 6. Total density matrix ───────────────────────────────────────────────
    P_a = C_ao_a[:, :n_occ_a] @ C_ao_a[:, :n_occ_a].T
    P_b = C_ao_b[:, :n_occ_b] @ C_ao_b[:, :n_occ_b].T
    P_total = P_a + P_b
    n_elec = n_occ_a + n_occ_b
    print(f"\n6. Total density from {n_occ_a} alpha + {n_occ_b} beta occupied MOs")
    print(f"   tr(S P_total) = {np.trace(S @ P_total):.4f}  (should be {n_elec})")

    # ── 7. Compute NAOs ───────────────────────────────────────────────────────
    if nao_basis not in ("shared", "per_spin"):
        raise ValueError(f"nao_basis must be 'shared' or 'per_spin', got {nao_basis!r}")
    if nao_basis == "shared":
        print("\n7. Computing NAOs from TOTAL density (shared basis)...")
        nao_coeffs_ao, nao_occupancies, nao_tiers = compute_nao_from_density(
            P_total, S, ao_list, return_tiers=True)
        nao_occ_a = nao_occ_b = nao_occupancies
        nao_c_a = nao_c_b = nao_coeffs_ao
        nao_tier_a = nao_tier_b = nao_tiers
        print(f"   NAO total occupation: {nao_occupancies.sum():.4f}  (should be ~{n_elec})")
    else:
        print("\n7. Computing NAOs per spin (SEPARATE bases)...")
        nao_c_a, nao_occ_a, nao_tier_a = compute_nao_from_density(P_a, S, ao_list, return_tiers=True)
        nao_c_b, nao_occ_b, nao_tier_b = compute_nao_from_density(P_b, S, ao_list, return_tiers=True)
        nao_coeffs_ao, nao_occupancies = nao_c_a, nao_occ_a
        print(f"   alpha occupation {nao_occ_a.sum():.4f} (~{n_occ_a}), "
              f"beta {nao_occ_b.sum():.4f} (~{n_occ_b})")
        print("   NOTE: alpha and beta are in DIFFERENT bases; a cross-spin "
              "comparison of G_rs is not basis-controlled.")

    # ── 8. Transform BOTH spin channels to NAO basis ──────────────────────────
    print("\n8. Transforming alpha MO coefficients to NAO basis...")
    C_nao_a = transform_mos_to_nao_basis(C_ao_a, nao_c_a, S)
    print(f"   C_nao_alpha shape: {C_nao_a.shape}")

    print("   Transforming beta MO coefficients to NAO basis...")
    C_nao_b = transform_mos_to_nao_basis(C_ao_b, nao_c_b, S)
    print(f"   C_nao_beta  shape: {C_nao_b.shape}")

    print(f"\n{'='*60}")
    print("NAO COMPUTATION COMPLETE (UNRESTRICTED)")
    print(f"{'='*60}\n")

    return {
        "C_nao_alpha":     C_nao_a,
        "C_nao_beta":      C_nao_b,
        "energies_alpha":  energies_a,
        "energies_beta":   energies_b,
        "n_occ_alpha":     n_occ_a,
        "n_occ_beta":      n_occ_b,
        "ao_list":         ao_list,
        "elements":        elements,
        "coords_ang":      coords_ang,
        "nao_occupancies": nao_occupancies,
        "nao_occupancies_alpha": nao_occ_a,
        "nao_occupancies_beta":  nao_occ_b,
        "nao_tiers_alpha": nao_tier_a,
        "nao_tiers_beta":  nao_tier_b,
        "nao_basis":       nao_basis,
    }


def _parse_gto_lines(lines: List[str], elements: List[str]) -> List[AOBasisFunction]:
    """
    Parse GTO section lines (tag line already removed) into an AO basis list.
    """
    ao_list: List[AOBasisFunction] = []
    atom_idx_0 = -1
    ctr_count: Dict[Tuple[int, int], int] = {}   # (atom_idx, l) → next ctr_idx
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line:
            continue

        parts = line.split()
        if not parts:
            continue

        # Atom header: digits followed by optional "0"
        if parts[0].isdigit() and (len(parts) == 1 or (len(parts) >= 2 and parts[1] == '0')):
            atom_idx_0 = int(parts[0]) - 1
            continue

        if atom_idx_0 < 0 or atom_idx_0 >= len(elements):
            continue

        element = elements[atom_idx_0]
        sh_type_str = parts[0].lower()

        # Shell header requires at least 2 tokens: type + n_prim
        if len(parts) < 2:
            continue
        try:
            n_prim = int(parts[1])
        except ValueError:
            continue

        # Skip primitive lines
        for _ in range(n_prim):
            if i < len(lines):
                i += 1

        # Map to angular momentum list
        if sh_type_str == 's':
            l_list = [0]
        elif sh_type_str == 'p':
            l_list = [1]
        elif sh_type_str == 'd':
            l_list = [2]
        elif sh_type_str == 'f':
            l_list = [3]
        elif sh_type_str == 'g':
            l_list = [4]
        elif sh_type_str == 'sp':
            l_list = [0, 1]
        else:
            continue

        for l in l_list:
            key = (atom_idx_0, l)
            ctr_idx = ctr_count.get(key, 0)
            ctr_count[key] = ctr_idx + 1

            m_order = MOLDEN_M_ORDER.get(l, [0])
            for m in m_order:
                ao_list.append(AOBasisFunction(atom_idx_0, element, l, m, ctr_idx))

    return ao_list


# ── Standalone check ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compute_nao_from_orca.py <file.molden.input>")
        sys.exit(1)

    result = compute_nao_from_orca(sys.argv[1])
    print(f"C_nao shape:    {result['C_nao'].shape}")
    print(f"n_occ:          {result['n_occ']}")
    print(f"n_atoms:        {len(result['elements'])}")
    print(f"Elements:       {result['elements']}")
    homo = result['n_occ'] - 1
    lumo = result['n_occ']
    print(f"HOMO energy:    {result['energies'][homo]:.6f} H")
    print(f"LUMO energy:    {result['energies'][lumo]:.6f} H")
    print(f"NAO occ sum:    {result['nao_occupancies'].sum():.4f}")
