#!/usr/bin/env python3
"""
Compute Natural Atomic Orbitals (NAOs) from Turbomole output.

Since Turbomole's NBO interface doesn't output NAOMO coefficients,
we compute NAOs directly by diagonalizing atomic blocks of the density matrix.

This gives us the NAO basis and the transformation coefficients from NAOs to MOs.
"""

import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

# Natural minimal basis and core counts per element, used by the three-tier
# classification. Module level so derive_nao_tiers() can be called standalone.
_NMB_TABLE: dict = {
    'H' : {0: 1, 1: 0},
    'He': {0: 1},
    'Li': {0: 2, 1: 0},
    'Be': {0: 2, 1: 0},
    'B' : {0: 2, 1: 1},
    'C' : {0: 2, 1: 1},
    'N' : {0: 2, 1: 1},
    'O' : {0: 2, 1: 1},
    'F' : {0: 2, 1: 1},
    'Ne': {0: 2, 1: 1},
    'Na': {0: 3, 1: 2},
    'Mg': {0: 3, 1: 2},
    'Al': {0: 3, 1: 2},
    'Si': {0: 3, 1: 2},
    'P' : {0: 3, 1: 2},
    'S' : {0: 3, 1: 2},
    'Cl': {0: 3, 1: 2},
    'Ar': {0: 3, 1: 2},
    'Sc': {0: 4, 1: 2, 2: 1},
    'Ti': {0: 4, 1: 2, 2: 1},
    'V' : {0: 4, 1: 2, 2: 1},
    'Cr': {0: 4, 1: 2, 2: 1},
    'Mn': {0: 4, 1: 2, 2: 1},
    'Fe': {0: 4, 1: 2, 2: 1},
    'Co': {0: 4, 1: 2, 2: 1},
    'Ni': {0: 4, 1: 2, 2: 1},
    'Cu': {0: 4, 1: 2, 2: 1},
    'Zn': {0: 4, 1: 2, 2: 1},
    # K and Ca are group 1/2, not transition metals: the NBO 7 change that moved the np
    # into the Rydberg manifold applies to the d block, so these keep a valence 4p, as
    # Na/Mg keep a valence 3p and Ga-Kr a valence 4p.
    'K' : {0: 4, 1: 3},
    'Ca': {0: 4, 1: 3},
    'Ga': {0: 4, 1: 3, 2: 1},
    'Ge': {0: 4, 1: 3, 2: 1},
    'As': {0: 4, 1: 3, 2: 1},
    'Se': {0: 4, 1: 3, 2: 1},
    'Br': {0: 4, 1: 3, 2: 1},
    'Kr': {0: 4, 1: 3, 2: 1},
    'I' : {0: 5, 1: 4, 2: 2},
}

# 4d and 5d transition metals under the standard def2 ECPs, which remove the deep
# core: ncore=28 strips [Ar]3d10 (Y-Cd), ncore=60 strips [Kr]4d10 4f14 (Hf-Hg). In
# both cases the surviving NMB is (n-1)s,ns / (n-1)p / (n-1)d: two s shells, ONE p
# shell and one d, with the (n-1)s and (n-1)p as semi-core, so these metals have no
# valence p -- the same NBO 7 convention applied to Sc-Zn above. (An earlier draft of
# this comment read "(n-1)p,np", which is the NBO 3.1 convention this table deliberately
# does not implement; reconciling the table to that wording would reintroduce the
# valence np and with it the 78x anomaly.) These are ECP counts, NOT all-electron ones: an
# all-electron 4d/5d calculation has a different NMB and must not use this table.
_NMB_ECP: dict = {0: 2, 1: 1, 2: 1}
for _e in ('Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
           'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'):
    _NMB_TABLE[_e] = dict(_NMB_ECP)
del _e

_CORE_TABLE: dict = {
    'H' : {},
    'He': {},
    'Li': {0: 1},
    'Be': {0: 1},
    'B' : {0: 1},
    'C' : {0: 1},
    'N' : {0: 1},
    'O' : {0: 1},
    'F' : {0: 1},
    'Ne': {0: 1},
    'Na': {0: 2, 1: 1},
    'Mg': {0: 2, 1: 1},
    'Al': {0: 2, 1: 1},
    'Si': {0: 2, 1: 1},
    'P' : {0: 2, 1: 1},
    'S' : {0: 2, 1: 1},
    'Cl': {0: 2, 1: 1},
    'Ar': {0: 2, 1: 1},
    'Sc': {0: 3, 1: 2},
    'Ti': {0: 3, 1: 2},
    'V' : {0: 3, 1: 2},
    'Cr': {0: 3, 1: 2},
    'Mn': {0: 3, 1: 2},
    'Fe': {0: 3, 1: 2},
    'Co': {0: 3, 1: 2},
    'Ni': {0: 3, 1: 2},
    'Cu': {0: 3, 1: 2},
    'Zn': {0: 3, 1: 2},
    'K' : {0: 3, 1: 2},
    'Ca': {0: 3, 1: 2},
    'Ga': {0: 3, 1: 2, 2: 1},
    'Ge': {0: 3, 1: 2, 2: 1},
    'As': {0: 3, 1: 2, 2: 1},
    'Se': {0: 3, 1: 2, 2: 1},
    'Br': {0: 3, 1: 2, 2: 1},
    'Kr': {0: 3, 1: 2, 2: 1},
    'I' : {0: 4, 1: 3, 2: 2},
}

_CORE_ECP: dict = {0: 1, 1: 1}
for _e in ('Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
           'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'):
    _CORE_TABLE[_e] = dict(_CORE_ECP)
del _e


def _sym(elem: str) -> str:
    """Canonical element symbol for a table lookup.

    Turbomole writes lowercase symbols ('fe', 'c') and build_ao_basis_turbomole accepts
    them, capitalising for its own basis lookup while storing the raw case on the
    AOBasisFunction. The minimal-basis tables are keyed on capitalised symbols, so every
    lookup normalises here -- otherwise a supported input dies in _require_nmb telling
    the user to add a duplicate lowercase entry.
    """
    return elem.capitalize() if elem else elem


def _require_nmb(elem: str) -> None:
    """Fail loudly on an element the natural minimal basis does not cover.

    A missing element used to be silent: every one of its NAOs stayed at the
    initialised tier 2 (Rydberg), organize_coefficients then dropped the atom, and
    all of its G_rs pairs vanished with no error and no warning. A wrong answer that
    looks like a right one is worse than a stack trace.
    """
    elem = _sym(elem)
    if elem in _NMB_TABLE and elem not in _CORE_TABLE:
        raise ValueError(
            f"Element {elem!r} is in _NMB_TABLE but not _CORE_TABLE. Its core count "
            f"would default to zero, so every core NAO on that atom would be "
            f"classified valence and admitted into G_rs. Add {elem!r} to _CORE_TABLE "
            f"in compute_nao_from_turbomole.py."
        )
    if elem not in _NMB_TABLE:
        raise ValueError(
            f"No natural-minimal-basis entry for element {elem!r}. Every NAO on that "
            f"atom would be classified Rydberg and the atom would be dropped from "
            f"G_rs silently. Add {elem!r} to _NMB_TABLE and _CORE_TABLE in "
            f"compute_nao_from_turbomole.py (core+valence shell counts per l). "
            f"Covered elements: {', '.join(sorted(_NMB_TABLE))}."
        )


@dataclass
class AOBasisFunction:
    """Represents an AO basis function."""
    atom_idx: int  # 0-indexed atom number
    element: str
    angular_momentum: int  # 0=s, 1=p, 2=d, 3=f
    magnetic_quantum: int  # m_l value
    contraction_idx: int  # Which contraction of this shell
    
    def get_orbital_type(self) -> str:
        """Get orbital type label (s, px, py, pz, dxy, etc.)."""
        if self.angular_momentum == 0:
            return 's'
        elif self.angular_momentum == 1:
            # p orbitals: px, py, pz (m=-1,0,1)
            return ['py', 'pz', 'px'][self.magnetic_quantum + 1]
        elif self.angular_momentum == 2:
            # d orbitals: dxy, dyz, dz2, dxz, dx2y2 (m=-2,-1,0,1,2)
            return ['dxy', 'dyz', 'dz2', 'dxz', 'dx2y2'][self.magnetic_quantum + 2]
        elif self.angular_momentum == 3:
            return 'f'
        else:
            return f'l{self.angular_momentum}'


def parse_mo_coefficients_matrix(mo_file: str,
                                  max_mo: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Parse Turbomole MO file and extract coefficient matrix.

    Args:
        mo_file: Path to alpha or beta MO file
        max_mo:  If given, stop reading after this many MOs (useful when only
                 occupied + a few virtual orbitals are needed — much faster for
                 large systems since the file is read line-by-line).

    Returns:
        mo_coeffs: MO coefficient matrix (n_ao x n_mo)
        mo_energies: MO energies (n_mo,)
        n_ao: Number of AO basis functions
    """
    mo_energies = []
    all_coeffs = []
    n_ao = None

    with open(mo_file, 'r') as f:
        pending_header: Optional[Tuple[float, int]] = None   # (eigenvalue, nsaos)
        coeffs: list = []

        for line in f:
            # ── Early exit once max_mo reached ──────────────────────────────
            if max_mo is not None and len(mo_energies) >= max_mo:
                break

            # ── Orbital header line ──────────────────────────────────────────
            if 'eigenvalue=' in line:
                # Flush previous orbital if complete
                if pending_header is not None:
                    ev, nsaos = pending_header
                    if len(coeffs) == nsaos:
                        mo_energies.append(ev)
                        all_coeffs.append(coeffs)
                    coeffs = []
                    pending_header = None

                ev_m    = re.search(r'eigenvalue=([-+]?\d*\.?\d+D[+-]\d+)', line)
                nsao_m  = re.search(r'nsaos=(\d+)', line)
                if ev_m and nsao_m:
                    ev    = float(ev_m.group(1).replace('D', 'e'))
                    nsaos = int(nsao_m.group(1))
                    if n_ao is None:
                        n_ao = nsaos
                    pending_header = (ev, nsaos)
                continue

            # ── Coefficient line ─────────────────────────────────────────────
            if pending_header is not None:
                stripped = line.strip()
                if not stripped:
                    continue
                for j in range(0, min(80, len(stripped)), 20):
                    tok = stripped[j:j+20].strip()
                    if tok:
                        try:
                            coeffs.append(float(tok.replace('D', 'e').replace('d', 'e')))
                        except ValueError:
                            pass

        # Flush final orbital
        if pending_header is not None:
            ev, nsaos = pending_header
            if len(coeffs) == nsaos:
                mo_energies.append(ev)
                all_coeffs.append(coeffs)

    # Convert to numpy arrays
    mo_coeffs   = np.array(all_coeffs).T    # Shape: (n_ao, n_mo)
    mo_energies = np.array(mo_energies)

    return mo_coeffs, mo_energies, n_ao


def parse_basis_structure(basis_path: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    Parse a Turbomole basis file and extract the contracted shell structure
    per element.

    Returns:
        {element: [(l, n_contractions), ...]} sorted by ascending l.
        E.g. for Fe def2-SVP: {'fe': [(0,5),(1,3),(2,2),(3,1)]}
    """
    L_MAP = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5}
    result: Dict[str, List[Tuple[int, int]]] = {}

    with open(basis_path, 'r') as f:
        lines = f.readlines()

    current_element = None
    l_counts: Dict[int, int] = {}

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('$'):
            continue

        if stripped == '*':
            if current_element is not None:
                if l_counts:
                    shells = sorted(l_counts.items())
                    result[current_element] = shells
                    current_element = None
                    l_counts = {}
            continue

        if stripped.startswith('#'):
            continue

        if current_element is None and not stripped[0].isdigit():
            parts = stripped.split()
            if parts:
                current_element = parts[0].capitalize()
                l_counts = {}
            continue

        if current_element is not None:
            parts = stripped.split()
            if len(parts) == 2 and parts[0].isdigit() and parts[1].lower() in L_MAP:
                l = L_MAP[parts[1].lower()]
                l_counts[l] = l_counts.get(l, 0) + 1

    if current_element is not None and l_counts:
        shells = sorted(l_counts.items())
        result[current_element] = shells

    return result


def parse_ecp_ncore(control_path: str) -> Dict[str, int]:
    """
    Parse the $ecp section of a Turbomole control file.

    Returns:
        {element_capitalized: ncore} e.g. {'Ru': 28}
    """
    result: Dict[str, int] = {}
    try:
        with open(control_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        return result

    ecp_start = content.find('$ecp')
    if ecp_start == -1:
        return result

    ecp_section = content[ecp_start:]
    next_section = ecp_section.find('\n$', 1)
    if next_section != -1:
        ecp_section = ecp_section[:next_section]

    current_element = None
    for line in ecp_section.split('\n'):
        stripped = line.strip()
        if not stripped or stripped.startswith('$ecp') or stripped.startswith('#') or stripped == '*':
            continue
        if 'ncore' in stripped.lower():
            m = re.search(r'ncore\s*=\s*(\d+)', stripped, re.IGNORECASE)
            if m and current_element:
                result[current_element] = int(m.group(1))
            continue
        parts = stripped.split()
        if len(parts) >= 2 and not parts[0][0].isdigit():
            current_element = parts[0].capitalize()

    return result


def find_basis_file(work_dir: str = '.') -> Optional[str]:
    """Find the Turbomole basis file, checking control for $basis file= directive."""
    control_path = Path(work_dir) / 'control'
    if control_path.exists():
        with open(control_path, 'r') as f:
            for line in f:
                if '$basis' in line and 'file=' in line:
                    m = re.search(r'file=(\S+)', line)
                    if m:
                        p = Path(work_dir) / m.group(1)
                        if p.exists():
                            return str(p)
    basis_path = Path(work_dir) / 'basis'
    if basis_path.exists():
        return str(basis_path)
    return None


def build_ao_basis_def2svp(atoms: List[Tuple[str, int]],
                           h_polarized: bool = True) -> Tuple[List[AOBasisFunction], List[int]]:
    """
    Build AO basis function list for def2-SVP or def2-SV(P) basis set.

    Args:
        atoms:        List of (element_symbol, atom_index) tuples
        h_polarized:  If True (default), use def2-SVP: H has [2s1p] = 5 AOs.
                      If False, use def2-SV(P): H has [2s] = 2 AOs only.
                      Also affects Fe: SVP has [5s3p2d1f]=31 AOs; SV(P) has [5s3p2d]=24 AOs.

    Returns:
        ao_basis_list: List of AOBasisFunction objects
        ao_to_atom_map: List mapping AO index to atom index
    """
    ao_basis_list = []
    ao_to_atom_map = []

    # def2-SVP basis set structure (spherical harmonics)
    # Format: list of (angular_momentum, n_contractions) tuples.
    # Total AOs per atom = sum over shells of n_contractions × (2l+1).
    # Verified against Turbomole basen/ files and build_ao_labels_from_atoms in HERMES.
    #
    # Period 1:  H, He  → [2s1p]             5 AOs  (def2-SVP)
    #                   → [2s]               2 AOs  (def2-SV(P), h_polarized=False)
    # Period 2:  Li, Be → [3s2p] (no d!)      9 AOs
    #            B–Ne   → [3s2p1d]            14 AOs
    # Period 3:  Na     → [4s2p1d] (only 2p!) 15 AOs
    #            Mg–Ar  → [4s3p1d]            18 AOs
    # Period 4:  K, Ca  → [5s3p2d]            24 AOs
    #            Sc–Zn  → [5s3p2d1f] (3d TM)  31 AOs  (def2-SVP)
    #                   → [5s3p2d]            24 AOs  (def2-SV(P), h_polarized=False)
    #            Ga–Kr  → [5s4p3d]            32 AOs

    h_structure  = [(0, 2), (1, 1)] if h_polarized else [(0, 2)]   # 5 or 2 AOs
    fe_structure = [(0, 5), (1, 3), (2, 2), (3, 1)] if h_polarized else [(0, 5), (1, 3), (2, 2)]

    basis_structure = {
        # Period 1
        'H':  h_structure,                                          #  5 or 2 AOs
        'He': [(0, 2), (1, 1)],                                    #  5 AOs [2s1p]
        # Period 2 — Li/Be have NO d polarization
        'Li': [(0, 3), (1, 2)],                    #  9 AOs [3s2p]
        'Be': [(0, 3), (1, 2)],                    #  9 AOs [3s2p]
        # Period 2, B–Ne: [3s2p1d]
        'B':  [(0, 3), (1, 2), (2, 1)],            # 14 AOs
        'C':  [(0, 3), (1, 2), (2, 1)],            # 14 AOs
        'N':  [(0, 3), (1, 2), (2, 1)],            # 14 AOs
        'O':  [(0, 3), (1, 2), (2, 1)],            # 14 AOs
        'F':  [(0, 3), (1, 2), (2, 1)],            # 14 AOs
        'Ne': [(0, 3), (1, 2), (2, 1)],            # 14 AOs
        # Period 3 — Na exception (only 2 p-sets)
        'Na': [(0, 4), (1, 2), (2, 1)],            # 15 AOs [4s2p1d]
        # Period 3, Mg–Ar: [4s3p1d]
        'Mg': [(0, 4), (1, 3), (2, 1)],            # 18 AOs
        'Al': [(0, 4), (1, 3), (2, 1)],            # 18 AOs
        'Si': [(0, 4), (1, 3), (2, 1)],            # 18 AOs
        'P':  [(0, 4), (1, 3), (2, 1)],            # 18 AOs
        'S':  [(0, 4), (1, 3), (2, 1)],            # 18 AOs
        'Cl': [(0, 4), (1, 3), (2, 1)],            # 18 AOs
        'Ar': [(0, 4), (1, 3), (2, 1)],            # 18 AOs
        # Period 4, 3d transition metals (Sc–Zn): [5s3p2d1f]
        'Sc': [(0, 5), (1, 3), (2, 2), (3, 1)],   # 31 AOs
        'Ti': [(0, 5), (1, 3), (2, 2), (3, 1)],   # 31 AOs
        'V':  [(0, 5), (1, 3), (2, 2), (3, 1)],   # 31 AOs
        'Cr': [(0, 5), (1, 3), (2, 2), (3, 1)],   # 31 AOs
        'Mn': [(0, 5), (1, 3), (2, 2), (3, 1)],   # 31 AOs
        'Fe': fe_structure,                        # 31 (SVP) or 24 (SV(P)) AOs
        'Co': [(0, 5), (1, 3), (2, 2), (3, 1)],   # 31 AOs
        'Ni': [(0, 5), (1, 3), (2, 2), (3, 1)],   # 31 AOs
        'Cu': [(0, 5), (1, 3), (2, 2), (3, 1)],   # 31 AOs
        'Zn': [(0, 5), (1, 3), (2, 2), (3, 1)],   # 31 AOs
    }

    # Turbomole stores functions within each shell in this m_l order.
    # This MUST match the actual ordering in the mos / alpha / beta files,
    # which is the same ordering used by build_ao_labels_from_atoms in HERMES:
    #   P_ORBS = ['px', 'py', 'pz']  → m = +1, -1, 0
    #   D_ORBS: m = 0, +1, -1, -2, +2  → dz2, dxz, dyz, dxy, dx2y2
    #   F_ORBS = ['f+0','f+1','f-1','f+2','f-2','f+3','f-3'] → m = 0,1,-1,2,-2,3,-3
    #
    # The d order was verified against NBO7 Fe 3d occupancies on two systems whose
    # degeneracy patterns differ, which pins all five positions: ferrocene fixes
    # m=0, -1, +2 and porphyrinFe2 fixes m=-2, +2. It matches Gaussian/Molden
    # (0, +1, -1, +2, -2) except that the |m|=2 pair is written in the opposite
    # order. The previous table permuted dz2/dxy/dx2y2, which left the d-shell
    # total correct but mislabelled the components -- invisible to any shell-summed
    # quantity, fatal to per-component ones (plane selection, axis-projected
    # sigma/pi/delta).
    TURBO_M_ORDER = {
        0: [0],
        1: [1, -1, 0],              # px, py, pz
        2: [0, 1, -1, -2, 2],      # dz2, dxz, dyz, dxy, dx2y2
        3: [0, 1, -1, 2, -2, 3, -3],  # f (7 real spherical harmonics)
    }

    for element, atom_idx in atoms:
        if element not in basis_structure:
            raise ValueError(f"Element '{element}' not in def2-SVP basis table. "
                             f"Add it to build_ao_basis_def2svp() in compute_nao_from_turbomole.py.")

        for angular_momentum, n_contractions in basis_structure[element]:
            m_values = TURBO_M_ORDER.get(
                angular_momentum,
                list(range(-angular_momentum, angular_momentum + 1))
            )
            for contraction_idx in range(n_contractions):
                for m in m_values:
                    ao_func = AOBasisFunction(
                        atom_idx=atom_idx,
                        element=element,
                        angular_momentum=angular_momentum,
                        magnetic_quantum=m,
                        contraction_idx=contraction_idx
                    )
                    ao_basis_list.append(ao_func)
                    ao_to_atom_map.append(atom_idx)
    
    return ao_basis_list, ao_to_atom_map


def build_ao_basis_turbomole(atoms: List[Tuple[str, int]],
                             basis_path: Optional[str] = None,
                             h_polarized: bool = True) -> Tuple[List[AOBasisFunction], List[int]]:
    """
    Build AO basis function list, parsing from Turbomole basis file when available.

    Tries to parse the basis file for shell structure. Falls back to the
    hardcoded def2-SVP/SV(P) table for elements it covers.
    """
    # d order: see the note in build_ao_basis_def2svp(); verified against NBO7.
    TURBO_M_ORDER = {
        0: [0],
        1: [1, -1, 0],
        2: [0, 1, -1, -2, 2],      # dz2, dxz, dyz, dxy, dx2y2
        3: [0, 1, -1, 2, -2, 3, -3],
    }

    parsed_basis = {}
    if basis_path:
        try:
            parsed_basis = parse_basis_structure(basis_path)
        except Exception:
            pass

    elements_needed = {elem for elem, _ in atoms}
    all_in_parsed = all(elem.capitalize() in parsed_basis or elem in parsed_basis
                        for elem in elements_needed)

    if not all_in_parsed:
        try:
            return build_ao_basis_def2svp(atoms, h_polarized=h_polarized)
        except ValueError:
            pass
        if not parsed_basis:
            raise ValueError(
                f"Cannot determine basis structure: no basis file available and "
                f"elements {elements_needed - set(parsed_basis.keys())} not in "
                f"hardcoded def2-SVP table. Provide a Turbomole basis file.")

    ao_basis_list = []
    ao_to_atom_map = []

    for element, atom_idx in atoms:
        key = element.capitalize() if element.capitalize() in parsed_basis else element
        if key in parsed_basis:
            structure = parsed_basis[key]
        else:
            raise ValueError(
                f"Element '{element}' not found in basis file or hardcoded table.")

        for angular_momentum, n_contractions in structure:
            m_values = TURBO_M_ORDER.get(
                angular_momentum,
                list(range(-angular_momentum, angular_momentum + 1))
            )
            for contraction_idx in range(n_contractions):
                for m in m_values:
                    ao_func = AOBasisFunction(
                        atom_idx=atom_idx,
                        element=element,
                        angular_momentum=angular_momentum,
                        magnetic_quantum=m,
                        contraction_idx=contraction_idx
                    )
                    ao_basis_list.append(ao_func)
                    ao_to_atom_map.append(atom_idx)

    return ao_basis_list, ao_to_atom_map


def compute_overlap_matrix_identity(n_ao: int) -> np.ndarray:
    """
    Use identity matrix as overlap approximation.
    
    NOTE: This is an approximation! For accurate NAOs, we need the real
    overlap matrix. However, for initial testing and when the basis is
    nearly orthogonal, this can provide reasonable results.
    
    Args:
        n_ao: Number of AO basis functions
    
    Returns:
        Identity matrix of size (n_ao, n_ao)
    """
    print("WARNING: Using identity matrix for overlap. Results will be approximate!")
    print("For accurate NAOs, the real overlap matrix should be extracted from Turbomole.")
    return np.eye(n_ao)


def extract_overlap_from_turbomole() -> Optional[np.ndarray]:
    """
    Extract overlap matrix from Turbomole using aoforce tool.
    
    aoforce generates the overlap matrix and writes it to a file.
    We run it with minimal options just to get the overlap.
    
    Returns:
        Overlap matrix or None if extraction fails
    """
    import subprocess
    import os
    from pathlib import Path
    
    print("   Attempting to extract overlap matrix from Turbomole...")
    
    try:
        # Run aoforce with input to skip frequency calculation
        # We just want it to write the overlap matrix
        result = subprocess.run(
            ['aoforce'],
            input='\n',  # Just press enter to defaults
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # aoforce may write overlap to various files
        # Try to find overlap in output files
        
        # Check for overlap in aoforce output
        overlap_files = ['overlap', 'sao', 'aoforce.out']
        
        for fname in overlap_files:
            if Path(fname).exists():
                print(f"   Found {fname}, attempting to parse...")
                # Try to parse - this is format-dependent
                # For now, return None and use identity
                pass
        
        print("   ⚠ Overlap matrix extraction not yet implemented")
        print("   Falling back to identity approximation")
        return None
        
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"   ⚠ Could not run aoforce: {e}")
        return None


def compute_nao_from_density(density_matrix: np.ndarray,
                             overlap_matrix: np.ndarray,
                             ao_basis_list: List,
                             return_tiers: bool = False):
    """See below. With return_tiers=True an extra int array is returned:
    0 = core, 1 = valence (NMB minus core), 2 = Rydberg (NRB). This is the
    classification the orthogonalisation already uses internally; exposing it
    lets downstream code select valence NAOs by tier instead of by thresholding
    a raw occupancy, which is not a spin- or convention-independent criterion."""
    """
    Compute Natural Atomic Orbitals from density matrix — NBO7-consistent algorithm.

    Matches the Reed-Curtiss-Weinhold (1988) / NBO7 NAO procedure:

      Step 1 — Intra-atomic Löwdin orthogonalization.
               For each atom A, compute W_A = S_AA^{-1/2} from the diagonal
               block of the overlap matrix. Build block-diagonal W =
               diag(W_A, W_B, …).  Transform the density:
               P_partial = W P W  (P in intra-atomic OAO basis).

      Step 2 — Per-(atom, l, m_l) block diagonalization.
               Diagonalise each shell block of P_partial → pre-NAOs in the
               intra-atomic OAO basis, sorted by occupancy (descending) and
               phase-normalised (largest |coefficient| positive).

      Step 3 — Back-transform pre-NAOs to AO basis.
               nao_ao_pre = W^{-1} nao_partial = diag(S_AA^{+1/2}) nao_partial.
               Pre-NAOs from different atoms are still non-orthogonal.

      Step 4 — Three-step inter-atomic Löwdin orthogonalization.
               Implements the NBO7 Jan-2017 modified algorithm: core shells
               are orthogonalised first, then valence, then Rydberg (NRB).
               Each tier is projected against all previously completed tiers
               before its own symmetric Löwdin step.

               4a. Symmetric Löwdin on CORE pre-NAOs (innermost NMB shells,
                   identified via element-specific _CORE counts).
               4b. Project core out of VALENCE pre-NAOs; symmetric Löwdin
                   on the residual (outermost NMB = chemical valence shells).
               4c. Project core+valence out of NRB pre-NAOs; symmetric Löwdin
                   on the residual Rydberg manifold.

    The result satisfies  nao_ao^T S nao_ao = I  (orthonormal in AO metric).

    Output column i of nao_coeffs_ao corresponds to ao_basis_list[i].

    Args:
        density_matrix: AO density matrix P  (n_ao × n_ao)
        overlap_matrix: AO overlap matrix S  (n_ao × n_ao)
        ao_basis_list:  list of AOBasisFunction objects (atom_idx, l, m per AO)

    Returns:
        nao_coeffs_ao:   NAO coefficient matrix in AO basis  (n_ao × n_ao),
                         orthonormal in S-metric; column i ↔ ao_basis_list[i]
        nao_occupancies: pre-NAO occupancies  (n_ao,), index i ↔ ao_basis_list[i]
    """
    from collections import defaultdict

    n_ao = len(ao_basis_list)
    S    = overlap_matrix

    # ── NMB count per (element, |l|): number of NMB NAOs per m-value ────────
    # Minimal-basis occupancy: 1s for H; 1s(core)+2s(val)+2p(val) for first-row;
    # includes 3d for first-row TMs.  Extend as needed for new elements.
    # NOTE 2026-09-11: the transition-metal entries previously carried one more p
    # shell than NBO7 does, which made the (n)p a VALENCE function. NBO7 places it in
    # the Rydberg manifold -- verified against our own reference outputs, where Fe 4p
    # is labelled Ryd( 4p) at 0.001-0.011 e. Ours reported it valence at ~0.22 e, the
    # "78x anomaly" of the metals section. Corrected; the comment below always
    # described the NBO7 convention correctly, the table did not implement it.
    #
    # NOTE 2026-09-15: the ECP entries for Y-Cd and Hf-Hg used to be written into the
    # module table from inside this function, through an alias. They therefore existed
    # only once this function had been called, and derive_nao_tiers -- which Stage 2
    # calls WITHOUT ever calling this one -- never saw them. They are defined at module
    # scope now, so both paths agree at import.
    _NMB = _NMB_TABLE

    # ── CORE count per (element, |l|): innermost NMB shells classified as core
    # NBO7 Jan-2017 modified algorithm orthogonalizes core → valence → Rydberg.
    # Core = all NMB shells below the chemical valence shell:
    #   H, He : no core (1s is valence)
    #   Li–Ne : 1s is core        → {0: 1}
    #   Na–Ar : 1s,2s,2p are core → {0: 2, 1: 1}
    #   3d TMs: 1s,2s,3s,2p,3p are core; 3d and 4s are valence → {0: 3, 1: 2}
    #   Br    : [Ar]+3d are core  → {0: 3, 1: 2, 2: 1}
    #   I     : [Kr]+4d are core  → {0: 4, 1: 3, 2: 2}
    #   4d TMs (ECP ncore=28): 4s,4p are semi-core → {0: 1, 1: 1}
    #   5d TMs (ECP ncore=60): 5s,5p are semi-core → {0: 1, 1: 1}
    _CORE = _CORE_TABLE

    # ── Step 1: intra-atomic Löwdin W = diag(S_AA^{-1/2}) ──────────────────
    atom_ao_groups: dict = defaultdict(list)
    for i, ao in enumerate(ao_basis_list):
        atom_ao_groups[ao.atom_idx].append(i)

    W     = np.zeros((n_ao, n_ao))   # block-diag S_AA^{-1/2}
    W_inv = np.zeros((n_ao, n_ao))   # block-diag S_AA^{+1/2}

    for indices in atom_ao_groups.values():
        idx  = np.array(indices)
        S_AA = S[np.ix_(idx, idx)]
        evals, evecs = np.linalg.eigh(S_AA)
        evals = np.maximum(evals, 1e-15)
        W    [np.ix_(idx, idx)] = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
        W_inv[np.ix_(idx, idx)] = evecs @ np.diag(      np.sqrt(evals)) @ evecs.T

    # Occupation matrix Occ = S D S (Reed 1985, Eq. A6).
    # In an orthonormal basis, occupation and bond-order matrices coincide.
    # Using the FULL overlap (including inter-atomic blocks) is essential:
    # Occ_AA = (S D S)_AA ≠ S_AA D_AA S_AA because of inter-atomic coupling.
    Occ = S @ density_matrix @ S

    # Occ in intra-atomic OAO (Löwdin) basis: W Occ W = S^{-1/2} Occ S^{-1/2}
    # Within each atom block this equals S_AA^{-1/2} Occ_AA S_AA^{-1/2}.
    Occ_partial = W @ Occ @ W

    # ── Step 2: per-(atom, l) block diagonalization with m-averaging ───────
    # Reed 1985, Eq. A11-A12: symmetry-average the occupation matrix over the
    # (2l+1) magnetic quantum numbers before diagonalising.  The same
    # eigenvectors are then applied to every m_l component.
    shell_groups: dict = defaultdict(list)
    for i, ao in enumerate(ao_basis_list):
        key = (ao.atom_idx, ao.angular_momentum, ao.magnetic_quantum)
        shell_groups[key].append(i)

    # Build (atom, l) → list of (m, [global_indices]) for symmetry averaging
    al_groups: dict = defaultdict(list)
    for (atom_idx, ang_mom, mag_q), ao_indices in shell_groups.items():
        al_groups[(atom_idx, abs(ang_mom))].append((mag_q, ao_indices))

    nao_partial     = np.zeros((n_ao, n_ao))
    nao_occupancies = np.zeros(n_ao)

    for (atom_idx, l_val), m_list in al_groups.items():
        m_list_sorted = sorted(m_list, key=lambda x: x[0])
        n_m = len(m_list_sorted)
        n_shell = len(m_list_sorted[0][1])  # number of radial functions

        if n_shell == 1:
            # Single radial function per m: identity transformation
            for m_val, ao_indices in m_list_sorted:
                i = ao_indices[0]
                nao_partial[i, i]   = 1.0
                nao_occupancies[i]  = Occ_partial[i, i]
        else:
            # Symmetry-average Occ_partial over m values (Reed 1985, Eq. A11)
            idx_first = np.array(m_list_sorted[0][1])
            Occ_avg = np.zeros((n_shell, n_shell))
            for m_val, ao_indices in m_list_sorted:
                idx = np.array(ao_indices)
                Occ_avg += Occ_partial[np.ix_(idx, idx)]
            Occ_avg /= n_m

            eigvals, eigvecs = np.linalg.eigh(Occ_avg)

            # Sort descending (valence → Rydberg).
            order   = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

            # Phase convention: most-diffuse AO (last in the shell group) is
            # positive in the AO basis (after W back-transform).
            # Back-transform is now W (= S^{-1/2}), not W_inv.
            W_sub = W[np.ix_(idx_first, idx_first)]
            for col in range(n_shell):
                ao_vec = W_sub @ eigvecs[:, col]
                if ao_vec[-1] < 0:
                    eigvecs[:, col] *= -1

            # Apply the SAME eigenvectors to ALL m components
            for m_val, ao_indices in m_list_sorted:
                idx = np.array(ao_indices)
                # Per-m occupancies (using m-specific Occ block, shared eigvecs)
                Occ_m = Occ_partial[np.ix_(idx, idx)]
                m_eigvals = np.diag(eigvecs.T @ Occ_m @ eigvecs)

                for col, ao_out in enumerate(ao_indices):
                    nao_vec               = np.zeros(n_ao)
                    nao_vec[idx]          = eigvecs[:, col]
                    nao_partial[:, ao_out] = nao_vec
                    nao_occupancies[ao_out] = m_eigvals[col]

    # ── Step 3: back-transform pre-NAOs to AO basis ─────────────────────────
    # nao_partial lives in the intra-OAO basis; W = diag(S_AA^{-1/2})
    # gives S-orthonormal pre-NAOs: (W u)^T S (W u) = u^T u = I.
    nao_ao_pre = W @ nao_partial

    # Record the phase convention from Step 2 as it stands in the AO basis
    # AFTER the W back-transform.  NBO7 does NOT re-normalise after
    # inter-atomic Löwdin; the sign set in the pre-NAO step is preserved.
    # We store +1 or -1 for each NAO column.
    pre_nao_signs = np.empty(n_ao)
    for i in range(n_ao):
        mr = np.argmax(np.abs(nao_ao_pre[:, i]))
        pre_nao_signs[i] = 1.0 if nao_ao_pre[mr, i] >= 0.0 else -1.0

    # ── Step 4: three-step inter-atomic Löwdin (NBO7 Jan-2017 algorithm) ────
    # Split NMB into core (innermost filled shells) and valence, then
    # orthogonalise in the order: core → valence → Rydberg (NRB).
    # Each group is projected against all previously orthogonalised groups
    # before its own symmetric Löwdin step.
    core_set: set = set()
    val_set:  set = set()
    for (atom_idx, l, m), ao_indices in shell_groups.items():
        elem       = ao_basis_list[ao_indices[0]].element
        _require_nmb(elem)
        nmb_count  = _NMB.get(_sym(elem),  {}).get(abs(l), 0)
        core_count = _CORE.get(_sym(elem), {}).get(abs(l), 0)
        val_count  = nmb_count - core_count
        if nmb_count == 0:
            continue
        # Sort by occupancy descending: highest-occ = most core-like
        sorted_by_occ = sorted(ao_indices, key=lambda i: -nao_occupancies[i])
        core_set.update(sorted_by_occ[:core_count])
        val_set.update(sorted_by_occ[core_count : core_count + val_count])

    core_idx = np.array(sorted(core_set), dtype=int)
    val_idx  = np.array(sorted(val_set),  dtype=int)
    nrb_idx  = np.array(
        [i for i in range(n_ao) if i not in core_set and i not in val_set],
        dtype=int
    )

    # Tier labels, in the same order as the NAO columns: 0 core, 1 valence, 2 Rydberg.
    nao_tiers = np.full(n_ao, 2, dtype=np.int8)
    nao_tiers[core_idx] = 0
    nao_tiers[val_idx]  = 1

    # ── Symmetry-averaged occupancy weights (Reed 1985, Eq. A11) ─────────
    # Average occupancy over the (2l+1) m_l values for each (atom, l, RANK).
    # Rank = position within the sorted eigenvectors of each (atom, l, m_l)
    # block: rank 0 = highest-occ (e.g. Val 2p), rank 1 = next (e.g. Ryd 3p).
    # This ensures rotational invariance: all m_l components of the same
    # physical subshell get the same weight in the OWSO.
    sym_avg_occ = np.zeros(n_ao)
    # Build rank map: for each (atom, l, m_l), ao_indices are already ordered
    # by descending occupancy (from Step 2 sorting). rank = position in list.
    rank_groups: dict = defaultdict(list)  # (atom_idx, l, rank) -> [ao_idx]
    for (atom_idx, ang_mom, mag_q), ao_indices in shell_groups.items():
        l = abs(ang_mom)
        for rank, ao_idx in enumerate(ao_indices):
            rank_groups[(atom_idx, l, rank)].append(ao_idx)
    for (atom_idx, l, rank), indices in rank_groups.items():
        avg = np.mean(np.abs(nao_occupancies[indices]))
        for i in indices:
            sym_avg_occ[i] = avg

    import os
    _OWSO_MODE = os.environ.get('HERMES_NAO_OWSO', '1')
    # '0' = plain Löwdin, '1' = W(WSW)^{-1/2}, '2' = W^{1/2}(W^{1/2}σW^{1/2})^{-1/2}

    def _owso(mat: np.ndarray, occ: np.ndarray) -> np.ndarray:
        """Occupancy-Weighted Symmetric Orthogonalization (OWSO).

        Mode 1: O_w = W (W σ W)^{-1/2}  (Reed 1985, Eq. A16/A24)
        Mode 2: O_w = W^{1/2} (W^{1/2} σ W^{1/2})^{-1/2}  (alternative formula)
        Mode 0: plain Löwdin σ^{-1/2}

        Set env HERMES_NAO_OWSO=0/1/2
        """
        if mat.shape[1] == 0:
            return mat
        sigma = mat.T @ S @ mat

        if _OWSO_MODE == '0':
            ev, ec = np.linalg.eigh(sigma)
            ev = np.maximum(ev, 1e-15)
            X = ec @ np.diag(1.0 / np.sqrt(ev)) @ ec.T
            return mat @ X

        w = np.maximum(np.abs(occ), 1e-10)

        if _OWSO_MODE == '2':
            # Alternative formula: W^{1/2} (W^{1/2} σ W^{1/2})^{-1/2}
            w_sqrt = np.diag(np.sqrt(w))
            wsw = w_sqrt @ sigma @ w_sqrt
            ev, ec = np.linalg.eigh(wsw)
            ev = np.maximum(ev, 1e-15)
            wsw_inv_sqrt = ec @ np.diag(1.0 / np.sqrt(ev)) @ ec.T
            X = w_sqrt @ wsw_inv_sqrt
            return mat @ X

        # Mode 1 (default): W (WSW)^{-1/2}
        W_diag = np.diag(w)
        wsw = W_diag @ sigma @ W_diag
        ev, ec = np.linalg.eigh(wsw)
        ev = np.maximum(ev, 1e-15)
        wsw_inv_sqrt = ec @ np.diag(1.0 / np.sqrt(ev)) @ ec.T
        X = W_diag @ wsw_inv_sqrt
        return mat @ X

    def _project_out(nao_done: np.ndarray, pre: np.ndarray) -> np.ndarray:
        """Remove components of already-orthogonalised NAOs from pre-NAOs."""
        if nao_done.shape[1] == 0:
            return pre
        return pre - nao_done @ (nao_done.T @ S @ pre)

    # Use raw pre-NAO occupancies (not symmetry-averaged) if HERMES_NAO_RAW_WEIGHTS=1
    import os as _os
    _use_raw = _os.environ.get('HERMES_NAO_RAW_WEIGHTS', '0') == '1'
    _weights = np.abs(nao_occupancies) if _use_raw else sym_avg_occ

    # 4a: OWSO on CORE pre-NAOs
    nao_core = _owso(nao_ao_pre[:, core_idx], _weights[core_idx])

    # 4b: project core out of VALENCE pre-NAOs, then OWSO
    nao_val = _owso(_project_out(nao_core, nao_ao_pre[:, val_idx]),
                    _weights[val_idx])

    # 4c: project core+valence out of NRB, then OWSO
    # Rydbergs with occupancy below t_w are excluded from OWSO and
    # Schmidt+Löwdin orthogonalised separately (Reed 1985, p.745).
    nao_nmb  = np.hstack([nao_core, nao_val])      # all NMB NAOs assembled
    nrb_pre  = _project_out(nao_nmb, nao_ao_pre[:, nrb_idx])

    # 4c.1 (NBO7 Step 5): Rediag Rydberg density per (atom, l, m) block
    # After Schmidt projection, re-diagonalise the occupation matrix within
    # each (atom, l, m) Rydberg sub-block to restore natural ordering.
    # This determines the correct heavy/light Rydberg split for OWSO.
    Occ_ryd = nrb_pre.T @ S @ density_matrix @ S @ nrb_pre
    nrb_set_local = set(nrb_idx.tolist())
    nrb_g2l = {g: l for l, g in enumerate(nrb_idx)}
    for key, ao_indices in shell_groups.items():
        ryd_local = [nrb_g2l[i] for i in ao_indices if i in nrb_set_local]
        if len(ryd_local) <= 1:
            continue
        rl = np.array(ryd_local)
        P_sub = Occ_ryd[np.ix_(rl, rl)]
        ev_r, ec_r = np.linalg.eigh(P_sub)
        order_r = np.argsort(ev_r)[::-1]
        ev_r = ev_r[order_r]
        ec_r = ec_r[:, order_r]
        new_cols = nrb_pre[:, rl] @ ec_r
        for col in range(len(rl)):
            mr = np.argmax(np.abs(new_cols[:, col]))
            if new_cols[mr, col] < 0:
                new_cols[:, col] *= -1
        nrb_pre[:, rl] = new_cols

    # Recompute Rydberg weights after step 5 rediag
    Occ_ryd_new = nrb_pre.T @ S @ density_matrix @ S @ nrb_pre
    nrb_occ_new = np.diag(Occ_ryd_new)
    ryd_rank_groups: dict = defaultdict(list)
    for key, ao_indices in shell_groups.items():
        atom_idx_k, ang_k, mag_k = key
        l_k = abs(ang_k)
        ryd_local = [(nrb_g2l[i], i) for i in ao_indices if i in nrb_set_local]
        if not ryd_local:
            continue
        ryd_sorted = sorted(ryd_local, key=lambda x: -nrb_occ_new[x[0]])
        for rank, (li, gi) in enumerate(ryd_sorted):
            ryd_rank_groups[(atom_idx_k, l_k, rank)].append(li)
    nrb_weights = np.zeros(len(nrb_idx))
    for (a, l, r), idxs in ryd_rank_groups.items():
        avg = np.mean(np.abs(nrb_occ_new[idxs]))
        for i in idxs:
            nrb_weights[i] = avg

    t_w = 1e-4
    heavy_mask = nrb_weights >= t_w
    light_mask = ~heavy_mask
    if np.any(heavy_mask) and np.any(light_mask):
        # OWSO on heavy Rydbergs
        heavy_idx_local = np.where(heavy_mask)[0]
        light_idx_local = np.where(light_mask)[0]
        nao_nrb_heavy = _owso(nrb_pre[:, heavy_idx_local],
                              nrb_weights[heavy_idx_local])
        # Schmidt orthogonalise light Rydbergs against heavy, then Löwdin
        light_pre = _project_out(nao_nrb_heavy, nrb_pre[:, light_idx_local])
        sigma_light = light_pre.T @ S @ light_pre
        ev_l, ec_l = np.linalg.eigh(sigma_light)
        ev_l = np.maximum(ev_l, 1e-15)
        lowdin_light = ec_l @ np.diag(1.0 / np.sqrt(ev_l)) @ ec_l.T
        nao_nrb_light = light_pre @ lowdin_light
        # Reassemble in original order
        nao_nrb = np.zeros_like(nrb_pre)
        nao_nrb[:, heavy_idx_local] = nao_nrb_heavy
        nao_nrb[:, light_idx_local] = nao_nrb_light
    else:
        nao_nrb = _owso(nrb_pre, nrb_weights)

    # Assemble full NAO matrix (column i ↔ ao_basis_list[i])
    nao_coeffs_ao = np.zeros((n_ao, n_ao))
    if len(core_idx) > 0:
        nao_coeffs_ao[:, core_idx] = nao_core
    nao_coeffs_ao[:, val_idx]  = nao_val
    nao_coeffs_ao[:, nrb_idx]  = nao_nrb

    # ── Step 5: re-diagonalize density in NAO basis per (atom, l) block ─────
    # Reed 1985, Eq. 8 / Step 9 in the 3-tier procedure:
    # After inter-atomic orthogonalization, re-diagonalize the density matrix
    # within each (atom, l) block of the NAO basis.  This produces the final
    # NAO occupancies AND rotates the NAO vectors within each block.
    #
    # P_NAO = N^T S P S N  (not right — see below)
    # Actually: P_NAO_ij = <NAO_i | ρ̂ | NAO_j> = Σ_k c_{ik} c_{jk}
    # where c_{ik} = NAO_i^T S C_k (MO k in NAO basis).
    # Since we only have the density matrix P = C_occ C_occ^T,
    # P_NAO = (N^T S) P_AO (S N)... but that's not right either.
    #
    # For an orthonormal basis N (N^T S N = I):
    #   ρ̂ expressed as P_AO: ρ̂ = Σ_{μν} P_{μν} |χ_μ><χ_ν| S
    #   Not quite. The AO density matrix satisfies n_e = tr(S P).
    #   The 1-RDM in NAO basis: γ_{ij} = Σ_{μναβ} N_{μi} S_{μα} P_{αβ} S_{βν} N_{νj}
    #   = (N^T S P S N)_{ij}
    #   No — P_AO is the AO density: P_{μν} = Σ_k C_{μk} C_{νk} (occ MOs).
    #   γ_{ij} = Σ_k (N^T S C)_{ik} (N^T S C)_{jk}
    #   = (N^T S C_occ) (N^T S C_occ)^T
    #   = N^T S (C_occ C_occ^T) S N
    #   = N^T S P_AO S N
    #   But this requires P_AO = C_occ C_occ^T, and the input density_matrix IS this.
    #   So: P_NAO = N^T S P S N  where P = density_matrix.
    P_NAO = nao_coeffs_ao.T @ S @ density_matrix @ S @ nao_coeffs_ao

    # Build (atom, l, m) groups: all NAO column indices for each (atom, l, m)
    # Using (atom, l, m) rather than (atom, l) to preserve px/py/pz identity.
    atom_l_groups: dict = defaultdict(list)
    for i, ao in enumerate(ao_basis_list):
        atom_l_groups[(ao.atom_idx, ao.angular_momentum, ao.magnetic_quantum)].append(i)

    for key, block_indices in atom_l_groups.items():
        idx = np.array(block_indices)
        if len(idx) == 1:
            # Single NAO in block: occupancy is just the diagonal element
            nao_occupancies[idx[0]] = P_NAO[idx[0], idx[0]]
            continue

        # Extract density sub-block in NAO basis
        P_block = P_NAO[np.ix_(idx, idx)]
        eigvals, eigvecs = np.linalg.eigh(P_block)

        # Sort descending (highest occupancy first)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Phase convention: largest |coefficient| positive in AO basis
        # Apply rotation to NAO vectors: N_new = N_old @ U
        new_nao_cols = nao_coeffs_ao[:, idx] @ eigvecs

        for col in range(len(idx)):
            mr = np.argmax(np.abs(new_nao_cols[:, col]))
            if new_nao_cols[mr, col] < 0:
                new_nao_cols[:, col] *= -1

        # Update NAO matrix and occupancies
        nao_coeffs_ao[:, idx] = new_nao_cols
        for j, ao_out in enumerate(idx):
            nao_occupancies[ao_out] = eigvals[j]

    if return_tiers:
        return nao_coeffs_ao, nao_occupancies, nao_tiers
    return nao_coeffs_ao, nao_occupancies



def derive_nao_tiers(ao_basis_list, nao_occupancies):
    """Return the core/valence/Rydberg tier of every NAO: 0 core, 1 valence, 2 Rydberg.

    This is the same rule compute_nao_from_density applies internally, expressed so
    it can be recovered from an existing checkpoint (AO basis + occupancies) without
    rebuilding the NAO basis. Within each (atom, l, m_l) group the NAOs are ranked by
    occupancy; the innermost _CORE[elem][l] are core, the next
    _NMB[elem][l] - _CORE[elem][l] are valence, the rest are Rydberg.
    """
    from collections import defaultdict
    import numpy as _np
    shell_groups = defaultdict(list)
    for i, ao in enumerate(ao_basis_list):
        shell_groups[(ao.atom_idx, ao.angular_momentum, ao.magnetic_quantum)].append(i)
    tiers = _np.full(len(ao_basis_list), 2, dtype=_np.int8)
    for (atom_idx, l, m), idxs in shell_groups.items():
        elem = ao_basis_list[idxs[0]].element
        _require_nmb(elem)
        nmb  = _NMB_TABLE.get(_sym(elem), {}).get(abs(l), 0)
        core = _CORE_TABLE.get(_sym(elem), {}).get(abs(l), 0)
        if nmb == 0:
            continue
        order = sorted(idxs, key=lambda i: -nao_occupancies[i])
        for i in order[:core]:
            tiers[i] = 0
        for i in order[core:nmb]:
            tiers[i] = 1
    return tiers

def transform_mos_to_nao_basis(mo_coeffs_ao: np.ndarray,
                                nao_coeffs_ao: np.ndarray,
                                overlap_matrix: np.ndarray = None) -> np.ndarray:
    """
    Project MO coefficients onto the NAO basis.

    With NBO7-style NAOs (nao_coeffs_ao orthonormal in AO metric:
    nao_coeffs_ao.T @ S @ nao_coeffs_ao = I), the expansion coefficient of
    MO k in NAO r is the AO metric inner product:

        c_{r,k} = nao_ao_r^T @ S @ C_k

    When overlap_matrix is None, falls back to bare dot product (legacy).

    Args:
        mo_coeffs_ao:   MO coefficient matrix (n_ao × n_mo)
        nao_coeffs_ao:  NAO coefficient matrix (n_ao × n_nao)
        overlap_matrix: AO overlap matrix S (n_ao × n_ao); required for
                        NBO7-style NAOs

    Returns:
        mo_coeffs_nao: MO coefficients in NAO basis (n_nao × n_mo)
    """
    if overlap_matrix is not None:
        return nao_coeffs_ao.T @ (overlap_matrix @ mo_coeffs_ao)
    return nao_coeffs_ao.T @ mo_coeffs_ao


def compute_nao_coefficients_turbomole(alpha_file: str, beta_file: str,
                                        atoms: List[Tuple[str, int]],
                                        n_occ_alpha: int, n_occ_beta: int,
                                        overlap_matrix: Optional[np.ndarray] = None,
                                        max_mo: Optional[int] = None,
                                        nao_basis: str = "shared") -> Dict:
    """
    Compute NAO coefficients and transform MO coefficients to NAO basis.

    Args:
        alpha_file: Path to alpha MO file
        beta_file: Path to beta MO file
        atoms: List of (element, atom_index) tuples
        n_occ_alpha: Number of occupied alpha orbitals
        n_occ_beta: Number of occupied beta orbitals
        overlap_matrix: Optional AO overlap matrix (uses identity if None)
        max_mo: If given, read only this many MOs from file (for performance on
                large systems where only occupied + a few virtuals are needed).
                When set, overlap is approximated as identity since C is rectangular.

    Returns:
        Dictionary containing:
          - 'nao_basis': List of AOBasisFunction objects
          - 'nao_occupancies_alpha': Alpha NAO occupancies
          - 'nao_occupancies_beta': Beta NAO occupancies
          - 'mo_in_nao_alpha': Alpha MO coeffs in NAO basis (n_nao x n_mo)
          - 'mo_in_nao_beta': Beta MO coeffs in NAO basis (n_nao x n_mo)
          - 'mo_energies_alpha': Alpha MO energies
          - 'mo_energies_beta': Beta MO energies
    """
    print("\n" + "="*70)
    print("COMPUTING NATURAL ATOMIC ORBITALS FROM TURBOMOLE")
    print("="*70)

    # Step 1: Parse MO coefficients
    print("\n1. Parsing MO coefficients...")
    if max_mo is not None:
        print(f"   Reading only first {max_mo} MOs (max_mo={max_mo}) for performance")
    mo_coeffs_alpha, mo_energies_alpha, n_ao = parse_mo_coefficients_matrix(alpha_file, max_mo=max_mo)
    if beta_file is None:
        # Restricted calculation: alpha = beta
        mo_coeffs_beta, mo_energies_beta = mo_coeffs_alpha, mo_energies_alpha
        print(f"   ✓ Restricted calculation: using alpha MOs for both spins")
    else:
        mo_coeffs_beta, mo_energies_beta, _ = parse_mo_coefficients_matrix(beta_file, max_mo=max_mo)
    print(f"   ✓ Alpha: {mo_coeffs_alpha.shape[1]} orbitals, {n_occ_alpha} occupied")
    print(f"   ✓ Beta:  {mo_coeffs_beta.shape[1]} orbitals, {n_occ_beta} occupied")
    print(f"   ✓ AO basis size: {n_ao}")
    
    # Step 2: Build AO basis structure — try basis file, then hardcoded fallback
    print("\n2. Building AO basis structure...")
    basis_path = find_basis_file('.')
    if basis_path:
        print(f"   Found basis file: {basis_path}")

    ao_basis_list, ao_to_atom_map = build_ao_basis_turbomole(
        atoms, basis_path=basis_path, h_polarized=True)
    if len(ao_basis_list) != n_ao:
        ao_basis_list_alt, ao_to_atom_map_alt = build_ao_basis_turbomole(
            atoms, basis_path=basis_path, h_polarized=False)
        if len(ao_basis_list_alt) == n_ao:
            ao_basis_list, ao_to_atom_map = ao_basis_list_alt, ao_to_atom_map_alt
            print(f"   ✓ Auto-detected def2-SV(P) basis (H: 2s only, no p-polarization)")
        else:
            raise ValueError(
                f"Basis size mismatch: h_polarized=True gives {len(ao_basis_list)}, "
                f"h_polarized=False gives {len(ao_basis_list_alt)}, expected {n_ao}. "
                f"Check basis set or add element support."
            )
    else:
        print(f"   ✓ Basis matched ({len(ao_basis_list)} AOs)")
    print(f"   ✓ Generated {len(ao_basis_list)} AO basis functions")
    print(f"   ✓ Atoms: {len(set(ao_to_atom_map))}")
    
    # Step 3: Get overlap matrix
    print("\n3. Obtaining overlap matrix...")
    if overlap_matrix is None:
        overlap_matrix = extract_overlap_from_turbomole()
        if overlap_matrix is None:
            # Derive from MO orthonormality: MOs satisfy C.T @ S @ C = I.
            # When the mos/alpha/beta file contains ALL n_ao MOs (C is square, n_ao × n_ao),
            # S = inv(C @ C.T) exactly.  For UHF, S is spin-independent (it only depends on
            # the AO basis), so we try alpha first, then fall back to beta if alpha is non-square
            # (e.g. Turbomole occasionally writes n_ao-1 MOs to one spin file).
            def _try_recover_S(C, label):
                if C.shape[0] == C.shape[1]:
                    print(f"   Computing S from MO orthonormality ({label}): S = inv(C @ C.T) ...")
                    try:
                        S = np.linalg.inv(C @ C.T)
                        print(f"   ✓ Overlap matrix recovered from {label} ({n_ao}×{n_ao})")
                        return S
                    except np.linalg.LinAlgError:
                        print(f"   ⚠ Inversion failed for {label}")
                else:
                    print(f"   {label} MO matrix not square ({C.shape}), skipping")
                return None

            overlap_matrix = _try_recover_S(mo_coeffs_alpha, 'alpha')
            if overlap_matrix is None and mo_coeffs_beta is not mo_coeffs_alpha:
                overlap_matrix = _try_recover_S(mo_coeffs_beta, 'beta')
            if overlap_matrix is None:
                print(f"   Falling back to identity approximation")
                overlap_matrix = compute_overlap_matrix_identity(n_ao)
    else:
        print("   ✓ Using provided overlap matrix")
    
    # Step 4: Compute density matrices
    print("\n4. Computing density matrices from occupied orbitals...")
    C_occ_alpha = mo_coeffs_alpha[:, :n_occ_alpha]
    C_occ_beta = mo_coeffs_beta[:, :n_occ_beta]
    
    P_alpha = C_occ_alpha @ C_occ_alpha.T
    P_beta = C_occ_beta @ C_occ_beta.T
    
    print(f"   ✓ Alpha density matrix: {P_alpha.shape}")
    print(f"   ✓ Beta density matrix: {P_beta.shape}")
    
    # Step 5: Compute NAOs
    print("\n5. Computing Natural Atomic Orbitals...")
    if nao_basis not in ("shared", "per_spin"):
        raise ValueError(f"nao_basis must be 'shared' or 'per_spin', got {nao_basis!r}")
    if nao_basis == "shared":
        print("   Building ONE basis from P_total = P_alpha + P_beta (shared)")
        nao_coeffs_alpha, nao_occ_alpha, nao_tier_alpha = compute_nao_from_density(
            P_alpha + P_beta, overlap_matrix, ao_basis_list, return_tiers=True
        )
        nao_coeffs_beta, nao_occ_beta = nao_coeffs_alpha, nao_occ_alpha
        nao_tier_beta = nao_tier_alpha
    else:
        print("   Building SEPARATE bases from P_alpha and P_beta (per_spin)")
        nao_coeffs_alpha, nao_occ_alpha, nao_tier_alpha = compute_nao_from_density(
            P_alpha, overlap_matrix, ao_basis_list, return_tiers=True
        )
        nao_coeffs_beta, nao_occ_beta, nao_tier_beta = compute_nao_from_density(
            P_beta, overlap_matrix, ao_basis_list, return_tiers=True
        )
        print("   NOTE: alpha and beta are in DIFFERENT bases; a cross-spin "
              "comparison of G_rs is not basis-controlled.")
    
    print(f"   ✓ Alpha NAOs: {len(nao_occ_alpha)} orbitals")
    print(f"   ✓ Beta NAOs: {len(nao_occ_beta)} orbitals")
    print(f"   ✓ Alpha total population: {nao_occ_alpha.sum():.4f}")
    print(f"   ✓ Beta total population: {nao_occ_beta.sum():.4f}")
    
    # Step 6: Transform MO coefficients to NAO basis
    print("\n6. Transforming MO coefficients to NAO basis...")
    mo_in_nao_alpha = transform_mos_to_nao_basis(mo_coeffs_alpha, nao_coeffs_alpha, overlap_matrix)
    mo_in_nao_beta  = transform_mos_to_nao_basis(mo_coeffs_beta,  nao_coeffs_beta,  overlap_matrix)
    
    print(f"   ✓ Alpha MO→NAO: {mo_in_nao_alpha.shape}")
    print(f"   ✓ Beta MO→NAO: {mo_in_nao_beta.shape}")
    
    print("\n" + "="*70)
    print("NAO COMPUTATION COMPLETE")
    print("="*70 + "\n")
    
    return {
        'ao_basis': ao_basis_list,
        'nao_occupancies_alpha': nao_occ_alpha,
        'nao_occupancies_beta': nao_occ_beta,
        'mo_in_nao_alpha': mo_in_nao_alpha,
        'mo_in_nao_beta': mo_in_nao_beta,
        'mo_energies_alpha': mo_energies_alpha,
        'mo_energies_beta': mo_energies_beta,
        'nao_basis': nao_basis,
        'nao_tiers_alpha': nao_tier_alpha,
        'nao_tiers_beta': nao_tier_beta,
    }


def organize_nao_coefficients_by_atom(nao_result: Dict,
                                      selection: str = 'auto') -> Dict:
    """
    Organize NAO-based MO coefficients by atom and orbital type.
    
    This matches the format expected by compute_greens_function_turbomole.py
    
    Returns:
        {"alpha": {atom_num: {orbital_type: [coeff_list]}},
         "beta": {atom_num: {orbital_type: [coeff_list]}}}
    """
    ao_basis = nao_result['ao_basis']
    mo_in_nao_alpha = nao_result['mo_in_nao_alpha']
    mo_in_nao_beta = nao_result['mo_in_nao_beta']
    nao_occ_alpha = nao_result['nao_occupancies_alpha']
    nao_occ_beta = nao_result['nao_occupancies_beta']
    
    n_mo_alpha = mo_in_nao_alpha.shape[1]
    n_mo_beta = mo_in_nao_beta.shape[1]
    
    # Initialize structure
    coeffs_alpha = {}
    coeffs_beta = {}
    
    # Valence occupancy threshold (to distinguish valence from Rydberg)
    VAL_THRESHOLD = 0.1
    _tiers = nao_result.get('nao_tiers_alpha')
    if _tiers is None:
        try:
            _tiers = derive_nao_tiers(nao_result['ao_basis'],
                                      nao_result['nao_occupancies_alpha'])
        except Exception:
            _tiers = None
    _sel = selection
    if _sel == 'auto':
        _sel = 'tier' if _tiers is not None else 'occupancy'
    if _sel == 'tier' and _tiers is None:
        raise ValueError("selection='tier' requires NAO tiers, which could not be derived.")
    
    # Process each NAO
    for nao_idx, ao_func in enumerate(ao_basis):
        atom_num = ao_func.atom_idx + 1  # Convert to 1-indexed
        orb_type = ao_func.get_orbital_type()
        
        # Only include valence s, p, and d orbitals
        if orb_type not in ['s', 'px', 'py', 'pz', 'dxy', 'dxz', 'dyz', 'dx2y2', 'dz2']:
            continue
        
        # Check if this is a valence NAO (significant occupancy)
        occ_alpha = nao_occ_alpha[nao_idx]
        occ_beta = nao_occ_beta[nao_idx]
        
        if _sel == 'tier':
            if _tiers[nao_idx] != 1:          # 0 core, 1 valence, 2 Rydberg
                continue
        elif occ_alpha < VAL_THRESHOLD and occ_beta < VAL_THRESHOLD:
            continue  # Skip Rydberg orbitals
        
        # Initialize atom in dictionary if needed
        if atom_num not in coeffs_alpha:
            coeffs_alpha[atom_num] = {}
        if atom_num not in coeffs_beta:
            coeffs_beta[atom_num] = {}
        
        # Initialize orbital type if needed
        if orb_type not in coeffs_alpha[atom_num]:
            coeffs_alpha[atom_num][orb_type] = [0.0] * n_mo_alpha
        if orb_type not in coeffs_beta[atom_num]:
            coeffs_beta[atom_num][orb_type] = [0.0] * n_mo_beta
        
        # Add NAO contributions to this orbital type
        # (sum over multiple NAOs of same type if present)
        for mo_idx in range(n_mo_alpha):
            coeffs_alpha[atom_num][orb_type][mo_idx] += mo_in_nao_alpha[nao_idx, mo_idx]
        
        for mo_idx in range(n_mo_beta):
            coeffs_beta[atom_num][orb_type][mo_idx] += mo_in_nao_beta[nao_idx, mo_idx]
    
    return {"alpha": coeffs_alpha, "beta": coeffs_beta}


if __name__ == "__main__":
    print("""
================================================================================
NAO COMPUTATION MODULE
================================================================================

This module computes Natural Atomic Orbitals from Turbomole output files.

Usage:
    from compute_nao_from_turbomole import compute_nao_coefficients_turbomole
    
    # Define atoms (element, 0-indexed atom number)
    atoms = [('Fe', 0), ('C', 1), ('C', 2), ...]
    
    # Compute NAOs
    nao_result = compute_nao_coefficients_turbomole(
        alpha_file='alpha',
        beta_file='beta',
        atoms=atoms,
        n_occ_alpha=47,
        n_occ_beta=46
    )
    
    # Organize for Green's function calculation
    coefficients = organize_nao_coefficients_by_atom(nao_result)

================================================================================
    """)
