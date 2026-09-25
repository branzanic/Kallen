#!/usr/bin/env python3
"""
HERMES G_rs from Checkpoint — Stage 2 of the two-stage HERMES pipeline.

Loads a pre-computed NAO checkpoint (from hermes_nao_save.py) and computes G_rs
for any MO window, plane, orbital family, and threshold — without re-reading
the multi-GB MO files or recomputing NAOs.

Usage:
    # Interactive (prompts for all settings):
    python hermes_grs_from_checkpoint.py

    # Non-interactive with full control:
    python hermes_grs_from_checkpoint.py \\
        --checkpoint hermes_nao_checkpoint.npz \\
        --window HOMO-10:LUMO+10 \\
        --planes XY YZ XZ \\
        --family pd \\
        --spin both \\
        --threshold 0.01 \\
        --output-prefix turbo_turbo

    # Occupation-aware mode (skip non-aufbau empty MOs):
    python hermes_grs_from_checkpoint.py --occupation-aware

A. M. V. Branzanic — 2026
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# HERMES imports
_hermes_dir = Path(__file__).resolve().parent
if str(_hermes_dir) not in sys.path:
    sys.path.insert(0, str(_hermes_dir))

from compute_nao_from_turbomole import AOBasisFunction

# Plane-to-orbital mapping (same as compute_greens_function_turbomole.py)
PLANE_ORBITAL_MAP = {
    'XY': {'s': ['s'], 'p': ['pz'], 'd': ['dxz', 'dyz', 'dz2']},
    'YZ': {'s': ['s'], 'p': ['px'], 'd': ['dxy', 'dxz', 'dx2y2']},
    'XZ': {'s': ['s'], 'p': ['py'], 'd': ['dxy', 'dyz', 'dx2y2']},
    'SHELL_SUM': {'s': ['s'], 'p': ['px', 'py', 'pz'], 'd': ['dxy', 'dxz', 'dyz', 'dx2y2', 'dz2']},
}
# SHELL_SUM is Sum_type |G_rs| -- the absolute value is taken per angular component
# BEFORE summing. That makes it robust to NAO sign/phase conventions, which is
# why it serves as a cross-framework checksum, and it simultaneously destroys
# rotation invariance: |x|+|y|+|z| is an L1 norm, not a trace. Validation
# quantity only. See seeds/seed_05_known_issues.md.
SHELL_SUM_WARNING = (
    "\n  !!  SHELL_SUM selected: this is \u03a3_type |G_rs|, a phase-robust VALIDATION CHECKSUM.\n"
    "      It is NOT rotation-invariant (|x|+|y|+|z| is an L1 norm, not a trace)\n"
    "      and must NOT be used as physical coupling data. For an orientation-free\n"
    "      measure use the axis-projected \u03c3/\u03c0/\u03b4 mode:\n"
    "          hermes_grs_from_checkpoint.py --decompose axis\n"
)

HA_TO_EV = 27.211386

CHANNEL_LABELS = {'s': 'σ(s)', 'p': 'π(p)', 'd': 'δ(d)'}


def parse_qpenergies(filepath: str) -> np.ndarray:
    """Parse Turbomole qpenergies.dat → array of QP energies in Hartree.

    The file has columns: eps(eV)  QP-eps(eV)  Sigma_c  Sigma_x  Vxc
    We extract column 2 (QP-eps) and convert eV → Hartree.
    Assumes C1 symmetry (orbitals labeled '1a', '2a', ...).
    """
    qp_energies = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('$'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            label = parts[0]
            idx_match = re.match(r"(\d+)", label)
            if not idx_match:
                continue
            orb_idx = int(idx_match.group(1)) - 1  # 0-based
            qp_ev = float(parts[2])  # QP-eps in eV
            qp_energies[orb_idx] = qp_ev / HA_TO_EV  # convert to Hartree

    n_orb = max(qp_energies.keys()) + 1
    result = np.zeros(n_orb)
    for idx, energy in qp_energies.items():
        result[idx] = energy
    return result


def perp_basis(e: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build orthonormal perpendicular basis {u1, u2} for bond-axis vector e."""
    tmp = np.array([1.0, 0.0, 0.0]) if abs(e[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u1 = tmp - np.dot(tmp, e) * e
    u1 /= np.linalg.norm(u1)
    return u1, np.cross(e, u1)


def build_d_rotation_matrix(e: np.ndarray, u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
    """Build 5x5 Wigner D^2 rotation matrix for d-orbital axis projection.

    Transforms d-orbital coefficients from lab frame {x,y,z} to bond frame
    {u1, u2, e} where e is the bond axis (z'), u1 is perp-1 (x'), u2 is perp-2 (y').
    Derived from rank-2 symmetric traceless tensor transformation under rotation.

    Ordering: [dxy, dxz, dyz, dx2-y2, dz2] — matches HERMES convention.
    Bond-frame classification: dz2'->sigma(m=0), dxz'/dyz'->pi(|m|=1), dxy'/dx2y2'->delta(|m|=2).
    """
    R = np.array([u1, u2, e])
    D = np.empty((5, 5))
    s3 = np.sqrt(3.0)

    # Row 0: bond dxy (delta) — from x'y' = (u1.r)(u2.r)
    D[0, 0] = R[0, 0]*R[1, 1] + R[0, 1]*R[1, 0]
    D[0, 1] = R[0, 0]*R[1, 2] + R[0, 2]*R[1, 0]
    D[0, 2] = R[0, 1]*R[1, 2] + R[0, 2]*R[1, 1]
    D[0, 3] = R[0, 0]*R[1, 0] - R[0, 1]*R[1, 1]
    D[0, 4] = (-R[0, 0]*R[1, 0] - R[0, 1]*R[1, 1] + 2*R[0, 2]*R[1, 2]) / s3

    # Row 1: bond dxz (pi) — from x'z' = (u1.r)(e.r)
    D[1, 0] = R[0, 0]*R[2, 1] + R[0, 1]*R[2, 0]
    D[1, 1] = R[0, 0]*R[2, 2] + R[0, 2]*R[2, 0]
    D[1, 2] = R[0, 1]*R[2, 2] + R[0, 2]*R[2, 1]
    D[1, 3] = R[0, 0]*R[2, 0] - R[0, 1]*R[2, 1]
    D[1, 4] = (-R[0, 0]*R[2, 0] - R[0, 1]*R[2, 1] + 2*R[0, 2]*R[2, 2]) / s3

    # Row 2: bond dyz (pi) — from y'z' = (u2.r)(e.r)
    D[2, 0] = R[1, 0]*R[2, 1] + R[1, 1]*R[2, 0]
    D[2, 1] = R[1, 0]*R[2, 2] + R[1, 2]*R[2, 0]
    D[2, 2] = R[1, 1]*R[2, 2] + R[1, 2]*R[2, 1]
    D[2, 3] = R[1, 0]*R[2, 0] - R[1, 1]*R[2, 1]
    D[2, 4] = (-R[1, 0]*R[2, 0] - R[1, 1]*R[2, 1] + 2*R[1, 2]*R[2, 2]) / s3

    # Row 3: bond dx2-y2 (delta) — from (x'^2-y'^2)/2
    D[3, 0] = R[0, 0]*R[0, 1] - R[1, 0]*R[1, 1]
    D[3, 1] = R[0, 0]*R[0, 2] - R[1, 0]*R[1, 2]
    D[3, 2] = R[0, 1]*R[0, 2] - R[1, 1]*R[1, 2]
    D[3, 3] = (R[0, 0]**2 - R[1, 0]**2 - R[0, 1]**2 + R[1, 1]**2) / 2
    D[3, 4] = (-R[0, 0]**2 + R[1, 0]**2 - R[0, 1]**2 + R[1, 1]**2
               + 2*R[0, 2]**2 - 2*R[1, 2]**2) / (2*s3)

    # Row 4: bond dz2 (sigma) — from (2z'^2-x'^2-y'^2)/(2sqrt3)
    D[4, 0] = s3 * R[2, 0] * R[2, 1]
    D[4, 1] = s3 * R[2, 0] * R[2, 2]
    D[4, 2] = s3 * R[2, 1] * R[2, 2]
    D[4, 3] = s3 * (R[2, 0]**2 - R[2, 1]**2) / 2
    D[4, 4] = (3*R[2, 2]**2 - 1) / 2

    return D


@dataclass
class AtomData:
    number: int
    element: str
    x: float
    y: float
    z: float

    def distance_to(self, other: 'AtomData') -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)


def load_checkpoint(npz_path: str) -> Tuple[dict, dict]:
    """Load .npz arrays and .json metadata from checkpoint.

    Returns:
        (arrays_dict, metadata_dict)
    """
    npz_path = Path(npz_path)
    json_path = npz_path.with_suffix('.json')

    if not npz_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {npz_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Metadata not found: {json_path}")

    arrays = dict(np.load(npz_path))
    with open(json_path) as f:
        metadata = json.load(f)

    return arrays, metadata


def rebuild_ao_basis(ao_basis_json: List[dict]) -> List[AOBasisFunction]:
    """Reconstruct AOBasisFunction list from JSON metadata."""
    return [
        AOBasisFunction(
            atom_idx=d['atom_idx'],
            element=d['element'],
            angular_momentum=d['angular_momentum'],
            magnetic_quantum=d['magnetic_quantum'],
            contraction_idx=d['contraction_idx'],
        )
        for d in ao_basis_json
    ]


def organize_coefficients(
    mo_in_nao: np.ndarray,
    nao_occ: np.ndarray,
    ao_basis: List[AOBasisFunction],
    val_threshold: float = 0.1,
    nao_tiers: Optional[np.ndarray] = None,
    selection: str = "auto",
) -> Dict[int, Dict[str, List[float]]]:
    """Organize NAO-projected MO coefficients by atom and orbital type.

    selection:
      "tier"      -- admit an NAO iff the NAO construction classified it VALENCE
                     (tier 1). Requires nao_tiers. This is the documented
                     three-tier scheme and is the correct criterion.
      "occupancy" -- admit an NAO iff its occupancy exceeds val_threshold.
                     Legacy. This is NOT equivalent: it admits core functions
                     whose occupancy is high (Fe 2p and 3p sit near 2 e and pass
                     a 0.1 e cut, though the tiering calls them core) and it
                     rejects genuine valence functions whose occupancy happens to
                     fall below the cut. Because the cut is absolute while
                     occupancies scale with the density, an orbital near the
                     threshold changes status between the per-spin and shared
                     NAO basis conventions, which is not a physical difference.
      "auto"      -- "tier" when nao_tiers is supplied, else "occupancy" with a
                     warning. Checkpoints written before the tiers were stored
                     therefore keep their original behaviour.
    """
    """Organize NAO-projected MO coefficients by atom and orbital type.

    Same logic as organize_nao_coefficients_by_atom() but works from arrays.

    Returns:
        {atom_number(1-indexed): {orbital_type: [coeff_for_each_MO]}}
    """
    n_nao, n_mo = mo_in_nao.shape
    if selection not in ("auto", "tier", "occupancy"):
        raise ValueError(f"selection must be 'auto', 'tier' or 'occupancy', got {selection!r}")
    if selection == "tier" and nao_tiers is None:
        raise ValueError("selection='tier' requires nao_tiers; regenerate the checkpoint "
                         "with a reader that stores them.")
    _sel = selection
    if _sel == "auto":
        _sel = "tier" if nao_tiers is not None else "occupancy"
        if _sel == "occupancy":
            print("   [organize_coefficients] no NAO tiers in this checkpoint; falling back "
                  "to the legacy occupancy cut. Regenerate to use the three-tier scheme.")
    coeffs = {}

    for nao_idx, ao_func in enumerate(ao_basis):
        orb_type = ao_func.get_orbital_type()

        # Only s, p, and d orbitals
        if orb_type not in ('s', 'px', 'py', 'pz', 'dxy', 'dxz', 'dyz', 'dx2y2', 'dz2'):
            continue

        # Valence selection
        if _sel == "tier":
            if nao_tiers[nao_idx] != 1:      # 0 core, 1 valence, 2 Rydberg
                continue
        else:
            if nao_occ[nao_idx] < val_threshold:
                continue

        atom_num = ao_func.atom_idx + 1  # 1-indexed

        if atom_num not in coeffs:
            coeffs[atom_num] = {}
        if orb_type not in coeffs[atom_num]:
            coeffs[atom_num][orb_type] = np.zeros(n_mo)

        # Sum over multiple NAOs of same type on same atom
        coeffs[atom_num][orb_type] += mo_in_nao[nao_idx, :]

    # Convert numpy arrays to lists for compatibility with existing G_rs code
    for atom_num in coeffs:
        for orb_type in coeffs[atom_num]:
            coeffs[atom_num][orb_type] = coeffs[atom_num][orb_type].tolist()

    return coeffs


def compute_grs_per_spin(
    atom1_num: int, atom2_num: int,
    coeffs_spin: Dict[int, Dict[str, List[float]]],
    ef: float,
    energies: np.ndarray,
    start_idx: int, end_idx: int,
    orbital_types: List[str],
    abs_per_type: bool = False,
    occupation_aware: bool = False,
    occupations: Optional[np.ndarray] = None,
    mo_indices: Optional[List[int]] = None,
) -> Optional[float]:
    """Compute G_rs for one atom pair, one spin channel.

    Args:
        occupation_aware: If True, skip MOs with occupation < 0.5 within the
                         occupied range (fixes non-aufbau violations).
        occupations: MO occupation array (1.0 for occupied, 0.0 for virtual).
                    Required when occupation_aware=True.
        mo_indices: If provided, iterate over these specific MO indices instead
                   of range(start_idx, end_idx+1).
    """
    if atom1_num not in coeffs_spin or atom2_num not in coeffs_spin:
        return None

    total_grs = 0.0
    found_any = False

    iter_indices = mo_indices if mo_indices is not None else range(start_idx, end_idx + 1)

    for orb_type in orbital_types:
        if orb_type not in coeffs_spin[atom1_num] or orb_type not in coeffs_spin[atom2_num]:
            continue

        c1 = coeffs_spin[atom1_num][orb_type]
        c2 = coeffs_spin[atom2_num][orb_type]
        if not c1 or not c2:
            continue

        found_any = True
        grs_type = 0.0

        for k in iter_indices:
            if k >= len(c1) or k >= len(c2) or k >= len(energies):
                continue

            # Occupation-aware: skip MOs that are nominally in the occupied
            # range but actually unoccupied (non-aufbau violations)
            if occupation_aware and occupations is not None:
                if k < len(occupations) and occupations[k] < 0.5:
                    continue

            denom = ef - energies[k]
            if abs(denom) > 1e-10:
                grs_type += c1[k] * c2[k] / denom

        total_grs += abs(grs_type) if abs_per_type else grs_type

    return total_grs if found_any else None


def compute_grs_axis_decompose(
    atom1_num: int, atom2_num: int,
    coeffs_spin: Dict[int, Dict[str, List[float]]],
    ef: float, energies: np.ndarray,
    start_idx: int, end_idx: int,
    e_rs: np.ndarray, u1: np.ndarray, u2: np.ndarray,
    occupation_aware: bool = False,
    occupations: Optional[np.ndarray] = None,
    mo_indices: Optional[List[int]] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Axis-projected chemical σ/π/δ decomposition for one atom pair.

    σ = G(s) + G(p∥e_rs) + G(d_σ)    head-on overlap (m=0)
    π = G(p⊥e_rs) + G(d_π)           sideways overlap (|m|=1)
    δ = G(d_δ)                        face-on overlap (|m|=2)
    Invariant: σ + π + δ == G_total(spd)
    """
    if atom1_num not in coeffs_spin or atom2_num not in coeffs_spin:
        return None, None, None

    c1 = coeffs_spin[atom1_num]
    c2 = coeffs_spin[atom2_num]
    n_mo = len(energies)
    zero = np.zeros(n_mo)

    s1 = np.asarray(c1.get('s', zero))
    s2 = np.asarray(c2.get('s', zero))
    p1 = np.stack([np.asarray(c1.get('px', zero)),
                   np.asarray(c1.get('py', zero)),
                   np.asarray(c1.get('pz', zero))])
    p2 = np.stack([np.asarray(c2.get('px', zero)),
                   np.asarray(c2.get('py', zero)),
                   np.asarray(c2.get('pz', zero))])
    d1 = np.stack([np.asarray(c1.get('dxy', zero)),
                   np.asarray(c1.get('dxz', zero)),
                   np.asarray(c1.get('dyz', zero)),
                   np.asarray(c1.get('dx2y2', zero)),
                   np.asarray(c1.get('dz2', zero))])
    d2 = np.stack([np.asarray(c2.get('dxy', zero)),
                   np.asarray(c2.get('dxz', zero)),
                   np.asarray(c2.get('dyz', zero)),
                   np.asarray(c2.get('dx2y2', zero)),
                   np.asarray(c2.get('dz2', zero))])

    has_any1 = np.any(s1) or np.any(p1) or np.any(d1)
    has_any2 = np.any(s2) or np.any(p2) or np.any(d2)
    if not (has_any1 and has_any2):
        return None, None, None

    if mo_indices is not None:
        idx = np.array(mo_indices)
    else:
        idx = np.arange(start_idx, end_idx + 1)
    idx = idx[idx < n_mo]

    if occupation_aware and occupations is not None:
        idx = idx[occupations[idx] >= 0.5]

    denom = ef - energies[idx]
    valid = np.abs(denom) > 1e-10
    idx = idx[valid]
    denom = denom[valid]

    if len(idx) == 0:
        return None, None, None

    inv_denom = 1.0 / denom

    # s → σ
    G_s = float(np.sum(s1[idx] * s2[idx] * inv_denom))

    # p∥e_rs → σ   (head-on: projection onto bond axis)
    proj1_par = e_rs @ p1[:, idx]
    proj2_par = e_rs @ p2[:, idx]
    G_sig_p = float(np.sum(proj1_par * proj2_par * inv_denom))

    # p⊥e_rs → π   (sideways: projection onto perpendicular plane)
    proj1_u1 = u1 @ p1[:, idx]
    proj2_u1 = u1 @ p2[:, idx]
    proj1_u2 = u2 @ p1[:, idx]
    proj2_u2 = u2 @ p2[:, idx]
    G_pi_p = float(np.sum((proj1_u1 * proj2_u1 + proj1_u2 * proj2_u2) * inv_denom))

    # d → σ/π/δ via Wigner D² rotation to bond frame
    G_sig_d = 0.0
    G_pi_d = 0.0
    G_delta = 0.0
    has_d = np.any(d1) and np.any(d2)
    if has_d:
        D2 = build_d_rotation_matrix(e_rs, u1, u2)
        d1_rot = D2 @ d1[:, idx]  # (5, n_idx)
        d2_rot = D2 @ d2[:, idx]
        # σ: dz2' (row 4, m=0)
        G_sig_d = float(np.sum(d1_rot[4] * d2_rot[4] * inv_denom))
        # π: dxz' + dyz' (rows 1,2, |m|=1)
        G_pi_d = float(np.sum((d1_rot[1] * d2_rot[1] + d1_rot[2] * d2_rot[2]) * inv_denom))
        # δ: dxy' + dx2y2' (rows 0,3, |m|=2)
        G_delta = float(np.sum((d1_rot[0] * d2_rot[0] + d1_rot[3] * d2_rot[3]) * inv_denom))

    return G_s + G_sig_p + G_sig_d, G_pi_p + G_pi_d, G_delta


def parse_window(window_str: str, homo_idx: int, lumo_idx: int, n_mo: int) -> Tuple[int, int]:
    """Parse a window specification string.

    Examples:
        'HOMO-10:LUMO+10'  -> (homo_idx-10, lumo_idx+10)
        '0:HOMO'           -> (0, homo_idx)
        '0:LUMO'           -> (0, lumo_idx)
        'HOMO:LUMO'        -> (homo_idx, lumo_idx)
        '3260:3280'        -> (3260, 3280)
        'all'              -> (0, n_mo-1)
    """
    window_str = window_str.strip()

    if window_str.lower() == 'all':
        return 0, n_mo - 1

    def resolve_token(tok: str) -> int:
        tok = tok.strip()
        # HOMO+N or HOMO-N
        if tok.startswith('HOMO'):
            rest = tok[4:]
            if not rest:
                return homo_idx
            return homo_idx + int(rest)
        # LUMO+N or LUMO-N
        if tok.startswith('LUMO'):
            rest = tok[4:]
            if not rest:
                return lumo_idx
            return lumo_idx + int(rest)
        return int(tok)

    parts = window_str.split(':')
    if len(parts) != 2:
        raise ValueError(f"Window must be 'start:end', got '{window_str}'")

    start = max(0, resolve_token(parts[0]))
    end = min(n_mo - 1, resolve_token(parts[1]))
    return start, end


def parse_mo_indices(indices_str: str, homo_idx: int, lumo_idx: int, n_mo: int) -> List[int]:
    """Parse comma-separated MO indices with HOMO/LUMO notation.

    Examples:
        'HOMO-2,HOMO,LUMO+1,LUMO+3' -> [homo-2, homo, lumo+1, lumo+3]
        '459,461,464,466'            -> [459, 461, 464, 466]
    """
    def resolve_token(tok: str) -> int:
        tok = tok.strip()
        if tok.startswith('HOMO'):
            rest = tok[4:]
            return homo_idx + int(rest) if rest else homo_idx
        if tok.startswith('LUMO'):
            rest = tok[4:]
            return lumo_idx + int(rest) if rest else lumo_idx
        return int(tok)

    indices = []
    for tok in indices_str.split(','):
        tok = tok.strip()
        if not tok:
            continue
        idx = resolve_token(tok)
        if 0 <= idx < n_mo:
            indices.append(idx)
    return sorted(set(indices))


def get_orbital_types(plane: str, family: str) -> List[str]:
    # "P_SUM" is the pre-2026-09-17 name for "SHELL_SUM". Accept it here so that
    # direct callers outside the CLI (analysis scripts, notebooks) keep working;
    # the CLI normalises earlier and never reaches this.
    plane = "SHELL_SUM" if plane == "P_SUM" else plane
    orb_types = []
    for letter in ('s', 'p', 'd'):
        if letter in family:
            orb_types.extend(PLANE_ORBITAL_MAP[plane][letter])
    return orb_types if orb_types else PLANE_ORBITAL_MAP[plane]['p']


def build_occupation_array(n_occ: int, n_mo: int) -> np.ndarray:
    """Build idealized occupation array: 1.0 for indices < n_occ, 0.0 above."""
    occ = np.zeros(n_mo)
    occ[:n_occ] = 1.0
    return occ


def write_results(results: List[tuple],
                  plane: str, spin: str, ef: float,
                  orb_range: Tuple[int, int], homo_idx: int, lumo_idx: int,
                  n_mo: int, threshold: float, orbital_types: List[str],
                  orbital_family: str, output_prefix: str, output_dir: Path,
                  is_unrestricted: bool, occupation_aware: bool,
                  rooted: bool = False, mo_indices: Optional[List[int]] = None,
                  decompose: bool = False, active_channels: Optional[List[str]] = None):
    """Write .txt and .out files matching existing HERMES output format."""

    spin_suffix = f"_{spin}" if is_unrestricted and spin in ('alpha', 'beta') else ""
    txt_path = output_dir / f"{output_prefix}_{plane}{spin_suffix}.txt"
    out_path = output_dir / f"{output_prefix}_{plane}{spin_suffix}.out"

    # .txt (simple: atom1 atom2 G_rs distance — unchanged for backward compatibility)
    with open(txt_path, 'w') as f:
        for a1, a2, grs, dist, _channels in results:
            f.write(f"{a1:3d} {a2:3d} {grs:15.6f} {dist:15.6f}\n")

    # .out (summary)
    with open(out_path, 'w') as f:
        f.write("=" * 80 + "\n")
        if plane == 'axis':
            f.write(f"YOSHIZAWA GREEN'S FUNCTION — AXIS-PROJECTED σ/π/δ DECOMPOSITION\n")
        else:
            f.write(f"YOSHIZAWA GREEN'S FUNCTION CALCULATION - {plane} PLANE\n")
        f.write(f"From HERMES NAO Checkpoint (hermes_grs_from_checkpoint.py)\n")
        f.write("=" * 80 + "\n\n")

        f.write("CALCULATION PARAMETERS\n")
        f.write("-" * 80 + "\n")
        if plane == 'axis':
            f.write(f"Analysis mode:         Axis-projected σ/π/δ (rotation-invariant)\n")
            f.write(f"  σ = s + p∥(bond axis) + d(m=0)   [head-on overlap]\n")
            f.write(f"  π = p⊥(bond axis) + d(|m|=1)     [sideways overlap]\n")
            f.write(f"  δ = d(|m|=2)                      [face-on overlap]\n")
        else:
            f.write(f"Analysis plane:        {plane}\n")
        orb_str = ', '.join(orbital_types)
        f.write(f"Orbital types:         {orb_str}\n")
        family_label = '+'.join(orbital_family)
        f.write(f"Orbital family:        {family_label}\n")
        if decompose and active_channels:
            if 'σ' in active_channels:
                f.write(f"Decomposition:         Chemical σ/π (axis-projected)\n")
            else:
                ch_labels = ', '.join(CHANNEL_LABELS.get(ch, ch) for ch in active_channels)
                f.write(f"Channel decomposition: {ch_labels}\n")
        if is_unrestricted:
            f.write(f"Calculation type:      Unrestricted DFT\n")
            f.write(f"Spin channel:          {spin}\n")
        else:
            f.write(f"Calculation type:      Restricted\n")
        f.write(f"TD Approach:           {'Rooted state-specific' if rooted else 'Standard'}\n")
        f.write(f"Occupation-aware:      {'yes' if occupation_aware else 'no'}\n")
        f.write(f"Fermi energy:          {ef:.10f} H ({ef * HA_TO_EV:.4f} eV)\n")
        f.write(f"Threshold:             {threshold}\n\n")

        f.write("ORBITAL INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total MOs available:   {n_mo}\n")
        f.write(f"HOMO index (0-based):  {homo_idx}\n")
        f.write(f"LUMO index (0-based):  {lumo_idx}\n\n")

        if mo_indices is not None:
            f.write("SPECIFIC MO INDICES FOR CALCULATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"MO indices (0-based):    {', '.join(str(i) for i in mo_indices)}\n")
            f.write(f"Total orbitals in sum:   {len(mo_indices)}\n\n")
        else:
            f.write("ORBITAL RANGE FOR CALCULATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Start orbital (0-based): {orb_range[0]}\n")
            f.write(f"End orbital (0-based):   {orb_range[1]}\n")
            f.write(f"Total orbitals in sum:   {orb_range[1] - orb_range[0] + 1}\n\n")

        f.write("RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Pairs above threshold: {len(results)}\n\n")

        if results:
            if decompose and active_channels:
                ch_hdrs = ''.join(f"{'G('+ch+')':>12s}" for ch in active_channels)
                f.write(f"Top 20 pairs by |G_rs| with channel decomposition:\n")
                f.write(f"  {'A1':>5s} - {'A2':>5s}  {'G_rs':>12s}  {'dist':>8s}  {ch_hdrs}\n")
                for a1, a2, grs, dist, channels in results[:20]:
                    ch_vals = ''
                    for ch in active_channels:
                        ch_vals += f"{channels.get(ch, 0.0):12.6f}"
                    f.write(f"  {a1:5d} - {a2:5d}  {grs:12.6f}  {dist:8.3f}  {ch_vals}\n")
            else:
                f.write("Top 20 pairs by |G_rs|:\n")
                for a1, a2, grs, dist, _channels in results[:20]:
                    f.write(f"  {a1:5d} - {a2:5d}  G_rs = {grs:12.6f}  dist = {dist:8.3f} A\n")

    print(f"  {txt_path.name}: {len(results)} pairs")


def interactive_window(homo_idx: int, lumo_idx: int, n_mo: int) -> Tuple[int, int]:
    """Prompt user for MO window selection."""
    print(f"\nORBITAL WINDOW SELECTION:")
    print(f"  HOMO index: {homo_idx} (0-based)")
    print(f"  LUMO index: {lumo_idx} (0-based)")
    print(f"  Total MOs:  {n_mo}")
    print(f"\nOptions:")
    print(f"  1) 0 to HOMO (occupied only)")
    print(f"  2) 0 to LUMO (occupied + LUMO)")
    print(f"  3) HOMO-5 to LUMO+5")
    print(f"  4) HOMO-10 to LUMO+10")
    print(f"  5) Custom window (e.g., HOMO-15:LUMO+15)")
    print(f"  6) All MOs")

    while True:
        choice = input("\nSelect window (1-6, default 4): ").strip()
        if not choice or choice == '4':
            s = max(0, homo_idx - 10)
            e = min(n_mo - 1, lumo_idx + 10)
            print(f"  -> MOs {s} to {e} (HOMO-10 to LUMO+10, {e - s + 1} MOs)")
            return s, e
        elif choice == '1':
            print(f"  -> MOs 0 to {homo_idx} ({homo_idx + 1} MOs)")
            return 0, homo_idx
        elif choice == '2':
            print(f"  -> MOs 0 to {lumo_idx} ({lumo_idx + 1} MOs)")
            return 0, lumo_idx
        elif choice == '3':
            s = max(0, homo_idx - 5)
            e = min(n_mo - 1, lumo_idx + 5)
            print(f"  -> MOs {s} to {e} ({e - s + 1} MOs)")
            return s, e
        elif choice == '5':
            spec = input("  Enter window (e.g., HOMO-15:LUMO+15): ").strip()
            try:
                s, e = parse_window(spec, homo_idx, lumo_idx, n_mo)
                print(f"  -> MOs {s} to {e} ({e - s + 1} MOs)")
                return s, e
            except ValueError as err:
                print(f"  Error: {err}")
        elif choice == '6':
            print(f"  -> MOs 0 to {n_mo - 1} (all {n_mo} MOs)")
            return 0, n_mo - 1
        else:
            print("  Invalid choice.")


def interactive_plane() -> List[str]:
    """Prompt user for plane selection."""
    print(f"\nPLANE SELECTION:")
    print(f"  1) XY (pz orbitals)")
    print(f"  2) YZ (px orbitals)")
    print(f"  3) XZ (py orbitals)")
    print(f"  4) ALL (XY + YZ + XZ)")
    print("  5) SHELL_SUM = Σ_type |G| over every type in the family — VALIDATION CHECKSUM ONLY")
    print(f"     (phase-robust, but NOT rotation-invariant and NOT physical coupling)")

    while True:
        choice = input("\nSelect plane (1-5, default 4): ").strip()
        if not choice or choice == '4':
            return ['XY', 'YZ', 'XZ']
        elif choice == '1':
            return ['XY']
        elif choice == '2':
            return ['YZ']
        elif choice == '3':
            return ['XZ']
        elif choice == '5':
            print(SHELL_SUM_WARNING)
            return ['SHELL_SUM']
        else:
            print("  Invalid choice.")


def interactive_family() -> str:
    """Prompt user for orbital family."""
    print(f"\nORBITAL FAMILY:")
    print(f"  1) p-orbitals only (default)")
    print(f"  2) d-orbitals only")
    print(f"  3) p+d combined")
    print(f"  4) s+p (σ-channel, saturated bridges)")
    print(f"  5) s+p+d (all valence)")
    print(f"  6) Custom (type any combination of s, p, d)")

    while True:
        choice = input("\nSelect family (1-6, default 1): ").strip()
        if not choice or choice == '1':
            return 'p'
        elif choice == '2':
            return 'd'
        elif choice == '3':
            return 'pd'
        elif choice == '4':
            return 'sp'
        elif choice == '5':
            return 'spd'
        elif choice == '6':
            return _select_custom_orbital_family()
        else:
            print("  Invalid choice.")


def _select_custom_orbital_family() -> str:
    """Prompt user for a custom orbital family string."""
    while True:
        raw = input("  Enter orbital types (e.g. 's', 'sd', 'spd'): ").strip().lower()
        family = ''.join(c for c in 'spd' if c in raw)
        if family:
            label = '+'.join(family)
            print(f"  → Selected: {label}")
            return family
        print("  Invalid input. Use any combination of s, p, d.")


def _parse_excitation_energies(out_file: Path) -> List[float]:
    """Parse TD/BSE excitation energies (Hartree) from ORCA, Gaussian or Turbomole output.

    ORCA:      STATE  N:  E=   X.XXXXXX au  ...
    Gaussian:  Excited State   N:  Singlet-A   4.2246 eV  ...
    Turbomole: "Excitation energy:   0.1786016732935153"  (Hartree, from escf
               TD-DFT or 2c/BSE output; one such line per excitation block).

    Returns list ordered by state index (Hartree).
    """
    orca_pat = re.compile(r'STATE\s+\d+:\s+E=\s+([\d.]+)\s+au')
    gauss_pat = re.compile(r'Excited State\s+\d+:\s+\S+\s+([\d.]+)\s+eV')
    # Turbomole escf/BSE: the Hartree line is "Excitation energy:  <val>".
    # Must NOT match "Excitation energy / eV:" — the ':' immediately after
    # "energy" excludes the eV line (which reads "energy / eV:").
    turbo_pat = re.compile(r'Excitation energy:\s+([-\d.]+(?:[eE][-+]?\d+)?)')
    eV_to_Ha = 1.0 / 27.211386
    energies: List[float] = []
    try:
        with open(out_file) as fh:
            for line in fh:
                m = orca_pat.search(line)
                if m:
                    energies.append(float(m.group(1)))
                    continue
                m = gauss_pat.search(line)
                if m:
                    energies.append(float(m.group(1)) * eV_to_Ha)
                    continue
                m = turbo_pat.search(line)
                if m:
                    energies.append(float(m.group(1)))  # already Hartree
    except Exception:
        pass
    return energies


def apply_rooted_approach(
    energies: np.ndarray,
    lumo_idx: int,
    window_end: int,
    rooted_dir: Path,
) -> Tuple[np.ndarray, bool]:
    """Replace virtual orbital energies using rooted TD-DFT approach.

    For each virtual orbital i in the window:
        eps_eff(LUMO+i) = eps_HOMO + omega_{i+1}

    where omega is parsed from Root{i}_*.out files in rooted_dir.

    Args:
        energies:   1-D orbital energy array (modified copy returned)
        lumo_idx:   0-based LUMO index
        window_end: last orbital index in the MO window
        rooted_dir: directory containing Root{i}_*.out files

    Returns:
        (modified_energies, was_applied)
    """
    n_roots = window_end - lumo_idx + 1
    if n_roots <= 0:
        return energies, False

    # Find Root output files (.out for ORCA, .log for Gaussian)
    root_files: List[Optional[Path]] = []
    for i in range(1, n_roots + 1):
        matches = sorted(
            f for f in list(rooted_dir.glob("*.out")) + list(rooted_dir.glob("*.log"))
            if f.name.lower().startswith(f'root{i}_')
        )
        root_files.append(matches[0] if matches else None)

    if all(f is None for f in root_files):
        return energies, False

    found = [(i + 1, f) for i, f in enumerate(root_files) if f is not None]
    missing = [i + 1 for i, f in enumerate(root_files) if f is None]

    print(f"\n{'=' * 60}")
    print("ROOTED TD-DFT APPROACH")
    print(f"{'=' * 60}")
    print(f"+ Found {len(found)}/{n_roots} Root output files:")
    for idx, f in found:
        print(f"  Root{idx}: {f.name}")
    if missing:
        print(f"x Missing: {', '.join(f'Root{i}' for i in missing)}")
        print("  (keeping base TD energy for missing orbitals)")

    # Parse excitation energies from the first valid Root file
    exc_energies: List[float] = []
    for rf in root_files:
        if rf is not None:
            parsed = _parse_excitation_energies(rf)
            if parsed and any(e > 0 for e in parsed):
                exc_energies = parsed
                print(f"  Excitation energies from: {rf.name}")
                break

    if not exc_energies:
        print("  WARNING: No excitation energies found — cannot apply rooted approach")
        return energies, False

    homo_energy = float(energies[lumo_idx - 1])
    print(f"  E_HOMO = {homo_energy:.6f} H")
    print(f"  Using eps_eff(LUMO+i) = E_HOMO + omega_i")

    composite = energies.copy()
    for i in range(n_roots):
        virtual_idx = lumo_idx + i
        if virtual_idx >= len(composite):
            break
        if i < len(exc_energies):
            eff_energy = homo_energy + exc_energies[i]
            composite[virtual_idx] = eff_energy
            print(f"  Root{i+1}: orbital {virtual_idx} (LUMO+{i}) "
                  f"energy = {eff_energy:.6f} H  "
                  f"(base: {float(energies[virtual_idx]):.6f} H)")
        else:
            print(f"  Root{i+1}: no excitation energy available, keeping base")

    print("+ Composite orbital energies built")
    return composite, True


def main():
    ap = argparse.ArgumentParser(
        description='HERMES G_rs from Checkpoint — compute G_rs with arbitrary MO windows'
    )
    ap.add_argument('--checkpoint', '-c', type=str, default='hermes_nao_checkpoint.npz',
                    help='Path to .npz checkpoint (default: hermes_nao_checkpoint.npz)')
    ap.add_argument('--window', '-w', type=str, default=None,
                    help='MO window, e.g., "HOMO-10:LUMO+10", "0:HOMO", "all" (interactive if omitted)')
    ap.add_argument('--mo-indices', type=str, default=None,
                    help='Specific MO indices (comma-separated), e.g., "HOMO-2,HOMO,LUMO+1,LUMO+3" '
                         'or "459,461,464,466". Overrides --window.')
    ap.add_argument('--planes', nargs='+', default=None,
                    choices=['XY', 'YZ', 'XZ', 'SHELL_SUM', 'P_SUM'],
                    help='Planes to compute (interactive if omitted)')
    ap.add_argument('--family', type=str, default=None,
                    help='Orbital family: any combination of s, p, d '
                         '(e.g. "p", "sp", "pd", "spd", "sd"). Interactive if omitted.')
    ap.add_argument('--spin', type=str, default=None,
                    choices=['alpha', 'beta', 'both'],
                    help='Spin channel for unrestricted (interactive if omitted)')
    ap.add_argument('--threshold', '-t', type=float, default=None,
                    help='|G_rs| threshold (interactive if omitted)')
    ap.add_argument('--ef', type=float, default=None,
                    help='Custom Fermi energy in Hartree (default: midpoint of HOMO-LUMO gap)')
    ap.add_argument('--ef-alpha', type=float, default=None,
                    help='Custom alpha Fermi energy in Hartree')
    ap.add_argument('--ef-beta', type=float, default=None,
                    help='Custom beta Fermi energy in Hartree')
    ap.add_argument('--output-prefix', '-o', type=str, default=None,
                    help='Output file prefix (default: derived from directory)')
    ap.add_argument('--output-dir', type=str, default=None,
                    help='Output directory (default: current directory)')
    ap.add_argument('--occupation-aware', action='store_true',
                    help='Skip MOs with zero occupation in occupied range (fixes non-aufbau)')
    ap.add_argument('--rooted-dir', type=str, default=None,
                    help='Directory with Root{i}_*.out files for rooted TD-DFT '
                         '(auto-detects in cwd if omitted)')
    ap.add_argument('--no-rooted', action='store_true',
                    help='Disable rooted auto-detection')
    ap.add_argument('--excitation-energies', type=str, default=None,
                    help='Explicit excitation energies in eV (comma-separated), '
                         'e.g. "4.013,4.246,5.327". Bypasses Root file parsing.')
    ap.add_argument('--decompose', type=str, default=None,
                    choices=['axis'],
                    help='Decomposition mode: "axis" for true chemical σ/π/δ '
                         '(projects s/p/d orbitals onto the internuclear axis per pair)')
    ap.add_argument('--valence-selection', choices=['auto','tier','occupancy'], default='auto',
                    help="how the G_rs step picks valence NAOs: 'tier' uses the "
                         "core/valence/Rydberg classification the NAO construction "
                         "computed (correct); 'occupancy' thresholds the raw occupancy "
                         "at --val-threshold (legacy); 'auto' (default) uses tiers when "
                         "the checkpoint carries them and falls back with a warning.")
    ap.add_argument('--val-threshold', type=float, default=0.1, metavar='OCC',
                    help="occupancy cut (electrons) admitting an NAO as valence, used "
                         "only by --valence-selection occupancy (or by 'auto' when the "
                         "checkpoint carries no tiers). Default 0.1, which is the "
                         "historical HERMES value; change it only to reproduce an "
                         "older result.")
    ap.add_argument('--qpenergies', type=str, default=None,
                    help='Path to Turbomole qpenergies.dat (from GW calculation). '
                         'Replaces KS eigenvalues with GW quasiparticle energies.')
    args = ap.parse_args()
    # P_SUM is the pre-2026-09-17 name for SHELL_SUM; accept it silently so
    # existing scripts and result directories keep working.
    if getattr(args, 'planes', None) and 'P_SUM' in args.planes:
        print("  !!  P_SUM is the former name of SHELL_SUM; reading it as SHELL_SUM.\n"
              "      Output is written as *_SHELL_SUM.txt, NOT *_P_SUM.txt -- a script\n"
              "      that regenerates and then reads back *_P_SUM.txt will read a stale file.")
    if getattr(args, 'planes', None):
        args.planes = ['SHELL_SUM' if p == 'P_SUM' else p for p in args.planes]


    print("=" * 70)
    print("HERMES G_rs FROM CHECKPOINT — Stage 2: Fast G_rs Computation")
    print("=" * 70)

    # --- Load checkpoint ---
    print(f"\nLoading checkpoint: {args.checkpoint}")
    arrays, meta = load_checkpoint(args.checkpoint)

    is_unrestricted = meta['is_unrestricted']
    n_atoms = meta['n_atoms']
    homo_alpha = meta['homo_idx_alpha']
    homo_beta = meta['homo_idx_beta']
    lumo_alpha = meta['lumo_idx_alpha']
    lumo_beta = meta['lumo_idx_beta']
    n_occ_alpha = meta['n_occ_alpha']
    n_occ_beta = meta['n_occ_beta']

    # MO coefficient arrays
    mo_in_nao_alpha = arrays['mo_in_nao_alpha']
    mo_in_nao_beta = arrays['mo_in_nao_beta']
    nao_occ_alpha = arrays['nao_occ_alpha']
    nao_occ_beta = arrays['nao_occ_beta']

    # Energies: prefer full_energies (covers all MOs) if available
    if 'full_energies_alpha' in arrays:
        energies_alpha = arrays['full_energies_alpha']
        energies_beta = arrays['full_energies_beta']
    else:
        energies_alpha = arrays['mo_energies_alpha']
        energies_beta = arrays['mo_energies_beta']

    # --- GW quasiparticle energy substitution ---
    if args.qpenergies:
        print(f"\n  Loading GW quasiparticle energies: {args.qpenergies}")
        qp = parse_qpenergies(args.qpenergies)
        n_qp = len(qp)
        n_ks = len(energies_alpha)
        n_replace = min(n_qp, n_ks)
        print(f"    QP energies available: {n_qp} orbitals")
        if homo_alpha < n_qp:
            ks_homo = energies_alpha[homo_alpha] * HA_TO_EV
            qp_homo = qp[homo_alpha] * HA_TO_EV
            ks_lumo = energies_alpha[lumo_alpha] * HA_TO_EV if lumo_alpha < n_ks else float('nan')
            qp_lumo = qp[lumo_alpha] * HA_TO_EV if lumo_alpha < n_qp else float('nan')
            print(f"    HOMO: KS = {ks_homo:.3f} eV → QP = {qp_homo:.3f} eV "
                  f"(Δ = {qp_homo - ks_homo:+.3f} eV)")
            print(f"    LUMO: KS = {ks_lumo:.3f} eV → QP = {qp_lumo:.3f} eV "
                  f"(Δ = {qp_lumo - ks_lumo:+.3f} eV)")
            print(f"    KS gap:  {ks_lumo - ks_homo:.3f} eV")
            print(f"    QP gap:  {qp_lumo - qp_homo:.3f} eV")
        energies_alpha[:n_replace] = qp[:n_replace]
        energies_beta[:n_replace] = qp[:n_replace]
        print(f"    Replaced {n_replace} eigenvalues with QP energies")

    n_mo_alpha = len(energies_alpha)
    n_mo_beta = len(energies_beta)

    # Reconstruct AO basis
    ao_basis = rebuild_ao_basis(meta['ao_basis'])

    # Reconstruct atoms
    atoms = [AtomData(**a) for a in meta['atoms']]

    print(f"  Atoms: {n_atoms}")
    print(f"  Unrestricted: {is_unrestricted}")
    if is_unrestricted:
        print(f"  Alpha: HOMO={homo_alpha}, LUMO={lumo_alpha}, {n_mo_alpha} MOs, "
              f"coeffs for {mo_in_nao_alpha.shape[1]} MOs")
        print(f"  Beta:  HOMO={homo_beta}, LUMO={lumo_beta}, {n_mo_beta} MOs, "
              f"coeffs for {mo_in_nao_beta.shape[1]} MOs")
    else:
        print(f"  HOMO={homo_alpha}, LUMO={lumo_alpha}, {n_mo_alpha} MOs")

    # --- Configure parameters (interactive or from CLI) ---
    axis_mode = args.decompose == 'axis'

    # Planes
    if axis_mode:
        planes = ['axis']
        print(f"\nAxis-projected σ/π mode — rotation-invariant (no plane selection)")
    elif args.planes:
        planes = args.planes
        if 'SHELL_SUM' in planes:
            print(SHELL_SUM_WARNING)
    else:
        planes = interactive_plane()
    if not axis_mode:
        print(f"\nPlanes: {', '.join(planes)}")

    # Orbital family
    if axis_mode:
        family = 'spd'
        print(f"Axis σ/π/δ: s+p+d family (s→σ, p∥axis→σ, p⊥axis→π, d→σ/π/δ via Wigner D²)")
    elif args.family:
        family = ''.join(c for c in 'spd' if c in args.family.lower())
        if not family:
            print(f"Invalid --family '{args.family}'. Use any combination of s, p, d.")
            sys.exit(1)
    else:
        family = interactive_family()
    if not axis_mode:
        family_label = '+'.join(family)
        print(f"Family: {family_label}")

    # MO window or specific indices
    mo_indices_alpha = None
    mo_indices_beta = None

    if args.mo_indices:
        # Specific MO indices mode — overrides --window
        mo_indices_alpha = parse_mo_indices(args.mo_indices, homo_alpha, lumo_alpha, n_mo_alpha)
        if is_unrestricted:
            mo_indices_beta = parse_mo_indices(args.mo_indices, homo_beta, lumo_beta, n_mo_beta)
        else:
            mo_indices_beta = mo_indices_alpha
        # Set window to span for rooted/E_F purposes
        window_alpha = (min(mo_indices_alpha), max(mo_indices_alpha))
        window_beta = (min(mo_indices_beta), max(mo_indices_beta))
        print(f"\nSpecific MO indices:")
        print(f"  Alpha: {mo_indices_alpha} ({len(mo_indices_alpha)} MOs)")
        if is_unrestricted:
            print(f"  Beta:  {mo_indices_beta} ({len(mo_indices_beta)} MOs)")
    elif args.window:
        window_alpha = parse_window(args.window, homo_alpha, lumo_alpha, n_mo_alpha)
        window_beta = parse_window(args.window, homo_beta, lumo_beta, n_mo_beta)
    else:
        print(f"\n--- Alpha window ---")
        window_alpha = interactive_window(homo_alpha, lumo_alpha, n_mo_alpha)
        if is_unrestricted:
            print(f"\n--- Beta window ---")
            use_same = input("Use same window offset for beta? (Y/n): ").strip().lower()
            if use_same in ('', 'y', 'yes'):
                # Apply same offset from HOMO
                offset_start = window_alpha[0] - homo_alpha
                offset_end = window_alpha[1] - lumo_alpha
                window_beta = (
                    max(0, homo_beta + offset_start),
                    min(n_mo_beta - 1, lumo_beta + offset_end),
                )
                print(f"  Beta window: MOs {window_beta[0]} to {window_beta[1]}")
            else:
                window_beta = interactive_window(homo_beta, lumo_beta, n_mo_beta)
        else:
            window_beta = window_alpha

    # Check window doesn't exceed stored coefficients
    max_coeff_alpha = mo_in_nao_alpha.shape[1] - 1
    max_coeff_beta = mo_in_nao_beta.shape[1] - 1
    if window_alpha[1] > max_coeff_alpha:
        print(f"\n  WARNING: Alpha window end {window_alpha[1]} exceeds stored coefficients "
              f"({max_coeff_alpha}). Clamping.")
        window_alpha = (window_alpha[0], max_coeff_alpha)
        if mo_indices_alpha:
            mo_indices_alpha = [i for i in mo_indices_alpha if i <= max_coeff_alpha]
    if window_beta[1] > max_coeff_beta:
        print(f"\n  WARNING: Beta window end {window_beta[1]} exceeds stored coefficients "
              f"({max_coeff_beta}). Clamping.")
        window_beta = (window_beta[0], max_coeff_beta)
        if mo_indices_beta:
            mo_indices_beta = [i for i in mo_indices_beta if i <= max_coeff_beta]

    # Spin channel
    if is_unrestricted:
        if args.spin:
            spin_channels = [args.spin] if args.spin != 'both' else ['alpha', 'beta']
        else:
            print(f"\nSPIN CHANNEL:")
            print(f"  1) Alpha only")
            print(f"  2) Beta only")
            print(f"  3) Both (default)")
            choice = input("Select (1-3, default 3): ").strip()
            if choice == '1':
                spin_channels = ['alpha']
            elif choice == '2':
                spin_channels = ['beta']
            else:
                spin_channels = ['alpha', 'beta']
    else:
        spin_channels = ['restricted']

    # --- Rooted TD-DFT approach ---
    rooted_applied = False
    if args.excitation_energies:
        eV_to_Ha = 1.0 / 27.211386
        exc_eV = [float(x.strip()) for x in args.excitation_energies.split(',')]
        exc_Ha = [e * eV_to_Ha for e in exc_eV]
        print(f"\n{'=' * 60}")
        print("ROOTED TD-DFT — EXPLICIT EXCITATION ENERGIES")
        print(f"{'=' * 60}")
        # Applied PER SPIN. Each channel has its own HOMO energy and LUMO index, so a
        # composite built from alpha and copied to beta gives the beta channel alpha's
        # spectrum -- silently harmless for closed shells, badly wrong for open ones.
        # (Bug fixed 2026-08-27; the Root-file branch below was already per-spin.)
        def _composite(energies, lumo_idx, label):
            homo_energy = float(energies[lumo_idx - 1])
            print(f"  [{label}] E_HOMO = {homo_energy:.6f} H")
            comp = energies.copy()
            for i, omega in enumerate(exc_Ha):
                virtual_idx = lumo_idx + i
                if virtual_idx >= len(comp):
                    break
                eff = homo_energy + omega
                comp[virtual_idx] = eff
                print(f"    [{label}] Root{i+1}: ω = {exc_eV[i]:.4f} eV → "
                      f"ε_eff = {eff:.6f} H (base: {float(energies[virtual_idx]):.6f} H)")
            return comp

        energies_alpha = _composite(energies_alpha, lumo_alpha, "alpha")
        if is_unrestricted:
            energies_beta = _composite(energies_beta, lumo_beta, "beta")
        else:
            energies_beta = energies_alpha.copy()
        rooted_applied = True
        print("+ Composite orbital energies built from explicit excitation energies")
    elif not args.no_rooted:
        rooted_dir = Path(args.rooted_dir) if args.rooted_dir else Path.cwd()
        # Apply to alpha energies using alpha window end
        energies_alpha, rooted_alpha = apply_rooted_approach(
            energies_alpha, lumo_alpha, window_alpha[1], rooted_dir)
        # Apply to beta energies using beta window end
        energies_beta, rooted_beta = apply_rooted_approach(
            energies_beta, lumo_beta, window_beta[1], rooted_dir)
        rooted_applied = rooted_alpha or rooted_beta

    # Fermi energy — uses rooted LUMO if rooted was applied
    homo_e_alpha = energies_alpha[homo_alpha]
    lumo_e_alpha = energies_alpha[lumo_alpha]
    ef_alpha_default = (homo_e_alpha + lumo_e_alpha) / 2.0

    if is_unrestricted:
        homo_e_beta = energies_beta[homo_beta]
        lumo_e_beta = energies_beta[lumo_beta]
        ef_beta_default = (homo_e_beta + lumo_e_beta) / 2.0

        ef_alpha = args.ef_alpha if args.ef_alpha is not None else (args.ef if args.ef is not None else ef_alpha_default)
        ef_beta = args.ef_beta if args.ef_beta is not None else (args.ef if args.ef is not None else ef_beta_default)

        print(f"\nFermi energy{' (using rooted LUMO)' if rooted_applied else ''}:")
        print(f"  Alpha: {ef_alpha:.10f} H ({ef_alpha * HA_TO_EV:.4f} eV)")
        print(f"  Beta:  {ef_beta:.10f} H ({ef_beta * HA_TO_EV:.4f} eV)")
    else:
        ef_alpha = args.ef if args.ef is not None else ef_alpha_default
        ef_beta = ef_alpha
        print(f"\nFermi energy{' (using rooted LUMO)' if rooted_applied else ''}: "
              f"{ef_alpha:.10f} H ({ef_alpha * HA_TO_EV:.4f} eV)")

    # Threshold
    if args.threshold is not None:
        threshold = args.threshold
    else:
        thr_input = input(f"\nG_rs threshold (default 0.001): ").strip()
        threshold = float(thr_input) if thr_input else 0.001
    print(f"Threshold: {threshold}")

    # Output
    if args.output_prefix:
        output_prefix = args.output_prefix
    else:
        # Derive from directory name
        cwd = Path.cwd()
        parent = cwd.parent
        if parent.name == "turbomole":
            output_prefix = parent.parent.name + "_turbo"
        else:
            output_prefix = parent.name + "_turbo"

    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    occupation_aware = args.occupation_aware
    if occupation_aware:
        print(f"\nOccupation-aware mode: ON (skipping non-aufbau empty MOs)")

    # --- Organize coefficients ---
    print(f"\nOrganizing NAO coefficients by atom...")
    t0 = time.time()
    # A key can be present and still carry nothing: the save scripts default to
    # np.zeros(0, dtype=np.int8) when a reader supplied no tiers, and a zero-length
    # array passed to organize_coefficients indexes out of bounds on the first NAO --
    # a traceback far from the cause. Treat empty as absent and derive instead.
    def _stored(key):
        a = arrays[key] if key in arrays else None
        return a if a is not None and getattr(a, 'size', 0) else None

    tiers_alpha = _stored('nao_tiers_alpha')
    tiers_beta  = _stored('nao_tiers_beta')
    if tiers_alpha is None or tiers_beta is None:
        # Checkpoint predates tier storage; the classification is recoverable from the
        # AO basis and occupancies alone, so derive it rather than fall back silently.
        try:
            from compute_nao_from_turbomole import derive_nao_tiers
            # derive BOTH: a file carrying only one channel's tiers would otherwise
            # filter alpha by tier and beta by the occupancy cut, which is exactly the
            # uncontrolled comparison the shared NAO basis exists to prevent
            tiers_alpha = derive_nao_tiers(ao_basis, nao_occ_alpha)
            tiers_beta  = derive_nao_tiers(ao_basis, nao_occ_beta)
            print("   NAO tiers derived from the checkpoint (not stored, or stored empty).")
        except Exception as _e:
            print(f"   Could not derive NAO tiers ({_e}); valence selection falls back.")
    coeffs_alpha = organize_coefficients(mo_in_nao_alpha, nao_occ_alpha, ao_basis,
                                         nao_tiers=tiers_alpha, selection=args.valence_selection,
                                         val_threshold=args.val_threshold)
    coeffs_beta = organize_coefficients(mo_in_nao_beta, nao_occ_beta, ao_basis,
                                        nao_tiers=tiers_beta, selection=args.valence_selection,
                                        val_threshold=args.val_threshold)
    print(f"  Done in {time.time() - t0:.1f} s")
    print(f"  Alpha: {len(coeffs_alpha)} atoms with valence p/d orbitals")
    print(f"  Beta:  {len(coeffs_beta)} atoms with valence p/d orbitals")

    # Build occupation arrays for occupation-aware mode
    occ_array_alpha = build_occupation_array(n_occ_alpha, len(energies_alpha))
    occ_array_beta = build_occupation_array(n_occ_beta, len(energies_beta))

    # --- Compute G_rs ---
    print(f"\n{'=' * 70}")
    print(f"COMPUTING GREEN'S FUNCTION VALUES")
    print(f"{'=' * 70}")

    if axis_mode:
        decompose = True
        active_channels = ['σ', 'π', 'δ']
    else:
        decompose = len(family) > 1
        active_channels = [ch for ch in 'spd' if ch in family]
    n_pairs = n_atoms * (n_atoms - 1) // 2

    for plane in planes:
        if axis_mode:
            orbital_types = ['s', 'px', 'py', 'pz', 'dxy', 'dxz', 'dyz', 'dx2y2', 'dz2']
            print(f"\n{'─' * 70}")
            print(f"AXIS-PROJECTED σ/π/δ DECOMPOSITION")
            print(f"  σ = s + p∥(bond axis) + d(m=0)")
            print(f"  π = p⊥(bond axis) + d(|m|=1)")
            print(f"  δ = d(|m|=2)")
            print(f"{'─' * 70}")
        else:
            orbital_types = get_orbital_types(plane, family)
            if len(planes) > 1:
                print(f"\n{'─' * 70}")
                print(f"PLANE: {plane}  orbitals: {', '.join(orbital_types)}")
                print(f"{'─' * 70}")

        channel_types_for_plane = {}
        if decompose and not axis_mode:
            for ch in active_channels:
                channel_types_for_plane[ch] = PLANE_ORBITAL_MAP[plane][ch]

        for spin in spin_channels:
            if spin == 'alpha' or spin == 'restricted':
                coeffs_spin = coeffs_alpha
                energies_spin = energies_alpha
                ef_spin = ef_alpha
                window = window_alpha
                occ_spin = occ_array_alpha
                mo_idx = mo_indices_alpha
            else:
                coeffs_spin = coeffs_beta
                energies_spin = energies_beta
                ef_spin = ef_beta
                window = window_beta
                occ_spin = occ_array_beta
                mo_idx = mo_indices_beta

            spin_label = spin.upper() if spin != 'restricted' else 'RESTRICTED'
            if mo_idx is not None:
                print(f"\n  {spin_label} spin, specific MOs {mo_idx} "
                      f"({len(mo_idx)} MOs)...")
            else:
                print(f"\n  {spin_label} spin, window MOs {window[0]}-{window[1]} "
                      f"({window[1] - window[0] + 1} MOs)...")

            t0 = time.time()
            results = []

            for i, atom1 in enumerate(atoms):
                for j in range(i + 1, len(atoms)):
                    atom2 = atoms[j]

                    if axis_mode:
                        pos1 = np.array([atom1.x, atom1.y, atom1.z])
                        pos2 = np.array([atom2.x, atom2.y, atom2.z])
                        d = pos2 - pos1
                        norm_d = np.linalg.norm(d)
                        if norm_d < 1e-10:
                            continue
                        e_rs = d / norm_d
                        u1_ax, u2_ax = perp_basis(e_rs)
                        sigma, pi, delta = compute_grs_axis_decompose(
                            atom1_num=atom1.number, atom2_num=atom2.number,
                            coeffs_spin=coeffs_spin, ef=ef_spin,
                            energies=energies_spin,
                            start_idx=window[0], end_idx=window[1],
                            e_rs=e_rs, u1=u1_ax, u2=u2_ax,
                            occupation_aware=occupation_aware,
                            occupations=occ_spin,
                            mo_indices=mo_idx,
                        )
                        if sigma is not None:
                            grs = sigma + pi + delta
                            channels = {'σ': sigma, 'π': pi, 'δ': delta}
                        else:
                            grs = None
                            channels = {}
                    else:
                        grs_kwargs = dict(
                            atom1_num=atom1.number, atom2_num=atom2.number,
                            coeffs_spin=coeffs_spin, ef=ef_spin,
                            energies=energies_spin,
                            start_idx=window[0], end_idx=window[1],
                            abs_per_type=(plane == 'SHELL_SUM'),
                            occupation_aware=occupation_aware,
                            occupations=occ_spin,
                            mo_indices=mo_idx,
                        )

                        if decompose:
                            channels = {}
                            for ch, ch_types in channel_types_for_plane.items():
                                ch_grs = compute_grs_per_spin(
                                    orbital_types=ch_types, **grs_kwargs)
                                if ch_grs is not None:
                                    channels[ch] = ch_grs
                            grs = sum(channels.values()) if channels else None
                        else:
                            grs = compute_grs_per_spin(
                                orbital_types=orbital_types, **grs_kwargs)
                            channels = {}

                    if grs is not None and abs(grs) >= threshold:
                        dist = atom1.distance_to(atom2)
                        results.append((atom1.number, atom2.number, grs, dist, channels))

            results.sort(key=lambda x: abs(x[2]), reverse=True)
            elapsed = time.time() - t0

            print(f"  {len(results)} pairs above threshold ({elapsed:.1f} s)")

            write_results(
                results, plane, spin, ef_spin, window,
                homo_alpha if spin in ('alpha', 'restricted') else homo_beta,
                lumo_alpha if spin in ('alpha', 'restricted') else lumo_beta,
                n_mo_alpha if spin in ('alpha', 'restricted') else n_mo_beta,
                threshold, orbital_types, family, output_prefix, output_dir,
                is_unrestricted, occupation_aware,
                rooted=rooted_applied,
                mo_indices=mo_idx,
                decompose=decompose,
                active_channels=active_channels,
            )

            if results:
                print(f"  Top 5:")
                for a1, a2, grs, dist, channels in results[:5]:
                    line = f"    {a1:5d}-{a2:5d}: G_rs={grs:12.6f}, dist={dist:8.3f} A"
                    if channels:
                        parts = []
                        for ch in active_channels:
                            if ch in channels:
                                pct = channels[ch] / grs * 100 if abs(grs) > 1e-15 else 0.0
                                parts.append(f"{ch}={channels[ch]:+.4f} ({pct:+.0f}%)")
                        line += f"  [{' '.join(parts)}]"
                    print(line)

    print(f"\n{'=' * 70}")
    print(f"CALCULATION COMPLETE")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
