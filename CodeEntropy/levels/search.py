"""These functions find neighbors.

There are different functions for different types of neighbor searching.
Currently RAD is the default with grid as an alternative.
"""

import MDAnalysis as mda
import numpy as np
import os
from numba import njit


@njit
def _distances_without_temporaries(coms, center, dimensions):
    """Fuse the reference distance arithmetic without fastmath or changed PBC."""
    result = np.empty(len(coms), dtype=np.float64)
    half = 0.5 * dimensions
    for i in range(len(coms)):
        squared = 0.0
        for d in range(3):
            delta = coms[i, d] - center[d]
            if delta > half[d]:
                delta -= dimensions[d]
            if delta < -half[d]:
                delta += dimensions[d]
            squared += delta * delta
        result[i] = np.sqrt(squared)
    return result


def _nearest_stable(distances, limit=30):
    """Return exactly the prefix of a stable full sort, including cutoff ties."""
    if len(distances) <= limit or not np.isfinite(distances).all():
        return np.argsort(distances, kind="mergesort")[:limit]
    cutoff = np.partition(distances, limit - 1)[limit - 1]
    candidates = np.flatnonzero(distances <= cutoff)
    return candidates[np.argsort(distances[candidates], kind="mergesort")[:limit]]


@njit
def _apply_pbc(vec, dimensions, half_dimensions):
    """
    Apply minimum image convention for periodic boundary conditions.

    Args:
        vec (np.ndarray):
            Vector to wrap.
        dimensions (np.ndarray):
            Simulation box dimensions.
        half_dimensions (np.ndarray):
            Half box lengths.

    Returns:
        np.ndarray:
            Wrapped vector.
    """
    for d in range(3):
        if vec[d] > half_dimensions[d]:
            vec[d] -= dimensions[d]
        elif vec[d] < -half_dimensions[d]:
            vec[d] += dimensions[d]
    return vec


@njit
def _rad_blocking_loop(i_coords, sorted_indices, sorted_distances, coms, dimensions):
    """
    Perform RAD neighbor selection using a blocking criterion.

    This is a Numba-compiled implementation of the RAD algorithm, which
    determines whether a molecule j is a neighbor of molecule i by checking
    whether any closer molecule k blocks j based on angular and distance
    relationships.

    The criterion is based on:

        (1 / r_ij)^2 > (1 / r_ik)^2 * cos(theta_jik)

    where k blocks j if the inequality holds.

    Args:
        i_coords (np.ndarray):
            Coordinates of the central molecule.
        sorted_indices (np.ndarray):
            Indices of molecules sorted by distance from i.
        sorted_distances (np.ndarray):
            Distances corresponding to sorted_indices.
        coms (np.ndarray):
            Precomputed center of mass coordinates for all molecules.
        dimensions (np.ndarray):
            Simulation box dimensions for periodic boundary conditions.

    Returns:
        np.ndarray:
            Indices of molecules that belong to the RAD neighbor shell.
    """
    n = sorted_indices.shape[0]
    limit = min(n, 30)

    half_dimensions = 0.5 * dimensions

    inv_r2 = 1.0 / (sorted_distances * sorted_distances)

    shell = np.empty(limit, dtype=np.int64)
    count = 0

    for y in range(limit):
        j_idx = sorted_indices[y]
        r_ij = sorted_distances[y]
        j_coords = coms[j_idx]

        ba = j_coords - i_coords
        ba = _apply_pbc(ba, dimensions, half_dimensions)

        blocked = False

        for z in range(y):
            k_idx = sorted_indices[z]
            r_ik = sorted_distances[z]

            if r_ik > r_ij:
                continue

            k_coords = coms[k_idx]

            ac = k_coords - j_coords
            ac = _apply_pbc(ac, dimensions, half_dimensions)

            dist_ac2 = (ac * ac).sum()

            denom = -2.0 * r_ik * r_ij
            if denom == 0.0:
                continue

            costheta = (dist_ac2 - r_ik * r_ik - r_ij * r_ij) / denom

            if inv_r2[y] < inv_r2[z] * costheta:
                blocked = True
                break

        if not blocked:
            shell[count] = j_idx
            count += 1

    return shell[:count]


class Search:
    """
    Class for functions to find neighbors.
    """

    def __init__(self, backend=None):
        """
        Initialize the Search class.

        This class includes frame-safe caching of fragment COMs and
        system dimensions to avoid recomputation while preserving
        identical results to the original implementation.
        """
        self._cached_frame = None
        self._cached_fragments = None
        self._cached_coms = None
        self._cached_dimensions = None
        self._cached_universe = None
        self._backend = backend or os.environ.get("CODEENTROPY_RAD_BACKEND", "cpu")
        if self._backend not in ("cpu", "gpu"):
            raise ValueError("CODEENTROPY_RAD_BACKEND must be 'cpu' or 'gpu'")
        self._gpu_candidates = None

    def _update_cache(self, universe, frame_index: int) -> None:
        """Update cached MDAnalysis data for a specific selected frame.

        Args:
            universe: MDAnalysis universe object containing the active analysis system.
            frame_index: Frame index in the active analysis-universe index space.

        Returns:
            None.
        """
        frame_index = int(frame_index)

        if self._cached_frame == frame_index and self._cached_universe is universe:
            return

        fragments = universe.atoms.fragments
        coms = np.array([frag.center_of_mass() for frag in fragments])

        self._cached_fragments = fragments
        self._cached_coms = coms
        self._cached_dimensions = universe.dimensions[:3]
        self._cached_frame = frame_index
        self._cached_universe = universe
        self._gpu_candidates = None
        if self._backend == "gpu":
            from CodeEntropy.levels.gpu_search import prepare_candidates
            self._gpu_candidates = prepare_candidates(coms, self._cached_dimensions,
                                                       self._get_distances)

    def _get_distances(self, coms, i_coords, dimensions):
        """
        Compute distances between a central coordinate and all fragment COMs
        using periodic boundary conditions.

        Args:
            coms (np.ndarray):
                Array of fragment center of mass coordinates.
            i_coords (np.ndarray):
                Coordinates of the reference (central) molecule.
            dimensions (np.ndarray):
                Simulation box dimensions.

        Returns:
            np.ndarray:
                Distances from the central molecule to all fragments.
        """
        return _distances_without_temporaries(coms, i_coords, dimensions)

    def get_RAD_neighbors(self, universe, mol_id, frame_source, frame_index):
        """Find RAD neighbours of a molecule at one selected frame.

        Args:
            universe: MDAnalysis universe object for the active analysis system.
            mol_id: Index of the central molecule.
            frame_source: FrameSource controlling selected trajectory access.
            frame_index: Frame index in the active analysis-universe index space.

        Returns:
            Indices of neighbouring molecules identified by RAD.
        """
        frame_index = int(frame_index)
        # Keep the first seek: covariance processing may have made molecules whole
        # in-place. Reload once before caching the reference-frame COMs, then reuse
        # those values for every molecule in this frame.
        if (self._cached_frame != frame_index
                or self._cached_universe is not universe
                or universe.trajectory.ts.frame != frame_index):
            frame_source.seek(frame_index)
        self._update_cache(universe, frame_index)

        fragments = self._cached_fragments
        coms = self._cached_coms
        dimensions = self._cached_dimensions

        number_molecules = len(fragments)
        central_position = coms[mol_id]

        if self._gpu_candidates is not None:
            candidates = self._gpu_candidates[mol_id]
            if candidates is not None:
                return _rad_blocking_loop(central_position, candidates[0], candidates[1],
                                          coms, dimensions)

        distances_array = self._get_distances(coms, central_position, dimensions)

        indices = np.arange(number_molecules)
        mask = indices != mol_id
        filtered_indices = indices[mask]
        filtered_distances = distances_array[mask]

        # The blocking loop consumes only the nearest 30 molecules. Preserve the
        # stable reference prefix exactly without sorting the entire solvent box.
        order = _nearest_stable(filtered_distances)

        sorted_indices = filtered_indices[order]
        sorted_distances = filtered_distances[order]

        neighbor_indices = _rad_blocking_loop(
            central_position,
            sorted_indices,
            sorted_distances,
            coms,
            dimensions,
        )

        return neighbor_indices

    def get_grid_neighbors(
        self, universe, mol_id, highest_level, frame_source, frame_index
    ):
        """Find neighbours using MDAnalysis grid search at one selected frame.

        Args:
            universe: MDAnalysis universe object for the active analysis system.
            mol_id: Index of the molecule of interest.
            highest_level: Molecule level, e.g. ``"united_atom"`` or ``"residue"``.
            frame_source: FrameSource controlling selected trajectory access.
            frame_index: Frame index in the active analysis-universe index space.

        Returns:
            Fragment indices of neighbouring molecules.
        """
        frame_index = int(frame_index)
        frame_source.seek(frame_index)

        fragments = universe.atoms.fragments
        fragment = fragments[mol_id]

        search_object = mda.lib.NeighborSearch.AtomNeighborSearch(universe.atoms)

        selection_string = f"index {fragment.indices[0]}:{fragment.indices[-1]}"
        molecule_atom_group = universe.select_atoms(selection_string)

        if highest_level == "united_atom":
            search = mda.lib.NeighborSearch.AtomNeighborSearch.search(
                search_object,
                molecule_atom_group,
                radius=3.0,
                level="A",
            )
            neighbors = search - molecule_atom_group
        else:
            search = mda.lib.NeighborSearch.AtomNeighborSearch.search(
                search_object,
                molecule_atom_group,
                radius=3.5,
                level="R",
            )
            neighbors = search - fragment.residues
            neighbors = neighbors.atoms

        return neighbors.fragindices
