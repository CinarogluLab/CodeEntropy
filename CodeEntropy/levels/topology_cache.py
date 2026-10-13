"""Per-Universe cache of bond-adjacency data.

Shared by hierarchy.py and levels/nodes/axes_topology.py, which each need
"atoms bonded to atom N" during static (one-time, per-Universe) topology
construction. Bonds are static for the lifetime of a Universe - they don't
change frame to frame - so this is computed once and reused, rather than
each caller independently querying MDAnalysis's Atom.bonded_atoms property
(which itself does a non-trivial bond-graph lookup) for the same atoms
repeatedly across different call sites.
"""

import weakref

import numpy as np

_BOND_ADJACENCY_CACHE = weakref.WeakKeyDictionary()


def get_bonded_indices(system) -> list[np.ndarray]:
    """Return a list where element i is the sorted array of atom indices
    bonded to atom i, cached per Universe.

    :param system: mdanalysis Universe (or AtomGroup with a .universe)
    """
    universe = getattr(system, "universe", system)
    cache = _BOND_ADJACENCY_CACHE.get(universe)
    if cache is None:
        n_atoms = universe.atoms.n_atoms
        bonded_lists: list[list[int]] = [[] for _ in range(n_atoms)]
        bond_indices = universe.atoms.bonds.to_indices()
        for a, b in bond_indices:
            bonded_lists[a].append(b)
            bonded_lists[b].append(a)
        cache = [np.asarray(sorted(lst), dtype=int) for lst in bonded_lists]
        _BOND_ADJACENCY_CACHE[universe] = cache
    return cache
