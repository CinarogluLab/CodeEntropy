from unittest.mock import MagicMock

import pytest

from CodeEntropy.levels.hierarchy import HierarchyBuilder


def _heavy_atoms_group(n_atoms: int, n_residues: int):
    heavy = MagicMock()
    heavy.__len__.return_value = n_atoms
    heavy.residues = [MagicMock() for _ in range(n_residues)]
    return heavy


def test_select_levels_assigns_expected_levels():
    hb = HierarchyBuilder()

    u = MagicMock()
    u.atoms = MagicMock()
    frag0 = MagicMock()
    frag1 = MagicMock()
    frag2 = MagicMock()

    frag0.select_atoms.return_value = _heavy_atoms_group(n_atoms=1, n_residues=1)
    frag1.select_atoms.return_value = _heavy_atoms_group(n_atoms=2, n_residues=1)
    frag2.select_atoms.return_value = _heavy_atoms_group(n_atoms=3, n_residues=2)

    u.atoms.fragments = [frag0, frag1, frag2]

    n_mols, levels = hb.select_levels(u)

    assert n_mols == 3
    assert levels[0] == ["united_atom"]
    assert levels[1] == ["united_atom", "residue"]
    assert levels[2] == ["united_atom", "residue", "polymer"]


def test_get_beads_unknown_level_raises():
    hb = HierarchyBuilder()
    with pytest.raises(ValueError):
        hb.get_beads(MagicMock(), "nonsense")


def test_get_beads_polymer_returns_single_all_selection():
    hb = HierarchyBuilder()
    mol = MagicMock()
    mol.select_atoms.return_value = "ALL"
    out = hb.get_beads(mol, "polymer")
    assert out == ["ALL"]
    mol.select_atoms.assert_called_once_with("all")


def test_get_beads_residue_returns_residue_atomgroups():
    hb = HierarchyBuilder()

    mol = MagicMock()
    r0 = MagicMock()
    r1 = MagicMock()
    r0.atoms = "R0_ATOMS"
    r1.atoms = "R1_ATOMS"
    mol.residues = [r0, r1]

    out = hb.get_beads(mol, "residue")
    assert out == ["R0_ATOMS", "R1_ATOMS"]


def test_get_beads_united_atom_no_heavy_atoms_falls_back_to_all():
    hb = HierarchyBuilder()

    mol = MagicMock()
    heavy = MagicMock()
    heavy.__len__.return_value = 0
    mol.select_atoms.side_effect = lambda sel: (
        heavy if sel == "prop mass > 1.1" else "ALL"
    )
    out = hb.get_beads(mol, "united_atom")

    assert out == ["ALL"]


def test_united_atom_beads_match_selection_for_partial_and_unsorted_containers():
    import MDAnalysis as mda
    import numpy as np

    u = mda.Universe.empty(6)
    u.add_TopologyAttr('masses', [12., 1.008, 16., 1.008, 1.008, 12.])
    u.add_TopologyAttr('bonds', [(0, 1), (0, 2), (2, 3), (2, 4)])
    hb = HierarchyBuilder()
    for indices in ([0, 1, 2, 3, 4, 5], [4, 2, 0], [5, 2, 2, 3]):
        mol = u.atoms[indices]
        expected = [mol.select_atoms(
            f'index {a.index} or ((prop mass <= 1.1) and bonded index {a.index})'
        ).indices for a in mol.select_atoms('prop mass > 1.1')]
        actual = hb.get_beads(mol, 'united_atom')
        assert len(actual) == len(expected)
        for bead, reference in zip(actual, expected):
            np.testing.assert_array_equal(bead.indices, reference)
