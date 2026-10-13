from types import SimpleNamespace
from unittest.mock import MagicMock

from CodeEntropy.entropy.water import WaterEntropy


def _make_fake_universe_with_water():
    universe = MagicMock()

    water_selection = MagicMock()

    water_residues = MagicMock()
    water_residues.resnames = ["WAT"]
    water_residues.__len__.return_value = 1

    water_selection.residues = water_residues
    water_selection.atoms = [1, 2, 3]

    universe.select_atoms.return_value = water_selection
    return universe


def test_water_entropy_calls_solver_and_logs_components():
    args = SimpleNamespace(temperature=298.0)
    reporter = MagicMock()

    Sorient_dict = {1: {"WAT": [3.0, 5]}}

    covariances = SimpleNamespace(counts={(7, "WAT"): 2})

    vibrations = SimpleNamespace(
        translational_S={(7, "WAT"): [1.0, 2.0]},
        rotational_S={(7, "WAT"): 4.0},
    )

    solver = MagicMock(return_value=(Sorient_dict, covariances, vibrations, None, 123))

    we = WaterEntropy(args=args, reporter=reporter, solver=solver)

    we._solute_id_to_resname = MagicMock(return_value="SOL")

    universe = _make_fake_universe_with_water()

    we.calculate_and_log(universe=universe, start=0, end=10, step=1, group_id=9)

    solver.assert_called_once()

    reporter.add_residue_data.assert_any_call(
        9, "WAT", "Water", "Orientational", 5, 3.0
    )

    reporter.add_residue_data.assert_any_call(
        9, "SOL", "Water", "Transvibrational", 2, 3.0
    )

    reporter.add_residue_data.assert_any_call(
        9, "SOL", "Water", "Rovibrational", 2, 4.0
    )

    reporter.add_group_label.assert_called_once()


def test_water_entropy_handles_empty_solver_results_gracefully():
    args = SimpleNamespace(temperature=298.0)
    reporter = MagicMock()

    solver = MagicMock(
        return_value=(
            {},
            SimpleNamespace(counts={}),
            SimpleNamespace(translational_S={}, rotational_S={}),
            None,
            0,
        )
    )

    we = WaterEntropy(args=args, reporter=reporter, solver=solver)
    we._solute_id_to_resname = MagicMock(return_value="SOL")

    universe = _make_fake_universe_with_water()

    we.calculate_and_log(universe=universe, start=0, end=10, step=1, group_id=1)

    reporter.add_residue_data.assert_not_called()
    reporter.add_group_label.assert_called_once()


def test_water_group_label_handles_multiple_water_resnames():
    args = SimpleNamespace(temperature=298.0)
    reporter = MagicMock()

    solver = MagicMock(
        return_value=(
            {},
            SimpleNamespace(counts={}),
            SimpleNamespace(translational_S={}, rotational_S={}),
            None,
            0,
        )
    )
    we = WaterEntropy(args=args, reporter=reporter, solver=solver)
    we._solute_id_to_resname = MagicMock(return_value="SOL")

    universe = MagicMock()
    water_selection = MagicMock()

    water_residues = MagicMock()
    water_residues.resnames = ["WAT", "TIP3"]
    water_residues.__len__.return_value = 2

    water_selection.residues = water_residues
    water_selection.atoms = [1, 2, 3, 4]

    universe.select_atoms.return_value = water_selection

    we.calculate_and_log(universe=universe, start=0, end=10, step=1, group_id=1)

    reporter.add_group_label.assert_called_once()


def test_log_group_label_defaults_to_WAT_when_no_residue_names_match():
    args = SimpleNamespace(temperature=298.0)
    reporter = MagicMock()

    solver = MagicMock(
        return_value=(
            {},
            SimpleNamespace(counts={}),
            SimpleNamespace(translational_S={}, rotational_S={}),
            None,
            0,
        )
    )
    we = WaterEntropy(args=args, reporter=reporter, solver=solver)

    universe = MagicMock()

    water_selection = MagicMock()
    water_residues = MagicMock()
    water_residues.resnames = ["WAT"]
    water_residues.__len__.return_value = 1

    water_selection.residues = water_residues
    water_selection.atoms = [1, 2, 3]
    universe.select_atoms.return_value = water_selection

    Sorient_dict = {1: {"TIP3": [1.0, 1]}}

    we._log_group_label(universe, Sorient_dict, group_id=7)

    reporter.add_group_label.assert_called_once()
    _, residue_group, *_ = reporter.add_group_label.call_args.args
    assert residue_group == "WAT"


def test_log_group_label_defaults_to_WAT_when_no_names_match():
    args = SimpleNamespace(temperature=298.0)
    reporter = MagicMock()
    we = WaterEntropy(args=args, reporter=reporter, solver=MagicMock())

    universe = MagicMock()
    water_selection = MagicMock()

    residues = MagicMock()
    residues.resnames = ["WAT"]
    residues.__len__.return_value = 2

    water_selection.residues = residues
    water_selection.atoms = [1, 2, 3, 4]
    universe.select_atoms.return_value = water_selection

    Sorient_dict = {1: {"TIP3": [1.0, 2]}}

    we._log_group_label(universe, Sorient_dict, group_id=7)

    reporter.add_group_label.assert_called_once()

    assert reporter.add_group_label.call_args.args[1] == "WAT"


def test_solute_id_to_resname_strips_suffix_after_last_underscore():
    assert WaterEntropy._solute_id_to_resname("ALA_0") == "ALA"
    assert WaterEntropy._solute_id_to_resname("ALA_BLA_12") == "ALA_BLA"


def test_solute_id_to_resname_returns_string_when_no_underscore():
    assert WaterEntropy._solute_id_to_resname("WAT") == "WAT"
    assert WaterEntropy._solute_id_to_resname(123) == "123"


def test_water_entropy_export_labels_all_fields_and_real_counts():
    args = SimpleNamespace(temperature=298.0)
    reporter = SimpleNamespace(
        add_residue_data=MagicMock(),
        add_group_label=MagicMock(),
    )
    orient_fields = [7.0, 40, 5.5, 5.0, 3.2, 0.21, 9.1, 8.7]
    Sorient_dict = {12: {"LIG": orient_fields}}
    covariances = SimpleNamespace(counts={("LIG_12", "SOL"): 40})
    vibrations = SimpleNamespace(
        translational_S={("LIG_12", "SOL"): [10.0, 20.0, 16.0]},
        rotational_S={("LIG_12", "SOL"): [5.0, 6.0, 7.0]},
    )
    solver = MagicMock(return_value=(Sorient_dict, covariances, vibrations, None, 4))
    we = WaterEntropy(args=args, reporter=reporter, solver=solver)

    we.calculate_and_log(
        universe=_make_fake_universe_with_water(), start=0, end=4, step=1, group_id=2
    )

    export = reporter.water_entropy_export["2"]
    assert export["n_frames"] == 4
    (orient,) = export["orientational"]
    assert orient["nearest_resid"] == 12 and orient["nearest_resname"] == "LIG"
    assert [orient[name] for name in WaterEntropy.ORIENTATIONAL_FIELDS] == orient_fields
    assert orient["mean_waters"] == 10.0
    (vib,) = export["vibrational"]
    assert vib["nearest_resname"] == "LIG" and vib["nearest_resid"] == "12"
    assert vib["solvent_resname"] == "SOL"
    assert vib["S_trans"] == 46.0 and vib["S_rot"] == 18.0
    assert vib["observations"] == 40 and vib["mean_waters"] == 10.0


def test_water_entropy_export_never_invents_counts():
    reporter = SimpleNamespace(add_residue_data=MagicMock(), add_group_label=MagicMock())
    vibrations = SimpleNamespace(
        translational_S={("MET_1", "SOL"): [1.0]}, rotational_S={("MET_1", "SOL"): [1.0]}
    )
    solver = MagicMock(
        return_value=({}, SimpleNamespace(counts={}), vibrations, None, 3)
    )
    we = WaterEntropy(args=SimpleNamespace(temperature=298.0), reporter=reporter, solver=solver)

    we.calculate_and_log(
        universe=_make_fake_universe_with_water(), start=0, end=3, step=1, group_id=0
    )

    (vib,) = reporter.water_entropy_export["0"]["vibrational"]
    assert vib["observations"] is None and vib["mean_waters"] is None


def test_bulk_export_keys_match_between_vibrational_and_orientational():
    reporter = SimpleNamespace()
    universe = _make_fake_universe_with_water()
    universe.trajectory = list(range(20))
    bulk = MagicMock(
        return_value=(
            {"SOL": {"SOL": [1.0, 30, 4.5, 4.5, 2.0, 0.2, 6.0, 6.0]}},
            SimpleNamespace(counts={("SOL", "SOL"): 30}),
            SimpleNamespace(translational_S={("SOL", "SOL"): [15.0, 16.0, 16.0]},
                            rotational_S={("SOL", "SOL"): [6.0, 6.0, 5.0]}),
        )
    )
    we = WaterEntropy(args=SimpleNamespace(temperature=298.0), reporter=reporter,
                      solver=MagicMock(), bulk_solver=bulk)

    we.calculate_bulk_export(universe=universe, start=0, end=20, step=2, group_id=0)

    bulk.assert_called_once_with(universe, 0, 20, 2, 298.0)
    export = reporter.water_entropy_bulk_export["0"]
    assert export["n_frames"] == 10
    (vib,), (orient,) = export["vibrational"], export["orientational"]
    assert (vib["nearest_resname"], vib["nearest_resid"]) == ("SOL", "SOL")
    assert (orient["nearest_resname"], str(orient["nearest_resid"])) == ("SOL", "SOL")
    assert vib["mean_waters"] == orient["mean_waters"] == 3.0
    assert "S_orient_non_hb_Nw" in export["field_definitions"]


def test_export_lists_ambiguous_solute_residue_keys():
    from CodeEntropy.entropy.water import _ambiguous_residue_keys

    residues = [SimpleNamespace(resname=n, resid=i) for n, i in
                (("ALA", 1), ("GLY", 2), ("ALA", 1), ("LIG", 1))]
    universe = MagicMock()
    universe.select_atoms.return_value = SimpleNamespace(residues=residues)

    assert _ambiguous_residue_keys(universe) == ["ALA_1"]
    universe.select_atoms.assert_called_once_with("not water")


def test_export_records_solver_input_composition():
    from CodeEntropy.entropy.water import _input_composition

    water = SimpleNamespace(residues=[object()] * 5)

    class Residues(list):
        resnames = ["LIG", "ALA", "ALA"]

    non_water = SimpleNamespace(residues=Residues([1, 2, 3]))
    universe = MagicMock()
    universe.atoms = [0] * 30
    universe.filename = "sys.tpr"
    universe.trajectory = MagicMock(filename="sys.trr", __len__=lambda self: 7)
    universe.select_atoms.side_effect = lambda sel: water if sel == "water" else non_water

    assert _input_composition(universe) == {
        "n_atoms": 30, "n_water_residues": 5, "n_non_water_residues": 3,
        "non_water_resnames": ["ALA", "LIG"],
        "non_water_resname_counts": {"ALA": 2, "LIG": 1}, "topology_file": "sys.tpr",
        "trajectory_file": "sys.trr", "trajectory_n_frames": 7,
    }
