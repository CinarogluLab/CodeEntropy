"""Unit tests for the EE-MCC binding entropy assembly."""

from __future__ import annotations

import copy
import json
import math

import pytest

from CodeEntropy.results.binding_entropy import (
    GAS_CONSTANT,
    BindingEntropyError,
    assemble,
)

S_POS = GAS_CONSTANT * math.log(55.5)


def _composition(resnames, n_water=1000):
    counts = {}
    for name in resnames:
        counts[name] = counts.get(name, 0) + 1
    return {"n_atoms": 3 * n_water + 10 * len(resnames), "n_water_residues": n_water,
            "n_non_water_residues": len(resnames), "non_water_resnames": sorted(set(resnames)),
            "non_water_resname_counts": dict(sorted(counts.items())),
            "topology_file": "t.tpr", "trajectory_file": "t.trr", "trajectory_n_frames": 10}


ROLE_RESNAMES = {"complex": ["MET", "LIG"], "receptor": ["MET"], "ligand": ["LIG"]}


def _shell(entries, n_frames=10, resnames=("MET", "LIG")):
    """entries: (resname, resid, mean_waters, S_trans, S_rot, S_orient)."""
    vib, orient = [], []
    for resname, resid, n, s_t, s_r, s_o in entries:
        vib.append(
            {"solute_key": f"{resname}_{resid}", "nearest_resname": resname,
             "nearest_resid": str(resid), "solvent_resname": "SOL", "S_trans": s_t,
             "S_rot": s_r, "observations": n * n_frames, "mean_waters": n}
        )
        orient.append(
            {"nearest_resname": resname, "nearest_resid": resid, "observations": n * n_frames,
             "mean_waters": n, "S_orient_hb_biased": -1.0, "S_orient_non_hb_Nc": s_o,
             "S_orient_non_hb_Nw": -2.0}
        )
    return {"9": {"n_frames": n_frames, "vibrational": vib, "orientational": orient,
                  "input_composition": _composition(list(resnames))}}


def _run(groups, shell=None, temperature=298.0):
    run = {"args": {"temperature": temperature},
           "groups": {g: {"components": c} for g, c in groups.items()}}
    if shell is not None:
        run["water_entropy"] = shell
    return run


def _write(tmp_path, name, run):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(run))
    return str(path)


CHOICES = {
    "water_orientational_field": "S_orient_non_hb_Nc",
    "bulk_water_source": "mcc_water_group",
    "unbound_ligand_orientational": "codeentropy",
    "bound_ligand_orientational_J_per_molK": -3.0,
    "ligand_positional": "paper_1M",
    "shell_scope": "all",
    "replicates": "independent",
    "population_tolerance": 1e-9,
}


@pytest.fixture
def config(tmp_path):
    complex_run = _run(
        {"0": {"polymer:FTmat-Transvibrational": 100.0, "polymer:Orientational": 999.0},
         "1": {"residue:FTmat-Transvibrational": 50.0, "residue:Orientational": 888.0},
         "2": {"united_atom:Transvibrational": 12345.0}},
        _shell([("MET", 1, 10.0, 40.0, 15.0, 5.0), ("LIG", 2, 2.0, 45.0, 16.0, 7.0)]),
    )
    receptor_run = _run(
        {"0": {"polymer:FTmat-Transvibrational": 101.0, "polymer:Orientational": 777.0}},
        _shell([("MET", 1, 13.0, 41.0, 15.0, 5.0)], resnames=ROLE_RESNAMES["receptor"]),
    )
    ligand_run = _run(
        {"0": {"residue:FTmat-Transvibrational": 52.0, "residue:Orientational": 20.0}},
        _shell([("LIG", 1, 4.0, 46.0, 16.0, 6.0)], resnames=ROLE_RESNAMES["ligand"]),
    )
    water_runs = [
        _run({"0": {"united_atom:Transvibrational": 47.0, "united_atom:Rovibrational": 17.0,
                    "united_atom:Orientational": 11.0}}),
        _run({"0": {"united_atom:Transvibrational": 48.0, "united_atom:Rovibrational": 17.0,
                    "united_atom:Orientational": 11.0}}),
    ]
    return {
        "ligand_resname": "LIG",
        "receptor_resnames": ["MET"],
        "complex": {"runs": [_write(tmp_path, "c", complex_run)],
                    "receptor_group": "0", "ligand_group": "1"},
        "receptor": {"runs": [_write(tmp_path, "r", receptor_run)], "receptor_group": "0"},
        "ligand": {"runs": [_write(tmp_path, "l", ligand_run)], "ligand_group": "0"},
        "water": {"runs": [_write(tmp_path, f"w{i}", w) for i, w in enumerate(water_runs)],
                  "water_group": "0"},
        "choices": dict(CHOICES),
    }


def test_delta_s_matches_hand_calculation_and_eq3_bookkeeping(config):
    result = assemble(config)

    s_bulk = 75.5
    assert result["bulk_water_S_per_molecule"]["mean"] == pytest.approx(s_bulk)
    assert result["bulk_water_S_per_molecule"]["sem"] == pytest.approx(0.5)
    assert result["N_water_released"] == pytest.approx(13 + 4 - 12)
    assert result["systems"]["complex"]["X"]["mean"] == pytest.approx(-23.0)
    assert result["systems"]["receptor"]["X"]["mean"] == pytest.approx(-87.5)
    assert result["systems"]["ligand"]["X"]["mean"] == pytest.approx(42.0 + S_POS)

    bound = 100 + 50 - 3 + (10 * 60 + 2 * 68) + 5 * s_bulk
    unbound = (101 + 13 * 61) + (52 + S_POS + 20 + 4 * 68)
    assert result["delta_S_J_per_molK"]["mean"] == pytest.approx(bound - unbound)
    assert result["T_delta_S"]["mean"]["kJ_per_mol"] == pytest.approx(
        298.0 * (bound - unbound) / 1000
    )


def test_paper_eq7_ligand_orientational_uses_ligand_shell_count(config):
    config["choices"]["unbound_ligand_orientational"] = {
        "method": "paper_eq7", "symmetry_number": 1, "p_corr": 1.0,
    }
    s_or = GAS_CONSTANT * math.log(4.0**1.5 * math.pi**0.5)

    result = assemble(config)

    assert result["systems"]["ligand"]["X"]["mean"] == pytest.approx(22.0 + S_POS + s_or)


def test_selected_water_orientational_field_is_used(config):
    config["choices"]["water_orientational_field"] = "S_orient_hb_biased"

    result = assemble(config)

    assert result["systems"]["receptor"]["X"]["mean"] == pytest.approx(-87.5 - 13 * 6.0)


@pytest.mark.parametrize("choice", sorted(CHOICES))
def test_every_scientific_choice_is_required(config, choice):
    del config["choices"][choice]

    with pytest.raises(BindingEntropyError, match="missing required choices"):
        assemble(config)


def test_population_mismatch_between_vibrational_and_orientational_is_rejected(
    config, tmp_path
):
    run = json.loads(open(config["receptor"]["runs"][0]).read())
    run["water_entropy"]["9"]["orientational"][0]["mean_waters"] = 12.0
    config["receptor"]["runs"] = [_write(tmp_path, "bad", run)]

    with pytest.raises(BindingEntropyError, match="population mismatch"):
        assemble(config)


def test_solvated_run_without_export_is_rejected(config, tmp_path):
    run = json.loads(open(config["ligand"]["runs"][0]).read())
    del run["water_entropy"]
    config["ligand"]["runs"] = [_write(tmp_path, "noexport", run)]

    with pytest.raises(BindingEntropyError, match="water_entropy export"):
        assemble(config)


def test_shell_scope_include_keeps_listed_residues_and_ligand(config):
    config["choices"]["shell_scope"] = {"include": []}

    result = assemble(config)

    assert result["systems"]["complex"]["n_ws"]["mean"] == pytest.approx(2.0)
    assert result["systems"]["receptor"]["n_ws"]["mean"] == pytest.approx(0.0)


def test_paired_and_independent_sems(config, tmp_path):
    def replicate(path, delta):
        run = json.loads(open(path).read())
        for comps in run["groups"].values():
            for key in comps["components"]:
                comps["components"][key] += delta
        return run

    for system, gid_key in (("complex", "receptor_group"), ("receptor", "receptor_group"),
                            ("ligand", "ligand_group")):
        base = config[system]["runs"][0]
        config[system]["runs"] = [
            _write(tmp_path, f"{system}{i}", replicate(base, d)) for i, d in enumerate((0, 2))
        ]

    independent = assemble(copy.deepcopy(config))
    config["choices"]["replicates"] = "paired"
    paired = assemble(config)

    assert independent["delta_S_J_per_molK"]["mean"] == pytest.approx(
        paired["delta_S_J_per_molK"]["mean"]
    )
    n_wb = 5.0
    # Replicate 2 adds 2 to every component. Counted shifts in X: complex +4 (receptor
    # and ligand non-orientational), receptor +2, ligand +4 (non-orientational plus its
    # orientational term in 'codeentropy' mode). SEM of two values a, a+s is s/2.
    sem_c, sem_r, sem_l = 2.0, 1.0, 2.0
    assert independent["delta_S_J_per_molK"]["sem"] == pytest.approx(
        math.sqrt(sem_c**2 + sem_r**2 + sem_l**2 + (n_wb * 0.5) ** 2)
    )
    diffs_sem = abs((4 - 2 - 4) / 2)
    assert paired["delta_S_J_per_molK"]["sem"] == pytest.approx(
        math.sqrt(diffs_sem**2 + (n_wb * 0.5) ** 2)
    )


def test_example_config_placeholders_are_all_rejected():
    from pathlib import Path

    from CodeEntropy.results.binding_entropy import validate_choices

    example = Path(__file__).resolve().parents[4] / "examples" / "binding_entropy_config.example.json"
    placeholders = json.loads(example.read_text())["choices"]
    for name, value in placeholders.items():
        if isinstance(value, str) and value.startswith("CHOOSE"):
            choices = dict(CHOICES)
            choices[name] = value
            with pytest.raises(BindingEntropyError):
                validate_choices(choices)


@pytest.mark.parametrize(
    "name, bad",
    [
        ("bound_ligand_orientational_J_per_molK", "0"),
        ("bound_ligand_orientational_J_per_molK", float("nan")),
        ("ligand_positional", "1M"),
        ("unbound_ligand_orientational", {"method": "paper_eq7", "symmetry_number": 0,
                                          "p_corr": 1.0}),
        ("shell_scope", {"include": "MET_1"}),
        ("population_tolerance", -1),
    ],
)
def test_invalid_choice_values_are_rejected(config, name, bad):
    config["choices"][name] = bad

    with pytest.raises(BindingEntropyError):
        assemble(config)


def test_cli_reports_errors_without_traceback(tmp_path):
    from CodeEntropy.results.binding_entropy import main

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"choices": {}}))

    with pytest.raises(SystemExit, match="^error: missing required choices"):
        main([str(bad)])


def _bulk_run(s_trans, s_rot=17.0, s_orient=11.0, n=100.0, non_water=()):
    export = {
        "n_frames": 5,
        "input_composition": _composition(list(non_water), n_water=int(n)),
        "vibrational": [{"solute_key": "SOL", "nearest_resname": "SOL", "nearest_resid": "SOL",
                         "solvent_resname": "SOL", "S_trans": s_trans, "S_rot": s_rot,
                         "observations": n * 5, "mean_waters": n}],
        "orientational": [{"nearest_resname": "SOL", "nearest_resid": "SOL",
                           "observations": n * 5, "mean_waters": n,
                           "S_orient_non_hb_Nc": s_orient, "S_orient_hb_biased": 0.0,
                           "S_orient_non_hb_Nw": s_orient}],
    }
    return {"args": {"temperature": 298.0}, "groups": {}, "water_entropy_bulk": {"0": export}}


def test_waterentropy_bulk_export_source_matches_same_per_molecule_values(config, tmp_path):
    config["choices"]["bulk_water_source"] = "waterentropy_bulk_export"
    config["water"] = {"runs": [_write(tmp_path, "b0", _bulk_run(47.0)),
                                _write(tmp_path, "b1", _bulk_run(48.0))]}

    result = assemble(config)

    assert result["bulk_source_matched_kernel"] is True
    assert result["bulk_water_S_per_molecule"]["mean"] == pytest.approx(75.5)
    assert result["systems"]["complex"]["X"]["mean"] == pytest.approx(-23.0)


def test_mcc_bulk_source_is_flagged_as_not_matched_kernel(config):
    assert assemble(config)["bulk_source_matched_kernel"] is False


def test_undefined_bulk_sem_propagates_as_nan(config, tmp_path):
    config["water"]["runs"] = config["water"]["runs"][:1]
    for system in ("complex", "receptor", "ligand"):
        config[system]["runs"] = config[system]["runs"] * 2

    result = assemble(config)

    assert math.isnan(result["bulk_water_S_per_molecule"]["sem"])
    assert math.isnan(result["delta_S_J_per_molK"]["sem"])


def _mutate_shell(config, tmp_path, system, name, mutate):
    run = json.loads(open(config[system]["runs"][0]).read())
    mutate(run["water_entropy"]["9"])
    config[system]["runs"] = [_write(tmp_path, name, run)]


def test_duplicate_shell_entries_are_rejected(config, tmp_path):
    _mutate_shell(config, tmp_path, "receptor", "dup",
                  lambda ex: ex["vibrational"].append(dict(ex["vibrational"][0])))

    with pytest.raises(BindingEntropyError, match="duplicate vibrational entry"):
        assemble(config)


def test_multiple_solvent_populations_are_rejected(config, tmp_path):
    def add_other_solvent(ex):
        other = dict(ex["vibrational"][0], solvent_resname="MOH", nearest_resid="2")
        ex["vibrational"].append(other)

    _mutate_shell(config, tmp_path, "receptor", "solv", add_other_solvent)

    with pytest.raises(BindingEntropyError, match="multiple solvent populations"):
        assemble(config)


def test_ambiguous_residue_keys_in_scope_are_rejected(config, tmp_path):
    _mutate_shell(config, tmp_path, "receptor", "amb",
                  lambda ex: ex.__setitem__("ambiguous_residue_keys", ["MET_1"]))

    with pytest.raises(BindingEntropyError, match="identify more than one residue"):
        assemble(config)


def test_ambiguous_residue_keys_outside_scope_are_allowed(config, tmp_path):
    _mutate_shell(config, tmp_path, "receptor", "amb2",
                  lambda ex: ex.__setitem__("ambiguous_residue_keys", ["MET_1"]))
    config["choices"]["shell_scope"] = {"include": []}

    assert assemble(config)["systems"]["receptor"]["n_ws"]["mean"] == pytest.approx(0.0)


def test_bulk_export_from_input_with_solute_is_rejected(config, tmp_path):
    config["choices"]["bulk_water_source"] = "waterentropy_bulk_export"
    config["water"] = {"runs": [_write(tmp_path, "b0", _bulk_run(47.0)),
                                _write(tmp_path, "bmix", _bulk_run(48.0, non_water=("MET", "LIG")))]}

    with pytest.raises(BindingEntropyError, match="water-only simulation"):
        assemble(config)


def test_export_without_input_composition_is_rejected(config, tmp_path):
    _mutate_shell(config, tmp_path, "receptor", "noprov",
                  lambda ex: ex.pop("input_composition"))

    with pytest.raises(BindingEntropyError, match="no input_composition"):
        assemble(config)


@pytest.mark.parametrize(
    "role, resnames",
    [("receptor", ["MET", "LIG"]), ("complex", ["MET"]), ("complex", ["LIG"]),
     ("ligand", ["MET"])],
)
def test_runs_whose_input_does_not_match_their_role_are_rejected(
    config, tmp_path, role, resnames
):
    _mutate_shell(config, tmp_path, role, f"wrong_{role}",
                  lambda ex: ex.__setitem__("input_composition", _composition(resnames)))

    with pytest.raises(BindingEntropyError, match=f"listed as a {role} run"):
        assemble(config)


def _set_composition(config, tmp_path, role, resnames):
    _mutate_shell(config, tmp_path, role, f"comp_{role}_{'_'.join(resnames) or 'none'}",
                  lambda ex: ex.__setitem__("input_composition", _composition(resnames)))


@pytest.mark.parametrize(
    "role, resnames",
    [("receptor", ["NA", "CL"]), ("complex", ["LIG", "NA"]), ("ligand", ["LIG", "MET"])],
)
def test_role_uses_explicit_receptor_names_not_any_other_residue(
    config, tmp_path, role, resnames
):
    _set_composition(config, tmp_path, role, resnames)

    with pytest.raises(BindingEntropyError, match=f"listed as a {role} run"):
        assemble(config)


def test_unlisted_ions_are_ignored_for_roles(config, tmp_path):
    _set_composition(config, tmp_path, "complex", ["MET", "LIG", "NA", "NA"])
    _set_composition(config, tmp_path, "receptor", ["MET", "NA"])
    _set_composition(config, tmp_path, "ligand", ["LIG", "CL"])

    assert assemble(config)["delta_S_J_per_molK"]["mean"] == pytest.approx(22.5 - S_POS)


def test_receptor_residue_count_must_match_between_complex_and_receptor(config, tmp_path):
    _set_composition(config, tmp_path, "complex", ["MET", "MET", "LIG"])

    with pytest.raises(BindingEntropyError, match="receptor residue count differs"):
        assemble(config)


def test_ligand_residue_count_must_match_between_complex_and_ligand(config, tmp_path):
    _set_composition(config, tmp_path, "ligand", ["LIG", "LIG"])

    with pytest.raises(BindingEntropyError, match="ligand residue count differs"):
        assemble(config)


@pytest.mark.parametrize(
    "value", [None, [], "MET", ["CHOOSE: names"], ["MET", "LIG"], ["MET", 3]]
)
def test_receptor_resnames_must_be_an_explicit_valid_list(config, value):
    if value is None:
        del config["receptor_resnames"]
    else:
        config["receptor_resnames"] = value

    with pytest.raises(BindingEntropyError, match="receptor_resnames"):
        assemble(config)


def test_composition_without_residue_counts_is_rejected(config, tmp_path):
    def drop_counts(ex):
        ex["input_composition"].pop("non_water_resname_counts")

    _mutate_shell(config, tmp_path, "receptor", "nocounts", drop_counts)

    with pytest.raises(BindingEntropyError, match="non_water_resname_counts"):
        assemble(config)
